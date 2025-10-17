"""
InternVLA-M1 Agent for InternManip framework.

This agent handles observation conversion, action prediction, and ensemble logic
for InternVLA-M1 model evaluation.
"""

from collections import deque
import json
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
import torch
from PIL import Image
from typing import List, Dict, Any

from huggingface_hub import snapshot_download

from internmanip.agent.base import BaseAgent
from internmanip.configs import AgentCfg
from internmanip.configs.dataset.data_config import DATA_CONFIG_MAP
from internmanip.dataset.embodiment_tags import EmbodimentTag
from internmanip.dataset.schema import DatasetMetadata
from internmanip.dataset.transform.base import ComposedModalityTransform
from internmanip.utils.agent_utils.io_utils import unsqueeze_dict_values, squeeze_dict_values


class InternVLAM1Agent(BaseAgent):
    """
    Agent for InternVLA-M1 model that handles:
    - Converting environment observations to model input format
    - Predicting actions using InternVLA-M1
    - Converting model output back to environment action format
    - Action ensembling for smooth execution
    """
    
    def __init__(self, config: AgentCfg):
        super().__init__(config)
        
        if torch.cuda.is_available():
            self.policy_model = self.policy_model.cuda()
        
        # Get transform configuration
        model_transform, observation_indices, action_indices = self.policy_model.config.transform()
        
        # Setup data config and transforms
        self.data_config = config.agent_settings['data_config']
        data_config_cls = DATA_CONFIG_MAP[self.data_config]
        transforms = data_config_cls.transform()
        
        if model_transform is not None:
            transforms.append(model_transform)
        
        self.transforms = ComposedModalityTransform(transforms=transforms)
        self.transforms.eval()
        
        # Setup embodiment tag
        self.embodiment_tag = EmbodimentTag(config.agent_settings['embodiment_tag'])
        self._load_metadata(config)
        
        # Action prediction settings
        self.pred_action_horizon = config.agent_settings.get('pred_action_horizon', 16)
        self.action_ensemble = config.agent_settings.get('action_ensemble', False)
        self.adaptive_ensemble_alpha = config.agent_settings.get('adaptive_ensemble_alpha', 0.5)
        
        # Ensemble tracking
        self.ensembler_list = []
        
        # Evaluation type (e.g., 'realarx', 'genmanip')
        self.eval_type = config.agent_settings.get('eval_type', 'genmanip')
    
    def step(self, inputs: List[Dict]) -> List[Dict]:
        """
        Process a batch of observations and return actions.
        
        Args:
            inputs: List of observation dicts from environment
            
        Returns:
            List of action dicts for environment
        """
        outputs = []
        
        # Ensure we have enough ensemblers
        while len(self.ensembler_list) < len(inputs):
            if self.action_ensemble:
                self.ensembler_list.append(
                    AdaptiveEnsembler(
                        pred_action_horizon=self.pred_action_horizon,
                        adaptive_ensemble_alpha=self.adaptive_ensemble_alpha,
                    )
                )
            else:
                self.ensembler_list.append(None)
        
        # Process each environment
        for env_idx, input_obs in enumerate(inputs):
            if input_obs == {}:
                outputs.append({})
                continue
            
            # Reset ensembler at start of episode
            if input_obs['robot']['step'] == 0:
                if self.action_ensemble:
                    self.ensembler_list[env_idx].reset()
                print(f'Episode started with instruction: {input_obs["robot"]["instruction"]}')
            
            # Convert observation to model input format
            converted_input = self.convert_input(input_obs)
            
            # Predict action
            if self.action_ensemble:
                # Predict every step and ensemble
                model_pred = self.inference(converted_input)
                pred_action = self.ensembler_list[env_idx].ensemble_action(model_pred[0])
                pred_action = torch.tensor(pred_action)
            else:
                # Predict every N steps and cache
                if input_obs['robot']['step'] % self.pred_action_horizon == 0:
                    model_pred = self.inference(converted_input)
                    self.ensembler_list[env_idx] = model_pred[0]
                pred_action = self.ensembler_list[env_idx][
                    input_obs['robot']['step'] % self.pred_action_horizon
                ]
            
            # Denormalize and convert output
            unnormalized_output = self.transforms.unapply({'action': pred_action})
            squeezed_output = squeeze_dict_values(unnormalized_output)
            converted_output = self.convert_output(squeezed_output, converted_input)
            outputs.append(converted_output)
        
        return outputs
    
    def reset(self):
        """Reset the agent state."""
        self.ensembler_list = []
    
    def inference(self, input_data: Dict) -> torch.Tensor:
        """
        Run model inference on input data.
        
        Args:
            input_data: Dict with 'image' (List[PIL.Image]) and 'lang' (str)
            
        Returns:
            Predicted actions as torch.Tensor [T, action_dim]
        """
        # Prepare batch format
        batch_images = [input_data['image']]  # Batch of 1
        batch_lang = [input_data['lang']]
        
        # Run inference
        with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            result = self.policy_model.inference(image=batch_images, lang=batch_lang)
            action_pred = result['action_pred'].float().cpu()
        
        return action_pred
    
    def convert_input(self, obs: Dict) -> Dict:
        """
        Convert environment observation to model input format.
        
        Args:
            obs: Environment observation dict
            
        Returns:
            Dict with 'image' and 'lang' keys in model format
        """
        if self.data_config == 'arx' or self.data_config == 'aloha_v4':
            # For ALOHA robot with multiple cameras
            # Collect images from different camera views
            images = []
            
            # Map camera names (environment may use different names)
            camera_mappings = {
                'left_camera': ['left_camera', 'hand_left'],
                'right_camera': ['right_camera', 'hand_right'],
                'top_camera': ['top_camera', 'head'],
            }
            
            for cam_key, possible_names in camera_mappings.items():
                img = None
                for name in possible_names:
                    if name in obs['robot']['sensors']:
                        img = obs['robot']['sensors'][name]['rgb']
                        break
                
                if img is not None:
                    # Convert to PIL Image if needed
                    if isinstance(img, np.ndarray):
                        img = Image.fromarray(img.astype(np.uint8))
                    images.append(img)
            
            converted = {
                'image': images,
                'lang': obs['robot']['instruction'],
            }
            
        elif self.data_config == 'genmanip_v1':
            # For GenManip with 3 camera views
            images = []
            
            camera_keys = ['realsense', 'obs_camera', 'obs_camera_2']
            for cam_key in camera_keys:
                if cam_key in obs['robot']['sensors']:
                    img = obs['robot']['sensors'][cam_key]['rgb']
                    if isinstance(img, np.ndarray):
                        img = Image.fromarray(img.astype(np.uint8))
                    images.append(img)
            
            converted = {
                'image': images,
                'lang': obs['robot']['instruction'],
            }
        else:
            raise ValueError(f'Unsupported data config: {self.data_config}')
        
        return converted
    
    def convert_output(self, output: Dict, input_data: Dict) -> Dict:
        """
        Convert model output to environment action format.
        
        Args:
            output: Model output dict with action components
            input_data: Original input data for context
            
        Returns:
            Action dict in environment format
        """
        if self.data_config == 'arx' or self.data_config == 'aloha_v4':
            # For ALOHA dual-arm robot
            # InternVLA-M1 outputs normalized actions that need to be mapped
            # to left/right arm joint positions and gripper commands
            
            # Extract action components (assuming the model outputs all joints)
            # This is a simplified version - adjust based on actual output format
            action = output.get('action', None)
            
            if action is None:
                # Try to reconstruct from individual components
                left_arm = output.get('action.left_arm_delta_qpos', np.zeros(6))
                right_arm = output.get('action.right_arm_delta_qpos', np.zeros(6))
                left_gripper = output.get('action.left_gripper', 0.5)
                right_gripper = output.get('action.right_gripper', 0.5)
            else:
                # Split the action tensor
                left_arm = action[:6]
                right_arm = action[6:12] if len(action) >= 12 else action[:6]
                left_gripper = action[12] if len(action) > 12 else 0.5
                right_gripper = action[13] if len(action) > 13 else 0.5
            
            converted = {
                'left_arm_action': left_arm.tolist() if hasattr(left_arm, 'tolist') else list(left_arm),
                'right_arm_action': right_arm.tolist() if hasattr(right_arm, 'tolist') else list(right_arm),
                'left_gripper_action': float(left_gripper) * 2 - 1,  # Convert [0,1] to [-1,1]
                'right_gripper_action': float(right_gripper) * 2 - 1,
            }
            
        elif self.data_config == 'genmanip_v1':
            # For GenManip with end-effector control
            # Extract delta pose and gripper action
            delta_ee_pos = output.get('action.delta_ee_pos', np.zeros(3))
            delta_ee_rot = output.get('action.delta_ee_rot', np.zeros(3))
            gripper = output.get('action.gripper', 0.5)
            
            # Apply deltas (would need current state in full implementation)
            converted = {
                'eef_position': delta_ee_pos.tolist() if hasattr(delta_ee_pos, 'tolist') else list(delta_ee_pos),
                'eef_orientation': [1, 0, 0, 0],  # Placeholder - would need proper rotation handling
                'gripper_action': float(gripper) * 2 - 1,
            }
        else:
            raise ValueError(f'Unsupported data config: {self.data_config}')
        
        return converted
    
    def _load_metadata(self, config: AgentCfg):
        """Load normalization metadata from model checkpoint."""
        if Path(config.base_model_path).exists():
            metadata_path = Path(config.base_model_path) / 'experiment_cfg' / 'metadata.json'
        else:
            snapshot_path = snapshot_download(
                repo_id=config.base_model_path,
                cache_dir=config.model_kwargs['HF_cache_dir'],
                local_files_only=True,
                allow_patterns='experiment_cfg/metadata.json'
            )
            metadata_path = Path(snapshot_path) / 'experiment_cfg' / 'metadata.json'
        
        with open(metadata_path, 'r') as f:
            metadatas = json.load(f)
        
        # Get metadata for the specific embodiment
        metadata_dict = metadatas.get(self.embodiment_tag.value)
        if metadata_dict is None:
            print(f'Warning: No metadata found for embodiment tag: {self.embodiment_tag.value}')
            print(f'Available tags: {list(metadatas.keys())}')
            # Use default metadata if available
            if metadatas:
                metadata_dict = list(metadatas.values())[0]
            else:
                return
        
        # Convert lists to numpy arrays
        def convert_lists_to_arrays(obj):
            if isinstance(obj, list):
                return np.array(obj)
            if isinstance(obj, dict):
                return {k: convert_lists_to_arrays(v) for k, v in obj.items()}
            return obj
        
        metadata_dict = convert_lists_to_arrays(metadata_dict)
        metadata = DatasetMetadata.model_validate(metadata_dict)
        self.transforms.set_metadata(metadata)
        self.metadata = metadata


class AdaptiveEnsembler:
    """
    Adaptive action ensembler that combines predictions using cosine similarity weighting.
    """
    
    def __init__(self, pred_action_horizon: int, adaptive_ensemble_alpha: float = 0.0):
        self.pred_action_horizon = pred_action_horizon
        self.action_history = deque(maxlen=self.pred_action_horizon)
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
    
    def reset(self):
        """Clear action history."""
        self.action_history.clear()
    
    def ensemble_action(self, cur_action: np.ndarray) -> np.ndarray:
        """
        Ensemble current action with history using adaptive weighting.
        
        Args:
            cur_action: Current predicted action [T, action_dim] or [action_dim]
            
        Returns:
            Ensembled action for current timestep
        """
        self.action_history.append(cur_action)
        num_actions = len(self.action_history)
        
        if cur_action.ndim == 1:
            curr_act_preds = np.stack(list(self.action_history))
        else:
            curr_act_preds = np.stack([
                pred_actions[i]
                for i, pred_actions in zip(range(num_actions - 1, -1, -1), self.action_history)
            ])
        
        # Calculate cosine similarity with current prediction
        ref = curr_act_preds[num_actions - 1, :]
        previous_pred = curr_act_preds
        
        dot_product = np.sum(previous_pred * ref, axis=1)
        norm_previous_pred = np.linalg.norm(previous_pred, axis=1)
        norm_ref = np.linalg.norm(ref)
        cos_similarity = dot_product / (norm_previous_pred * norm_ref + 1e-7)
        
        # Compute weights
        weights = np.exp(self.adaptive_ensemble_alpha * cos_similarity)
        weights = weights / weights.sum()
        
        # Weighted average
        ensembled = np.sum(weights[:, None] * curr_act_preds, axis=0)
        return ensembled
