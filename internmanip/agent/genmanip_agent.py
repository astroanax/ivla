from collections import deque
from copy import deepcopy
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from scipy.spatial.transform import Rotation
import torch

from huggingface_hub import snapshot_download

from internmanip.agent.base import BaseAgent
from internmanip.utils.agent_utils.io_utils import unsqueeze_dict_values, squeeze_dict_values
from internmanip.configs import AgentCfg
from internmanip.configs.dataset.data_config import DATA_CONFIG_MAP
from internmanip.dataset.embodiment_tags import EmbodimentTag
from internmanip.dataset.schema import DatasetMetadata
from internmanip.dataset.transform.base import ComposedModalityTransform


class GenmanipAgent(BaseAgent):
    def __init__(self, config: AgentCfg):
        super().__init__(config)
        self.policy_model.compute_dtype = 'bfloat16'
        self.policy_model.config.compute_dtype = 'bfloat16'
        self.policy_model = self.policy_model.to(torch.bfloat16)
        if torch.cuda.is_available():
            self.policy_model = self.policy_model.cuda()

        model_transform, observation_indices, action_indices = self.policy_model.config.transform()
        self.data_config = config.agent_settings['data_config']
        data_config_cls = DATA_CONFIG_MAP[self.data_config]
        transforms = data_config_cls.transform()
        self.action_transforms = transforms[-2]
        if model_transform is not None:
            transforms.append(model_transform)
        self.transforms = ComposedModalityTransform(transforms=transforms)
        self.transforms.eval()
        self.embodiment_tag = EmbodimentTag(config.agent_settings['embodiment_tag'])
        self._load_metadata(config)

        self.pred_action_horizon = config.agent_settings['pred_action_horizon']
        self.adaptive_ensemble_alpha = config.agent_settings['adaptive_ensemble_alpha']
        self.ensembler_list = []

        self.episode_count = []
        self.step_count = []
        # self.save_folder = ""
        # self.output_history_list = []

    def step(self, inputs: list[dict]) -> list[dict]:
        while len(self.ensembler_list) < len(inputs):
            self.ensembler_list.append(
                AdaptiveEnsembler(
                    pred_action_horizon=self.pred_action_horizon,
                    adaptive_ensemble_alpha=self.adaptive_ensemble_alpha,
                )
            )
            self.episode_count.append(0)
            self.step_count.append(0)

        outputs = []
        # GenManip has only one environment, so this loop will iterate only once
        for env, input in enumerate(inputs):
            if input == {}:
                outputs.append({})
                continue

            if input['robot']['step'] == 0:
                self.reset_env(env)
            self.step_count[env] = input['robot']['step']

            converted_input = self.convert_input(input)
            unsqueezed_input = unsqueeze_dict_values(converted_input)
            transformed_input = self.transforms(unsqueezed_input)
            pred_actions = self.policy_model.inference(transformed_input)['action_pred'][0].cpu().float()
            output = self.ensembler_list[env].ensemble_action(pred_actions)
            converted_output = self.convert_output(output, converted_input)
            outputs.append(converted_output)

        # self._debug_print_data(inputs, title=f"Input Data {env}")
        # self._debug_print_data(outputs, title=f"Output Data {env}")
        # self._record_outputs_data(outputs)
        return outputs

    def reset(self):
        for ensembler in self.ensembler_list:
            ensembler.reset()
        self.ensembler_list = []
        self.episode_count = []
        self.step_count = []
        # self.output_history_list = []

    def reset_env(self, env):
        self.ensembler_list[env].reset()
        print(f'Reset env{env}')
        # self.plot_output_history(env)
        self.episode_count[env] += 1
        # self.output_history_list[env] = []

    def convert_input(self, input: dict):
        if self.data_config == 'genmanip_v1':
            quat_wxyz = input['robot']['eef_pose'][1]
            quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
            ee_rot = Rotation.from_quat(quat_xyzw).as_euler('xyz', degrees=False)
            converted_data = {
                'video.ego_view': np.array([input['robot']['sensors']['realsense']['rgb']]),
                'video.base_view': np.array([input['robot']['sensors']['obs_camera']['rgb']]),
                'video.base_2_view': np.array([input['robot']['sensors']['obs_camera_2']['rgb']]),
                'state.joints': np.array([input['robot']['joints_state']['positions'][:7]]),
                'state.gripper': np.array([input['robot']['joints_state']['positions'][7:]]),
                'state.joints_vel': np.array([input['robot']['joints_state']['velocities'][:7]]),
                'state.gripper_vel': np.array([input['robot']['joints_state']['velocities'][7:]]),
                'state.ee_pos': np.array([input['robot']['eef_pose'][0]]),
                'state.ee_rot': np.array([ee_rot]),
                'annotation.human.action.task_description': input['robot']['instruction'],
            }
        elif self.data_config == 'aloha_v3':
            left_arm_joint_indices = [12, 14, 16, 18, 20, 22]
            right_arm_joint_indices = [13, 15, 17, 19, 21, 23]
            left_gripper_joint_indices = [24, 25]
            right_gripper_joint_indices = [26, 27]
            arm_qpos = (input['robot']['joints_state']['positions'][:12].tolist()
                     + [input['robot']['joints_state']['positions'][idx] for idx in left_arm_joint_indices]
                     + [input['robot']['joints_state']['positions'][idx] for idx in left_gripper_joint_indices]
                     + [input['robot']['joints_state']['positions'][idx] for idx in right_arm_joint_indices]
                     + [input['robot']['joints_state']['positions'][idx] for idx in right_gripper_joint_indices])
            converted_data = {
                'video.left_view': np.array([input['robot']['sensors']['left_camera']['rgb']]),
                'video.right_view': np.array([input['robot']['sensors']['right_camera']['rgb']]),
                'video.top_view': np.array([input['robot']['sensors']['top_camera']['rgb']]),
                'state.arm_qpos': np.array([arm_qpos]),
                'annotation.human.action.task_description': input['robot']['instruction'],
            }
        else:
            raise ValueError(f'Unsupported data config class: {self.data_config}')
        return converted_data

    def convert_output(self, output: np.ndarray, input: dict):
        if self.data_config == 'genmanip_v1':
            converted_data = {
                'action.gripper': torch.from_numpy(output[:1]),
                'action.delta_ee_pos': torch.from_numpy(output[1:4]),
                'action.delta_ee_rot': torch.from_numpy(output[4:7]),
            }
            converted_data = self.action_transforms.unapply(deepcopy(converted_data))
            converted_data = squeeze_dict_values(converted_data)
            ee_rot = (converted_data['action.delta_ee_rot'] + input['state.ee_rot'])[0].tolist()
            quat_xyzw = Rotation.from_euler('xyz', ee_rot, degrees=False).as_quat()
            quat_wxyz = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
            eef_position = (converted_data['action.delta_ee_pos'] + input['state.ee_pos'])[0].tolist()
            eef_orientation = quat_wxyz
            gripper_action = converted_data['action.gripper']*2-1
            converted_data = {
                'eef_position': eef_position,
                'eef_orientation': eef_orientation,
                'gripper_action': gripper_action,
            }
        elif self.data_config == 'aloha_v3':
            converted_data = {
                'action.left_arm_delta_qpos': torch.from_numpy(output[:12]),
                'action.right_arm_delta_qpos': torch.from_numpy(output[:12]),
                'action.left_gripper_close': torch.from_numpy(output[12:14]),
                'action.right_gripper_close': torch.from_numpy(output[12:14]),
            }
            converted_data = self.action_transforms.unapply(deepcopy(converted_data))
            converted_data = squeeze_dict_values(converted_data)
            left_arm_action = (converted_data['action.left_arm_delta_qpos'][:6] + torch.tensor(input['state.arm_qpos'][0][12:18])).tolist()
            left_gripper_action = converted_data['action.left_gripper_close'][0]*2-1
            right_arm_action = (converted_data['action.right_arm_delta_qpos'][6:] + torch.tensor(input['state.arm_qpos'][0][20:26])).tolist()
            right_gripper_action = converted_data['action.right_gripper_close'][1]*2-1
            converted_data = {
                'left_arm_action': left_arm_action,
                'left_gripper_action': left_gripper_action,
                'right_arm_action': right_arm_action,
                'right_gripper_action': right_gripper_action,
            }
        else:
            raise ValueError(f'Unsupported data config class: {self.data_config}')
        return converted_data

    def _load_metadata(self, config: AgentCfg):
        # Load metadata for normalization stats
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
            raise ValueError(
                f'No metadata found for embodiment tag: {self.embodiment_tag.value}',
                f'make sure the metadata.json file is present at {metadata_path}',
            )
        else:
            # deserialize the ndarray
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

    def _debug_print_data(self, data, title='Data Debug'):
        print(f'\n=== {title} ===')
        print(data)
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    print(f'{key}: shape={value.shape}, dtype={value.dtype}')
                elif isinstance(value, np.ndarray):
                    print(f'{key}: shape={value.shape}, dtype={value.dtype}')
                elif isinstance(value, list):
                    print(f'{key}: type=list, length={len(value)}')
                else:
                    print(f'{key}: type={type(value)}')
        else:
            if hasattr(data, 'shape'):
                print(f"shape={data.shape}, dtype={getattr(data, 'dtype', 'unknown')}")
            else:
                print(f'type={type(data)}')

    def _record_outputs_data(self, outputs):
        while len(self.output_history_list) < len(outputs):
            self.output_history_list.append([])
        for env, output in enumerate(outputs):
            if output is not None:
                record_data = {
                    'step': self.step_count[env],
                    'arm_action': output['arm_action'],
                    'gripper_action': output['gripper_action']
                }
                self.output_history_list[env].append(record_data)

    def plot_output_history(self, env):
        if not self.output_history_list:
            print('No output history to plot.')
            return
        if not self.output_history_list[env]:
            print(f'No output history for environment {env}.')
            return

        steps = [data['step'] for data in self.output_history_list[env]]
        arm_actions = [data['arm_action'] for data in self.output_history_list[env]]
        arm_actions = np.array(arm_actions)
        gripper_actions = [data['gripper_action'] for data in self.output_history_list[env]]
        gripper_actions = np.array(gripper_actions)

        plt.figure(figsize=(12, 8))
        plt.suptitle(f'Environment {env} - Episode {self.episode_count[env]}', fontsize=16)

        for i in range(7):
            plt.subplot(3, 3, i+1)
            plt.plot(steps, arm_actions[:, i])
            plt.title(f'Joint {i+1}')
            plt.xlabel('Step')
            plt.ylabel('Action')
            plt.grid(True)

        plt.subplot(3, 3, 8)
        plt.plot(steps, gripper_actions, 'r-')
        plt.title('Gripper Action')
        plt.xlabel('Step')
        plt.ylabel('Action')
        plt.grid(True)

        save_folder = self.save_folder + f'/env{env}'
        os.makedirs(save_folder, exist_ok=True)
        save_path = f'{save_folder}/episode{self.episode_count[env]}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Environment {env} plot saved: {save_path}')


class AdaptiveEnsembler:
    def __init__(self, pred_action_horizon, adaptive_ensemble_alpha=0.0):
        self.pred_action_horizon = pred_action_horizon
        self.action_history = deque(maxlen=self.pred_action_horizon)
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha

    def reset(self):
        self.action_history.clear()

    def ensemble_action(self, cur_action):
        self.action_history.append(cur_action)
        num_actions = len(self.action_history)
        if cur_action.ndim == 1:
            curr_act_preds = np.stack(self.action_history)
        else:
            curr_act_preds = np.stack(
                [pred_actions[i] for (i, pred_actions) in zip(range(num_actions - 1, -1, -1), self.action_history)]
            )
        # calculate cosine similarity between the current prediction and all previous predictions
        ref = curr_act_preds[num_actions-1, :]
        previous_pred = curr_act_preds
        dot_product = np.sum(previous_pred * ref, axis=1)
        norm_previous_pred = np.linalg.norm(previous_pred, axis=1)
        norm_ref = np.linalg.norm(ref)
        cos_similarity = dot_product / (norm_previous_pred * norm_ref + 1e-7)
        # compute the weights for each prediction
        weights = np.exp(self.adaptive_ensemble_alpha * cos_similarity)
        weights = weights / weights.sum()
        # compute the weighted average across all predictions for this timestep
        cur_action = np.sum(weights[:, None] * curr_act_preds, axis=0)
        return cur_action
