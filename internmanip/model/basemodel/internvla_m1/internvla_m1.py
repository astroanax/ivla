"""
InternVLA-M1: Spatially Guided Vision-Language-Action Framework

This module adapts the InternVLA-M1 model for use within the InternManip framework.
It integrates:
- Qwen2.5 VL backbone for vision-language understanding
- DINO encoder for dense multi-view spatial features  
- Layer-wise QFormer for multi-layer feature aggregation
- DiT diffusion head for action prediction

Reference: https://github.com/InternRobotics/InternVLA-M1
"""

import sys
import os
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from functools import partial
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from PIL import Image

# Add InternVLA-M1 to path if available
internvla_path = Path(__file__).parent.parent.parent.parent.parent / 'InternVLA-M1'
if internvla_path.exists():
    sys.path.insert(0, str(internvla_path))

try:
    from InternVLA.model.framework.M1 import InternVLA_M1 as InternVLA_M1_Base
    from InternVLA.model.modules.vlm.QWen2_5 import get_qwen2_5_interface
    from InternVLA.model.modules.projector.QFormer import get_layerwise_qformer
    from InternVLA.model.modules.action_model.DiTActionHeader import get_action_model
    from InternVLA.model.modules.dino_model.dino import get_dino_model
    from InternVLA.training.trainer_utils.metrics import resize_images
    INTERNVLA_AVAILABLE = True
except ImportError as e:
    print(f"Warning: InternVLA-M1 modules not available: {e}")
    print("Please ensure InternVLA-M1 is in the parent directory or PYTHONPATH")
    INTERNVLA_AVAILABLE = False

from internmanip.model.basemodel.base import BasePolicyModel
from internmanip.configs.model.internvla_m1_cfg import InternVLA_M1_Config
from internmanip.model.data_collator_registry import DataCollatorRegistry


def collate_internvla_m1(features: List[dict]) -> dict:
    """
    Collate function for InternVLA-M1 batching.
    
    Args:
        features: List of dicts containing 'image', 'lang', and optionally 'action'
        
    Returns:
        Batched dictionary ready for model input
    """
    batch = {}
    
    # Collect images and language
    batch['image'] = [f['image'] for f in features]  # List of List[PIL.Image]
    batch['lang'] = [f['lang'] for f in features]    # List of str
    
    # Collect actions if present (training mode)
    if 'action' in features[0]:
        actions = [f['action'] for f in features]
        batch['action'] = actions  # Will be converted to tensor in model
    
    return batch


class InternVLA_M1(BasePolicyModel):
    """
    InternVLA-M1 model wrapper for InternManip framework.
    
    This class wraps the original InternVLA-M1 implementation and provides
    the interface required by InternManip for training and inference.
    """
    
    config_class = InternVLA_M1_Config
    
    def __init__(self, config: InternVLA_M1_Config, **kwargs):
        """
        Initialize InternVLA-M1 model.
        
        Args:
            config: InternVLA_M1_Config instance
            **kwargs: Additional arguments
        """
        super().__init__(config, **kwargs)
        
        if not INTERNVLA_AVAILABLE:
            raise ImportError(
                "InternVLA-M1 modules are required but not available. "
                "Please install InternVLA-M1 dependencies and ensure the "
                "InternVLA-M1 repository is accessible."
            )
        
        self.config = config
        
        # Convert InternManip config to InternVLA format
        internvla_config = self._build_internvla_config()
        
        # Initialize InternVLA-M1 base model
        self.model = InternVLA_M1_Base(config=internvla_config)
        
        # Store configuration parameters
        self.future_action_window_size = config.action_horizon
        self.num_ddim_steps = config.num_ddim_steps
        self.cfg_scale = config.cfg_scale
        self.use_ddim = config.use_ddim
        self.image_size = config.image_size
    
    def _build_internvla_config(self):
        """
        Convert InternManip config to InternVLA config format.
        """
        from omegaconf import OmegaConf
        
        # Build nested config structure expected by InternVLA
        internvla_cfg = {
            'framework': {
                'qwenvl': self.config.framework_cfg.get('qwen_vl', {}),
                'dino': self.config.framework_cfg.get('dino', {}),
                'layer_qformer': self.config.framework_cfg.get('layer_qformer', {}),
                'action_model': self.config.framework_cfg.get('action_model', {}),
            },
            'datasets': {
                'vla_data': {
                    'image_size': self.config.image_size,
                }
            },
            'trainer': {
                'repeated_diffusion_steps': 4,
            }
        }
        
        return OmegaConf.create(internvla_cfg)
    
    def forward(
        self,
        image: List[List[Image.Image]],
        lang: List[str],
        action: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.
        
        Args:
            image: List[List[PIL.Image]] - batch of multi-view images
            lang: List[str] - batch of language instructions
            action: Optional[torch.Tensor] - ground truth actions [B, T, action_dim]
            
        Returns:
            Dict containing 'action_loss'
        """
        # Prepare examples in InternVLA format
        examples = []
        for i in range(len(image)):
            example = {
                'image': image[i],
                'lang': lang[i],
            }
            if action is not None:
                if isinstance(action, list):
                    example['action'] = action[i]
                else:
                    example['action'] = action[i].cpu().numpy()
            examples.append(example)
        
        # Call InternVLA-M1 forward
        return self.model.forward(examples=examples)
    
    @torch.inference_mode()
    def inference(
        self,
        image: List[List[Image.Image]],
        lang: List[str],
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Inference method for action prediction.
        
        Args:
            image: List[List[PIL.Image]] - batch of multi-view images
            lang: List[str] - batch of language instructions
            
        Returns:
            Dict containing 'action_pred' with shape [B, T, action_dim]
        """
        # Call InternVLA-M1 predict_action
        result = self.model.predict_action(
            batch_images=image,
            instructions=lang,
            cfg_scale=self.cfg_scale,
            use_ddim=self.use_ddim,
            num_ddim_steps=self.num_ddim_steps,
            resize_image=self.image_size,
        )
        
        # Extract normalized actions and convert to tensor
        normalized_actions = result['normalized_actions']
        action_pred = torch.from_numpy(normalized_actions)
        
        return {'action_pred': action_pred}
    
    def get_action(self, observations: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Compatibility method for direct action prediction.
        
        Args:
            observations: Dict with 'image' and 'lang' keys
            
        Returns:
            Dict containing 'action_pred'
        """
        image = observations.get('image')
        lang = observations.get('lang')
        
        if not isinstance(image, list):
            image = [image]
        if not isinstance(lang, list):
            lang = [lang]
        
        return self.inference(image=image, lang=lang)


# Register the model with transformers AutoModel
AutoConfig.register('internvla_m1', InternVLA_M1_Config)
AutoModel.register(InternVLA_M1_Config, InternVLA_M1)

# Register the collator
DataCollatorRegistry.register_fn(
    InternVLA_M1_Config.model_type,
    collate_internvla_m1
)


def build_internvla_m1_model(config: InternVLA_M1_Config, **kwargs) -> InternVLA_M1:
    """
    Factory function to build InternVLA-M1 model.
    
    Args:
        config: InternVLA_M1_Config instance
        **kwargs: Additional model arguments
        
    Returns:
        InternVLA_M1 model instance
    """
    return InternVLA_M1(config=config, **kwargs)
