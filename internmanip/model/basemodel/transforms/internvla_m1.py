"""
Transform class for InternVLA-M1 model.

Handles:
- Multi-view image preprocessing
- Language instruction formatting  
- Action normalization/denormalization
"""

import numpy as np
import torch
from pydantic import Field, PrivateAttr
from typing import Optional, Dict

from internmanip.dataset.transform.base import InvertibleModalityTransform
from internmanip.dataset.schema import DatasetMetadata


class InternVLAM1Transform(InvertibleModalityTransform):
    """
    Transform for InternVLA-M1 model that handles multi-view images,
    language instructions, and action sequences.
    """
    
    apply_to: list[str] = Field(
        default_factory=list,
        description='Not used in this transform, kept for compatibility.'
    )
    
    training: bool = Field(
        default=True,
        description='Whether to apply the transform in training mode.'
    )
    
    state_horizon: int = Field(
        default=1,
        description='Number of state observations to use.'
    )
    
    action_horizon: int = Field(
        default=16,
        description='Number of future action steps to predict.'
    )
    
    image_size: list = Field(
        default_factory=lambda: [224, 224],
        description='Target image size [height, width].'
    )
    
    default_instruction: str = Field(
        default='Complete the task.',
        description='Default instruction if none provided.'
    )
    
    # Private attributes
    _language_key: Optional[str] = PrivateAttr(default=None)
    _metadata: Optional[DatasetMetadata] = PrivateAttr(default=None)
    
    def set_metadata(self, dataset_metadata: DatasetMetadata):
        """Set the metadata for normalization statistics."""
        super().set_metadata(dataset_metadata)
        self._metadata = dataset_metadata
    
    def check_keys_and_batch_size(self, data: dict):
        """Determine if data is batched and extract batch size."""
        grouped_keys = {}
        for key in data.keys():
            if 'annotation' in key:
                modality = 'language'
            else:
                try:
                    modality, _ = key.split('.')
                except:
                    modality = 'others'
            
            if modality not in grouped_keys:
                grouped_keys[modality] = []
            grouped_keys[modality].append(key)
        
        # Use video key to determine batch size
        video_ndim = data['video'].ndim
        if video_ndim == 5:  # [T, V, H, W, C]
            is_batched = False
            batch_size = 1
        elif video_ndim == 6:  # [B, T, V, H, W, C]
            is_batched = True
            batch_size = data['video'].shape[0]
        else:
            raise ValueError(f'Unsupported video dimensions: {video_ndim}')
        
        # Handle language keys
        if 'language' in grouped_keys:
            language_keys = grouped_keys['language']
            assert len(language_keys) == 1, f'Multiple language keys: {language_keys}'
            self._language_key = language_keys[0]
        
        return is_batched, batch_size
    
    def _prepare_images(self, data: dict):
        """
        Prepare multi-view images for InternVLA-M1.
        
        Expected input: data['video'] with shape [T, V, H, W, C]
        Returns: List of PIL Images for each view
        """
        images = data['video']  # [T, V, H, W, C]
        
        # InternVLA-M1 expects the latest frame from each view
        # Take the last timestep: [V, H, W, C]
        if images.shape[0] > 0:
            images = images[-1]  # Take last timestep
        else:
            raise ValueError('No images in video tensor')
        
        # Convert to numpy and ensure uint8
        if isinstance(images, torch.Tensor):
            images = (images * 255).to(torch.uint8).cpu().numpy()
        images = images.astype(np.uint8)
        
        # Convert to list of PIL Images
        from PIL import Image
        pil_images = []
        for v in range(images.shape[0]):  # Iterate over views
            pil_images.append(Image.fromarray(images[v]))
        
        return pil_images
    
    def _prepare_language(self, data: dict) -> str:
        """Extract and format language instruction."""
        if self._language_key is not None and self._language_key in data:
            raw_language = data[self._language_key]
            if isinstance(raw_language, list):
                raw_language = raw_language[0]
            return raw_language.lower() if raw_language else self.default_instruction
        return self.default_instruction
    
    def _prepare_action(self, data: dict):
        """
        Prepare action sequence.
        
        Returns:
            action: np.ndarray of shape [action_horizon, action_dim]
        """
        if 'action' not in data:
            # Return dummy action if not in training mode
            return np.zeros((self.action_horizon, 7))  # Assume 7-DOF action
        
        action = data['action']
        
        # Ensure correct shape
        if action.shape[0] != self.action_horizon:
            # Pad or truncate as needed
            if action.shape[0] < self.action_horizon:
                padding = np.zeros((self.action_horizon - action.shape[0], action.shape[1]))
                action = np.concatenate([action, padding], axis=0)
            else:
                action = action[:self.action_horizon]
        
        return action
    
    def apply_single(self, data: dict) -> dict:
        """Apply transform to a single sample."""
        transformed = {}
        
        # Prepare images
        images = self._prepare_images(data)
        transformed['image'] = images
        
        # Prepare language
        lang = self._prepare_language(data)
        transformed['lang'] = lang
        
        # Prepare action (only during training)
        if self.training and 'action' in data:
            action = self._prepare_action(data)
            transformed['action'] = action
        
        return transformed
    
    def apply_batch(self, data: dict, batch_size: int) -> dict:
        """Apply transform to a batch of samples."""
        # Split batch
        data_split = []
        for i in range(batch_size):
            single_data = {}
            for key, value in data.items():
                if isinstance(value, str):
                    single_data[key] = value
                else:
                    try:
                        single_data[key] = value[i]
                    except (TypeError, IndexError):
                        single_data[key] = value
            data_split.append(single_data)
        
        # Process each sample
        processed = [self.apply_single(elem) for elem in data_split]
        
        # Stack results
        batched = {}
        for key in processed[0].keys():
            values = [p[key] for p in processed]
            if key in ['image', 'lang']:
                batched[key] = values  # Keep as list
            else:
                batched[key] = np.stack(values)
        
        return batched
    
    def apply(self, data: dict) -> dict:
        """Apply transform to data (auto-detects batched vs single)."""
        is_batched, batch_size = self.check_keys_and_batch_size(data)
        if is_batched:
            return self.apply_batch(data, batch_size)
        else:
            return self.apply_single(data)
    
    def unapply(self, data: dict) -> dict:
        """
        Unapply transform to recover original action space.
        
        Args:
            data: dict with 'action' key containing normalized actions
        
        Returns:
            dict with denormalized actions
        """
        # For now, assume actions are already in the correct space
        # In a full implementation, this would use metadata statistics
        # to denormalize actions
        return data
    
    def __call__(self, data: dict) -> dict:
        """Make the transform callable."""
        return self.apply(data)
