#!/usr/bin/env python

# Copyright 2024 Columbia Artificial Intelligence, Robotics Lab,
# and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Diffusion Policy as per "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"

TODO(alexander-soare):
  - Remove reliance on diffusers for DDPMScheduler and LR scheduler.
"""

import math
from collections import deque
from typing import Callable, Iterator, Any, Dict, List, Optional
from torch.nn import Parameter

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch import Tensor, nn
from transformers import AutoConfig, AutoModel,BatchFeature
from transformers.data.data_collator import DataCollatorMixin

from internmanip.model.basemodel.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE
from internmanip.model.basemodel.base import BasePolicyModel
from internmanip.configs.model.dp_cfg import DiffusionConfig
from internmanip.model.action_head.diffusion_action_head import DiffusionActionHead
from internmanip.model.data_collator_registry import DataCollatorRegistry


@DataCollatorRegistry.register(DiffusionConfig.model_type)
def data_collator_base(features):
    batch = {}
    # Transpose the video data to match the expected input shape for the model, from [B, T, H, W, C] to [B, T, C, H, W]
    video_data = torch.stack([f['video'] for f in features])
    batch['observation.images'] = video_data
    # Change the data type of state and action to float32 to avoid type mismatch in mixed precision training
    batch['observation.state'] = torch.stack([f['state'] for f in features]).float()
    batch['action'] = torch.stack([f['action'] for f in features]).float()
    batch['action_is_pad'] = torch.stack([torch.from_numpy(f['action_pad']) for f in features])
    batch['language'] = [f['annotation.human.action.task_description'] for f in features]
    return batch



class DiffusionModel(BasePolicyModel):
    """
    Diffusion Policy as per "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"
    (paper: https://huggingface.co/papers/2303.04137, code: https://github.com/real-stanford/diffusion_policy).
    """

    config_class = DiffusionConfig
    name: str = 'dp_clip'

    def __init__(
        self,
        config: DiffusionConfig,
        local_model_path: str=None,
        **kwargs
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """
        super().__init__(config)
        self.config = config
        self.local_model_path = local_model_path
        self.action_head = DiffusionActionHead(config)
        self.action_head.set_trainable_parameters(
            tune_visual=config.tune_visual, tune_llm=config.tune_llm
        )


    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Run the batch through the model and compute the loss for training or validation."""
        # batch = self.normalize_inputs(batch)
        # if self.config.image_features:
        #     batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
        #     batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
            # batch[]
            # batch['observation.images'] = batch[OBS_IMAGES]

        # Handle language input if language conditioning is enabled
        if self.config.use_language_conditioning and 'language' in batch:
            # Ensure language is properly formatted
            if isinstance(batch['language'], list):
                # For list format, ensure each item is a string
                batch['language'] = [instructions[0] for instructions in batch['language']]  # type: ignore
            elif isinstance(batch['language'], torch.Tensor):
                # For tensor format, convert to list of strings for processing
                batch['language'] = [str(desc.item()) for desc in batch['language']]  # type: ignore

        # batch = self.normalize_targets(batch)
        loss_dict = self.action_head.compute_loss(batch)
        # Return the loss dictionary containing both main loss and action loss
        return loss_dict

    def inference(self, batch: dict[str, Tensor]) -> Tensor:
        """Generate actions for inference given observations.

        Args:
            batch: Dictionary containing observation data with keys:
                - "video": (B, n_obs_steps, N, C, H, W)
                - "state": (B, n_obs_steps, state_dim)
                - "annotation.human.action.task_description": List[str] - language instructions for conditioning

        Returns:
            BatchFeature: Generated actions of shape (B, n_action_steps, action_dim)
        """
        # 获取模型所在的设备和数据类型
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        # Convert input format and move tensors to model device with correct dtype
        inference_batch = {}
        inference_batch['observation.images'] = batch['video'].to(device, dtype=dtype)
        inference_batch['observation.state'] = batch['state'].to(device, dtype=dtype)
        inference_batch['language'] = batch['annotation.human.action.task_description']


        # Generate actions using the action head
        actions = self.action_head.generate_actions(inference_batch) # return (B, n_action_steps, action_dim)

        # return actions.detach().cpu()
        return BatchFeature(data={'action_pred': actions})




    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        tune_visual = kwargs.pop('tune_visual', True)
        tune_llm = kwargs.pop('tune_llm', False)

        pretrained_model = super().from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )

        pretrained_model.action_head.set_trainable_parameters(
            tune_visual=tune_visual, tune_llm=tune_llm
        )
        print('Total number of parameters: ', int(pretrained_model.num_parameters()/1024/1024), 'M')
        print('Total trainable number of parameters: ', int(sum(p.numel() for p in pretrained_model.parameters() if p.requires_grad)/1024/1024), 'M')
        return pretrained_model

AutoConfig.register('dp_clip', DiffusionConfig)
AutoModel.register(DiffusionConfig, DiffusionModel)
