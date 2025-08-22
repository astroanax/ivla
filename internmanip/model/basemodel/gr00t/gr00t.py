# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Tuple


import numpy as np
import torch
import tree
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError, RepositoryNotFoundError
from transformers import AutoConfig, AutoModel
from transformers.feature_extraction_utils import BatchFeature
from internmanip.configs.model.gr00t_cfg import GR00T_N1_5_Config, GR00T_N1_Config
from internmanip.model.backbone.eagle_backbone import EagleBackbone, EagleBackbone1_5
from internmanip.model.data_collator_registry import DataCollatorRegistry
from internmanip.model.basemodel.transforms.gr00t_n1 import collate_gr00t_n1, collate_gr00t_n15
from ...action_head.flow_matching_action_head import (
    FlowmatchingActionHead,
    FlowmatchingActionHead_1_5,
    FlowmatchingActionHeadConfig,
    FlowmatchingActionHeadConfig_1_5,
)

from ..base import BasePolicyModel

BACKBONE_FEATURE_KEY = 'backbone_features'
ACTION_KEY = 'action_pred'
LOSS_KEY = 'loss'
ERROR_MSG = 'Error: unexpected input/output'
N_COLOR_CHANNELS = 3


DataCollatorRegistry.register_fn(GR00T_N1_Config.model_type, collate_gr00t_n1)
DataCollatorRegistry.register_fn(GR00T_N1_5_Config.model_type, collate_gr00t_n15)

# real model
class GR00T_N1_5(BasePolicyModel):
    supports_gradient_checkpointing = True
    config_class = GR00T_N1_5_Config
    """
    we expect the backbone output to have a key 'backbone_features' with shape (batch_size, n, hidden_size)
    here n is variable and can be e.g. time, 1 or user specified
    we expect the action head output to have a key 'action_pred' with shape (batch_size, time, action_dim) during inference time
    we expect these to have type BatchFeature, and they can of course have many other user specified keys too
    """

    def __init__(
        self,
        config: GR00T_N1_5_Config,
        local_model_path: str,
    ):
        assert isinstance(config.backbone_cfg, dict)
        assert isinstance(config.action_head_cfg, dict)

        super().__init__(config)
        self.local_model_path = local_model_path

        self.backbone = EagleBackbone1_5(**config.backbone_cfg)
        action_head_cfg = FlowmatchingActionHeadConfig_1_5(**config.action_head_cfg)
        self.action_head = FlowmatchingActionHead_1_5(action_head_cfg)

        self.action_horizon = config.action_horizon
        self.action_dim = config.action_dim
        self.compute_dtype = config.compute_dtype

    def validate_inputs(self, inputs):
        # NOTE -- this should be handled internally by the model
        # however, doing that will likely be breaking changes -- so we'll need to do it after the deadline

        detected_error = False
        error_msg = ERROR_MSG
        if 'action' in inputs:
            action = inputs['action']
            type_ok = isinstance(action, torch.Tensor)
            shape_ok = (
                len(action.shape) == 3
                and action.shape[1] == self.action_horizon
                and action.shape[2] == self.action_dim
            )
            if not type_ok:
                error_msg += f'\n{action.dtype=}'
                detected_error = True
            if not shape_ok:
                error_msg += f'\n{action.shape=}'
                detected_error = True

        if 'video' in inputs:
            video = inputs['video']
            type_ok = isinstance(video, np.ndarray)
            dtype_ok = video.dtype == np.uint8
            shape_ok = len(video.shape) == 6 and video.shape[3] == N_COLOR_CHANNELS
            if not type_ok:
                error_msg += f'\n{type(video)=}'
                detected_error = True
            if not dtype_ok:
                error_msg += f'\n{video.dtype=}'
                detected_error = True
            if not shape_ok:
                error_msg += f'\n{video.shape=}'
                detected_error = True

        if detected_error:
            raise ValueError(error_msg)

    def validate_data(self, action_head_outputs, backbone_outputs, is_training):
        fail_backbone = (
            not isinstance(backbone_outputs, BatchFeature)
            or BACKBONE_FEATURE_KEY not in backbone_outputs
        )

        if fail_backbone:
            error_msg = ERROR_MSG
            error_msg += f'\n{isinstance(backbone_outputs, BatchFeature)=}'
            error_msg += f'\n{BACKBONE_FEATURE_KEY in backbone_outputs=}'
            error_msg += f'\n{backbone_outputs[BACKBONE_FEATURE_KEY].shape=}'
            raise ValueError(error_msg)

        fail_action_head = (not isinstance(action_head_outputs, BatchFeature)) or not (
            (
                LOSS_KEY in action_head_outputs and is_training
            )  # there might not be an action prediction during training
            or (
                ACTION_KEY in action_head_outputs
                and action_head_outputs[ACTION_KEY].shape[1] == self.action_horizon
                and action_head_outputs[ACTION_KEY].shape[2] == self.action_dim
            )
        )

        if fail_action_head:
            error_msg = ERROR_MSG
            error_msg += f'\n{isinstance(action_head_outputs, BatchFeature)=}'
            error_msg += f'\n{LOSS_KEY in action_head_outputs=}'
            error_msg += f'\n{action_head_outputs[ACTION_KEY].shape=}'
            error_msg += f'\n{self.action_horizon=}'
            error_msg += f'\n{self.action_dim=}'
            raise ValueError(error_msg)

    def forward(
        self,
        inputs: dict,
    ) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        action_head_outputs = self.action_head(backbone_outputs, action_inputs)
        self.validate_data(action_head_outputs, backbone_outputs, is_training=True)
        return action_head_outputs

    def inference(
        self,
        inputs: dict,
    ) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        # Because the behavior of backbones remains the same for training and inference, we can use `forward` for backbones.
        backbone_outputs = self.backbone(backbone_inputs)
        action_head_outputs = self.action_head.get_action(backbone_outputs, action_inputs)
        self.validate_data(action_head_outputs, backbone_outputs, is_training=False)
        return action_head_outputs

    def prepare_input(self, inputs) -> Tuple[BatchFeature, BatchFeature]:
        self.validate_inputs(inputs)
        backbone_inputs = self.backbone.prepare_input(inputs)
        action_inputs = self.action_head.prepare_input(inputs)

        def to_device_with_maybe_dtype(x):
            # Only cast to self.compute_dtype if the tensor is floating
            if torch.is_floating_point(x):
                return x.to(self.device, dtype=self.action_head.dtype)
            else:
                # Keep original dtype
                return x.to(self.device)

        backbone_inputs = tree.map_structure(to_device_with_maybe_dtype, backbone_inputs)
        action_inputs = tree.map_structure(to_device_with_maybe_dtype, action_inputs)
        return backbone_inputs, action_inputs

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        tune_visual = kwargs.pop('tune_visual', True)
        tune_llm = kwargs.pop('tune_llm', False)
        tune_projector = kwargs.pop('tune_projector', True)
        tune_diffusion_model = kwargs.pop('tune_diffusion_model', True)

        print(f'Loading pretrained dual brain from {pretrained_model_name_or_path}')
        print(f'Tune backbone vision tower: {tune_visual}')
        print(f'Tune backbone LLM: {tune_llm}')
        print(f'Tune action head projector: {tune_projector}')
        print(f'Tune action head DiT: {tune_diffusion_model}')

        # get the current model path being downloaded
        pretrained_model = super().from_pretrained(
            pretrained_model_name_or_path, torch_dtype=torch.bfloat16, **kwargs
        )

        pretrained_model.backbone.set_trainable_parameters(
            tune_visual=tune_visual, tune_llm=tune_llm
        )
        pretrained_model.action_head.set_trainable_parameters(
            tune_projector=tune_projector, tune_diffusion_model=tune_diffusion_model
        )

        pretrained_model.backbone.eagle_model.language_model.lm_head.weight.requires_grad = False
        pretrained_model.backbone.eagle_model.vision_model.vision_model.head.mlp.fc2.bias.requires_grad=False
        pretrained_model.backbone.eagle_model.vision_model.vision_model.head.mlp.fc2.weight.requires_grad = False
        pretrained_model.backbone.eagle_model.vision_model.vision_model.head.mlp.fc1.bias.requires_grad=False
        pretrained_model.backbone.eagle_model.vision_model.vision_model.head.mlp.fc1.weight.requires_grad=False
        pretrained_model.backbone.eagle_model.vision_model.vision_model.head.layernorm.bias.requires_grad=False
        pretrained_model.backbone.eagle_model.vision_model.vision_model.head.layernorm.weight.requires_grad=False
        pretrained_model.backbone.eagle_model.vision_model.vision_model.head.attention.out_proj.bias.requires_grad=False
        pretrained_model.backbone.eagle_model.vision_model.vision_model.head.attention.out_proj.weight.requires_grad=False
        pretrained_model.backbone.eagle_model.vision_model.vision_model.head.attention.in_proj_bias.requires_grad=False
        pretrained_model.backbone.eagle_model.vision_model.vision_model.head.attention.in_proj_weight.requires_grad=False
        pretrained_model.backbone.eagle_model.vision_model.vision_model.head.probe.requires_grad=False

        return pretrained_model


# register
AutoConfig.register('gr00t_n1_5', GR00T_N1_5_Config)
AutoModel.register(GR00T_N1_5_Config, GR00T_N1_5)

# real model
class GR00T_N1(BasePolicyModel):
    supports_gradient_checkpointing = True
    config_class = GR00T_N1_Config
    """
    we expect the backbone output to have a key 'backbone_features' with shape (batch_size, n, hidden_size)
    here n is variable and can be e.g. time, 1 or user specified
    we expect the action head output to have a key 'action_pred' with shape (batch_size, time, action_dim) during inference time
    we expect these to have type BatchFeature, and they can of course have many other user specified keys too
    """

    def __init__(
        self,
        config: GR00T_N1_Config,
        local_model_path: str=None,
        **kwargs
    ):
        assert isinstance(config.backbone_cfg, dict)
        assert isinstance(config.action_head_cfg, dict)

        super().__init__(config)
        self.local_model_path = local_model_path

        self.backbone = EagleBackbone(**config.backbone_cfg)
        action_head_cfg = FlowmatchingActionHeadConfig(**config.action_head_cfg)
        self.action_head = FlowmatchingActionHead(action_head_cfg)

        self.action_horizon = config.action_horizon
        self.action_dim = config.action_dim
        self.compute_dtype = config.compute_dtype

    def validate_inputs(self, inputs):
        # NOTE -- this should be handled internally by the model
        # however, doing that will likely be breaking changes -- so we'll need to do it after the deadline

        detected_error = False
        error_msg = ERROR_MSG
        if 'action' in inputs:
            action = inputs['action']
            type_ok = isinstance(action, torch.Tensor)
            shape_ok = (
                len(action.shape) == 3
                and action.shape[1] == self.action_horizon
                and action.shape[2] == self.action_dim
            )
            if not type_ok:
                error_msg += f'\n{action.dtype=}'
                detected_error = True
            if not shape_ok:
                error_msg += f'\n{action.shape=}'
                detected_error = True

        if 'video' in inputs:
            video = inputs['video']
            type_ok = isinstance(video, np.ndarray)
            dtype_ok = video.dtype == np.uint8
            shape_ok = len(video.shape) == 6 and video.shape[3] == N_COLOR_CHANNELS
            if not type_ok:
                error_msg += f'\n{type(video)=}'
                detected_error = True
            if not dtype_ok:
                error_msg += f'\n{video.dtype=}'
                detected_error = True
            if not shape_ok:
                error_msg += f'\n{video.shape=}'
                detected_error = True

        if detected_error:
            raise ValueError(error_msg)

    def validate_data(self, action_head_outputs, backbone_outputs, is_training):
        fail_backbone = (
            not isinstance(backbone_outputs, BatchFeature)
            or BACKBONE_FEATURE_KEY not in backbone_outputs
        )

        if fail_backbone:
            error_msg = ERROR_MSG
            error_msg += f'\n{isinstance(backbone_outputs, BatchFeature)=}'
            error_msg += f'\n{BACKBONE_FEATURE_KEY in backbone_outputs=}'
            error_msg += f'\n{backbone_outputs[BACKBONE_FEATURE_KEY].shape=}'
            raise ValueError(error_msg)

        fail_action_head = (not isinstance(action_head_outputs, BatchFeature)) or not (
            (
                LOSS_KEY in action_head_outputs and is_training
            )  # there might not be an action prediction during training
            or (
                ACTION_KEY in action_head_outputs
                and action_head_outputs[ACTION_KEY].shape[1] == self.action_horizon
                and action_head_outputs[ACTION_KEY].shape[2] == self.action_dim
            )
        )

        if fail_action_head:
            error_msg = ERROR_MSG
            error_msg += f'\n{isinstance(action_head_outputs, BatchFeature)=}'
            error_msg += f'\n{LOSS_KEY in action_head_outputs=}'
            error_msg += f'\n{action_head_outputs[ACTION_KEY].shape=}'
            error_msg += f'\n{self.action_horizon=}'
            error_msg += f'\n{self.action_dim=}'
            raise ValueError(error_msg)

    def forward(
        self,
        inputs: dict,
    ) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        action_head_outputs = self.action_head(backbone_outputs, action_inputs)
        self.validate_data(action_head_outputs, backbone_outputs, is_training=True)
        return action_head_outputs


    def calc_loss(
        self,
        inputs: dict,
    ) -> BatchFeature:
        """
        Calculate the loss.
        """
        action_head_outputs, action_inputs = inputs
        loss = self.action_head.calc_loss(action_head_outputs, action_inputs)
        return loss


    def inference(
        self,
        inputs: dict,
    ) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        # Because the behavior of backbones remains the same for training and inference, we can use `forward` for backbones.
        backbone_outputs = self.backbone(backbone_inputs)
        action_head_outputs = self.action_head.inference(backbone_outputs, action_inputs)
        self.validate_data(action_head_outputs, backbone_outputs, is_training=False)
        return action_head_outputs

    def prepare_input(self, inputs) -> Tuple[BatchFeature, BatchFeature]:
        self.validate_inputs(inputs)
        backbone_inputs = self.backbone.prepare_input(inputs)
        action_inputs = self.action_head.prepare_input(inputs)

        def to_device_with_maybe_dtype(x):
            # Only cast to self.compute_dtype if the tensor is floating
            if torch.is_floating_point(x):
                return x.to(self.device, dtype=self.action_head.dtype)
            else:
                # Keep original dtype
                return x.to(self.device)

        backbone_inputs = tree.map_structure(to_device_with_maybe_dtype, backbone_inputs)
        action_inputs = tree.map_structure(to_device_with_maybe_dtype, action_inputs)
        return backbone_inputs, action_inputs

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        tune_visual = kwargs.pop('tune_visual', True)
        tune_llm = kwargs.pop('tune_llm', False)
        tune_projector = kwargs.pop('tune_projector', True)
        tune_diffusion_model = kwargs.pop('tune_diffusion_model', True)

        print(f'Loading pretrained dual brain from {pretrained_model_name_or_path}')
        print(f'Tune backbone vision tower: {tune_visual}')
        print(f'Tune backbone LLM: {tune_llm}')
        print(f'Tune action head projector: {tune_projector}')
        print(f'Tune action head DiT: {tune_diffusion_model}')

        # get the current model path being downloaded
        pretrained_model = super().from_pretrained(
            pretrained_model_name_or_path,**kwargs
        )

        pretrained_model.backbone.set_trainable_parameters(
            tune_visual=tune_visual, tune_llm=tune_llm
        )
        pretrained_model.action_head.set_trainable_parameters(
            tune_projector=tune_projector, tune_diffusion_model=tune_diffusion_model
        )
        print('Total number of parameters: ', int(pretrained_model.num_parameters()/1024/1024), 'M')
        print('Total trainable number of parameters: ', int(sum(p.numel() for p in pretrained_model.parameters() if p.requires_grad)/1024/1024), 'M')
        return pretrained_model


# # register
AutoConfig.register('gr00t_n1', GR00T_N1_Config)
AutoModel.register(GR00T_N1_Config, GR00T_N1)
