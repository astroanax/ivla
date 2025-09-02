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
This is the action head for the diffusion policy.
"""


import math
from typing import Callable

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor, nn
from transformers import CLIPProcessor, CLIPModel

from internmanip.model.basemodel.constants import OBS_STATE
from internmanip.configs.model.dp_cfg import DiffusionConfig
from internmanip.model.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    get_output_shape,
)
from internmanip.model.backbone.diffusion_vision_backbone import DiffusionRgbEncoder

from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

class DiffusionSinusoidalPosEmb(torch.nn.Module):
    """1D sinusoidal positional embeddings as in Attention is All You Need."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class DiffusionActionHead(nn.Module):
    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.config: DiffusionConfig = config

        # Build observation encoders (depending on which observations are provided).
        global_cond_dim = self.config.robot_state_dim
        # if self.config.image_features:
        if self.config.image_features:
            num_images = self.config.image_features[0]
            if self.config.use_separate_rgb_encoder_per_camera:
                encoders = [DiffusionRgbEncoder(config) for _ in range(num_images)]
                self.rgb_encoder = nn.ModuleList(encoders)
                global_cond_dim += encoders[0].feature_dim * num_images
            else:
                self.rgb_encoder = DiffusionRgbEncoder(config)
                global_cond_dim += self.rgb_encoder.feature_dim * num_images
        # if self.config.env_state_feature:
        #     global_cond_dim += self.config.env_state_feature.shape[0]

        # Add language conditioning if enabled
        if self.config.use_language_conditioning:
            try:
                self.language_encoder = CLIPModel.from_pretrained(self.config.language_model_name)
                self.language_processor = CLIPProcessor.from_pretrained(self.config.language_model_name)
                print(f'Successfully loaded CLIP model from: {self.config.language_model_name}')
            except Exception as e:
                print(f'Failed to load CLIP model from {self.config.language_model_name}: {e}')
                print('Please ensure:')
                print('1. Internet connection is stable')
                print('2. Or use a local model path instead of HuggingFace identifier')
                print('3. Or download the model manually to local cache')
                raise RuntimeError(f'CLIP model loading failed: {e}')
            # Add a trainable projection layer to reduce text embedding dimension
            self.language_projection = nn.Linear(self.config.language_embedding_dim, self.config.language_projection_dim)
            # Add language projection dimension to global conditioning
            global_cond_dim += self.config.language_projection_dim
            self.language_dropout = nn.Dropout(self.config.language_dropout_prob)
        else:
            self.language_encoder = None
            self.language_processor = None
            self.language_projection = None
            self.language_dropout = None

        self.unet = DiffusionConditionalUnet1d(config, global_cond_dim=global_cond_dim * config.n_obs_steps)

        self.noise_scheduler = _make_noise_scheduler(  # type: ignore
            config.noise_scheduler_type,
            num_train_timesteps=config.num_train_timesteps, # type: ignore
            beta_start=config.beta_start, # type: ignore
            beta_end=config.beta_end, # type: ignore
            beta_schedule=config.beta_schedule, # type: ignore
            clip_sample=config.clip_sample, # type: ignore
            clip_sample_range=config.clip_sample_range, # type: ignore
            prediction_type=config.prediction_type, # type: ignore
        )

        if config.num_inference_steps is None:
            self.num_inference_steps: int = self.noise_scheduler.config.num_train_timesteps # type: ignore
        else:
            self.num_inference_steps: int = config.num_inference_steps

    def set_trainable_parameters(self, tune_visual: bool, tune_llm: bool):
        self.tune_visual = tune_visual
        self.tune_llm = tune_llm
        print(f'Tune action head visual: {self.tune_visual}')
        print(f'Tune action head LLM: {self.tune_llm}')

        # Handle visual encoder (rgb_encoder)
        if not tune_visual:
            if isinstance(self.rgb_encoder, nn.ModuleList):
                # If using separate encoders per camera
                for encoder in self.rgb_encoder:
                    for param in encoder.parameters():
                        param.requires_grad = False
            else:
                # If using single shared encoder
                for param in self.rgb_encoder.parameters():
                    param.requires_grad = False

        # Handle language encoder
        if not tune_llm:
            for param in self.language_encoder.parameters():
                param.requires_grad = False

        print(f'Tune action head visual: {self.tune_visual}')
        print(f'Tune action head LLM: {self.tune_llm}')

    def to(self, *args, **kwargs):
        """Override to() method to ensure noise_scheduler parameters are moved to the correct device."""
        super().to(*args, **kwargs)

        # Move noise scheduler parameters to the same device
        try:
            device = next(self.parameters()).device
        except StopIteration:
            # If model has no parameters, use cuda if available, otherwise cpu
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Move scheduler's internal tensors to the correct device
        for attr_name in dir(self.noise_scheduler):
            attr = getattr(self.noise_scheduler, attr_name)
            if isinstance(attr, torch.Tensor):
                setattr(self.noise_scheduler, attr_name, attr.to(device))

        # Move language encoder to the same device if it exists
        if self.language_encoder is not None:
            self.language_encoder = self.language_encoder.to(device) # type: ignore

        return self

    # ========= inference  ============
    def conditional_sample(
        self, batch_size: int, global_cond: Tensor | None = None, generator: torch.Generator | None = None
    ) -> Tensor:
        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)

        # Sample prior.
        sample = torch.randn(
            size=(batch_size, self.config.horizon, self.config.action_dim), # type: ignore
            dtype=dtype,
            device=device,
            generator=generator,
        )

        self.noise_scheduler.set_timesteps(self.num_inference_steps)

        for t in self.noise_scheduler.timesteps:
            # Predict model output.
            model_output = self.unet(
                sample,
                torch.full(sample.shape[:1], t, dtype=torch.long, device=sample.device),
                global_cond=global_cond,
            )
            # Compute previous image: x_t -> x_t-1
            sample = self.noise_scheduler.step(model_output, t, sample, generator=generator).prev_sample

        return sample

    def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        """Encode image features and concatenate them all together along with the state vector."""
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        global_cond_feats = [batch[OBS_STATE]]
        # Extract image features.
        if self.config.image_features:
            if self.config.use_separate_rgb_encoder_per_camera:
                # Combine batch and sequence dims while rearranging to make the camera index dimension first.
                images_per_camera = einops.rearrange(batch['observation.images'], 'b s n ... -> n (b s) ...')
                img_features_list = torch.cat(
                    [
                        encoder(images)
                        for encoder, images in zip(self.rgb_encoder, images_per_camera, strict=True)
                    ]
                )
                # Separate batch and sequence dims back out. The camera index dim gets absorbed into the
                # feature dim (effectively concatenating the camera features).
                img_features = einops.rearrange(
                    img_features_list, '(n b s) ... -> b s (n ...)', b=batch_size, s=n_obs_steps
                )
            else:
                # Combine batch, sequence, and "which camera" dims before passing to shared encoder.
                img_features = self.rgb_encoder(
                    einops.rearrange(batch['observation.images'], 'b s n ... -> (b s n) ...')
                )
                # Separate batch dim and sequence dim back out. The camera index dim gets absorbed into the
                # feature dim (effectively concatenating the camera features).
                img_features = einops.rearrange(
                    img_features, '(b s n) ... -> b s (n ...)', b=batch_size, s=n_obs_steps
                )
            global_cond_feats.append(img_features)

        # if self.config.env_state_feature:
        #     global_cond_feats.append(batch[OBS_ENV_STATE])

        # Add language conditioning if enabled
        if self.config.use_language_conditioning and 'language' in batch:
            language_texts = batch['language']

            # For list format, each batch item has its own instruction
            # We need to process each instruction separately and then stack
            text_embeddings_list = []

            # Ensure we have exactly batch_size instructions
            for i in range(batch_size):
                instruction = language_texts[i]
                if not isinstance(instruction, str):
                    instruction = str(instruction)

                # Process single instruction with CLIP
                inputs = self.language_processor(
                    text=instruction,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=77
                )
                # Move inputs to the same device as the model
                try:
                    device = next(self.parameters()).device
                except StopIteration:
                    # If model has no parameters, use cuda if available, otherwise cpu
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                inputs = {k: v.to(device) for k, v in inputs.items()}

                # Get text embeddings for this instruction (language encoder is frozen)
                with torch.no_grad():
                    text_emb = self.language_encoder.get_text_features(**inputs)

                # Project to lower dimension using trainable layer
                text_emb = self.language_projection(text_emb)

                # Apply dropout if in training mode
                if self.training:
                    text_emb = self.language_dropout(text_emb)

                text_embeddings_list.append(text_emb)

            # Stack all embeddings and expand to sequence dimension
            text_embeddings = torch.stack(text_embeddings_list, dim=0)  # (B, embed_dim)
            text_embeddings = text_embeddings.expand(-1, n_obs_steps, -1)  # (B, n_obs_steps, embed_dim)

            global_cond_feats.append(text_embeddings)

        # Concatenate all features along the feature dimension (dim=-1)
        # Each tensor in global_cond_feats has shape [B, n_obs_steps, feature_dim]
        # After concatenation: [B, n_obs_steps, total_feature_dim]
        # Then flatten the sequence dimension: [B, n_obs_steps * total_feature_dim]
        return torch.cat(global_cond_feats, dim=-1).flatten(start_dim=1)

    def generate_all_actions(self, batch: dict[str, Tensor]) -> Tensor:
        """
        This function expects `batch` to have:
        {
            "observation.state": (B, n_obs_steps, state_dim)

            "observation.images": (B, n_obs_steps, num_cameras, C, H, W)
                AND/OR
            "observation.environment_state": (B, environment_dim)
                AND/OR
            "language": List[str] or str - language instructions for conditioning
                - List[str]: Each batch item has its own instruction (recommended for training)
                - str: Single instruction applied to all batch items
        }
        Returns:
            (B, horizon, action_dim) Including the actions before the current observation and after the current observation
        """
        batch_size, n_obs_steps = batch['observation.state'].shape[:2]
        assert n_obs_steps == self.config.n_obs_steps

        # Encode image features and concatenate them all together along with the state vector.
        global_cond = self._prepare_global_conditioning(batch)  # (B, global_cond_dim)

        # run sampling
        actions = self.conditional_sample(batch_size, global_cond=global_cond)

        return actions

    def generate_actions(self, batch: dict[str, Tensor]) -> Tensor:
        """
        This function expects `batch` to have:
        {
            "observation.state": (B, n_obs_steps, state_dim)

            "observation.images": (B, n_obs_steps, num_cameras, C, H, W)
                AND/OR
            "observation.environment_state": (B, environment_dim)
                AND/OR
            "language": List[str] or str - language instructions for conditioning
                - List[str]: Each batch item has its own instruction (recommended for training)
                - str: Single instruction applied to all batch items
        }
        """
        batch_size, n_obs_steps = batch['observation.state'].shape[:2]
        assert n_obs_steps == self.config.n_obs_steps

        # Encode image features and concatenate them all together along with the state vector.
        global_cond = self._prepare_global_conditioning(batch)  # (B, global_cond_dim)

        # run sampling
        actions = self.conditional_sample(batch_size, global_cond=global_cond)

        # Extract `n_action_steps` steps worth of actions (from the current observation).
        start = n_obs_steps - 1
        end = start + self.config.n_action_steps
        actions = actions[:, start:end]

        return actions

    def compute_loss(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        This function expects `batch` to have (at least):
        {
            "state": (B, n_obs_steps, state_dim)

            "observation.images": (B, n_obs_steps, num_cameras, C, H, W)
                AND/OR
            "observation.environment_state": (B, environment_dim)
                AND/OR
            "language": List[str] or str - language instructions for conditioning
                - List[str]: Each batch item has its own instruction (recommended for training)
                - str: Single instruction applied to all batch items

            "action": (B, horizon, action_dim)
            "action_is_pad": (B, horizon)
        }
        """
        # Input validation.
        # assert set(batch).issuperset({"state", "action", "action_is_pad"})
        # assert "observation.images" in batch or "observation.environment_state" in batch
        n_obs_steps = batch['observation.state'].shape[1]
        horizon = batch['action'].shape[1]
        assert horizon == self.config.horizon
        assert n_obs_steps == self.config.n_obs_steps

        # Encode image features and concatenate them all together along with the state vector.
        global_cond = self._prepare_global_conditioning(batch)  # (B, global_cond_dim)

        # Forward diffusion.
        trajectory = batch['action']
        # Sample noise to add to the trajectory.
        eps = torch.randn(trajectory.shape, device=trajectory.device)
        # Sample a random noising timestep for each item in the batch.
        timesteps = torch.randint(
            low=0,
            high=self.noise_scheduler.config.num_train_timesteps, # type: ignore
            size=(trajectory.shape[0],),
            device=trajectory.device,
        ).long()
        # Add noise to the clean trajectories according to the noise magnitude at each timestep.
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, eps, timesteps) # type: ignore

        # Run the denoising network (that might denoise the trajectory, or attempt to predict the noise).
        pred = self.unet(noisy_trajectory, timesteps, global_cond=global_cond)

        # Compute the loss.
        # The target is either the original trajectory, or the noise.
        if self.config.prediction_type == 'epsilon':
            target = eps
        elif self.config.prediction_type == 'sample':
            target = batch['action']
        else:
            raise ValueError(f'Unsupported prediction type {self.config.prediction_type}')

        loss = F.mse_loss(pred, target, reduction='none')

        # Mask loss wherever the action is padded with copies (edges of the dataset trajectory).
        if self.config.do_mask_loss_for_padding:
            if 'action_is_pad' not in batch:
                raise ValueError(
                    "You need to provide 'action_is_pad' in the batch when "
                    f'{self.config.do_mask_loss_for_padding=}.'
                )
            in_episode_bound = ~batch['action_is_pad']
            loss = loss * in_episode_bound.unsqueeze(-1)

        loss_dict= {}
        loss_dict['loss'] = loss.mean()
        return loss_dict




class DiffusionConv1dBlock(nn.Module):
    """Conv1d --> GroupNorm --> Mish"""

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class DiffusionConditionalUnet1d(nn.Module):
    """A 1D convolutional UNet with FiLM modulation for conditioning.

    Note: this removes local conditioning as compared to the original diffusion policy code.
    """

    def __init__(self, config: DiffusionConfig, global_cond_dim: int):
        super().__init__()

        self.config = config

        # Encoder for the diffusion timestep.
        self.diffusion_step_encoder = nn.Sequential(
            DiffusionSinusoidalPosEmb(config.diffusion_step_embed_dim),
            nn.Linear(config.diffusion_step_embed_dim, config.diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(config.diffusion_step_embed_dim * 4, config.diffusion_step_embed_dim),
        )

        # The FiLM conditioning dimension.
        cond_dim = config.diffusion_step_embed_dim + global_cond_dim

        # In channels / out channels for each downsampling block in the Unet's encoder. For the decoder, we
        # just reverse these.
        in_out = [(config.action_dim, config.down_dims[0])] + list(
            zip(config.down_dims[:-1], config.down_dims[1:], strict=True)
        )

        # Unet encoder.
        common_res_block_kwargs = {
            'cond_dim': cond_dim,
            'kernel_size': config.kernel_size,
            'n_groups': config.n_groups,
            'use_film_scale_modulation': config.use_film_scale_modulation,
        }
        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(
                nn.ModuleList(
                    [
                        DiffusionConditionalResidualBlock1d(dim_in, dim_out, **common_res_block_kwargs),
                        DiffusionConditionalResidualBlock1d(dim_out, dim_out, **common_res_block_kwargs),
                        # Downsample as long as it is not the last block.
                        nn.Conv1d(dim_out, dim_out, 3, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
            )

        # Processing in the middle of the auto-encoder.
        self.mid_modules = nn.ModuleList(
            [
                DiffusionConditionalResidualBlock1d(
                    config.down_dims[-1], config.down_dims[-1], **common_res_block_kwargs
                ),
                DiffusionConditionalResidualBlock1d(
                    config.down_dims[-1], config.down_dims[-1], **common_res_block_kwargs
                ),
            ]
        )

        # Unet decoder.
        self.up_modules = nn.ModuleList([])
        for ind, (dim_out, dim_in) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(
                nn.ModuleList(
                    [
                        # dim_in * 2, because it takes the encoder's skip connection as well
                        DiffusionConditionalResidualBlock1d(dim_in * 2, dim_out, **common_res_block_kwargs),
                        DiffusionConditionalResidualBlock1d(dim_out, dim_out, **common_res_block_kwargs),
                        # Upsample as long as it is not the last block.
                        nn.ConvTranspose1d(dim_out, dim_out, 4, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.final_conv = nn.Sequential(
            DiffusionConv1dBlock(config.down_dims[0], config.down_dims[0], kernel_size=config.kernel_size),
            nn.Conv1d(config.down_dims[0], config.action_dim, 1),
        )

    def forward(self, x: Tensor, timestep: Tensor | int, global_cond=None) -> Tensor:
        """
        Args:
            x: (B, T, input_dim) tensor for input to the Unet.
            timestep: (B,) tensor of (timestep_we_are_denoising_from - 1).
            global_cond: (B, global_cond_dim)
            output: (B, T, input_dim)
        Returns:
            (B, T, input_dim) diffusion model prediction.
        """
        # For 1D convolutions we'll need feature dimension first.
        x = einops.rearrange(x, 'b t d -> b d t')

        timesteps_embed = self.diffusion_step_encoder(timestep)

        # If there is a global conditioning feature, concatenate it to the timestep embedding.
        if global_cond is not None:
            global_feature = torch.cat([timesteps_embed, global_cond], axis=-1)
        else:
            global_feature = timesteps_embed

        # Run encoder, keeping track of skip features to pass to the decoder.
        encoder_skip_features: list[Tensor] = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            encoder_skip_features.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        # Run decoder, using the skip features from the encoder.
        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, encoder_skip_features.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, 'b d t -> b t d')
        return x


class DiffusionConditionalResidualBlock1d(nn.Module):
    """ResNet style 1D convolutional block with FiLM modulation for conditioning."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
        # Set to True to do scale modulation with FiLM as well as bias modulation (defaults to False meaning
        # FiLM just modulates bias).
        use_film_scale_modulation: bool = False,
    ):
        super().__init__()

        self.use_film_scale_modulation = use_film_scale_modulation
        self.out_channels = out_channels

        self.conv1 = DiffusionConv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups)

        # FiLM modulation (https://huggingface.co/papers/1709.07871) outputs per-channel bias and (maybe) scale.
        cond_channels = out_channels * 2 if use_film_scale_modulation else out_channels
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, cond_channels))

        self.conv2 = DiffusionConv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups)

        # A final convolution for dimension matching the residual (if needed).
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """
        Args:
            x: (B, in_channels, T)
            cond: (B, cond_dim)
        Returns:
            (B, out_channels, T)
        """
        out = self.conv1(x)

        # Get condition embedding. Unsqueeze for broadcasting to `out`, resulting in (B, out_channels, 1).
        cond_embed = self.cond_encoder(cond).unsqueeze(-1)
        if self.use_film_scale_modulation:
            # Treat the embedding as a list of scales and biases.
            scale = cond_embed[:, : self.out_channels]
            bias = cond_embed[:, self.out_channels :]
            out = scale * out + bias
        else:
            # Treat the embedding as biases.
            out = out + cond_embed

        out = self.conv2(out)
        out = out + self.residual_conv(x)
        return out

def _make_noise_scheduler(name: str, **kwargs: dict) -> DDPMScheduler | DDIMScheduler:
    """
    Factory for noise scheduler instances of the requested type. All kwargs are passed
    to the scheduler.
    """
    if name == 'DDPM':
        return DDPMScheduler(**kwargs)  # type: ignore
    elif name == 'DDIM':
        return DDIMScheduler(**kwargs)  # type: ignore
    else:
        raise ValueError(f'Unsupported noise scheduler type {name}')
