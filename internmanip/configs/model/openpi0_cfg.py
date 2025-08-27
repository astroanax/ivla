# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import abc
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List


from internmanip.model.types import FeatureType, NormalizationMode, PolicyFeature

from internmanip.trainer.optim.optimizers import AdamWConfig
from internmanip.trainer.optim.schedulers import CosineDecayWithWarmupSchedulerConfig

from transformers import PretrainedConfig


@dataclass
class LoRAConfig:
    """LoRA configuration."""
    r: int = 32
    dropout: float = 0.0


@dataclass
class VisionConfig:
    """Vision configuration."""
    hidden_size: int = 1152
    projection_dim: int = 2048


@dataclass
class VisionProjectorConfig:
    """Vision projector configuration."""
    vision_config: VisionConfig = field(default_factory=VisionConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    use_quantize: bool = False
    use_lora: bool = False


@dataclass
class VisionModelConfig:
    """Vision model configuration."""
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    num_channels: int = 3
    image_size: int = 224
    patch_size: int = 14
    layer_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    num_image_tokens: int = 256
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    use_quantize: bool = False
    use_lora: bool = False


@dataclass
class MixtureComponentConfig:
    """Configuration for a single mixture component."""
    hidden_size: int = 1024
    intermediate_size: int = 4096
    use_final_norm: bool = True
    cache: bool = True
    use_quantize: bool = False
    use_lora: bool = False
    adaptive_mode: Optional[str] = None
    rope_theta: float = 100.0


@dataclass
class MixtureConfig:
    """Mixture configuration."""
    vlm: MixtureComponentConfig = field(default_factory=lambda: MixtureComponentConfig(
        hidden_size=2048,
        intermediate_size=16384,
        use_final_norm=False,
        cache=True,
        use_quantize=False,
        use_lora=False,
        adaptive_mode=None,
        rope_theta=10000.0
    ))
    proprio: MixtureComponentConfig = field(default_factory=lambda: MixtureComponentConfig(
        hidden_size=1024,
        intermediate_size=4096,
        use_final_norm=True,
        cache=True,
        use_quantize=False,
        use_lora=False,
        adaptive_mode=None,
        rope_theta=100.0
    ))
    action: MixtureComponentConfig = field(default_factory=lambda: MixtureComponentConfig(
        hidden_size=1024,
        intermediate_size=4096,
        use_final_norm=True,
        cache=False,
        use_quantize=False,
        use_lora=False,
        adaptive_mode=None,
        rope_theta=100.0
    ))

@dataclass
class JointConfig:
    action_expert_adaptive_mode: Optional[str] = None
    time_hidden_size: Optional[int] = None
    mixture: Optional[str] = None  # 或者可以用dict, 视你的实际结构
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    num_hidden_layers: int = 18
    num_attention_heads: int = 8
    num_key_value_heads: int = 1
    head_dim: int = 256
    rms_norm_eps: float = 1e-6
    attention_bias: bool = False
    attention_dropout: float = 0.0
    pad_token_id: Optional[int] = 0

# @PreTrainedConfig.register_subclass("pi0")
@dataclass
class OpenPI0Config(PretrainedConfig):
    model_type = "openpi0"
    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 4
    n_action_steps: int = 4

    normalization_mapping = None
    # normalization_mapping: dict[str, NormalizationMode] = field(
    #     default_factory=lambda: {
    #         "VISUAL": NormalizationMode.IDENTITY,
    #         "STATE": NormalizationMode.MEAN_STD,
    #         "ACTION": NormalizationMode.MEAN_STD,
    #     }
    # )

    # Shorter state and action vectors will be padded
    max_state_dim: int = 32
    max_action_dim: int = 32

    # Image preprocessing
    resize_imgs_with_padding: tuple[int, int] = (224, 224)

    # Add empty images. Used by pi0_aloha_sim which adds the empty
    # left and right wrist cameras in addition to the top camera.
    empty_cameras: int = 0

    # Converts the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi_aloha: bool = False

    # Converts joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions_aloha: bool = False

    # Tokenizer
    tokenizer_max_length: int = 48

    # Projector
    proj_width: int = 1024

    # Decoding
    num_steps: int = 10

    # Attention utils
    use_cache: bool = True
    attention_implementation: str = "eager"  # or fa2, flex

    # Finetuning settings
    freeze_vision_encoder: bool = True
    train_expert_only: bool = False
    train_state_proj: bool = True

    # Training presets
    optimizer_lr: float = 2.5e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6


    # Model architecture parameters
    action_expert_adaptive_mode: Optional[str] = None
    time_hidden_size: int = 256
    num_hidden_layers: int = 18
    num_attention_heads: int = 8
    num_key_value_heads: int = 1
    head_dim: int = 256
    rms_norm_eps: float = 1e-6
    attention_bias: bool = False
    attention_dropout: float = 0.0
    pad_token_id: int = 0

    # LoRA settings
    lora_r: int = 32
    lora_dropout: float = 0.0

    # TODO: Add EMA
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __post_init__(self):
        super().__post_init__()

        # TODO(Steven): Validate device and amp? in all policy configs?
        """Input validation (not exhaustive)."""
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.n_obs_steps != 1:
            raise ValueError(
                f"Multiple observation steps not handled yet. Got `nobs_steps={self.n_obs_steps}`"
            )

        if self.use_delta_joint_actions_aloha:
            raise NotImplementedError(
                "`use_delta_joint_actions_aloha` is used by pi0 for aloha real models. It is not ported yet in LeRobot."
            )

    def validate_features(self) -> None:
        # TODO: implement value error
        # if not self.image_features and not self.env_state_feature:
        #     raise ValueError("You must provide at least one image or the environment state among the inputs.")

        for i in range(self.empty_cameras):
            key = f"observation.images.empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 480, 640),
            )
            self.input_features[key] = empty_camera

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )
    
    def transform(self):
        transforms = None
        return transforms, list(range(self.n_obs_steps)), list(range(self.n_action_steps))

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None

    # Training configuration
    device: str = "cuda"
    n_nodes: int = 1
    seed: int = 42
    pretrained_model_path: str = "google/paligemma-3b-pt-224"
    load_pretrained_weights: bool = True
    resume_checkpoint_path: Optional[str] = None
    train_vlm: bool = True
    use_torch_compile: bool = True
    use_bf16: bool = True
    use_amp: bool = True
    quantize: bool = False
    debug: bool = False

    # EMA settings
    use_ema: bool = False
    ema_decay: float = 0.99
    ema_start: str = "${save_model_start}"
    ema_freq: int = 1
    ema_device: str = "cuda"
    use_swa: bool = False

    # Data configuration
    dataset_mix: str = "bridge"
    split: str = "train"
    window_size: int = 1
    action_horizon: int = 4
    skip_unlabeled: bool = True
    load_proprio: bool = True
    shuffle_buffer_size: int = 200000
    num_parallel_calls: int = 100
    traj_transform_threads: int = 10
    traj_read_threads: int = 10


    # Logging and evaluation
    log_freq: int = 16
    n_epochs: int = 30

    eval_freq: int = 2000
    eval_size: int = 1024
    # eval_thresholds: List[float] = field(default_factory=lambda: [0.05, 0.1, 0.2, 0.3, 0.5])
    eval_thresholds = [0.05, 0.1, 0.2, 0.3, 0.5]

    # Training hyperparameters
    global_batch_size: int = 1024
    per_device_batch_size: int = 16
    action_lr: float = 5e-5
    vlm_lr: float = 5e-5
    action_weight_decay: float = 0
    vlm_weight_decay: float = 0
    max_grad_norm: float = 1.0

    # Flow sampling
    flow_sampling: str = "beta"
    num_inference_steps: int = 10
    final_action_clip_value: float = 1.0

    # Model dimensions
    cond_steps: int = 1
    horizon_steps: int = 4
    action_dim: int = 7
    proprio_dim: int = 7
    max_seq_len: int = 276
    tokenizer_padding: str = "max_length"
    max_image_text_tokens: int = 276

    # Mixture configuration
    mixture_vlm_hidden_size: int = 2048
    mixture_vlm_intermediate_size: int = 16384
    mixture_vlm_use_final_norm: bool = False
    mixture_vlm_cache: bool = True
    mixture_vlm_use_quantize: bool = False
    mixture_vlm_use_lora: bool = False
    mixture_vlm_adaptive_mode: Optional[str] = None
    mixture_vlm_rope_theta: float = 10000.0

    mixture_proprio_hidden_size: int = 1024
    mixture_proprio_intermediate_size: int = 4096
    mixture_proprio_use_final_norm: bool = True
    mixture_proprio_cache: bool = True
    mixture_proprio_use_quantize: bool = False
    mixture_proprio_use_lora: bool = False
    mixture_proprio_adaptive_mode: Optional[str] = None
    mixture_proprio_rope_theta: float = 100.0

    mixture_action_hidden_size: int = 1024
    mixture_action_intermediate_size: int = 4096
    mixture_action_use_final_norm: bool = True
    mixture_action_cache: bool = False
    mixture_action_use_quantize: bool = False
    mixture_action_use_lora: bool = False
    mixture_action_adaptive_mode: Optional[str] = None
    mixture_action_rope_theta: float = 100.0

    # Time and expert settings
    time_max_period: float = 100.0
    action_expert_rope_theta: float = 100.0

    # Fixed token settings
    image_token_index: int = 257152
    vocab_size: int = 257216

    # Vision model configuration
    vision_hidden_size: int = 1152
    vision_intermediate_size: int = 4304
    vision_num_hidden_layers: int = 27
    vision_num_attention_heads: int = 16
    vision_num_channels: int = 3
    vision_image_size: int = 224
    vision_patch_size: int = 14
    vision_layer_norm_eps: float = 1e-6
    vision_attention_dropout: float = 0.0
    vision_num_image_tokens: int = 256
    vision_use_quantize: bool = False
    vision_use_lora: bool = False

    # Vision projector configuration
    vision_projector_hidden_size: int = 1152
    vision_projector_projection_dim: int = 2048
    vision_projector_use_quantize: bool = False
    vision_projector_use_lora: bool = False

    # Joint model configuration

    def get_mixture_config(self) -> MixtureConfig:
        """Get mixture configuration as a dataclass."""
        return MixtureConfig(
            vlm=MixtureComponentConfig(
                hidden_size=self.mixture_vlm_hidden_size,
                intermediate_size=self.mixture_vlm_intermediate_size,
                use_final_norm=self.mixture_vlm_use_final_norm,
                cache=self.mixture_vlm_cache,
                use_quantize=self.mixture_vlm_use_quantize,
                use_lora=self.mixture_vlm_use_lora,
                adaptive_mode=self.mixture_vlm_adaptive_mode,
                rope_theta=self.mixture_vlm_rope_theta,
            ),
            proprio=MixtureComponentConfig(
                hidden_size=self.mixture_proprio_hidden_size,
                intermediate_size=self.mixture_proprio_intermediate_size,
                use_final_norm=self.mixture_proprio_use_final_norm,
                cache=self.mixture_proprio_cache,
                use_quantize=self.mixture_proprio_use_quantize,
                use_lora=self.mixture_proprio_use_lora,
                adaptive_mode=self.mixture_proprio_adaptive_mode,
                rope_theta=self.mixture_proprio_rope_theta,
            ),
            action=MixtureComponentConfig(
                hidden_size=self.mixture_action_hidden_size,
                intermediate_size=self.mixture_action_intermediate_size,
                use_final_norm=self.mixture_action_use_final_norm,
                cache=self.mixture_action_cache,
                use_quantize=self.mixture_action_use_quantize,
                use_lora=self.mixture_action_use_lora,
                adaptive_mode=self.mixture_action_adaptive_mode,
                rope_theta=self.mixture_action_rope_theta,
            )
        )

    def get_vision_config(self) -> VisionModelConfig:
        """Get vision model configuration as a dataclass."""
        return VisionModelConfig(
            hidden_size=self.vision_hidden_size,
            intermediate_size=self.vision_intermediate_size,
            num_hidden_layers=self.vision_num_hidden_layers,
            num_attention_heads=self.vision_num_attention_heads,
            num_channels=self.vision_num_channels,
            image_size=self.vision_image_size,
            patch_size=self.vision_patch_size,
            layer_norm_eps=self.vision_layer_norm_eps,
            attention_dropout=self.vision_attention_dropout,
            num_image_tokens=self.vision_num_image_tokens,
            lora=LoRAConfig(
                r=self.lora_r,
                dropout=self.lora_dropout,
            ),
            use_quantize=self.vision_use_quantize,
            use_lora=self.vision_use_lora,
        )

    def get_vision_projector_config(self) -> 'VisionProjectorConfig':
        """Get vision projector configuration as a dataclass."""
        return VisionProjectorConfig(
            vision_config=VisionConfig(
                hidden_size=self.vision_projector_hidden_size,
                projection_dim=self.vision_projector_projection_dim,
            ),
            lora=LoRAConfig(
                r=self.lora_r,
                dropout=self.lora_dropout,
            ),
            use_quantize=self.vision_projector_use_quantize,
            use_lora=self.vision_projector_use_lora,
        )

    def get_joint_config(self) -> 'VisionProjectorConfig':
        """Get vision projector configuration as a dataclass."""
        return JointConfig(
             action_expert_adaptive_mode = self.action_expert_adaptive_mode,
            time_hidden_size = self.time_hidden_size,
            mixture = self.get_mixture_config(),
            lora=LoRAConfig(
                r=self.lora_r,
                dropout=self.lora_dropout,
            ),
            num_hidden_layers = self.num_hidden_layers,
            num_attention_heads = self.num_attention_heads,
            num_key_value_heads = self.num_key_value_heads,
            head_dim = self.head_dim,
            rms_norm_eps = self.rms_norm_eps,
            attention_bias = self.attention_bias,
            attention_dropout = self.attention_dropout,
            pad_token_id = self.pad_token_id,
        )
