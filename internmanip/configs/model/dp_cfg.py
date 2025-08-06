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
from dataclasses import dataclass, field

from transformers.configuration_utils import PretrainedConfig

@dataclass
class DiffusionConfig(PretrainedConfig):
    """Configuration class for DiffusionPolicy.

    Defaults are configured for training with PushT providing proprioceptive and single camera observations.

    The parameters you will most likely need to change are the ones which depend on the environment / sensors.
    Those are: `input_shapes` and `output_shapes`.

    Notes on the inputs and outputs:
        - "observation.state" is required as an input key.
        - Either:
            - At least one key starting with "observation.image is required as an input.
              AND/OR
            - The key "observation.environment_state" is required as input.
        - If there are multiple keys beginning with "observation.image" they are treated as multiple camera
          views. Right now we only support all images having the same shape.
        - "action" is required as an output key.

    Args:
        n_obs_steps: Number of environment steps worth of observations to pass to the policy (takes the
            current step and additional steps going back).
        horizon: Diffusion model action prediction size as detailed in `DiffusionPolicy.select_action`.
        n_action_steps: The number of action steps to run in the environment for one invocation of the policy.
            See `DiffusionPolicy.select_action` for more details.
        input_shapes: A dictionary defining the shapes of the input data for the policy. The key represents
            the input data name, and the value is a list indicating the dimensions of the corresponding data.
            For example, "observation.image" refers to an input from a camera with dimensions [3, 96, 96],
            indicating it has three color channels and 96x96 resolution. Importantly, `input_shapes` doesn't
            include batch dimension or temporal dimension.
        output_shapes: A dictionary defining the shapes of the output data for the policy. The key represents
            the output data name, and the value is a list indicating the dimensions of the corresponding data.
            For example, "action" refers to an output shape of [14], indicating 14-dimensional actions.
            Importantly, `output_shapes` doesn't include batch dimension or temporal dimension.
        input_normalization_modes: A dictionary with key representing the modality (e.g. "observation.state"),
            and the value specifies the normalization mode to apply. The two available modes are "mean_std"
            which subtracts the mean and divides by the standard deviation and "min_max" which rescale in a
            [-1, 1] range.
        output_normalization_modes: Similar dictionary as `normalize_input_modes`, but to unnormalize to the
            original scale. Note that this is also used for normalizing the training targets.
        vision_backbone: Name of the torchvision resnet backbone to use for encoding images.
        crop_shape: (H, W) shape to crop images to as a preprocessing step for the vision backbone. Must fit
            within the image size. If None, no cropping is done.
        crop_is_random: Whether the crop should be random at training time (it's always a center crop in eval
            mode).
        pretrained_backbone_weights: Pretrained weights from torchvision to initialize the backbone.
            `None` means no pretrained weights.
        use_group_norm: Whether to replace batch normalization with group normalization in the backbone.
            The group sizes are set to be about 16 (to be precise, feature_dim // 16).
        spatial_softmax_num_keypoints: Number of keypoints for SpatialSoftmax.
        use_separate_rgb_encoder_per_camera: Whether to use a separate RGB encoder for each camera view.
        down_dims: Feature dimension for each stage of temporal downsampling in the diffusion modeling Unet.
            You may provide a variable number of dimensions, therefore also controlling the degree of
            downsampling.
        kernel_size: The convolutional kernel size of the diffusion modeling Unet.
        n_groups: Number of groups used in the group norm of the Unet's convolutional blocks.
        diffusion_step_embed_dim: The Unet is conditioned on the diffusion timestep via a small non-linear
            network. This is the output dimension of that network, i.e., the embedding dimension.
        use_film_scale_modulation: FiLM (https://huggingface.co/papers/1709.07871) is used for the Unet conditioning.
            Bias modulation is used be default, while this parameter indicates whether to also use scale
            modulation.
        noise_scheduler_type: Name of the noise scheduler to use. Supported options: ["DDPM", "DDIM"].
        num_train_timesteps: Number of diffusion steps for the forward diffusion schedule.
        beta_schedule: Name of the diffusion beta schedule as per DDPMScheduler from Hugging Face diffusers.
        beta_start: Beta value for the first forward-diffusion step.
        beta_end: Beta value for the last forward-diffusion step.
        prediction_type: The type of prediction that the diffusion modeling Unet makes. Choose from "epsilon"
            or "sample". These have equivalent outcomes from a latent variable modeling perspective, but
            "epsilon" has been shown to work better in many deep neural network settings.
        clip_sample: Whether to clip the sample to [-`clip_sample_range`, +`clip_sample_range`] for each
            denoising step at inference time. WARNING: you will need to make sure your action-space is
            normalized to fit within this range.
        clip_sample_range: The magnitude of the clipping range as described above.
        num_inference_steps: Number of reverse diffusion steps to use at inference time (steps are evenly
            spaced). If not provided, this defaults to be the same as `num_train_timesteps`.
        do_mask_loss_for_padding: Whether to mask the loss when there are copy-padded actions. See
            `LeRobotDataset` and `load_previous_and_future_frames` for more information. Note, this defaults
            to False as the original Diffusion Policy implementation does the same.
        # Language conditioning parameters
        use_language_conditioning: Whether to use language conditioning in the diffusion model.
        language_model_name: Name of the language model to use for text encoding (e.g., "openai/clip-vit-base-patch32").
        language_embedding_dim: Original dimension of the language embeddings from CLIP.
        language_projection_dim: Target dimension for text embeddings after projection (reduced from original).
        language_dropout_prob: Dropout probability for language embeddings during training.
    """

    model_type = 'dp_clip'

    # Inputs / output structure.
    n_obs_steps: int = field(default=1, metadata={'help': 'Number of observation steps'})
    horizon: int = field(default=16, metadata={'help': 'Diffusion model action prediction size'})
    n_action_steps: int = field(default=8, metadata={'help': 'Number of action steps'})

    # The original implementation doesn't sample frames for the last 7 steps,
    # which avoids excessive padding and leads to improved training results.
    drop_n_last_frames: int = field(default=7, metadata={'help': 'Drop n last frames'})  # horizon - n_action_steps - n_obs_steps + 1

    # Architecture / modeling.
    # Vision backbone.
    vision_backbone: str = field(default='resnet101', metadata={'help': 'Vision backbone name'})
    crop_shape: tuple[int, int] | None = field(default=None, metadata={'help': 'Crop shape for images'}) # cropping in this file is not used
    crop_is_random: bool = field(default=True, metadata={'help': 'Whether crop is random'})
    pretrained_backbone_weights: str | None = field(default=None, metadata={'help': 'Pretrained backbone weights'})
    use_group_norm: bool = field(default=True, metadata={'help': 'Whether to use group norm'})
    spatial_softmax_num_keypoints: int = field(default=64, metadata={'help': 'Number of keypoints for spatial softmax'})
    use_separate_rgb_encoder_per_camera: bool = field(default=False, metadata={'help': 'Whether to use separate RGB encoder per camera'})
    # Unet.
    down_dims: tuple[int, ...] = field(default=(128, 256, 256, 512), metadata={'help': 'Down dimensions for Unet'})
    kernel_size: int = field(default=5, metadata={'help': 'Kernel size for Unet'})
    n_groups: int = field(default=8, metadata={'help': 'Number of groups for group norm'})
    diffusion_step_embed_dim: int = field(default=128, metadata={'help': 'Diffusion step embedding dimension'})
    use_film_scale_modulation: bool = field(default=True, metadata={'help': 'Whether to use FiLM scale modulation'})
    # Noise scheduler.
    noise_scheduler_type: str = field(default='DDPM', metadata={'help': 'Noise scheduler type'})
    num_train_timesteps: int = field(default=100, metadata={'help': 'Number of training timesteps'})
    beta_schedule: str = field(default='squaredcos_cap_v2', metadata={'help': 'Beta schedule'})
    beta_start: float = field(default=0.0001, metadata={'help': 'Beta start value'})
    beta_end: float = field(default=0.02, metadata={'help': 'Beta end value'})
    prediction_type: str = field(default='epsilon', metadata={'help': 'Prediction type'}) # epsilon, sample
    clip_sample: bool = field(default=False, metadata={'help': 'Whether to clip sample'})
    clip_sample_range: float = field(default=1.0, metadata={'help': 'Clip sample range'})

    # Inference
    num_inference_steps: int | None = field(default=None, metadata={'help': 'Number of inference steps'})

    # Loss computation
    do_mask_loss_for_padding: bool = field(default=False, metadata={'help': 'Whether to mask loss for padding'})

    # Language conditioning parameters
    use_language_conditioning: bool = field(default=True, metadata={'help': 'Whether to use language conditioning'})
    language_model_name: str = field(default='openai/clip-vit-base-patch32', metadata={'help': 'Language model name'})  # You can also use local model path
    language_embedding_dim: int = field(default=512, metadata={'help': 'Language embedding dimension'})
    language_projection_dim: int = field(default=32, metadata={'help': 'Language projection dimension'})  # Reduced dimension for text embeddings
    language_dropout_prob: float = field(default=0.1, metadata={'help': 'Language dropout probability'})

    # Training presets
    optimizer_lr: float = field(default=1e-4, metadata={'help': 'Optimizer learning rate'})
    optimizer_betas: tuple = field(default=(0.95, 0.999), metadata={'help': 'Optimizer betas'})
    optimizer_eps: float = field(default=1e-8, metadata={'help': 'Optimizer epsilon'})
    optimizer_weight_decay: float = field(default=1e-6, metadata={'help': 'Optimizer weight decay'})
    scheduler_name: str = field(default='cosine', metadata={'help': 'Scheduler name'})
    scheduler_warmup_steps: int = field(default=500, metadata={'help': 'Scheduler warmup steps'})

    # Feature properties - these will be set during initialization
    robot_state_dim: int = field(default=0, metadata={'help': 'Robot state feature'})
    image_features: list[int] = field(default_factory=list, metadata={'help': 'Image features'})
    action_dim: int = field(default=0, metadata={'help': 'Action feature'})

    # Training parameters
    tune_visual: bool = field(default=True, metadata={'help': 'Whether to tune visual encoder'})
    tune_llm: bool = field(default=False, metadata={'help': 'Whether to tune language encoder'})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'data_config' in kwargs:
            self.set_dim(kwargs['data_config'])

        if 'tune_visual' in kwargs:
            self.tune_visual = kwargs['tune_visual']
        if 'tune_llm' in kwargs:
            self.tune_llm = kwargs['tune_llm']
        # for key, value in kwargs.items():
        #     setattr(self, key, value)

        # Ensure normalization_mapping attribute exists
        if not hasattr(self, 'normalization_mapping'):
            from internmanip.model.types import NormalizationMode
            self.normalization_mapping = {}

        if not hasattr(self, 'image_features'):
            self.image_features = [1,3,224,224]

    def transform(self):
        transforms = None
        return transforms, list(range(self.n_obs_steps)), list(range(self.horizon))

    def set_dim(self, data_config):
        # end effector is recommnened for diffusion model
        if data_config in ['genmanip_v1']:
            self.robot_state_dim = 24
            self.action_dim = 7
            self.image_features = [3,3,224,224]
        elif data_config in ['google','google_minmax','google_q99','google_robot']:
            self.robot_state_dim = 7
            self.action_dim = 7
            self.image_features = [1,3,224,224]
        elif data_config in ['widowx','widowx_minmax']:
            self.robot_state_dim = 7
            self.action_dim = 7
            self.image_features = [1,3,224,224]
        elif data_config in ['calvin']:
            self.robot_state_dim = 7
            self.action_dim = 7
            self.image_features = [2,3,224,224]
        elif data_config in ['sweep']:
            self.robot_state_dim = 6
            self.action_dim = 7
            self.image_features = [2,3,224,224]
        else:
            raise ValueError(f'Unsupported data config: {data_config}.Please add dimension here')
