from dataclasses import dataclass, field
from transformers import PretrainedConfig
from internmanip.model.basemodel.transforms.internvla_m1 import InternVLAM1Transform

@dataclass
class InternVLA_M1_Config(PretrainedConfig):
    """
    Configuration class for InternVLA-M1 model.
    
    InternVLA-M1 is a spatially guided vision-language-action framework that integrates:
    - Qwen2.5 VL backbone for vision-language understanding
    - DINO encoder for dense multi-view spatial features
    - Layer-wise QFormer for multi-layer feature aggregation
    - DiT diffusion head for future action sequence prediction
    """
    model_type = 'internvla_m1'
    
    # Framework configuration
    framework_cfg: dict = field(
        default_factory=lambda: {
            'qwen_vl': {
                'model_path': 'Qwen/Qwen2.5-7B-Instruct',
                'hidden_size': 3584,
                'select_layers': list(range(0, 28)),  # Use layers 0-27
            },
            'dino': {
                'dino_backbone': 'dinov2_vits14',
                'num_channels': 384,
            },
            'layer_qformer': {
                'qformer_start_layer': 0,
                'qformer_end_layer': 28,
                'num_query_tokens': 64,
                'hidden_size': 768,
            },
            'action_model': {
                'future_action_window_size': 16,
                'past_action_window_size': 0,
                'in_channels': 7,  # action dimension
                'hidden_size': 768,
                'num_layers': 12,
                'num_heads': 12,
            }
        },
        metadata={'help': 'Framework configuration for all submodules.'}
    )
    
    # Action configuration
    action_horizon: int = field(
        default=16,
        metadata={'help': 'Number of future action steps to predict.'}
    )
    
    action_dim: int = field(
        default=7,
        metadata={'help': 'Dimension of action space.'}
    )
    
    # Observation configuration
    observation_indices: list = field(
        default_factory=lambda: [0],
        metadata={'help': 'Indices of observations to use from history.'}
    )
    
    # Training configuration
    compute_dtype: str = field(
        default='bfloat16',
        metadata={'help': 'Compute dtype for training/inference.'}
    )
    
    # Image configuration
    image_size: list = field(
        default_factory=lambda: [224, 224],
        metadata={'help': 'Image size [height, width] for preprocessing.'}
    )
    
    # Diffusion configuration
    num_ddim_steps: int = field(
        default=5,
        metadata={'help': 'Number of DDIM sampling steps for inference.'}
    )
    
    cfg_scale: float = field(
        default=1.5,
        metadata={'help': 'Classifier-free guidance scale for inference.'}
    )
    
    use_ddim: bool = field(
        default=True,
        metadata={'help': 'Whether to use DDIM sampling for inference.'}
    )
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def transform(self):
        """
        Return the transform instance and horizon indices.
        
        Returns:
            tuple: (transform, observation_indices, action_indices)
        """
        transforms = InternVLAM1Transform(
            state_horizon=len(self.observation_indices),
            action_horizon=self.action_horizon,
            image_size=self.image_size,
        )
        action_indices = list(range(self.action_horizon))
        return transforms, self.observation_indices, action_indices
