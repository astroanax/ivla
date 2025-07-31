from pydantic import BaseModel

class TrainCfg(BaseModel):
    """Configuration List."""
    model_type: str = ''
    """pi0 gr00t_n1 gr00t_n1_5 dp_clip act_clip pi0fast"""
    # Dataset parameters
    dataset_path: str = './internmanip/demo_data/robot_sim_converted.PickNPlace/'
    """Path to the dataset directory."""

    HF_cache_dir: str = None
    """Path to user-defined HF cache"""

    output_dir: str = ''
    """Directory to save model checkpoints."""

    data_config: str = 'genmanip_joint'
    """Data configuration name from DATA_CONFIG_MAP."""

    # Training parameters
    batch_size: int = 16
    """Batch size per GPU for training."""

    max_steps: int = 10000
    """Maximum number of training steps."""

    num_gpus: int = 1
    """Number of GPUs to use for training."""

    save_steps: int = 500
    """Number of steps between saving checkpoints."""

    compute_dtype: str = 'bfloat16'
    """Data type for computation (e.g., 'float32', 'bfloat16')."""

    # Model parameters
    base_model_path: str = ''
    """Path or HuggingFace model ID for the base model."""

    tune_llm: bool = False
    """Whether to fine-tune the language model backbone."""

    tune_visual: bool = True
    """Whether to fine-tune the vision tower."""

    tune_projector: bool = True
    """Whether to fine-tune the projector."""

    tune_diffusion_model: bool = True
    """Whether to fine-tune the diffusion model."""

    resume_from_checkpoint: bool = False
    """Whether to resume from a checkpoint."""

    # Advanced training parameters
    learning_rate: float = 1e-4
    """Learning rate for training."""

    weight_decay: float = 1e-5
    """Weight decay for AdamW optimizer."""

    warmup_ratio: float = 0.05
    """Ratio of total training steps used for warmup."""

    lora_rank: int = 0
    """Rank for the LORA model."""

    lora_alpha: int = 16
    """Alpha value for the LORA model."""

    lora_dropout: float = 0.1
    """Dropout rate for the LORA model."""

    dataloader_num_workers: int = 8
    """Number of workers for data loading."""

    report_to: str = 'wandb'
    """Where to report training metrics (e.g., 'wandb', 'tensorboard')."""

    # Data loading parameters
    embodiment_tag: str = 'new_embodiment'
    """Embodiment tag to use for training. e.g. 'new_embodiment'"""

    video_backend: str = 'torchcodec'
    """Video backend to use for training. [torchcodec, decord, torchvision_av]"""

    augsteps: int = 4
    """number of extra steps for augmentation when gripper changes"""

    gradient_accumulation_steps: int = 1

    pad_center_crop: bool = False

    use_pretrained_model: bool = False
    """Whether to use a pretrained model."""

    skip_unlabeled: bool = False
