"""
Training configuration for InternVLA-M1 on IROS Challenge dataset.

Usage:
    torchrun --nnodes 1 --nproc_per_node 8 \\
        scripts/train/train.py \\
        --config run_configs/train/internvla_m1_iros.yaml
"""

# This is a Python config file that will be converted to YAML
# You can also create internvla_m1_iros.yaml directly

config = {
    # Model configuration
    'model_type': 'internvla_m1',
    
    # Dataset configuration
    'dataset_path': './data/dataset/IROS-2025-Challenge-Manip/train',
    'data_config': 'arx',  # or 'aloha_v4' for ALOHA robot
    
    # Model checkpoint (optional - for fine-tuning)
    'base_model_path': None,  # Set to checkpoint path for fine-tuning
    
    # Model architecture settings
    'model_config': {
        'action_horizon': 16,
        'action_dim': 7,
        'num_ddim_steps': 5,
        'cfg_scale': 1.5,
        'use_ddim': True,
        'image_size': [224, 224],
        'framework_cfg': {
            'qwen_vl': {
                'model_path': 'Qwen/Qwen2.5-7B-Instruct',
                'hidden_size': 3584,
                'select_layers': list(range(0, 28)),
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
                'in_channels': 7,
                'hidden_size': 768,
                'num_layers': 12,
                'num_heads': 12,
            }
        }
    },
    
    # Training hyperparameters
    'training': {
        'num_epochs': 50,
        'batch_size': 8,  # Per GPU
        'global_batch_size': 64,  # Total across all GPUs
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'warmup_steps': 1000,
        'gradient_accumulation_steps': 8,
        'max_grad_norm': 1.0,
        'mixed_precision': 'bf16',  # or 'fp16'
    },
    
    # Data loading
    'dataloader': {
        'num_workers': 4,
        'prefetch_factor': 2,
        'pin_memory': True,
    },
    
    # Optimizer
    'optimizer': {
        'type': 'adamw',
        'betas': [0.9, 0.999],
        'eps': 1e-8,
    },
    
    # Learning rate scheduler
    'scheduler': {
        'type': 'cosine',
        'warmup_ratio': 0.02,
        'min_lr': 1e-6,
    },
    
    # Checkpointing
    'checkpoint': {
        'save_dir': './checkpoints/internvla_m1',
        'save_steps': 1000,
        'save_total_limit': 5,
        'resume_from_checkpoint': None,
    },
    
    # Logging
    'logging': {
        'log_steps': 10,
        'use_wandb': True,
        'wandb_project': 'iros-2025-internvla-m1',
        'wandb_run_name': 'internvla_m1_iros',
    },
    
    # Evaluation during training
    'evaluation': {
        'eval_steps': 1000,
        'eval_dataset': './data/dataset/IROS-2025-Challenge-Manip/validation',
    },
    
    # HuggingFace cache
    'hf_cache_dir': None,  # Use default or specify custom path
}
