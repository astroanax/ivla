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

import os, sys
from typing import Any, Dict, List, Optional, Union
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import shutil
from dataclasses import dataclass
from pathlib import Path
import json
import tyro
import yaml
import torch

from transformers import TrainingArguments, TrainerCallback
from transformers import AutoModel, AutoConfig

from internmanip.trainer.base import BaseTrainer
from internmanip.configs.dataset.data_config import DATA_CONFIG_MAP
from internmanip.dataset.embodiment_tags import EmbodimentTag
from internmanip.dataset.transform.base import ComposedModalityTransform
from internmanip.dataset.transform.video import VideoCrop, VideoPadSquareCrop
from internmanip.dataset.base import LeRobotSingleDataset, LeRobotMixtureDataset
from internmanip.utils.peft import get_lora_model
from internmanip.model.data_collator_registry import DataCollatorRegistry

from run_configs.base_cfg import TrainCfg

POLICY_NAME_TO_ID = {
    'pi0': 'lerobot/pi0',
    'pi0fast': 'pi0fast_base',
    'gr00t_n1': 'nvidia/GR00T-N1-2B',
    'gr00t_n1_5': 'nvidia/GR00T-N1.5-3B',
    'dp_clip': None,  # No model_id associated
    'act_clip': None, # No model_id associated
}



#####################################################################################
# main training function
#####################################################################################

def main(config: TrainCfg):
    """Main training function."""
    # ------------ load model ------------


    kwargs = config.model_dump()
    kwargs.pop('model_type')

    if config.base_model_path == '':
        config.base_model_path = POLICY_NAME_TO_ID[config.model_type]
    if config.base_model_path is None:
        model_cfg = AutoConfig.for_model(config.model_type, **kwargs)
        model = AutoModel.from_config(model_cfg, **kwargs)
    else:
        # must ensure that if the path is a huggingface model, it should be a repo that has only one model weight
        model = AutoModel.from_pretrained(config.base_model_path, **kwargs)

    model.compute_dtype = config.compute_dtype
    model.config.compute_dtype = config.compute_dtype

    data_collator = DataCollatorRegistry.get_collator(config.model_type)

    # ------------ load dataset ------------
    embodiment_tag = EmbodimentTag(config.embodiment_tag)

    # modality configs and transforms
    data_config_cls = DATA_CONFIG_MAP[config.data_config]
    model_transform, observation_indices, action_indices = model.config.transform()
    modality_configs = data_config_cls.modality_config(observation_indices, action_indices)
    transforms = data_config_cls.transform()

    if config.pad_center_crop:
        transforms = [tr if not isinstance(tr, VideoCrop) else VideoPadSquareCrop(apply_to=tr.apply_to, scale=0.95) for tr in transforms]
    if model_transform is not None:
        transforms.append(model_transform)

    transforms = ComposedModalityTransform(transforms=transforms)
    # data_loader
    if isinstance(config.dataset_path, str):
        train_dataset = LeRobotSingleDataset(
            dataset_path=config.dataset_path,
            modality_configs=modality_configs,
            transforms=transforms,
            embodiment_tag=embodiment_tag,  # This will override the dataset's embodiment tag to "new_embodiment"
            video_backend=config.video_backend,
            cache_dir=config.HF_cache_dir,
            skip_unlabeled=config.skip_unlabeled
        )
    else:
        print('\n' + '='*30)
        print('⚠️  WARNING: MULTIPLE DATASETS DETECTED')
        print('='*30)
        print('You are about to train on multiple datasets simultaneously.')
        print('Please ensure that:')
        print('  1. All datasets have compatible and consistent modality configurations')
        print('  2. The datasets are from the same embodiment or compatible embodiments')
        print('  3. The datasets have similar data distributions and task objectives')
        print('='*30 + '\n')
        single_datasets = []
        for p in config.dataset_path:
            assert os.path.exists(p), f'Dataset path {p} does not exist'
            # We use the same transforms, modality configs, and embodiment tag for all datasets here
            dataset = LeRobotSingleDataset(
                dataset_path=p,
                modality_configs=modality_configs,
                transforms=transforms,
                embodiment_tag=embodiment_tag,
                video_backend=config.video_backend,
                cache_dir=config.HF_cache_dir,
                skip_unlabeled=config.skip_unlabeled
            )
            single_datasets.append(dataset)

        train_dataset = LeRobotMixtureDataset(
            data_mixture=[
                (dataset, 1.0)  # we will use equal weights for all datasets
                for dataset in single_datasets
            ],
            mode='train',
            balance_dataset_weights=config.balance_dataset_weights,
            balance_trajectory_weights=config.balance_trajectory_weights,
            seed=42,
            metadata_config={
                'percentile_mixing_method': 'weighted_average',
            },
        )
        print(f'Loaded {len(single_datasets)} datasets, with {config.dataset_path} ')

    if config.lora_rank > 0:
        model = get_lora_model(
            model,
            rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
        )
    # modify training args
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        run_name=None,
        remove_unused_columns=False,
        deepspeed='',
        gradient_checkpointing=False,
        bf16=True,
        tf32=True,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=False,
        dataloader_persistent_workers=True,
        optim='adamw_torch',
        adam_beta1=0.95,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type='cosine',
        logging_steps=10.0,
        num_train_epochs=300,
        max_steps=config.max_steps,
        save_strategy='steps',
        save_steps=config.save_steps,
        eval_strategy='no',
        save_total_limit=8,
        report_to=config.report_to,
        seed=42,
        do_eval=False,
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=100,
        torch_compile_mode=None,
    )
    # Create the trainer
    trainer = BaseTrainerWrapper(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    # Add checkpoint format callback to ensure experiment_cfg is copied to each checkpoint
    ckpt_format_callback = CheckpointFormatCallback(
        train_dataset=train_dataset, exp_cfg_dir=training_args.output_dir
    )
    trainer.add_callback(ckpt_format_callback)

    # Log dataloader information
    train_dl_len = len(trainer.get_train_dataloader())
    # eval_dl_len = len(trainer.get_eval_dataloader()) # @note (k2): How to manage eval dataloader?

    print(
        f'train dataloader length: {train_dl_len}\n'
        # f"eval dataloader length: {eval_dl_len}\n"
        f'train dataset length: {len(trainer.train_dataset)}\n'
        f'GPU memory before training: {torch.cuda.memory_allocated() / 1024 / 1024 / 1024} GB',
        flush=True,
    )
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('grm')
    print(f'🔢 Total parameters: {total:,}')
    print(f'🎯 Trainable parameters: {trainable:,}')
    print(f'📉 Non-trainable parameters: {total - trainable:,}')
    def print_model_parameters(model):
        print('\n🔍 [Trainable Parameters]')
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f'[✓] {name:<80} | dtype: {param.dtype}')

        print('\n🚫 [Frozen Parameters (not trainable)]')
        for name, param in model.named_parameters():
            if not param.requires_grad:
                print(f'[✗] {name:<80} | dtype: {param.dtype}')
    print_model_parameters(model)


    import torch.distributed as dist
    print('\n==============================')
    print(f'✅ torch.distributed.is_initialized: {dist.is_initialized()}')
    if dist.is_initialized():
        print(f'🔢 Rank: {dist.get_rank()} / World Size: {dist.get_world_size()}')
        print(f'📦 Backend: {dist.get_backend()}')
    else:
        print('❌ DDP not initialized!')
    print('==============================\n')

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    # trainer.evaluate(eval_dataset=train_dataset)
    trainer.save_state()


class BaseTrainerWrapper(BaseTrainer):
    import torch.nn as nn

    def prediction_step(self, model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]]):

        actions = model.inference(inputs)['action_pred']
        valid_action = inputs['action_mask']
        gt_action = inputs['action'][valid_action]
        pred_action = actions[valid_action].cpu()

        diff = pred_action - gt_action
        import matplotlib.pyplot as plt
        import numpy as np

        mean_diff = diff.abs().mean(dim=0)

        x = np.arange(mean_diff.shape[0])


        num_channels = mean_diff.shape[1]
        fig, axes = plt.subplots(num_channels, 1, figsize=(10, 3 * num_channels), sharex=True)

        x = list(range(pred_action.shape[1]))

        for i in range(num_channels):
            ax = axes[i]
            ax.plot(x, pred_action[0, :, i].cpu().numpy(), label='Predicted', color='blue')
            ax.plot(x, gt_action[0, :, i].cpu().numpy(), label='Ground Truth', color='orange')
            ax.set_title(f'Channel {i}')
            ax.set_ylabel('Mean Value')
            ax.grid(True)
            if i == num_channels - 1:
                ax.set_xlabel('Index (0~15)')
            if i == 0:
                ax.legend()

        plt.tight_layout()
        plt.savefig('debug_value_{}.png'.format(0))
        plt.close()

        return mean_diff


class CheckpointFormatCallback(TrainerCallback):
    """This callback format checkpoint to make them standalone. For now, it copies metadata
    files to /checkpoint-{step}/experiment_cfg/:
    - metadata.json
    """

    def __init__(self, train_dataset, exp_cfg_dir: str):
        """
        Args:
            exp_cfg_dir: Path to the directory containing all experiment metadata
        """
        self.exp_cfg_dir = exp_cfg_dir
        self.train_dataset = train_dataset

    def on_save(self, args, state, control, **kwargs):
        """Called after the trainer saves a checkpoint."""
        if state.is_world_process_zero:
            exp_cfg_dir = Path(self.exp_cfg_dir) / f'checkpoint-{state.global_step}/experiment_cfg'
            if not os.path.exists(exp_cfg_dir):
                os.makedirs(exp_cfg_dir)
            # Copy experiment config directory if provided
            metadata_json = {}
            if os.path.exists(exp_cfg_dir / 'metadata.json'):
                with open(exp_cfg_dir / 'metadata.json', 'r') as f:
                    metadata_json = json.load(f)
            if hasattr(self.train_dataset, 'metadata'):
                # Single dataset
                metadata_json.update(
                    {self.train_dataset.tag: self.train_dataset.metadata.model_dump(mode='json')}
                )
            elif hasattr(self.train_dataset, 'merged_metadata'):
                # Mixture dataset
                for tag, metadata in self.train_dataset.merged_metadata.items():
                    metadata_json.update(
                        {tag: metadata.model_dump(mode='json')}
                    )
            else:
                print(f'Warning: Unknown dataset type {type(self.train_dataset)}')

            with open(exp_cfg_dir / 'metadata.json', 'w') as f:
                json.dump(metadata_json, f, indent=4)

@dataclass
class Args:
    """Path to the training configuration YAML file. If provided, it will override the default values."""
    config: Optional[str] = None


if __name__ == '__main__':
    # Parse arguments using tyro, provide argument in the command
    args = tyro.cli(Args)
    if args.config is not None:
        # Load the configuration from the specified YAML file
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
        # Update the config with the loaded values
        config = TrainCfg(**cfg)
    else:
        config = TrainCfg()

    # Print the config
    print('\n' + '=' * 50)
    print('TRAINING CONFIGURATION:')
    print('=' * 50)
    for key, value in vars(config).items():
        print(f'{key}: {value}')
    print('=' * 50 + '\n')

    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1

    # Validate GPU configuration
    assert (
        config.num_gpus <= available_gpus
    ), f'Number of GPUs requested ({config.num_gpus}) is greater than the available GPUs ({available_gpus})'
    assert config.num_gpus > 0, 'Number of GPUs must be greater than 0'
    print(f'Using {config.num_gpus} GPUs')
    main(config)
