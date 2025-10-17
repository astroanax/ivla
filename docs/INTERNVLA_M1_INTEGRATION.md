# InternVLA-M1 Integration for InternManip

This document provides a complete guide for using InternVLA-M1 with the InternManip framework for the IROS 2025 Challenge.

## Overview

InternVLA-M1 is a spatially guided vision-language-action framework that combines:
- **Qwen2.5 VL backbone** for vision-language understanding
- **DINO encoder** for dense multi-view spatial features
- **Layer-wise QFormer** for multi-layer feature aggregation
- **DiT diffusion head** for future action sequence prediction

Reference: [InternVLA-M1 Paper](https://arxiv.org/abs/2510.13778) | [GitHub](https://github.com/InternRobotics/InternVLA-M1)

## Installation

### 1. Prerequisites

Ensure you have the InternManip environment set up:
```bash
cd /home/b231090pe/rehan/internmanip
source .venv/your_env/bin/activate
```

### 2. Install InternVLA-M1 Dependencies

The InternVLA-M1 repository should be in the parent directory. If not already installed:

```bash
cd InternVLA-M1
pip install -r requirements.txt
pip install qwen-vl-utils
```

### 3. Verify Installation

Check that InternVLA-M1 modules are accessible:
```bash
python -c "from InternVLA.model.framework.M1 import InternVLA_M1; print('✓ InternVLA-M1 available')"
```

## Model Files Structure

The integration adds the following files to InternManip:

```
internmanip/
├── configs/model/
│   └── internvla_m1_cfg.py              # Model configuration
├── model/
│   └── basemodel/
│       ├── transforms/
│       │   └── internvla_m1.py          # Transform class
│       └── internvla_m1/
│           ├── __init__.py
│           └── internvla_m1.py          # Model wrapper
├── agent/
│   ├── base.py                          # Updated with INTERNVLA_M1 registry
│   └── internvla_m1_agent.py            # Agent implementation
└── challenge/
    └── run_configs/eval/
        └── internvla_m1_on_real_dummy.py # Evaluation config
```

## Configuration

### Model Configuration

The model config is defined in `internmanip/configs/model/internvla_m1_cfg.py`:

```python
from internmanip.configs.model.internvla_m1_cfg import InternVLA_M1_Config

config = InternVLA_M1_Config(
    action_horizon=16,        # Future action steps to predict
    action_dim=7,             # Action space dimension
    num_ddim_steps=5,         # DDIM sampling steps
    cfg_scale=1.5,            # Classifier-free guidance scale
    image_size=[224, 224],    # Input image size
)
```

### Agent Settings

Configure the agent in evaluation configs:

```python
agent=AgentCfg(
    agent_type='internvla_m1_on_realarx',
    eval_type='internvla_m1',
    base_model_path='./data/model/internvla_m1',
    agent_settings={
        'data_config': 'arx',           # Dataset config: 'arx' or 'aloha_v4' or 'genmanip_v1'
        'embodiment_tag': 'new_embodiment',
        'pred_action_horizon': 16,
        'action_ensemble': False,       # Enable adaptive action ensembling
        'adaptive_ensemble_alpha': 0.5, # Ensembling weight parameter
    },
    model_kwargs={
        'HF_cache_dir': None,
        'torch_dtype': 'bfloat16'
    },
)
```

## Usage

### Evaluation with Dummy Environment

Test InternVLA-M1 integration with the dummy evaluator:

```bash
# Terminal 1: Start the agent server
python -m scripts.eval.start_agent_server --ports 5000

# Terminal 2: Run the dummy evaluator
python -m challenge.scripts.start_dummy_evaluator \
    --config challenge/run_configs/eval/internvla_m1_on_real_dummy.py \
    --server
```

### Evaluation with Real Robot

For real robot evaluation, ensure you have:
1. Downloaded the IROS 2025 validation dataset
2. Configured camera settings in the eval config
3. Started the agent server on the same network

```bash
python -m challenge.scripts.start_dummy_evaluator \
    --config challenge/run_configs/eval/internvla_m1_on_real_dummy.py \
    --dataset_path ./data/dataset/IROS-2025-Challenge-Manip/validation \
    --res_save_path ./results/internvla_m1 \
    --server
```

### Distributed Evaluation

For faster evaluation across multiple GPUs:

```bash
# Start Ray cluster
ray start --head --num-gpus=4

# Run distributed evaluation
python -m challenge.scripts.start_dummy_evaluator \
    --config challenge/run_configs/eval/internvla_m1_on_real_dummy.py \
    --distributed \
    --distributed_num_worker 4 \
    --server_port [5000,5001,5002,5003] \
    --server
```

## Customization

### Using Different Data Configs

InternVLA-M1 supports multiple robot embodiments:

**For ALOHA robot:**
```python
agent_settings={
    'data_config': 'arx',  # or 'aloha_v4'
    'embodiment_tag': 'new_embodiment',
}
```

**For GenManip environment:**
```python
agent_settings={
    'data_config': 'genmanip_v1',
    'embodiment_tag': 'new_embodiment',
}
```

### Adjusting Inference Parameters

Fine-tune inference for better performance:

```python
config = InternVLA_M1_Config(
    num_ddim_steps=10,      # More steps = higher quality but slower
    cfg_scale=2.0,          # Higher = more adherence to conditioning
    use_ddim=True,          # DDIM vs DDPM sampling
)
```

### Action Ensembling

Enable adaptive ensembling for smoother execution:

```python
agent_settings={
    'action_ensemble': True,
    'adaptive_ensemble_alpha': 0.5,  # 0 = uniform, higher = more recent bias
}
```

## Model Architecture

InternVLA-M1 processes observations through the following pipeline:

1. **Multi-view Image Encoding**
   - Qwen2.5 VL processes language + images → contextual embeddings
   - DINO extracts dense spatial features from each camera view

2. **Feature Fusion**
   - Layer-wise QFormer aggregates multi-layer VLM features
   - Concatenates with DINO spatial tokens

3. **Action Prediction**
   - DiT diffusion head predicts future action sequence
   - DDIM sampling generates smooth trajectories

## Troubleshooting

### Import Errors

**Error:** `ImportError: cannot import name 'InternVLA_M1'`

**Solution:** Ensure InternVLA-M1 is in the parent directory and added to PYTHONPATH:
```bash
export PYTHONPATH="/home/b231090pe/rehan/internmanip/InternVLA-M1:$PYTHONPATH"
```

### Model Loading Issues

**Error:** `HFValidationError` or `RepositoryNotFoundError`

**Solution:** Download the model checkpoint locally:
```bash
git lfs clone https://huggingface.co/InternRobotics/InternVLA-M1 ./data/model/internvla_m1
```

### CUDA Out of Memory

**Solution:** Reduce batch size or use gradient checkpointing:
- Set `torch_dtype='float16'` instead of `bfloat16`
- Reduce `num_ddim_steps` (e.g., from 10 to 5)
- Use single GPU evaluation instead of distributed

### Camera Configuration Issues

**Error:** Missing camera views

**Solution:** Update camera mappings in `internvla_m1_agent.py`:
```python
camera_mappings = {
    'left_camera': ['left_camera', 'hand_left', 'your_left_cam_name'],
    'right_camera': ['right_camera', 'hand_right', 'your_right_cam_name'],
    'top_camera': ['top_camera', 'head', 'your_top_cam_name'],
}
```

## Performance Tips

1. **Use bfloat16 precision** for faster inference on modern GPUs
2. **Enable action ensembling** for smoother robot motions
3. **Adjust DDIM steps** based on speed-quality tradeoff
4. **Use distributed evaluation** for large-scale benchmarking
5. **Cache model weights locally** to avoid repeated downloads

## Citation

If you use InternVLA-M1 in your work, please cite:

```bibtex
@article{internvla2024,
  title={InternVLA-M1: A Spatially Guided Vision-Language-Action Framework for Generalist Robot Policy},
  author={Chen, Xinyi and Chen, Yilun and Fu, Yanwei and Gao, Ning and Jia, Jiaya and Jin, Weiyang and Li, Hao and Mu, Yao and Pang, Jiangmiao and Qiao, Yu and Tian, Yang and Wang, Bin and Wang, Bolun and Wang, Fangjing and Wang, Hanqing and Wang, Tai and Wang, Ziqin and Wei, Xueyuan and Wu, Chao and Yang, Shuai and Ye, Jinhui and Yu, Junqiu and Zeng, Jia and Zhang, Jingjing and Zhang, Jinyu and Zhang, Shi and Zheng, Feng and Zhou, Bowen and Zhu, Yangkun},
  journal={arXiv preprint arXiv:2510.13778},
  year={2024}
}
```

## Support

For issues specific to InternManip integration:
- Check the InternManip documentation: https://internrobotics.github.io/
- Open an issue in the InternManip repository

For InternVLA-M1 model questions:
- Check the InternVLA-M1 repository: https://github.com/InternRobotics/InternVLA-M1
- Read the paper: https://arxiv.org/abs/2510.13778

## Competition Tips 🏆

For the IROS 2025 Challenge:

1. **Fine-tune on real robot data** for better sim-to-real transfer
2. **Use the adaptive ensemble** to handle varying action execution speeds
3. **Test with different cfg_scale values** to find optimal conditioning strength
4. **Monitor inference time** to ensure real-time performance
5. **Verify camera calibrations** match the competition setup

Good luck with the competition! 🚀
