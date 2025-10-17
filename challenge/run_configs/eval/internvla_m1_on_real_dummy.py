"""
Evaluation configuration for InternVLA-M1 on real robot (dummy test).

This config can be used to test InternVLA-M1 integration with:
    python -m challenge.scripts.start_dummy_evaluator \\
        --config challenge/run_configs/eval/internvla_m1_on_real_dummy.py \\
        --server
"""

from internmanip.configs import *
from internmanip.benchmarks.genmanip.config.env_config import AlohaSplitCameraEnable


eval_cfg = EvalCfg(
    eval_type='dummy',
    agent=AgentCfg(
        agent_type='internvla_m1_on_realarx',
        eval_type='internvla_m1',
        base_model_path='./data/model/internvla_m1',  # Path to InternVLA-M1 checkpoint
        agent_settings={
            'data_config': 'arx',  # or 'aloha_v4' for ALOHA robot
            'embodiment_tag': 'new_embodiment',
            'pred_action_horizon': 16,
            'action_ensemble': False,
            'adaptive_ensemble_alpha': 0,
            'eval_type': 'realarx',
        },
        model_kwargs={
            'HF_cache_dir': None,
            'torch_dtype': 'bfloat16'
        },
        server_cfg=ServerCfg(
            server_host='localhost',
            server_port=5000
        ),
    ),
    env=EnvCfg(
        env_type='dummy',
        env_settings=GenmanipEnvSettings(
            dataset_path='./data/dataset/IROS-2025-Challenge-Manip/validation',
            res_save_path='./results',
            is_save_img=False,
            aloha_split_camera_enable=AlohaSplitCameraEnable(
                top_camera=True,
                left_camera=True,
                right_camera=True
            ),
        ),
    ),
    distributed_cfg=DistributedCfg(
        num_workers=2,
        ray_head_ip='localhost'
    )
)
