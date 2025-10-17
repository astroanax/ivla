from internmanip.configs import *
from internmanip.benchmarks.genmanip.config.env_config import FrankaCameraEnable, AlohaSplitCameraEnable


eval_cfg = EvalCfg(
    eval_type='genmanip',
    agent=AgentCfg(
        agent_type='gr00t_n1_5_genmanip',
        base_model_path='./data/model',
        agent_settings={
            'data_config': 'aloha_v4',
            'embodiment_tag': 'new_embodiment',
            'pred_action_horizon': 16,
            'action_ensemble': False,
            'adaptive_ensemble_alpha': 0,
        },
        model_kwargs={
            'HF_cache_dir': None,
            'torch_dtype': 'float16'
        },
        server_cfg=ServerCfg(
            server_host='localhost',
            server_port=5000
        ),
    ),
    env=EnvCfg(
        env_type='genmanip',
        env_settings=GenmanipEnvSettings(
           # dataset_path='./data/dataset/IROS-2025-Challenge-Manip/validation',
            res_save_path='./results',
            is_save_img=False,
            aloha_split_camera_enable=AlohaSplitCameraEnable(
                top_camera=True, left_camera=True, right_camera=True
            ),
        ),
    ),
    distributed_cfg=DistributedCfg(
        num_workers=2,
        ray_head_ip='localhost'
    )
)
