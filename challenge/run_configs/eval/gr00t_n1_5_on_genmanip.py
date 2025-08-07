from internmanip.configs import *
from internmanip.benchmarks.genmanip.config.env_config import FrankaCameraEnable, AlohaSplitCameraEnable


eval_cfg = EvalCfg(
    eval_type='genmanip',
    agent=AgentCfg(
        agent_type='gr00t_n1_5_genmanip',
        base_model_path='./data/model',
        agent_settings={
            'data_config': 'aloha_v3',
            'embodiment_tag': 'new_embodiment',
            'pred_action_horizon': 16,
            'adaptive_ensemble_alpha': 0.5,
        },
        model_kwargs={
            'HF_cache_dir': None,
        },
        server_cfg=ServerCfg(
            server_host='localhost',
            server_port=5000
        ),
    ),
    env=EnvCfg(
        env_type='genmanip',
        env_settings=GenmanipEnvSettings(
            dataset_path='./data/dataset',
            res_save_path='./results',
            is_save_img=False,
            aloha_split_camera_enable=AlohaSplitCameraEnable(
                top_camera=True, left_camera=True, right_camera=True
            ),
        ),
    ),
)
