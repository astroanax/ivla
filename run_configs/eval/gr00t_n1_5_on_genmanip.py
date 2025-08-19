from internmanip.configs import *
from internmanip.benchmarks.genmanip.config.env_config import FrankaCameraEnable, AlohaSplitCameraEnable


eval_cfg = EvalCfg(
    eval_type='genmanip',
    agent=AgentCfg(
        agent_type='gr00t_n1_5_genmanip',
        base_model_path='/PATH/TO/YOUR/FINETUNED_CHECKPOINT',
        agent_settings={
            'data_config': 'genmanip_v1',
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
            # Optional (if you have local path to the dataset): 
            # dataset_path='./data/dataset/IROS-2025-Challenge-Manip/validation',
            res_save_path=f'{Path(__file__).absolute().parents[2]}/logs/eval/dp_on_genmanip',
            is_save_img=False,
            aloha_split_camera_enable=AlohaSplitCameraEnable(
                top_camera=True, left_camera=True, right_camera=True
            ),
        ),
    ),
)
