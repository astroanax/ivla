from internmanip.configs import *
from internmanip.configs.env.genmanip_env import CameraEnable


eval_cfg = EvalCfg(
    eval_type='genmanip',
    agent=AgentCfg(
        agent_type='gr00t_n15_genmanip',
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
            dataset_path='path/to/genmanip/benchmark_data',
            eval_tasks=['task1', 'task2', ...],
            res_save_path='path/to/save/results',
            is_save_image=True,
            camera_enable=CameraEnable(realsense=True, obs_camera=True, obs_camera_2=True),
            depth_obs=False,
            gripper_type='panda',
            env_num=1,
            max_step=500,
            max_success_step=50,
            physics_dt=1/60,
            rendering_dt=1/60,
            headless=True,
            ray_distribution=None,
        )
    ),
)
