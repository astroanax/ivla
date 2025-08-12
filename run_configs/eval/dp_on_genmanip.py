from internmanip.configs import *
from internmanip.benchmarks.genmanip.config.env_config import FrankaCameraEnable, AlohaSplitCameraEnable


eval_cfg = EvalCfg(
    eval_type='genmanip',
    agent=AgentCfg(
        agent_type='dp_clip',
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
            dataset_path='InternRobotics/InternData-GenmanipBench',
            eval_tasks=['instruction_16_1'],
            res_save_path=f'{Path(__file__).absolute().parents[2]}/logs/eval/dp_on_genmanip',
            is_save_img=True,
            robot_type='franka',
            gripper_type='panda',
            franka_camera_enable=FrankaCameraEnable(
                realsense=True, obs_camera=True, obs_camera_2=True
            ),
            aloha_split_camera_enable=AlohaSplitCameraEnable(
                top_camera=True, left_camera=True, right_camera=True
            ),
            depth_obs=False,
            max_step=500,
            max_success_step=50,
            env_num=1,
            physics_dt=1/30,
            rendering_dt=1/30,
            headless=True,
            ray_distribution=None,
        ),
    ),
)
