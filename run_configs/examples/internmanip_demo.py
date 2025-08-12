from internmanip.configs import *
from pathlib import Path


eval_cfg = EvalCfg(
    eval_type='simpler',
    agent=AgentCfg(
        agent_type='pi0',
        base_model_path='InternRobotics/Pi0_GoogleRobot_meanstd',
        agent_settings={
            'policy_setup': 'google_robot',
            'action_scale': 1.0,
            'exec_horizon': 1,
            'action_ensemble_temp': -0.8,
            'embodiment_tag': 'new_embodiment',
            'denoising_steps': 16,
        },
        model_kwargs={
            'HF_cache_dir': None,
        },
        server_cfg=ServerCfg(
            server_host='localhost',
            server_port=5000,
        ),
    ),
    env=EnvCfg(
        env_type='simpler',
        device_id=None,
        episodes_config_path=[
                f'{Path(__file__).absolute().parents[2]}/internmanip/benchmarks/utils/SimplerEnv/google_robot/variant_aggregation/move_near.json',
                f'{Path(__file__).absolute().parents[2]}/internmanip/benchmarks/utils/SimplerEnv/google_robot/variant_aggregation/open_and_close_drawer.json',
                f'{Path(__file__).absolute().parents[2]}/internmanip/benchmarks/utils/SimplerEnv/google_robot/variant_aggregation/pick_coke_can.json',
                f'{Path(__file__).absolute().parents[2]}/internmanip/benchmarks/utils/SimplerEnv/google_robot/variant_aggregation/place_in_drawer.json',

                f'{Path(__file__).absolute().parents[2]}/internmanip/benchmarks/utils/SimplerEnv/google_robot/visual_matching/move_near.json',
                f'{Path(__file__).absolute().parents[2]}/internmanip/benchmarks/utils/SimplerEnv/google_robot/visual_matching/open_and_close_drawer.json',
                f'{Path(__file__).absolute().parents[2]}/internmanip/benchmarks/utils/SimplerEnv/google_robot/visual_matching/pick_coke_can.json',
                f'{Path(__file__).absolute().parents[2]}/internmanip/benchmarks/utils/SimplerEnv/google_robot/visual_matching/place_in_drawer.json',

            ]
    ),
    logging_dir=f'{Path(__file__).absolute().parents[2]}/logs/demo/pi0_on_simpler/google_robot',
    distributed_cfg=DistributedCfg(
        num_workers=4,
        ray_head_ip='10.150.91.18', # or "auto"
        include_dashboard=True,
        dashboard_port=8265,
    )
)
