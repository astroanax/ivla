from internmanip.configs import *
from pathlib import Path

eval_cfg = EvalCfg(
    eval_type="calvin",
    agent=AgentCfg(
        agent_type="seer",
        model_name_or_path="/cpfs/user/tianshihan/ckpt/19.pth",
        model_kwargs={
            "finetune_type": "calvin",
            "device_id": 0,
            "vit_checkpoint_path": "/cpfs/user/tianshihan/ckpt/mae_pretrain_vit_base.pth",
            "sequence_length": 10,
            "num_resampler_query": 6,
            "num_obs_token_per_image": 9,
            "obs_pred": True,
            "atten_only_obs": False,
            "attn_robot_proprio_state": False,
            "atten_goal": False,
            "atten_goal_state": False,
            "mask_l_obs_ratio": 0.0,
            "calvin_input_image_size": 224,
            "patch_size": 16,
            "mask_ratio": 0.0,
            "num_token_per_timestep": 41,
            "input_self": False,
            "action_pred_steps": 3,
            "transformer_layers": 24,
            "hidden_dim": 384,
            "transformer_heads": 12,
            "phase": "evaluate",
            "gripper_width": False,
            "resume_from_checkpoint": "/cpfs/user/tianshihan/ckpt/19.pth",
            "cast_type": "float32",
            "calvin_eval_max_steps": 360,
        },
        server_cfg=ServerCfg(
            server_host="localhost",
            server_port=5000,
        ),
    ),
    env=EnvCfg(
        env_type="calvin",
        device_id=0,
        config_path=f"{Path(__file__).absolute().parents[2]}/internmanip/benchmarks/utils/calvin/merged_config.yaml",
        env_settings=CalvinEnvSettings(
            num_sequences=100,
        )
    ),
    logging_dir=f"{Path(__file__).absolute().parents[2]}/logs/eval/calvin",
    distributed_cfg=DistributedCfg(
        num_workers=4,
        ray_head_ip="10.150.91.18", # or "auto"
        include_dashboard=True,
        dashboard_port=8265,
    )
)