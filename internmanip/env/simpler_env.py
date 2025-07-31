from internmanip.env.base import EnvWrapper
from internmanip.configs import EnvCfg
from internmanip.configs.env.simpler_env import SimplerEnvSettings
from simpler_env.utils.env.env_builder import get_robot_control_mode, build_maniskill2_env
from simpler_env.evaluation.argparse import parse_range_tuple
import numpy as np
from sapien.core import Pose
from transforms3d.euler import euler2quat
from typing import Optional
import os


class SimplerEnv(EnvWrapper):
    def __init__(self, config: EnvCfg):
        super().__init__(config)

        egl_device = str(self.config.device_id) if self.config.device_id is not None else os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')[0]
        os.environ['EGL_VISIBLE_DEVICES'] = egl_device

        if self.config.env_settings is not None:
            if isinstance(self.config.env_settings, SimplerEnvSettings):
                print(f'simpler env settings: \n{self.config.env_settings.model_dump_json(indent=4)}')
            else:
                raise ValueError(f'Invalid env_settings type for simpler env: {type(self.config.env_settings)}')
        else:
            # default task: google_robot_pick_coke_can
            self.config.env_settings = SimplerEnvSettings(task_name='google_robot_pick_coke_can')

        # env will be re-built in each evaluation episode
        # self._build_env(self.config.env_settings)

        os.environ['DISPLAY'] = ''
        # prevent a single jax process from taking up all the GPU memory
        os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

    def _build_env(self, env_settings: SimplerEnvSettings):
        """
        Build the environment. SimplerEnv builds different environment for various tasks.
        """
        print(f'new env settings: \n{env_settings.model_dump_json(indent=4)}')
        self.env_settings = env_settings

        # env args: robot pose
        self.robot_init_xs = parse_range_tuple(self.env_settings.robot_init_x_range)
        self.robot_init_ys = parse_range_tuple(self.env_settings.robot_init_y_range)
        self.robot_init_quats = []
        for r in parse_range_tuple(self.env_settings.robot_init_rot_rpy_range[:3]):
            for p in parse_range_tuple(self.env_settings.robot_init_rot_rpy_range[3:6]):
                for y in parse_range_tuple(self.env_settings.robot_init_rot_rpy_range[6:]):
                    self.robot_init_quats.append((Pose(q=euler2quat(r, p, y)) * Pose(q=self.env_settings.robot_init_rot_quat_center)).q)
        # env args: object position
        if self.env_settings.obj_variation_mode == 'xy':
            self.obj_init_xs = parse_range_tuple(self.env_settings.obj_init_x_range)
            self.obj_init_ys = parse_range_tuple(self.env_settings.obj_init_y_range)
        # update logging info (additional_env_save_tags) if using a different camera from default
        if self.env_settings.obs_camera_name is not None:
            if self.env_settings.additional_env_save_tags is None:
                self.env_settings.additional_env_save_tags = f'obs_camera_{self.env_settings.obs_camera_name}'
            else:
                self.env_settings.additional_env_save_tags = self.env_settings.additional_env_save_tags + f'_obs_camera_{self.env_settings.obs_camera_name}'

        self.control_mode = get_robot_control_mode(self.env_settings.robot, None)

        # build environment
        if self.env_settings.additional_env_build_kwargs is None:
            self.env_settings.additional_env_build_kwargs = {}
        if self.env_settings.enable_raytracing:
            ray_tracing_dict = {'shader_dir': 'rt'}
            ray_tracing_dict.update(self.env_settings.additional_env_build_kwargs)
            # put raytracing dict keys before other keys for compatibility with existing result naming and metric calculation
            self.env_settings.additional_env_build_kwargs = ray_tracing_dict
        kwargs = dict(
            obs_mode='rgbd',
            robot=self.env_settings.robot,
            sim_freq=self.env_settings.sim_freq,
            control_mode=self.control_mode,
            control_freq=self.env_settings.control_freq,
            max_episode_steps=self.env_settings.max_episode_steps,
            scene_name=self.env_settings.scene_name,
            camera_cfgs={'add_segmentation': True},
            rgb_overlay_path=self.env_settings.rgb_overlay_path,
        )
        self.env = build_maniskill2_env(
            self.env_settings.env_name,
            **self.env_settings.additional_env_build_kwargs,
            **kwargs
        )

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        env_reset_options = {
            'robot_init_options': {
                'init_xy': np.array([options['robot_init_x'], options['robot_init_y']]),
                'init_rot_quat': options['robot_init_quat'],
            }
        }
        if options.get('obj_init_x') is not None:
            assert options.get('obj_init_y') is not None
            obj_variation_mode = 'xy'
            env_reset_options['obj_init_options'] = {
                'init_xy': np.array([options['obj_init_x'], options['obj_init_y']]),
            }
        else:
            assert options.get('obj_episode_id') is not None
            obj_variation_mode = 'episode'
            env_reset_options['obj_init_options'] = {
                'episode_id': options['obj_episode_id'],
            }
        obs, _ = self.env.reset(options=env_reset_options)
        return obs

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        return obs, reward, done, truncated, info

    def __getattr__(self, attr_name):
        # used to call some methods inside the env wrapper, such as self.env.get_language_instruction() and self.env.is_final_subtask()
        # NOTE: when gymnasium >= 1.0.0, some apis are not accessible directly through the env instance, need to use env.unwrapped.get_language_instruction() instead
        # Ref:https://github.com/simpler-env/SimplerEnv/issues/44
        import gymnasium
        if int(gymnasium.__version__.split('.')[0]) < 1:
            return getattr(self.env, attr_name)
        else:
            return getattr(self.env.unwrapped, attr_name)
