from internmanip.env.base import EnvWrapper
from internmanip.configs.env.env_cfg import EnvCfg
from internmanip.configs.env.calvin_env import CalvinEnvSettings
from omegaconf import OmegaConf
import hydra
from typing import Optional
import os
from pathlib import Path
import numpy as np


class CalvinEnv(EnvWrapper):
    def __init__(self, config: EnvCfg):
        super().__init__(config)
        
        # set egl device
        os.environ['PYOPENGL_PLATFORM'] = 'egl'
        egl_device = str(self.config.device_id) if self.config.device_id is not None else os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')[0]
        os.environ["EGL_VISIBLE_DEVICES"] = egl_device
        
        if self.config.env_settings is not None:
            if isinstance(self.config.env_settings, CalvinEnvSettings):
                print(f"calvin env settings: \n{self.config.env_settings.model_dump_json(indent=4)}")
            else:
                raise ValueError(f"Invalid env_settings type for calvin env: {type(self.config.env_settings)}")
        else:
            self.config.env_settings = CalvinEnvSettings()
        
        self.env_settings = self.config.env_settings

        self.env_config_path = self.config.config_path if self.config.config_path is not None else f"{Path(__file__).parents[1]}/benchmarks/utils/calvin/merged_config.yaml"
        if isinstance(self.env_config_path, list):
            raise ValueError(f"config_path for calvin env only supports single path, but got {type(self.env_config_path)}")
        
        show_gui = self.env_settings.show_gui
        render_conf = OmegaConf.load(self.env_config_path)
        if not hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
            hydra.initialize(".")
        self.env = hydra.utils.instantiate(render_conf.env, show_gui=show_gui, use_vr=False, use_scene_info=True, seed=self.config.env_settings.seed)
        np.random.seed(self.config.env_settings.seed)
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        obs = self.env.reset(robot_obs=options.get("robot_obs", None), scene_obs=options.get("scene_obs", None))
        return obs
    
    def get_obs(self):
        return self.env.get_obs()
    
    def get_info(self):
        return self.env.get_info()
    
    def step(self, action):
        return self.env.step(action)
