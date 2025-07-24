import os
import copy
import pickle

from internutopia.core.vec_env import Env
from internutopia.core.config import Config, SimConfig
from internutopia.core.config.distribution import RayDistributionCfg as DistributionCfg

from .config.env_config import *
from .config.task_config import *


class ConfigGenerator:
    """
    Configuration for the manipulation task.
    This class is used to define the task settings, including metrics, episodes, and task settings.
    """

    def __init__(self, env_settings: EnvSettings):
        self.env_settings = env_settings

    def create_metrics_cfg(self):
        manipulation_success_metric_cfg = ManipulationSuccessMetricCfg()
        
        return [manipulation_success_metric_cfg]
    
    def create_controllers_cfg(self):
        joint_controller_cfg = JointControllerCfg()
        ik_controller_cfg = InverseKinematicsControllerCfg()

        if self.env_settings.gripper_type=="panda":
            gripper_controller_cfg = GripperControllerCfg()
            return [joint_controller_cfg, gripper_controller_cfg, ik_controller_cfg]
        else:
            return [joint_controller_cfg, ik_controller_cfg]
    
    def create_sensors_cfg(self):
        camera_enable_list = self.env_settings.camera_enable
        assert camera_enable_list, "camera_enable cannot be empty"

        camera_config = []
        for camera_name, enable in camera_enable_list.model_dump().items():
            assert camera_name in ["realsense", "obs_camera", "obs_camera_2"]

            if not enable:
                continue

            per_camera_config = CameraCfg(
                name=camera_name,
                depth_obs=self.env_settings.depth_obs,
                gripper_type=self.env_settings.gripper_type
            )

            camera_config.append(per_camera_config)

        return camera_config
    
    def create_robots_cfg(self):
        if self.env_settings.gripper_type=="panda":
            franka_robot = FrankaPandaRobotCfg(
                controllers=self.create_controllers_cfg(),
                sensors=self.create_sensors_cfg()
            )
        else:
            franka_robot = FrankaRobotiqRobotCfg(
                controllers=self.create_controllers_cfg(),
                sensors=self.create_sensors_cfg()
            )

        return [franka_robot]

    def create_task_config(self):
        task_config = []
        for episode_info in self.env_settings.episode_list:
            meta_info_path = os.path.join(episode_info.episode_path, "meta_info.pkl")
            scene_asset_path = os.path.join(episode_info.episode_path, "scene.usd")

            if not os.path.exists(meta_info_path) or \
                not os.path.exists(scene_asset_path):
                continue

            with open(meta_info_path, "rb") as f:
                meta_info = pickle.load(f)

            per_task_config = ManipulationTaskCfg(
                metrics=self.create_metrics_cfg(),
                scene_asset_path = scene_asset_path,
                robots=self.create_robots_cfg(),
                max_step=self.env_settings.max_step,
                max_success_step=self.env_settings.max_success_step,
                prompt=meta_info["language_instruction"],
                target=meta_info["task_data"]["goal"],
                task_name=episode_info.task_name, # meta_info["task_name"]
                episode_name=episode_info.episode_name, # meta_info["episode_name"]
            )

            task_config.append(per_task_config)
        
        return task_config
    
    def create_simulator_config(self):
        simulator = SimConfig(
            physics_dt=self.env_settings.physics_dt, 
            rendering_dt=self.env_settings.rendering_dt,
            rendering_interval=0,
            use_fabric=False,
            headless=self.env_settings.headless,
            native=self.env_settings.headless
        )
    
        return simulator
    
    def create_env_config(self):
        config = Config(
            simulator=self.create_simulator_config(),
            env_num=self.env_settings.env_num,
            env_offset_size=50.0,
            task_configs=self.create_task_config()
        )

        if self.env_settings.ray_distribution is not None:
            return config.distribute(
                DistributionCfg(**self.env_settings.ray_distribution.model_dump())
            )
        else:
            return config


def import_extensions():
    from .robots import franka_panda
    from .robots import franka_robotiq
    from .sensors import camera
    from .tasks import manipulation_task
    from .controllers import ik_controller 
    from .controllers import joint_controller
    from .controllers import gripper_controller
    from .metrics import manipulation_success_metric


def create_env(env_settings: EnvSettings):
    """
    This function is used to create the environment.
    It initializes the simulator runtime and the environment with the given configuration.
    """
    config_generator = ConfigGenerator(env_settings)
    config = config_generator.create_env_config()

    import_extensions()

    return copy.deepcopy(config), Env(config)
