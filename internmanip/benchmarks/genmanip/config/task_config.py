import os
from pathlib import Path
from typing import List, Optional, Tuple

from internutopia.core.config.metric import MetricCfg
from internutopia.core.config.robot import ControllerCfg, RobotCfg, SensorCfg
from internutopia.core.config.task import TaskCfg

CUR_SCRIPT_DIR = Path(__file__).resolve().parent.parent


# Task config
class ManipulationTaskCfg(TaskCfg):
    type: Optional[str] = 'ManipulationTask'
    metrics: Optional[List[MetricCfg]] = []

    scene_asset_path: Optional[str]
    robots: Optional[List[RobotCfg]]

    max_step: int = 500
    max_success_step: int = 100
    prompt: Optional[str] = None
    target: Optional[List] = None
    task_name: Optional[str] = None
    episode_name: Optional[str] = None


# Sensor config
class CameraCfg(SensorCfg):
    type: Optional[str] = 'Camera'
    prim_path: Optional[str] = None
    enable: bool = False
    depth_obs: bool = False
    resolution: Optional[Tuple[int, int]] = (640, 480)
    robot_type: Optional[str] = None
    gripper_type: Optional[str] = None


# Controller config
class JointControllerCfg(ControllerCfg):
    type: Optional[str] = 'JointController'
    name: str = 'joint_controller'
    joint_names: List[str] = None


class InverseKinematicsControllerCfg(ControllerCfg):
    type: Optional[str] = 'InverseKinematicsController'
    name: str = 'arm_ik_controller'
    robot_description_path: str = os.path.join(
        CUR_SCRIPT_DIR, 'utils/robot_usd/franka/robot_descriptor.yaml'
    )
    robot_urdf_path: str = os.path.join(
        CUR_SCRIPT_DIR, 'utils/robot_usd/franka/lula_franka_gen.urdf'
    )
    end_effector_frame_name: str = 'panda_hand'
    threshold: float = 0.01
    reference: Optional[str] = None


class IKAlohaSplitCuroboControllerCfg(ControllerCfg):
    type: Optional[str] = 'IKAlohaSplitCuroboController'
    name: str = 'arm_ik_aloha_split_curobo_controller'
    curobo_cfg_path: str = os.path.join(
        CUR_SCRIPT_DIR, 'utils/curobo_cfg/piper100/piper100_left_arm.yml'
    )


class GripperControllerCfg(ControllerCfg):
    type: Optional[str] = 'GripperController'
    name: str = 'gripper_controller'


# Robot config
class FrankaPandaRobotCfg(RobotCfg):
    type: Optional[str] = 'FrankaPanda'
    name: Optional[str] = 'robot'
    prim_path: Optional[str] = '/franka'
    position: Optional[Tuple[float, float, float]] = (
        -0.41623175144195557,
        -0.0013529614079743624,
        0.9993137121200562,
    )
    usd_path: Optional[str] = os.path.join(
        CUR_SCRIPT_DIR, 'utils/robot_usd/franka/franka_panda_with_camera.usd'
    )
    controllers: Optional[List[ControllerCfg]] = None
    sensors: Optional[List[SensorCfg]] = None


class FrankaRobotiqRobotCfg(RobotCfg):
    type: Optional[str] = 'FrankaRobotiq'
    name: Optional[str] = 'robot'
    prim_path: Optional[str] = '/franka'
    position: Optional[Tuple[float, float, float]] = (
        -0.41623175144195557,
        -0.0013529614079743624,
        0.9993137121200562,
    )
    usd_path: Optional[str] = os.path.join(
        CUR_SCRIPT_DIR, 'utils/robot_usd/franka/franka_robotiq_with_camera.usd'
    )
    controllers: Optional[List[ControllerCfg]] = None
    sensors: Optional[List[SensorCfg]] = None


class AlohaSplitRobotCfg(RobotCfg):
    type: Optional[str] = 'AlohaSplit'
    name: Optional[str] = 'robot'
    prim_path: Optional[str] = '/aloha_split'
    position: Optional[Tuple[float, float, float]] = (-0.65, 0.0, 0.3)
    usd_path: Optional[str] = os.path.join(
        CUR_SCRIPT_DIR, 'utils/robot_usd/aloha_split/robot.usd'
    )
    controllers: Optional[List[ControllerCfg]] = None
    sensors: Optional[List[SensorCfg]] = None


# Metric config
class ManipulationSuccessMetricCfg(MetricCfg):
    type: Optional[str] = 'ManipulationSuccessMetric'
    name: str = 'manipulation_success_metric'
