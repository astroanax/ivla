from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field


class FrankaCameraEnable(BaseModel):
    realsense: bool = Field(
        default=False,
        description='the first-person perspective of the gripper\
            (follow the movement of the gripper)',
    )
    obs_camera: bool = Field(
        default=False,
        description='the third-person perspective from the side and \
            back of the table (facing the tabletop)',
    )
    obs_camera_2: bool = Field(
        default=False,
        description='the third-person perspective in front of the table (facing the tabletop)',
    )


class AlohaSplitCameraEnable(BaseModel):
    top_camera: bool = Field(
        default=False,
        description="the first-person perspective of the robot's head \
            (looking down at the table and follow the movement of the head)",
    )
    left_camera: bool = Field(
        default=False,
        description='the first-person perspective of the left gripper \
            (follow the movement of the left gripper)',
    )
    right_camera: bool = Field(
        default=False,
        description='the first-person perspective of the right gripper \
            (follow the movement of the right gripper)',
    )


class RayDistributionCfg(BaseModel):
    proc_num: Optional[int] = 1
    gpu_num_per_proc: Optional[float] = 1
    head_address: Optional[str] = None
    working_dir: Optional[str] = None


class EpisodeInfo(BaseModel):
    episode_path: str
    task_name: str
    episode_name: str


class EnvSettings(BaseModel):
    episode_list: List[EpisodeInfo] = Field(
        default=[],
        description='episode info list'
    )
    franka_camera_enable: FrankaCameraEnable = Field(
        default=FrankaCameraEnable(),
        description='whether to enable cameras in the \
            environment (robot_type must be franka)',
    )
    aloha_split_camera_enable: AlohaSplitCameraEnable = Field(
        default=AlohaSplitCameraEnable(),
        description='whether to enable cameras in the \
            environment (robot_type must be aloha_split)',
    )
    depth_obs: bool = Field(
        default=False, description='is needed depth image of each camera in obs'
    )
    robot_type: Literal['franka', 'aloha_split'] = Field(
        default='franka', description='type of robot'
    )
    gripper_type: Literal['panda', 'robotiq'] = Field(
        default='panda',
        description="type of gripper, must be 'panda' or 'robotiq' (under robot_type='franka')",
    )
    env_num: int = Field(default=1, description='the number of parallel execution environments')
    max_step: int = Field(default=500, description='max action step number per episode')
    max_success_step: int = Field(
        default=100,
        description='the max number of successful steps in each episode for early stopping',
    )
    physics_dt: float = Field(default=1 / 30, description='physics_dt of the simulator')
    rendering_dt: float = Field(default=1 / 30, description='rendering_dt of the simulator')
    headless: bool = Field(default=True, description='whether to show the gui')
    ray_distribution: Union[RayDistributionCfg, None] = Field(
        default=None, description='ray-based distributed parameters'
    )
