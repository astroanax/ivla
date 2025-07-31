from typing import List, Union, Optional, Literal

from pydantic import BaseModel, Field


class CameraEnable(BaseModel):
    realsense: bool = Field(default=False, description='the first-person perspective of the gripper(follow the movement of the gripper)')
    obs_camera: bool = Field(default=False, description='the third-person perspective from the side and back of the table (facing the tabletop)')
    obs_camera_2: bool = Field(default=False, description='the third-person perspective in front of the table (facing the tabletop)')


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
    episode_list: List[EpisodeInfo] = Field(description='episode info list')
    camera_enable: CameraEnable = Field(default=CameraEnable(), description='whether to enable cameras in the environment')
    depth_obs: bool = Field(default=False, description='is needed depth image of each camera in obs')
    gripper_type: Literal['panda', 'robotiq'] = Field(default='panda', description="type of gripper, must be 'panda' or 'robotiq'")
    env_num: int = Field(default=1, description='the number of parallel execution environments')
    max_step: int = Field(default=500, description='max action step number per episode')
    max_success_step: int = Field(default=100, description='the max number of successful steps in each episode for early stopping')
    physics_dt: float = Field(default=1/30, description='physics_dt of the simulator')
    rendering_dt: float = Field(default=1/30, description='rendering_dt of the simulator')
    headless: bool = Field(default=True, description='whether to show the gui')
    ray_distribution: Union[RayDistributionCfg, None] = Field(default=None, description='ray-based distributed parameters')
