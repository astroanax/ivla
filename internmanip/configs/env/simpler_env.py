from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
from pathlib import Path


class SimplerEnvSettings(BaseModel):
    policy_setup: Optional[str] = Field(default="google_robot", description="Policy model setup")
    task_name: str = Field(default="google_robot_pick_coke_can", description="Task name")
    env_name: Optional[str] = Field(default="GraspSingleOpenedCokeCanInScene-v0", description="Environment name")
    additional_env_save_tags: Optional[str] = Field(default=None, description="Additional tags to save the environment eval results")
    scene_name: Optional[str] = Field(default=None, description="Scene name")
    enable_raytracing: Optional[bool] = Field(default=False, description="Enable raytracing")
    robot: Optional[str] = Field(default="google_robot_static", description="Robot name")
    obs_camera_name: Optional[str] = Field(default=None, description="Obtain image observation from this camera for policy input. None = default")
    action_scale: Optional[float] = Field(default=1.0, description="Action scale")
    control_freq: Optional[int] = Field(default=3, description="Control frequency")
    sim_freq: Optional[int] = Field(default=513, description="Simulation frequency")
    max_episode_steps: Optional[int] = Field(default=80, description="Maximum episode steps")
    eval_setup: Optional[str] = Field(default="visual_matching", description="Evaluation setup mode for google robot")
    rgb_overlay_path: Optional[str] = Field(default=f"{Path(__file__).parents[2]}/benchmarks/SimplerEnv/ManiSkill2_real2sim/data/real_inpainting/google_coke_can_real_eval_1.png", description="RGB overlay image path")
    robot_init_x_range: Optional[List[float]] = Field(default = [0.35, 0.35, 1], description="Robot initial X range, it means [xmin, xmax, num]")
    robot_init_y_range: Optional[List[float]] = Field(default = [0.20, 0.20, 1], description="Robot initial Y range, it means [ymin, ymax, num]")
    robot_init_rot_quat_center: Optional[List[float]] = Field(default = [1, 0, 0, 0], description="Robot initial quaternion, it means [x, y, z, w]")
    robot_init_rot_rpy_range: Optional[List[float]] = Field(default = [0, 0, 1, 0, 0, 1, 0, 0, 1], description="Robot initial RPY(roll, pitch, yaw) range, it means [rmin, rmax, rnum, pmin, pmax, pnum, ymin, ymax, ynum]")
    obj_variation_mode: Optional[str] = Field(default="xy", description="Whether to vary the xy position of a single object, or to vary predetermined episodes")
    obj_episode_range: Optional[List[int]] = Field(default = [0, 60], description="Object episode range, it means [start, end]")
    obj_init_x_range: Optional[List[float]] = Field(default = [-0.35, -0.12, 5], description="Object initial X range, it means [xmin, xmax, num]")
    obj_init_y_range: Optional[List[float]] = Field(default = [-0.02, 0.42, 5], description="Object initial Y range, it means [ymin, ymax, num]")
    additional_env_build_kwargs: Optional[dict] = Field(default=None, description="Additional environment build kwargs")
    
    @field_validator('task_name')
    def check_task_name(cls, v):
        from simpler_env import ENVIRONMENTS
        if v not in ENVIRONMENTS:
            raise ValueError(f"Invalid task name: {v}")
        return v
    
    @field_validator('policy_setup')
    def check_policy_setup(cls, v):
        if v is not None and v not in ["widowx_bridge", "google_robot"]:
            raise ValueError(f"Invalid policy setup type: {v}")
        return v
    
    @field_validator('env_name')
    def check_env_name(cls, v):
        from simpler_env import ENVIRONMENT_MAP
        from mani_skill2_real2sim.utils.registration import REGISTERED_ENVS
        if v not in set([env_name for env_name, _ in ENVIRONMENT_MAP.values()] + list(REGISTERED_ENVS.keys())):
            raise ValueError(f"Invalid environment name: {v}")
        return v
    
    @field_validator('robot')
    def check_robot(cls, v):
        if v is not None and v not in ["google_robot_static", "widowx", "widowx_sink_camera_setup"]:
            raise ValueError(f"Invalid robot name: {v}")
        return v
    
    @field_validator('rgb_overlay_path')
    def check_rgb_overlay_path(cls, v):
        if v is not None and not Path(v).exists():
            raise ValueError(f"Invalid RGB overlay image path: {v}")
        return v
    
    @field_validator('obj_variation_mode')
    def check_obj_variation_mode(cls, v):
        if v not in ["xy", "episode"]:
            raise ValueError(f"Invalid object variation mode: {v}")
        return v
    
    @field_validator('robot_init_x_range')
    def validate_robot_init_x_range(cls, v):
        if v is not None and len(v) != 3:
            raise ValueError("robot_init_x_range must have exactly 3 elements: [xmin, xmax, num]")
        return v
    
    @field_validator('robot_init_y_range')
    def validate_robot_init_y_range(cls, v):
        if v is not None and len(v) != 3:
            raise ValueError("robot_init_y_range must have exactly 3 elements: [ymin, ymax, num]")
        return v
    
    @field_validator('robot_init_rot_quat_center')
    def validate_robot_init_rot_quat_center(cls, v):
        if v is not None and len(v) != 4:
            raise ValueError("robot_init_rot_quat_center must have exactly 4 elements: [x, y, z, w]")
        return v
    
    @field_validator('robot_init_rot_rpy_range')
    def validate_robot_init_rot_rpy_range(cls, v):
        if v is not None and len(v) != 9:
            raise ValueError("robot_init_rot_rpy_range must have exactly 9 elements: [rmin, rmax, rnum, pmin, pmax, pnum, ymin, ymax, ynum]")
        return v
    
    @field_validator('obj_episode_range')
    def validate_obj_episode_range(cls, v):
        if v is not None and len(v) != 2:
            raise ValueError("obj_episode_range must have exactly 2 elements: [start, end]")
        return v
    
    @field_validator('obj_init_x_range')
    def validate_obj_init_x_range(cls, v):
        if v is not None and len(v) != 3:
            raise ValueError("obj_init_x_range must have exactly 3 elements: [xmin, xmax, num]")
        return v
    
    @field_validator('obj_init_y_range')
    def validate_obj_init_y_range(cls, v):
        if v is not None and len(v) != 3:
            raise ValueError("obj_init_y_range must have exactly 3 elements: [ymin, ymax, num]")
        return v