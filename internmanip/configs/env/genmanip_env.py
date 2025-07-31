from typing import Optional
from pydantic import Field

from internmanip.benchmarks.genmanip.config.env_config import *

ALL_EVAL_TASKS = [
    'scene_seen/brush_paint',
    'scene_seen/colorful_cups',
    'scene_seen/object_select1',
    'scene_seen/object_select2',
    'scene_seen/object_select3',
    'scene_seen/object_select4',
    'scene_seen/ocr_box',
    'scene_seen/select_drink',
    'scene_seen/waste_split',
    'scene_unseen/brush_paint',
    'scene_unseen/colorful_cups',
    'scene_unseen/object_select1',
    'scene_unseen/object_select2',
    'scene_unseen/object_select3',
    'scene_unseen/object_select4',
    'scene_unseen/ocr_box',
    'scene_unseen/select_drink',
    'scene_unseen/waste_split',
]

class GenmanipEnvSettings(EnvSettings):
    dataset_path: str = Field(default=None, description='genmanip dataset path')
    eval_tasks: List[str] = Field(default=ALL_EVAL_TASKS, description='list of relative task path to be evaluated under dataset_path')
    episode_list: List[EpisodeInfo] = Field(default=None, description='episode info list')
    res_save_path: Optional[str] = Field(default=None, description='metirc & image & obs save path(if None, do not save)')
    is_save_img: bool = Field(default=False, description='when True, saves environment observation images (res_save_path must not be None)')
