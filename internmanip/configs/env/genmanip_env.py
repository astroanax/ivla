from typing import Optional
from pydantic import Field

from internmanip.benchmarks.genmanip.config.env_config import *


class GenmanipEnvSettings(EnvSettings):
    dataset_path: str = Field(
        default=None,
        description='genmanip dataset path'
    )
    eval_tasks: List[str] = Field(
        default=[],
        description='Specify some test tasks (list of relative task path to be \
            evaluated under dataset_path. e.g. IROS_C_V3_Aloha_seen/collect_three_glues)'
    )
    res_save_path: Optional[str] = Field(
        default=None,
        description='metirc & image & obs save path(if None, do not save)'
    )
    is_save_img: bool = Field(
        default=False,
        description='when True, saves environment observation images (res_save_path must not be None)'
    )
