from pydantic import BaseModel, Field, field_validator
from typing import Optional


class DistributedCfg(BaseModel):
    """
    Distributed settings of ray cluster for evaluation.
    """
    num_workers: int = Field(default=2, description='Number of workers')
    ray_head_ip: Optional[str] = Field(default=None, description='The head node ip of the Ray cluster')
    include_dashboard: Optional[bool] = Field(default=True, description='Whether to enable Ray dashboard')
    dashboard_port: Optional[int] = Field(default=8265, description='Ray dashboard port')

    @field_validator('num_workers')
    def validate_num_workers(cls, v):
        if v < 2:
            raise ValueError('Number of workers must be greater than 1 under distributed evaluation mode.')
        return v
