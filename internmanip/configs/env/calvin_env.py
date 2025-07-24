from pydantic import BaseModel, Field
from typing import Optional


class CalvinEnvSettings(BaseModel):
    show_gui: Optional[bool] = Field(default=False, description="Whether to show the gui")
    diverse_inst: Optional[bool] = Field(default=False, description="Whether to use diverse instances")
    num_sequences: Optional[int] = Field(default=1000, description="Number of eval sequences")
    episode_length: Optional[int] = Field(default=360, description="Episode length")
    debug: Optional[bool] = Field(default=False, description="Whether to use debug mode")
    reset: Optional[bool] = Field(default=False, description="Whether to reset the environment")
    seed: Optional[int] = Field(default=42, description="Set seed for calvin env")