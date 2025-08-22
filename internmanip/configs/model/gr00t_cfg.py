from dataclasses import dataclass, field
from transformers import PretrainedConfig
from internmanip.model.basemodel.transforms.gr00t_n1 import GR00TTransform, GR00TTransform_15
# config
@dataclass
class GR00T_N1_Config(PretrainedConfig):
    model_type = 'gr00t_n1'
    backbone_cfg: dict = field(init=False, metadata={'help': 'Backbone configuration.'})

    action_head_cfg: dict = field(init=False, metadata={'help': 'Action head configuration.'})

    action_horizon: int = field(init=False, metadata={'help': 'Action horizon.'})

    action_dim: int = field(init=False, metadata={'help': 'Action dimension.'})
    compute_dtype: str = field(default='float32', metadata={'help': 'Compute dtype.'})
    observation_indices = [0]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)

    def transform(self):
        transforms = GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(list(range(self.action_horizon))),
                max_state_dim=self.action_head_cfg['max_state_dim'],
                max_action_dim=self.action_head_cfg['max_action_dim'],
            )
        return transforms, self.observation_indices, list(range(self.action_horizon))

# config
@dataclass
class GR00T_N1_5_Config(PretrainedConfig):
    model_type = 'gr00t_n1_5'
    backbone_cfg: dict = field(init=False, metadata={'help': 'Backbone configuration.'})

    action_head_cfg: dict = field(init=False, metadata={'help': 'Action head configuration.'})

    action_horizon: int = field(init=False, metadata={'help': 'Action horizon.'})

    action_dim: int = field(init=False, metadata={'help': 'Action dimension.'})
    compute_dtype: str = field(default='float32', metadata={'help': 'Compute dtype.'})
    observation_indices = [0]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)

    def transform(self):
        transforms = GR00TTransform_15(
            state_horizon=len(self.observation_indices),
            action_horizon=len(list(range(self.action_horizon))),
            max_state_dim=self.action_head_cfg['max_state_dim'],
            max_action_dim=self.action_head_cfg['max_action_dim'],
        )
        return transforms, self.observation_indices, list(range(self.action_horizon))
