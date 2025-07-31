from internmanip.configs import AgentCfg
from enum import Enum


class AgentRegistry(Enum):
    """
    Registry of agent subclasses.
    The key is the agent type.
    The value is the agent subclass.
    """
    GR00T_N1 = 'Gr00t_N1_Agent'
    GR00T_N15 = 'Gr00t_N1_Agent'
    GR00T_N15_GENMANIP = 'Gr00tAgent_Genmanip'
    PI0 = 'PI0Agent'
    DP = 'DPAgent'

    @property
    def value(self):
        if self.name == 'GR00T_N1':
            from internmanip.agent.simpler_agent import SimplerAgent
            return SimplerAgent
        elif self.name == 'GR00T_N15':
            from internmanip.agent.simpler_agent import SimplerAgent
            return SimplerAgent
        elif self.name == 'GR00T_N15_GENMANIP':
            from internmanip.agent.genmanip_agent import GenmanipAgent
            return GenmanipAgent
        elif self.name == 'DP':
            from internmanip.agent.dp_agent_genmanip import DPAgent
            return DPAgent
        elif self.name == 'PI0':
            from internmanip.agent.simpler_agent import SimplerAgent
            return SimplerAgent
        else:
            raise ValueError(f'Invalid agent type: {self.name}. Only {[agent_type.name for agent_type in AgentRegistry]} are registered.')


class BaseAgent:

    def __init__(self, config: AgentCfg):
        self.config = config

        if self.config.server_cfg is not None:
            from internmanip.agent.utils import PolicyClient
            self.policy_model = PolicyClient(self.config)
        else:
            # TODO: should change the kwargs
            from internmanip import model
            from transformers import AutoModel
            if config.base_model_path is None:
                model = AutoModel.from_config(config.model_cfg, **config.model_kwargs)
            else:
                # must ensure that if the path is a huggingface model, it should be a repo that has only one model weight
                model = AutoModel.from_pretrained(config.base_model_path, **config.model_kwargs)
            self.policy_model = model

    def step(self):
        raise NotImplementedError('Not implemented in base agent class')

    def reset(self):
        raise NotImplementedError('Not implemented in base agent class')

    @classmethod
    def init(cls, config: AgentCfg):
        """
        Init a agent instance from a config.
        """
        print(f'Initializing agent {config.agent_type}')
        return AgentRegistry[config.agent_type].value(config)
