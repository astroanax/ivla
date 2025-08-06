from enum import Enum
from transformers import AutoModel

from internmanip import model
from internmanip.configs import AgentCfg



class AgentRegistry(Enum):
    """
    Registry of agent subclasses.
    The key is the agent type.
    The value is the agent subclass.
    """
    GR00T_N1 = 'Gr00t_N1_Agent'
    GR00T_N1_5 = 'Gr00t_N1_Agent'
    GR00T_N1_5_GENMANIP = 'Gr00tAgent_Genmanip'
    PI0 = 'PI0Agent'
    DP_CLIP = 'DPAgent'

    @property
    def value(self):
        if self.name == 'GR00T_N1':
            from internmanip.agent.simpler_agent import SimplerAgent
            return SimplerAgent
        elif self.name == 'GR00T_N1_5':
            from internmanip.agent.simpler_agent import SimplerAgent
            return SimplerAgent
        elif self.name == 'GR00T_N1_5_GENMANIP':
            from internmanip.agent.genmanip_agent import GenmanipAgent
            return GenmanipAgent
        elif self.name == 'DP_CLIP':
            from internmanip.agent.genmanip_agent import GenmanipAgent
            return GenmanipAgent
        elif self.name == 'PI0':
            from internmanip.agent.simpler_agent import SimplerAgent
            return SimplerAgent
        else:
            raise ValueError(f'Invalid agent type: {self.name}. Only {[agent_type.name for agent_type in AgentRegistry]} are registered.')


class BaseAgent:

    def __init__(self, config: AgentCfg):
        self.config = config

        if config.base_model_path is None:
            policy_model = AutoModel.from_config(config.model_cfg, **config.model_kwargs)
        else:
            # must ensure that if the path is a huggingface model, it should be a repo that has only one model weight
            policy_model = AutoModel.from_pretrained(config.base_model_path, **config.model_kwargs)
        self.policy_model = policy_model

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
