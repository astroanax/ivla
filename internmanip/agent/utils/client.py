from internmanip.configs import AgentCfg
from internmanip.agent.utils.io_utils import serialize_data, deserialize_data
import requests

class PolicyClient:
    """
    Client class for Policy service.
    """

    def __init__(self, config: AgentCfg):
        self.base_url = f'http://{config.server_cfg.server_host}:{config.server_cfg.server_port}'
        self.policy_name = self._initialize_policy_model(config)

    def _initialize_policy_model(self, config: AgentCfg):
        request_data = config.model_dump()
        response = requests.post(
            url=f'{self.base_url}/policy/initialize',
            json=request_data,
            headers={'Content-Type': 'application/json'}
        )
        response.raise_for_status()
        return response.json()['policy_name']

    def __getattr__(self, attr_name: str):
        def attribute(*args, **kwargs):
            request_data = {
                'args': serialize_data(args),
                'kwargs': serialize_data(kwargs)
            }
            response = requests.post(
                url=f'{self.base_url}/policy/{self.policy_name}/{attr_name}',
                json=request_data,
                headers={'Content-Type': 'application/json'}
            )
            response.raise_for_status()
            return deserialize_data(response.json())
        return attribute

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
