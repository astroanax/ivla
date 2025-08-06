from transformers import AutoModel

from typing import Dict, Any
import uvicorn
from fastapi import FastAPI, HTTPException, status, APIRouter

from internmanip.configs import ServerCfg, AgentCfg
from internmanip.agent.base import BaseAgent
from internmanip.utils.agent_utils.io_utils import deserialize_data, serialize_data

class AgentServer:
    """
    Server class for Agent service.
    """
    def __init__(self, config: ServerCfg):
        self.host = config.server_host
        self.port = config.server_port
        self.app = FastAPI(title='Agent Service')
        self.agent_instances: Dict[str, Any] = {}

        self._router = APIRouter(prefix='/agent')
        self._register_routes()
        self.app.include_router(self._router)

    def _register_routes(self):
        route_config = [
            ('/initialize', self.init_agent, ['POST'], status.HTTP_201_CREATED),
        ]

        for path, handler, methods, status_code in route_config:
            self._router.add_api_route(
                path=path,
                endpoint=handler,
                methods=methods,
                status_code=status_code
            )

        # dynamic register other called methods and attributes
        @self._router.post('/{agent_name}/{attribute_name}')
        async def dynamic_attribute(agent_name: str, attribute_name: str, request: Dict[str, Any]):
            self._validate_agent_exists(agent_name)
            agent = self.agent_instances[agent_name]

            if not hasattr(agent, attribute_name):
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f'Attribute {attribute_name} not found in agent {agent_name}'
                )

            attribute = getattr(agent, attribute_name)
            if callable(attribute):
                args = request.get('args', [])
                kwargs = request.get('kwargs', {})

                args = deserialize_data(args)
                kwargs = deserialize_data(kwargs)

                result = attribute(*args, **kwargs)
                return serialize_data(result)
            else:
                return serialize_data(attribute)

    async def init_agent(self, request: Dict[str, Any]):
        agent_config = AgentCfg(**request)
        agent = BaseAgent.init(agent_config)
        agent_name = agent_config.agent_type
        self.agent_instances[agent_name] = agent
        return {'status': 'success', 'agent_name': agent_name}

    def _validate_agent_exists(self, agent_name: str):
        if agent_name not in self.agent_instances:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail='Policy not found'
            )

    def run(self):
        uvicorn.run(self.app, host=self.host, port=self.port)
