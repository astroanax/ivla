from internmanip.configs import ServerCfg, AgentCfg
from internmanip.model.basemodel.base import BasePolicyModel
from internmanip.agent.utils.io_utils import deserialize_data, serialize_data
from typing import Dict, Any
import uvicorn
from fastapi import FastAPI, HTTPException, status, APIRouter


class PolicyServer:
    """
    Server class for Policy service.
    """
    def __init__(self, config: ServerCfg):
        self.host = config.server_host
        self.port = config.server_port
        self.app = FastAPI(title="Policy Service")
        self.policy_instances: Dict[str, BasePolicyModel] = {}
        
        self._router = APIRouter(prefix="/policy")
        self._register_routes()
        self.app.include_router(self._router)

    def _register_routes(self):
        route_config = [
            ("/initialize", self.init_policy, ["POST"], status.HTTP_201_CREATED),
        ]

        for path, handler, methods, status_code in route_config:
            self._router.add_api_route(
                path=path,
                endpoint=handler,
                methods=methods,
                status_code=status_code
            )

        # dynamic register other called methods and attributes
        @self._router.post("/{policy_name}/{attribute_name}")
        async def dynamic_attribute(policy_name: str, attribute_name: str, request: Dict[str, Any]):
            self._validate_policy_exists(policy_name)
            policy = self.policy_instances[policy_name]
            
            if not hasattr(policy, attribute_name):
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Attribute {attribute_name} not found in policy {policy_name}"
                )
                
            attribute = getattr(policy, attribute_name)
            if callable(attribute):
                args = request.get("args", [])
                kwargs = request.get("kwargs", {})
                
                args = deserialize_data(args)
                kwargs = deserialize_data(kwargs)

                result = attribute(*args, **kwargs)
                return serialize_data(result)
            else:
                return serialize_data(attribute)

    async def init_policy(self, request: Dict[str, Any]):
        policy_config = AgentCfg(**request)
        policy = BasePolicyModel.init(
            model_type=policy_config.agent_type,
            model_name_or_path=policy_config.model_name_or_path, 
            **policy_config.model_kwargs
        )
        policy_name = policy_config.agent_type
        self.policy_instances[policy_name] = policy
        return {"status": "success", "policy_name": policy_name}

    def _validate_policy_exists(self, policy_name: str):
        if policy_name not in self.policy_instances:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Policy not found"
            )

    def run(self):
        uvicorn.run(self.app, host=self.host, port=self.port)
