from pydantic import BaseModel


class ServerCfg(BaseModel):
    server_host: str = "localhost"
    server_port: int = 5000
