"""
This script is used to start the policy server for providing policy services to the evaluator.

Usage:
python scripts/eval/start_policy_server.py --host [host_address] --port [port_number]

Arguments:
--host: Server host address, defaults to localhost
--port: Server port number, defaults to 5000

Note:
- This server needs to work in conjunction with the evaluator
- Ensure the specified port is not already in use
"""

from internmanip.configs import ServerCfg
from internmanip.utils.agent_utils.server import AgentServer
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=5000)
    args = parser.parse_args()
    server_cfg = ServerCfg(server_host=args.host, server_port=args.port)

    server = AgentServer(server_cfg)
    server.run()

if __name__ == '__main__':
    main()
