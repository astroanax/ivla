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

import argparse
import multiprocessing
from typing import List
from internmanip.configs import ServerCfg
from internmanip.utils.agent_utils.server import AgentServer


def parse_ports(value: str) -> List[int]:
    """Parse both single port and port list formats"""
    try:
        if value.startswith('[') and value.endswith(']'):
            return [int(x.strip()) for x in value[1:-1].split(',')]
        return [int(value)]
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid port format: {value}") from e

def run_server(host: str, port: int):
    """Worker function for process pool"""
    try:
        server_cfg = ServerCfg(server_host=host, server_port=port)
        AgentServer(server_cfg).run()
    except OSError as e:
        print(f"Port {port} unavailable: {str(e)}")
    except Exception as e:
        print(f"Server error on port {port}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(
        description="Policy Server Cluster Starter",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--host',
        type=str,
        default='localhost',
        help='Server binding address'
    )
    parser.add_argument(
        '--ports',
        type=parse_ports,
        default='5000',
        help='Support int like 8000 or list like [8000,8001]'
    )
    args = parser.parse_args()

    processes = []
    for port in args.ports:
        p = multiprocessing.Process(
            target=run_server,
            args=(args.host, port),
            daemon=True
        )
        p.start()
        processes.append(p)
        print(f"Started server on {args.host}:{port} (PID: {p.pid})")

    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("\nReceived shutdown signal, terminating servers...")
        for p in processes:
            p.terminate()

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    main()
