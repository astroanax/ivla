from run_configs.eval.dp_on_genmanip import eval_cfg
from internmanip.agent.utils import PolicyServer


def main():
    server = PolicyServer(eval_cfg.agent.server_cfg)
    server.run()

if __name__ == "__main__":
    main()