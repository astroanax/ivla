

#!/bin/bash
# **The recommended finetuning configurations is to boost your batch size to the max, and train for 20k steps.**

env

echo $NODE_RANK


master_addr=$(scontrol show hostname ${node_list} | head -n1)

echo $master_addr

#bash kill.sh

MASTER_PORT=12111 PYTHONPATH=./ python \
scripts/eval/start_evaluator.py --config run_configs/examples/internmanip_demo.py