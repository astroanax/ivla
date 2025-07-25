

#!/bin/bash
# max_steps会覆盖epoch数量设定
# **The recommended finetuning configurations is to boost your batch size to the max, and train for 20k steps.**
#cd /mnt/petrelfs/houzhi/Code/grmanipulation

env

echo $NODE_RANK


master_addr=$(scontrol show hostname ${node_list} | head -n1)
# SH-IDC1-10-140-0-184  --master_addr=10.140.0.184 
echo $master_addr

#bash kill.sh

MASTER_PORT=12111 PYTHONPATH=./ python \
scripts/eval/start_evaluator.py --config run_configs/examples/internmanip_demo.py