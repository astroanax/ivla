#!/bin/bash
set -m
source ${CONDA_PREFIX}/etc/profile.d/conda.sh

declare -A params=(
  [server_conda_name]=""
  [config]=""
  [server]=false
  [distributed]=false
  [dataset_path]=""
  [res_save_path]=""
  [distributed_num_worker]=1
)

while [ $# -gt 0 ]; do
    case "$1" in
        --config)
            params[config]="$2"
            shift 2
            ;;
        --server)
            params[server]=true
            shift
            ;;
        --distributed)
            params[distributed]=true
            shift
            ;;
        --dataset_path)
            params[dataset_path]="$2"
            shift 2
            ;;
        --res_save_path)
            params[res_save_path]="$2"
            shift 2
            ;;
        --server_conda_name)
            params[server_conda_name]="$2"
            shift 2
            ;;
        --distributed_num_worker)
            params[distributed_num_worker]="$2"
            shift 2
            ;;
        *)
            echo "unknown params: $1" >&2
            exit 1
            ;;
    esac
done


find_free_ports() {
    local count=$1
    local ports=()
    
    for ((i=0; i<count; i++)); do
        local port=$(( RANDOM % 50001 + 10000 ))
        while lsof -i :$port >/dev/null 2>&1; do
            port=$(( RANDOM % 50001 + 10000 ))
        done
        ports+=($port)
    done
    
    echo "[$(IFS=,; echo "${ports[*]}")]"
}
FREE_PORTS=$(find_free_ports ${params[distributed_num_worker]})

echo "config: ${params[config]}"
echo "server: ${params[server]}"
echo "distributed: ${params[distributed]}"
echo "dataset_path: ${params[dataset_path]}"
echo "res_save_path: ${params[res_save_path]}"
echo "distributed_num_worker: ${params[distributed_num_worker]}"
echo "FREE_PORT: ${FREE_PORTS}"

if [ "${params[distributed_num_worker]}" -gt 1 ]; then
    conda activate genmanip
    ray disable-usage-stats
    ray stop
    ray start --head
fi

cd $(dirname $(dirname $(dirname $(readlink -f "$0"))))

mkdir -p ${params[res_save_path]}

conda activate ${params[server_conda_name]}
{
  python -m scripts.eval.start_agent_server --port $FREE_PORTS > "${params[res_save_path]}/server.log" 2>&1
  SERVER_PID=$!
  trap "kill -TERM $SERVER_PID" INT TERM EXIT
  wait $SERVER_PID
} &
trap "kill -TERM %1" INT TERM EXIT

sleep $((20 + 10 * ${params[distributed_num_worker]}))

conda activate genmanip
python -m scripts.eval.start_evaluator \
    --config "${params[config]}" \
    $([ "${params[server]}" = true ] && echo "--server") \
    $([ "${params[distributed]}" = true ] && echo "--distributed") \
    --dataset_path "${params[dataset_path]}" \
    --res_save_path "${params[res_save_path]}" \
    --server_port "$FREE_PORTS" \
    --distributed_num_worker "${params[distributed_num_worker]}" > "${params[res_save_path]}/eval.log" 2>&1
