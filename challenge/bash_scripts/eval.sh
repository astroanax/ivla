#!/bin/bash
set -m
source ${CONDA_PREFIX}/etc/profile.d/conda.sh

declare -A params=(
  [server_conda_name]=""
  [config]=""
  [server]=false
  [dataset_path]=""
  [res_save_path]=""
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
        *)
            echo "unknown params: $1" >&2
            exit 1
            ;;
    esac
done


find_free_port() {
    local port=$(( RANDOM % 50001 + 10000 ))
    while lsof -i :$port >/dev/null 2>&1; do
        (port=$(( RANDOM % 50001 + 10000 )))
    done
    echo $port
}
FREE_PORT=$(find_free_port)

echo "config: ${params[config]}"
echo "server: ${params[server]}"
echo "dataset_path: ${params[dataset_path]}"
echo "res_save_path: ${params[res_save_path]}"
echo "FREE_PORT: ${FREE_PORT}"


cd $(dirname $(dirname $(dirname $(readlink -f "$0"))))

mkdir -p ${params[res_save_path]}

conda activate ${params[server_conda_name]}
{
  python -m scripts.eval.start_agent_server --port $FREE_PORT > "${params[res_save_path]}/server.log" 2>&1
  SERVER_PID=$!
  trap "kill -TERM $SERVER_PID" INT TERM EXIT
  wait $SERVER_PID
} &
trap "kill -TERM %1" INT TERM EXIT

sleep 10

conda activate genmanip
python -m scripts.eval.start_evaluator \
    --config "${params[config]}" \
    $([ "${params[server]}" = true ] && echo "--server") \
    --dataset_path "${params[dataset_path]}" \
    --res_save_path "${params[res_save_path]}" \
    --server_port "$FREE_PORT" > "${params[res_save_path]}/eval.log" 2>&1
