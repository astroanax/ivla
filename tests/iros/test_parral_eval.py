import os
import signal
import shutil
import subprocess
import tempfile
import time
import psutil
import pytest


@pytest.fixture(scope="function")
def temp_result_dir():
    temp_dir = tempfile.mkdtemp()
    result_path = os.path.join(temp_dir, "results")
    os.makedirs(result_path, exist_ok=True)
    yield result_path
    shutil.rmtree(temp_dir, ignore_errors=True)


def kill_proc_tree(pid, sig=signal.SIGTERM, include_parent=True):
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    children = parent.children(recursive=True)
    for child in children:
        try:
            child.send_signal(sig)
        except psutil.NoSuchProcess:
            pass
    if include_parent:
        try:
            parent.send_signal(sig)
        except psutil.NoSuchProcess:
            pass

def test_eval_creates_results(temp_result_dir):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    dataset_path = os.path.join(project_root, "data/dataset/pytest_minidata")
    config_path = os.path.join(project_root, "challenge/run_configs/eval/gr00t_n1_5_on_genmanip.py")

    try:
        # start the server (gr00t env)
        server_cmd = [
            "conda", "run", "-n", "gr00t", "python",
            "-m", "scripts.eval.start_agent_server",
            "--port", "[17935,17936]"
        ]
        server_log = os.path.join(temp_result_dir, "server.log")
        with open(server_log, "w") as f:
            server_proc = subprocess.Popen(server_cmd, stdout=f, stderr=f)

        time.sleep(30)

        # Initialize the Ray cluster
        start_ray_cmd = "ray disable-usage-stats && ray stop && ray start --head"
        result = subprocess.run(
            ["conda", "run", "-n", "genmanip", "bash", "-c", start_ray_cmd],
            capture_output=True,
            text=True
        )

        # start the evaluator (genmanip env)
        eval_cmd = [
            "conda", "run", "-n", "genmanip", "python",
            "-m", "scripts.eval.start_evaluator",
            "--config", config_path,
            "--server",
            "--dataset_path", dataset_path,
            "--res_save_path", temp_result_dir,
            "--server_port", "[17935,17936]",
            "--distributed",
            "--distributed_num_worker", "2"
        ]
        eval_log = os.path.join(temp_result_dir, "eval.log")
        with open(eval_log, "w") as f:
            result = subprocess.run(eval_cmd, stdout=f, stderr=f)

        # confirm that the evaluator exited successfully
        assert result.returncode == 0, f"Evaluator failed: {result.stderr}"

        # check whether result.json is generated
        found_result_json = False
        for entry in os.listdir(temp_result_dir):
            subdir = os.path.join(temp_result_dir, entry)
            if os.path.isdir(subdir):
                result_json = os.path.join(subdir, "result.json")
                if os.path.isfile(result_json):
                    found_result_json = True
                    break

        assert found_result_json, f"No result.json found under {temp_result_dir}"

    finally:
        kill_proc_tree(server_proc.pid, sig=signal.SIGTERM)
        server_proc.wait(timeout=5)