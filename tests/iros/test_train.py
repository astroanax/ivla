import os
import shutil
import subprocess
import tempfile
import yaml
import pytest


@pytest.fixture(scope="function")
def temp_config(monkeypatch):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    original_config = os.path.join(project_root, "challenge/run_configs/train/gr00t_n1_5_aloha.yaml")
    temp_dir = tempfile.mkdtemp()
    temp_config_path = os.path.join(temp_dir, "test_config.yaml")

    # Read the original configuration and modify it
    with open(original_config, "r") as f:
        config = yaml.safe_load(f)

    config["batch_size"] = 1
    config["max_steps"] = 5
    config["save_steps"] = 5
    config["output_dir"] = os.path.join(temp_dir, "output")
    config["base_model_path"] = os.path.join(project_root, "data/model")

    with open(temp_config_path, "w") as f:
        yaml.safe_dump(config, f)

    # disabling wandb
    monkeypatch.setenv("WANDB_MODE", "disabled")

    yield temp_config_path, config["output_dir"]

    # clean
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_training_creates_checkpoint(temp_config):
    """
    Test whether training generates checkpoint folders and safetensors files
    """
    config_path, output_dir = temp_config

    # run the training command
    cmd = [
        "torchrun",
        "--nnodes", "1",
        "--nproc_per_node", "1",
        "scripts/train/train.py",
        "--config", config_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    # check if the training exited successfully
    assert result.returncode == 0, f"Training process failed: {result.stderr}"

    # check whether the checkpoint directory is generated
    checkpoints = [
        d for d in os.listdir(output_dir)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
    ]
    assert checkpoints, f"Checkpoint directory not found, output directory: {output_dir}"

    # check if there is a safetensors file in it
    ckpt_dir = os.path.join(output_dir, checkpoints[0])
    files = os.listdir(ckpt_dir)
    safetensor_files = [f for f in files if f.endswith(".safetensors")]
    assert safetensor_files, f"No safetensors file was found in the checkpoint directory: {ckpt_dir}"
