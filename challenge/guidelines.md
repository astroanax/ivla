# Challenge Participation Guidelines (WIP)

This document walks you through the end-to-end workflow: from pulling the base image, developing your model, to submitting and seeing your results on the leaderboard.


## 🧩 Environment Setup (Linux)  
### Pull the latest code
```bash
git clone --depth 1 --recurse-submodules https://github.com/InternRobotics/InternManip.git
```

### Prepare train & validation dataset
```bash
dataset_save_path=/custom/dataset/save/path # Please replace it with your local path. Note that at least 50G of storage space is required.
mkdir InternManip/data
ln -s ${dataset_save_path} InternManip/data/dataset

sudo apt-get install git git-lfs
git lfs install

git lfs clone https://huggingface.co/datasets/InternRobotics/IROS-2025-Challenge-Manip ${dataset_save_path}/IROS-2025-Challenge-Manip
```

### Pull base Docker image
```bash
docker pull crpi-mdum1jboc8276vb5.cn-beijing.personal.cr.aliyuncs.com/iros-challenge/internmanip:v1.0
```
  
### Run the container
```bash
xhost +local:root # Allow the container to access the display

cd InternManip

docker run --name internmanip -it --rm --privileged \
  --gpus all \
  --network host \
  -e "ACCEPT_EULA=Y" \
  -e "PRIVACY_CONSENT=Y" \
  -e "DISPLAY=${DISPLAY}" \
  --entrypoint /bin/bash \
  -w /root/InternManip \
  -v /tmp/.X11-unix/:/tmp/.X11-unix \
  -v ${PWD}:/root/InternManip \
  -v ${dataset_save_path}:/root/InternManip/data/dataset \
  -v ${HOME}/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
  -v ${HOME}/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
  -v ${HOME}/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
  -v ${HOME}/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
  -v ${HOME}/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
  -v ${HOME}/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
  -v ${HOME}/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
  -v ${HOME}/docker/isaac-sim/documents:/root/Documents:rw \
  crpi-mdum1jboc8276vb5.cn-beijing.personal.cr.aliyuncs.com/iros-challenge/internmanip:v1.0
```  
  



## 🤖 Observation Space & Action Space
To drive the robot with your policy, you need to train your policy model and implement the agent, especially the member function `step` (refer to [`internmanip/agent/genmanip_agent.py`](https://github.com/InternRobotics/InternManip/blob/master/internmanip/agent/genmanip_agent.py#L52) for an example). Here, we clarify the observation space and action space for the agent in this challenge to facilitate your agent design.

### Observation

At every step, the environment sends your agent a **list with one element** – a nested dictionary `obs` summarising the current observation.

```python
obs = [{
    "robot": {
        "robot_pose": (np.ndarray, np.ndarray),          # (pos, quat) base in world
        "joints_state": {
            "positions": np.ndarray,                     # (28,)  see DOF table below
            "velocities": np.ndarray                     # (28,)
        },
        "left_eef_pose":  (np.ndarray, np.ndarray),      # 7-D left gripper pose
        "right_eef_pose": (np.ndarray, np.ndarray),      # 7-D right gripper pose
        "sensors": {
            "top_camera":   {"rgb": np.ndarray, "depth": np.ndarray},   # (480,640,3) & (480,640)
            "left_camera":  {"rgb": np.ndarray, "depth": np.ndarray},
            "right_camera": {"rgb": np.ndarray, "depth": np.ndarray},
        },
        "instruction": str,                              # natural-language task prompt
        "metric": {
            "task_name": str,
            "episode_name": str,
            "episode_sr": int,
            "first_success_step": int,
            "episode_step": int
        },
        "step": int,
        "render": bool
    }
}]
```

The joint indices for `aloha_split` (28-DOF) robot are summarized in the following table.

| Index range | Component          |
|-------------|--------------------|
| 0-11        | **mobile base + lift** |
| 12,14,16,18,20,22 | **left arm** |
| 24,25       | **left gripper**   |
| 13,15,17,19,21,23 | **right arm** |
| 26,27       | **right gripper**  |

That is to say, you can extract arm/gripper joint states like this:

```python
q = obs["robot"]["joints_state"]["positions"]
left_arm_qpos   = q[[12,14,16,18,20,22]]
right_arm_qpos  = q[[13,15,17,19,21,23]]
left_gripper_q  = q[[24,25]]
right_gripper_q = q[[26,27]]
```

---

### Action Space

Your `step` method must return a **list with one dictionary** that specifies the next command for both arms.

Supported formats:

#### Format 1 – joint commands (recommended)

```python
action = [{
    "action": {
        "left_arm_action":    np.ndarray,  # (6,)  joint deltas or positions
        "left_gripper_action": float or [float,float],  # -1=close, 1=open
        "right_arm_action":   np.ndarray,  # (6,)
        "right_gripper_action": float or [float,float],
    }
}]
```

#### Format 2 – end-effector pose commands (optional)

```python
action = [{
    "action": {
        "left_eef_position":    np.ndarray,  # (3,)  x,y,z
        "left_eef_orientation": np.ndarray,  # (4,)  quaternion (w,x,y,z)
        "left_gripper_action":  float or [float,float],
        "right_eef_position":   np.ndarray,
        "right_eef_orientation": np.ndarray,
        "right_gripper_action": float or [float,float],
    }
}]
```

Return the list:

```python
return action
```

---

### Quick Integration Checklist

1. Edit `internmanip/agent/genmanip_agent.py` → implement `step(self, inputs)`.
2. Parse the observation as shown above.
3. Run your model and map the raw outputs to the required action dict.
4. Return the action list.

That’s it – Internmanip handles the rest (simulation, camera feeds, metrics logging).

#### Critical Notes
1. **Environment Parameters**  
   Refer to : [Evaluation Environment Configuration](https://internrobotics.github.io/user_guide/internmanip/tutorials/environment.html#evaluation-environment-configuration-parameters)  
   **Mandatory**: `robot_type` must be set to `aloha_split` for this challenge.

2. **I/O Specifications**  
   For GenManip environment data formats and action schemas: [I/O Documentation](https://internrobotics.github.io/user_guide/internmanip/tutorials/environment.html#when-robote-type-is-aloha-split)


## 🛠️ Train & Evaluate Your Policy Locally
### Implement your policy  

Please implement your model in `internmanip/model/basemodel/{model_name}/`
and its config in `internmanip/configs/model/{model_name}_cfg.py`

For detailed implementation steps and examples, please refer to the documentation: [✍🏻 Create a New Model](https://internrobotics.github.io/user_guide/internmanip/quick_start/add_model.html).
You can use `gr00t_n1_5` as a reference example.


### Train


Before training, please create a training configuration YAML file named `custom.yaml` under the `challenge/run_configs/train/` directory.
This YAML file specifies which model to train, dataset paths, hyperparameters, and cache settings. 

**Example Minimal YAML**
```yaml
model_type: custom_policy                 # registered model name
dataset_path:
  - InternRobotics/IROS-2025-Challenge-Manip/train/collec_three_glues
  - InternRobotics/IROS-2025-Challenge-Manip/train/collect_two_alarm_clocks
  - InternRobotics/IROS-2025-Challenge-Manip/train/collect_two_shoes
  - InternRobotics/IROS-2025-Challenge-Manip/train/gather_three_teaboxes
  - InternRobotics/IROS-2025-Challenge-Manip/train/make_sandwich
  - InternRobotics/IROS-2025-Challenge-Manip/train/oil_painting_recognition
  - InternRobotics/IROS-2025-Challenge-Manip/train/organize_colorful_cups
  - InternRobotics/IROS-2025-Challenge-Manip/train/purchase_gift_box
  - InternRobotics/IROS-2025-Challenge-Manip/train/put_drink_on_basket
  - InternRobotics/IROS-2025-Challenge-Manip/train/sort_waste
data_config: aloha_v3                     # pre-registered data config preset
base_model_path: nvidia/GR00T-N1.5-3B     # optional pretrained checkpoint path
hf_cache_dir: /your/custom/cache/path     # optional cache directory
```


You can base your config on existing ones (e.g., `gr00t_n1_5_aloha.yaml`) and adjust accordingly.

**Recommended Training Setup**

- Global batch size: 2048
- Training steps: 40,000 or more
- This setup typically requires multiple GPUs for efficient training.


Use the following command to start batch training on a single node with multiple GPUs:

```bash
conda activate your_agent_env_name
export PYTHONPATH="$(pwd):$PYTHONPATH"
torchrun --nnodes 1 --nproc_per_node 8 scripts/train/train.py --config run_configs/train/custom.yaml
```
If you use Slurm or multi-node clusters, refer to the official multi-node training scripts and procedures.
You will be prompted to log in to Weights & Biases (WandB) for monitoring training progress.


For more detailed instructions, advanced options, and troubleshooting, please refer to the full training documentation:
[🏃🏻‍♂️ Training and Evaluation](https://internrobotics.github.io/user_guide/internmanip/quick_start/train_eval.html) and [Tutorials of Training](https://internrobotics.github.io/user_guide/internmanip/tutorials/training.html).

  
### Evaluation
#### 1. Configuration File Preparation  
Create your custom evaluation configuration file based on the structure of:
`challenge/run_configs/eval/custom_on_genmanip.py`
You can refer to the existing `gr00t_n1_5_on_genmanip.py` as an example.


#### 2. Start Evaluation
There are two ways to start the evaluation:

**Approach 1️⃣: Start separately in manual**  
1. Open the terminal and start the agent server:  
```bash
conda activate your_agent_env_name
python -m scripts.eval.start_agent_server --host localhost --port 5000
```
2. Open another terminal and start the evaluator:
```bash
conda activate genmanip
python -m scripts.eval.start_evaluator \
  --config challenge/run_configs/eval/custom_on_genmanip.py \
  --server
```


We also provide a bash script to launch the agent server and evaluator in one command.

**Approach 2️⃣: Start in one command**  
```bash
./challenge/bash_scripts/eval.sh \
  --server_conda_name your_agent_env_name \
  --config challenge/run_configs/eval/custom_on_genmanip.py \
  --server \
  --dataset_path data/dataset/IROS-2025-Challenge-Manip/validation \
  --res_save_path ./results
```
You can check the results at `./results/server.log` and `./results/eval.log`.


## 📦 Packaging & Submission

### ✅ Create your image registry  
You can follow the following [`aliyun document`](https://help.aliyun.com/zh/acr/user-guide/create-a-repository-and-build-images?spm=a2c4g.11186623.help-menu-60716.d_2_15_4.75c362cbMywaYx&scm=20140722.H_60997._.OR_help-T_cn~zh-V_1) or [`Quay document`](https://quay.io/tutorial/) to create a free personal image registry. During the creation of the repository, please set it to **public** access.

### ✅ Build your submission image

Before creating an image, please note the following points:  
- **Make sure your Modified Code are correctly packaged in your submitted Docker image at `/root/InternManip`.**  
- **Your trained model weights are best placed in your submitted Docker image at `/root/InternManip/data/model`.**  
- **Please delete cache and other operations to reduce the image size.**  


> **NOTE !**  
> When using the `docker cp` command to move code files, first use the `umount /root/InternManip/data/dataset` and `umount /root/InternManip` commands in the container. Then delete all soft links in the local code repository, such as `rm InternManip/data/dataset`.

---  

You can build a new image by customizing the `Dockerfile`, or use command `docker commit`:

```bash
docker commit internmanip your-custom-image:v1
```

Push to your public registry
```bash
docker tag your-custom-image:v1 your-registry/your-repository:v1
docker push your-registry/your-repository:v1
```


### ✅ Submit and view the results

#### Submit your image URL on Eval.AI

After creating an account and team on eval.ai, please submit your entry **[here](https://eval.ai/web/challenges/challenge-page/2626/submission)**. In the "Make Submission" column at the bottom, you can select phase. Please select `Upload file` as the `submission type` and upload the **JSON file** shown below. If you select `private` for your `submission visibility`, the results will not be published on the leaderboard. You can select public again on the subsequent result viewing page.


Create a JSON file with your Docker image URL and team information. The submission must follow this exact structure:

```json
{
    "url": "your-registry/your-repository:v1",
    "args": {
        "agent_conda_name": "gr00t",
        "config_path": "/root/InternManip/challenge/run_configs/eval/gr00t_n1_5_on_genmanip.py"
    },
    "team": {
        "name": "your-team-name",
        "members": [
            {
                "name": "John Doe",
                "affiliation": "University of Example",
                "email": "john.doe@example.com",
                "leader": true
            },
            {
                "name": "Jane Smith",
                "affiliation": "Example Research Lab",
                "email": "jane.smith@example.com",
                "leader": false
            }
        ]
    }
}
```
 Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `url` | string | Complete Docker registry URL for your submission image |
| `agent_conda_name` | string | The name of the agent conda env being implemented |
| `config_path` | string | The path of test config file.<br>e.g. /root/InternManip/challenge/run_configs/eval/gr00t_n1_5_on_genmanip.py |
| `team.name` | string | Official team name for leaderboard display |
| `team.members` | array | List of all team members with their details |
| `members[].name` | string | Full name of team member |
| `members[].affiliation` | string | University or organization affiliation |
| `members[].email` | string | Valid contact email address |
| `members[].leader` | boolean | Team leader designation (exactly one must be `true`) |


For detailed submission guidelines and troubleshooting, refer to the official Eval.AI platform documentation.

#### Viewing Results

After submitting, you can view your submissions in the corresponding phase on the [`My Submissions`](https://eval.ai/web/challenges/challenge-page/2626/my-submission) page. Here, you can view the submission file, result file, and logs for each submission, and choose to publish it on the leaderboard. The leaderboard address is [here](https://eval.ai/web/challenges/challenge-page/2626/leaderboard).

> 😄 Good luck, and may the best vision-based policy win!
