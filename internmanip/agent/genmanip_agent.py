from collections import deque
from copy import deepcopy
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.spatial.transform import Rotation

from internmanip.agent.base import BaseAgent
from internmanip.configs import AgentCfg
from internmanip.configs.dataset.data_config import DATA_CONFIG_MAP
from internmanip.dataset.base import LeRobotSingleDataset
from internmanip.dataset.embodiment_tags import EmbodimentTag
from internmanip.dataset.transform.base import ComposedModalityTransform
from internmanip.model.basemodel.transforms.gr00t_n1 import DefaultDataCollator


class Gr00tAgent_Genmanip(BaseAgent):
    def __init__(self, config: AgentCfg):
        super().__init__(config)
        self.policy_model.compute_dtype = "bfloat16"
        self.policy_model.config.compute_dtype = "bfloat16"
        self.policy_model = self.model.to(torch.bfloat16)
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        dataset_path = config.agent_settings["dataset_path"]
        data_config_cls = DATA_CONFIG_MAP[config.agent_settings["data_config"]]
        modality_configs = data_config_cls.modality_config()
        embodiment_tag = EmbodimentTag(config.agent_settings["embodiment_tag"])
        video_backend = config.agent_settings["video_backend"]
        transforms = data_config_cls.transform()
        self.action_transforms = transforms[-2]
        transforms.append(self.agent.config.transform())
        transforms = ComposedModalityTransform(transforms=transforms)
        self.dataset = LeRobotSingleDataset(
            dataset_path=dataset_path,
            modality_configs=modality_configs,
            embodiment_tag=embodiment_tag,
            video_backend=video_backend,
            transforms=transforms,
        )

        self.data_collator = DefaultDataCollator()

        self.pred_action_horizon = config.agent_settings["pred_action_horizon"]
        self.adaptive_ensemble_alpha = config.agent_settings["adaptive_ensemble_alpha"]
        self.ensembler_list = []

        self.save_folder = "/root/grmanipulation/Data/image"
        self.episode_count = []
        self.step_count = []
        self.output_history_list = []

    def step(self, inputs: list[dict]) -> list[dict]:
        while len(self.ensembler_list) < len(inputs):
            self.ensembler_list.append(
                AdaptiveEnsembler(
                    pred_action_horizon=self.pred_action_horizon,
                    adaptive_ensemble_alpha=self.adaptive_ensemble_alpha,
                )
            )
            self.episode_count.append(0)
            self.step_count.append(0)

        outputs = []
        for env, input in enumerate(inputs):
            if input is None:
                outputs.append(None)
                continue

            if input["franka_robot"]["step"] == 0:
                self.reset_env(env)
            self.step_count[env] = input["franka_robot"]["step"]

            input = self.convert_input(input)
            pred_actions = self.policy_model.inference(input)["action_pred"][0].cpu().float()
            cur_action = self.ensembler_list[env].ensemble_action(pred_actions)
            output = self.convert_output(cur_action)
            outputs.append(output)
        # self._debug_print_data(inputs, title=f"Input Data {env}")
        # self._debug_print_data(outputs, title=f"Output Data {env}")
        # self._record_outputs_data(outputs)
        return outputs

    def reset(self):
        for ensembler in self.ensembler_list:
            ensembler.reset()
        self.ensembler_list = []
        self.episode_count = []
        self.step_count = []
        self.output_history_list = []

    def reset_env(self, env):
        self.ensembler_list[env].reset()
        print(f"Reset env{env}")
        # self.plot_output_history(env)
        # self.episode_count[env] += 1
        # self.output_history_list[env] = []

    def convert_input(self, input: dict):
        quat_wxyz = input["franka_robot"]["eef_pose"]["local_pose"][1]
        quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
        ee_rot = Rotation.from_quat(quat_xyzw).as_euler('xyz', degrees=False)
        converted_data = {
            "annotation.human.action.task_description": input["franka_robot"]["instruction"],
            "state.joints": np.array([input["franka_robot"]["joints_state"]["positions"][:7]]),
            "state.gripper": np.array([input["franka_robot"]["joints_state"]["positions"][7:]]),
            "state.joints_vel": np.array([input["franka_robot"]["joints_state"]["velocities"][:7]]),
            "state.gripper_vel": np.array([input["franka_robot"]["joints_state"]["velocities"][7:]]),
            "state.ee_pos": np.array([input["franka_robot"]["eef_pose"]["local_pose"][0]]),
            "state.ee_rot": np.array([ee_rot]),
            "video.base_view": np.array([input["franka_robot"]["sensors"]["obs_camera"]["rgb"]]),
            "video.base_2_view": np.array([input["franka_robot"]["sensors"]["obs_camera_2"]["rgb"]]),
            "video.ego_view": np.array([input["franka_robot"]["sensors"]["realsense"]["rgb"]]),
        }
        converted_data = self.dataset.transforms(converted_data)
        converted_data = self.data_collator([converted_data])
        return converted_data

    def convert_output(self, output:np.ndarray):
        converted_data = {
            "action.joints": torch.from_numpy(output[:7]),
            "action.gripper_w": torch.from_numpy(output[7:9]),
            "action.gripper": torch.from_numpy(output[9:10]),
            "action.ee_pos": torch.from_numpy(output[10:13]),
            "action.ee_rot": torch.from_numpy(output[13:16]),
            "action.delta_joints": torch.from_numpy(output[16:23]),
            "action.delta_ee_pos": torch.from_numpy(output[23:26]),
            "action.delta_ee_rot": torch.from_numpy(output[26:29]),
        }
        converted_data = self.action_transforms.unapply(deepcopy(converted_data))
        converted_data = {
            "arm_action": converted_data["action.joints"].tolist(),
            "gripper_action": converted_data["action.gripper"][0]*2-1,
        }
        # ee_pos = converted_data["action.ee_pos"].tolist()
        # ee_rot = converted_data["action.ee_rot"]
        # quat_xyzw = Rotation.from_euler('xyz', ee_rot, degrees=False).as_quat()
        # quat_wxyz = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
        # gripper = converted_data["action.gripper"][0]*2-1
        # converted_data = {
        #     "eef_position": ee_pos,
        #     "eef_orientation": quat_wxyz,
        #     "gripper_action": gripper,
        # }
        return converted_data

    def _debug_print_data(self, data, title="Data Debug"):
        print(f"\n=== {title} ===")
        print(data)
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    print(f"{key}: shape={value.shape}, dtype={value.dtype}")
                elif isinstance(value, np.ndarray):
                    print(f"{key}: shape={value.shape}, dtype={value.dtype}")
                elif isinstance(value, list):
                    print(f"{key}: type=list, length={len(value)}")
                else:
                    print(f"{key}: type={type(value)}")
        else:
            if hasattr(data, 'shape'):
                print(f"shape={data.shape}, dtype={getattr(data, 'dtype', 'unknown')}")
            else:
                print(f"type={type(data)}")

    def _record_outputs_data(self, outputs):
        while len(self.output_history_list) < len(outputs):
            self.output_history_list.append([])
        for env, output in enumerate(outputs):
            if output is not None:
                record_data = {
                    "step": self.step_count[env],
                    "arm_action": output["arm_action"],
                    "gripper_action": output["gripper_action"]
                }
                self.output_history_list[env].append(record_data)

    def plot_output_history(self, env):
        if not self.output_history_list:
            print("No output history to plot.")
            return
        if not self.output_history_list[env]:
            print(f"No output history for environment {env}.")
            return

        steps = [data["step"] for data in self.output_history_list[env]]
        arm_actions = [data["arm_action"] for data in self.output_history_list[env]]
        arm_actions = np.array(arm_actions)
        gripper_actions = [data["gripper_action"] for data in self.output_history_list[env]]
        gripper_actions = np.array(gripper_actions)

        plt.figure(figsize=(12, 8))
        plt.suptitle(f'Environment {env} - Episode {self.episode_count[env]}', fontsize=16)

        for i in range(7):
            plt.subplot(3, 3, i+1)
            plt.plot(steps, arm_actions[:, i])
            plt.title(f'Joint {i+1}')
            plt.xlabel('Step')
            plt.ylabel('Action')
            plt.grid(True)

        plt.subplot(3, 3, 8)
        plt.plot(steps, gripper_actions, 'r-')
        plt.title('Gripper Action')
        plt.xlabel('Step')
        plt.ylabel('Action')
        plt.grid(True)

        save_folder = self.save_folder + f"/env{env}"
        os.makedirs(save_folder, exist_ok=True)
        save_path = f"{save_folder}/episode{self.episode_count[env]}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Environment {env} plot saved: {save_path}")


class AdaptiveEnsembler:
    def __init__(self, pred_action_horizon, adaptive_ensemble_alpha=0.0):
        self.pred_action_horizon = pred_action_horizon
        self.action_history = deque(maxlen=self.pred_action_horizon)
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha

    def reset(self):
        self.action_history.clear()

    def ensemble_action(self, cur_action):
        self.action_history.append(cur_action)
        num_actions = len(self.action_history)
        if cur_action.ndim == 1:
            curr_act_preds = np.stack(self.action_history)
        else:
            curr_act_preds = np.stack(
                [pred_actions[i] for (i, pred_actions) in zip(range(num_actions - 1, -1, -1), self.action_history)]
            )
        # calculate cosine similarity between the current prediction and all previous predictions
        ref = curr_act_preds[num_actions-1, :]
        previous_pred = curr_act_preds
        dot_product = np.sum(previous_pred * ref, axis=1)  
        norm_previous_pred = np.linalg.norm(previous_pred, axis=1)  
        norm_ref = np.linalg.norm(ref)  
        cos_similarity = dot_product / (norm_previous_pred * norm_ref + 1e-7)
        # compute the weights for each prediction
        weights = np.exp(self.adaptive_ensemble_alpha * cos_similarity)
        weights = weights / weights.sum()
        # compute the weighted average across all predictions for this timestep
        cur_action = np.sum(weights[:, None] * curr_act_preds, axis=0)
        return cur_action
