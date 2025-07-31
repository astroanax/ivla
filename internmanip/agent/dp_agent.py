from internmanip.agent.utils.geometry import quat2mat, mat2euler
from internmanip.agent.base import BaseAgent
from internmanip.configs import AgentCfg
from internmanip.dataset.transform.video import VideoCrop, VideoResize, VideoToTensor
from internmanip.model.basemodel.diffusion_LMguided.modeling_diffusion import DiffusionModel
from typing import Optional, Sequence, List
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from transforms3d.euler import euler2axangle
from collections import deque
from PIL import Image
import cv2 as cv
from simpler_env.utils.action.action_ensemble import ActionEnsembler


class Pi0Agent(BaseAgent):
    def __init__(self, config: AgentCfg):
        super().__init__(config)
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

        # TODO config normalize
        # model_settings = config.model_cfg.model_settings
        model_settings = dict()
        agent_settings= config.agent_settings

        self.policy_setup = agent_settings.get('policy_setup', None)

        # TODO config normalize set the norm? or not
        self.unnorm_key = model_settings.get('unnorm_key', None)

        if self.policy_setup == 'widowx_bridge':
            self.unnorm_key = 'bridge_orig/1.0.0' if self.unnorm_key is None else self.unnorm_key
            self.action_ensemble = True
            self.sticky_gripper_num_repeat = 1
            # EE pose in Bridge data was relative to a top-down pose, instead of robot base
            # self.default_rot = np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]])  # https://github.com/rail-berkeley/bridge_data_robot/blob/b841131ecd512bafb303075bd8f8b677e0bf9f1f/widowx_envs/widowx_controller/src/widowx_controller/widowx_controller.py#L203
        elif self.policy_setup == 'google_robot':
            self.unnorm_key = (
                'fractal20220817_data/0.1.0' if self.unnorm_key is None else self.unnorm_key
            )
            self.action_ensemble = True
            self.sticky_gripper_num_repeat = 10
        else:
            raise NotImplementedError(
                f'Policy setup {self.policy_setup} not supported. The other datasets can be found in the huggingface config.json file.'
            )

        print(f'*** policy_setup: {self.policy_setup}, unnorm_key: {self.unnorm_key} ***')

        self.policy_client: DiffusionModel = self.policy_model

        self.image_size = model_settings.get('image_size', [224, 224])
        self.action_scale = model_settings.get('action_scale', 1.0)
        self.obs_horizon = 1
        self.obs_interval = 1
        self.pred_action_horizon = 5
        self.image_history = deque(maxlen=self.obs_horizon)
        self.exec_horizon = model_settings.get('exec_horizon', 4)

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

        self.action_ensemble_temp = model_settings.get('action_ensemble_temp', -0.8)

        if self.action_ensemble:
            self.action_ensembler = ActionEnsembler(
                self.pred_action_horizon, self.action_ensemble_temp
            )
        else:
            self.action_ensembler = None

        self.task = None
        self.task_description = None

    def reset(self, task_description: str) -> None:
        self.image_history.clear()
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.task_description = task_description
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None
        self.action_plan = deque()

    def step(
        self, image: np.ndarray, task_description: Optional[str] = None, *args, **kwargs
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Input:
            image: np.ndarray of shape (H, W, 3), uint8
            task_description: Optional[str], task description; if different from previous task description, policy state is reset
        Output:
            raw_action: dict; raw policy action output
            action: dict; processed action to be sent to the maniskill2 environment, with the following keys:
                - 'world_vector': np.ndarray of shape (3,), xyz translation of robot end-effector
                - 'rot_axangle': np.ndarray of shape (3,), axis-angle representation of end-effector rotation
                - 'gripper': np.ndarray of shape (1,), gripper action
                - 'terminate_episode': np.ndarray of shape (1,), 1 if episode should be terminated, 0 otherwise
        """
        if task_description is not None:
            if task_description != self.task_description:
                self.reset(task_description)

        assert image.dtype == np.uint8
        image = self._resize_image(image)
        self._add_image_to_history(image)
        images: List[Image.Image] = self._obtain_image_history()

        eef_pos = kwargs.get('eef_pos', None)
        # if self.policy_setup == "widowx_bridge":
        #     state = self.preprocess_widowx_proprio(eef_pos)
        # elif self.policy_setup == "google_robot":
        #     state = self.preprocess_google_robot_proprio(eef_pos)

        # TODO: transform state and images to the format of the inputs of the PI0Policy.inference

        frames_tensor = torch.from_numpy(np.asarray([np.asarray(img) for img in images])).to(torch.float32) / 255.0
        frames_tensor = frames_tensor.permute(0, 3, 1, 2)  # [T, C, H, W]

        import torchvision.transforms.v2 as T
        # center crop
        frames_tensor = T.CenterCrop((224, 224))(frames_tensor)

        transformed_image = frames_tensor[None, None,...].to(self.policy_client.device)
        if not self.action_plan:

            inputs = {
                'video': transformed_image,            # B 1 N_VIDEOS 3, 224, 224
                'state': torch.from_numpy(np.asarray(eef_pos)).to(self.policy_client.device)[None,None,:].to(torch.float32),         # B * 1 * 7
                'action_pad': torch.ones([1, self.pred_action_horizon, 1 ]).to(torch.bool).to(self.policy_client.device),
                'annotation.human.action.task_description': [[task_description]],
            }
            action_chunk = self.policy_client.inference(inputs)[:self.pred_action_horizon]
            self.action_plan.extend(action_chunk[: self.exec_horizon])

        raw_actions = self.action_plan.popleft()

        raw_action = {
            'world_vector': np.array(raw_actions[:3]),
            'rotation_delta': np.array(raw_actions[3:6]),
            'open_gripper': np.array(
                raw_actions[6:7]
            ),  # range [0, 1]; 1 = open; 0 = close
        }

        # process raw_action to obtain the action to be sent to the maniskill2 environment
        action = {}
        action['world_vector'] = raw_action['world_vector'] * self.action_scale
        action_rotation_delta = np.asarray(
            raw_action['rotation_delta'], dtype=np.float64
        )
        roll, pitch, yaw = action_rotation_delta
        action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
        action_rotation_axangle = action_rotation_ax * action_rotation_angle
        action['rot_axangle'] = action_rotation_axangle * self.action_scale

        if self.policy_setup == 'google_robot':
            action['gripper'] = 0
            current_gripper_action = raw_action['open_gripper']
            if self.previous_gripper_action is None:
                relative_gripper_action = np.array([0])
                self.previous_gripper_action = current_gripper_action
            else:
                relative_gripper_action = self.previous_gripper_action - current_gripper_action

            # fix a bug in the SIMPLER code here
            # self.previous_gripper_action = current_gripper_action

            if np.abs(relative_gripper_action) > 0.5 and (not self.sticky_action_is_on):
                self.sticky_action_is_on = True
                self.sticky_gripper_action = relative_gripper_action
                self.previous_gripper_action = current_gripper_action

            if self.sticky_action_is_on:
                self.gripper_action_repeat += 1
                relative_gripper_action = self.sticky_gripper_action

            if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
                self.sticky_action_is_on = False
                self.gripper_action_repeat = 0
                self.sticky_gripper_action = 0.0

            action['gripper'] = relative_gripper_action

        elif self.policy_setup == 'widowx_bridge':
            action['gripper'] = 2.0 * (raw_action['open_gripper'] > 0.5) - 1.0

        action['terminate_episode'] = np.array([0.0])
        return raw_action, action

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        image = cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)
        return image

    def _add_image_to_history(self, image: np.ndarray) -> None:
        if len(self.image_history) == 0:
            self.image_history.extend([image] * self.obs_horizon)
        else:
            self.image_history.append(image)

    def _obtain_image_history(self) -> List[Image.Image]:
        image_history = list(self.image_history)
        images = image_history[:: self.obs_interval]
        # images = [Image.fromarray(image).convert("RGB") for image in images]
        return images

    def visualize_epoch(
        self,
        predicted_raw_actions: Sequence[np.ndarray],
        images: Sequence[np.ndarray],
        save_path: str,
    ) -> None:
        images = [self._resize_image(image) for image in images]
        ACTION_DIM_LABELS = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'grasp']

        img_strip = np.concatenate(np.array(images[::3]), axis=1)

        # set up plt figure
        figure_layout = [['image'] * len(ACTION_DIM_LABELS), ACTION_DIM_LABELS]
        plt.rcParams.update({'font.size': 12})
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])

        # plot actions
        pred_actions = np.array(
            [
                np.concatenate(
                    [a['world_vector'], a['rotation_delta'], a['open_gripper']], axis=-1
                )
                for a in predicted_raw_actions
            ]
        )
        for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
            # actions have batch, horizon, dim, in this example we just take the first action for simplicity
            axs[action_label].plot(
                pred_actions[:, action_dim], label='predicted action'
            )
            axs[action_label].set_title(action_label)
            axs[action_label].set_xlabel('Time in one episode')

        axs['image'].imshow(img_strip)
        axs['image'].set_xlabel('Time in one episode (subsampled)')
        plt.legend()
        plt.savefig(save_path)
