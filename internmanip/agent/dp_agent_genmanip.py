from typing import Dict, Any

from internmanip.agent.base import BaseAgent
from internmanip.configs import AgentCfg
import torch
from collections import deque
import random
from internmanip.dataset.transform.base import ModalityTransform
from internmanip.configs.dataset.data_config_for_dp import DATA_CONFIG_MAP,BaseDataConfig
from internmanip.model.basemodel.base import BasePolicyModel
from internmanip.model.basemodel.diffusion_LMguided.modeling_diffusion import data_collator_base
from internmanip.dataset.base import LeRobotSingleDataset
from internmanip.dataset.embodiment_tags import EmbodimentTag
import numpy as np
from scipy.spatial.transform import Rotation
from typing import Sequence


def _set_eval_seed(seed: int = 42) -> None:
    """Fix random seeds for reproducible evaluation."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

_set_eval_seed(42)


def data_collator_single(features):
    """
    复用训练脚本中的data_collator逻辑，但处理单个样本而不是batch
    """
    # 将单个样本包装成列表，然后调用原始的data_collator
    batch = data_collator_base([features])
    return batch


def quaternion_to_euler_wxyz(
    quaternion: Sequence[float],
    order: str = 'xyz',
) -> np.ndarray:
    """
    将四元数 (w,x,y,z) 转换为欧拉角 (默认 roll-pitch-yaw，对应 order)。

    参数
    -------
    quaternion : Sequence[float]
        长度为 4，格式 [w, x, y, z]。
    order : str, default 'xyz'
        欧拉角轴顺序；可用 'xyz'、'zyx' 等 SciPy 支持的所有组合。

    返回
    -------
    np.ndarray
        欧拉角数组 (rad)，顺序同 `order`。
    """
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape[-1] != 4:
        raise ValueError('Quaternion must have 4 components [w, x, y, z]')
    # 归一化
    q = q / np.linalg.norm(q)
    # SciPy 需要 (x,y,z,w) ⇒ 做一次重排
    q_xyzw = np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)
    r = Rotation.from_quat(q_xyzw)
    return r.as_euler(order, degrees=False)


def euler_to_quaternion_wxyz(
    euler_angles: Sequence[float],
    order: str = 'xyz',
) -> np.ndarray:
    """
    将欧拉角转换成四元数，返回格式 [w, x, y, z]。

    参数
    -------
    euler_angles : Sequence[float]
        长度为 3，单位 rad。
    order : str, default 'xyz'
        同上。

    返回
    -------
    np.ndarray
        长度 4，格式 [w, x, y, z]。
    """
    r = Rotation.from_euler(order, euler_angles, degrees=False)
    q_xyzw = r.as_quat()                  # (x, y, z, w)
    q_wxyz = np.array(
        [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64
    )
    # 再做一次归一化（保险）
    return q_wxyz / np.linalg.norm(q_wxyz)


class DPAgent(BaseAgent):
    def __init__(self, config: AgentCfg):
        # 确保config有model_kwargs参数
        if not hasattr(config, 'model_kwargs') or config.model_kwargs is None:
            config.model_kwargs = {}

        super().__init__(config)

        self.policy_model.compute_dtype = 'bfloat16' # type: ignore
        self.policy_model.config.compute_dtype = 'bfloat16' # type: ignore
        # self.device = get_device_from_parameters(self.policy_model)  # 注释掉，改为动态获取

        # 适配新的配置格式
        if hasattr(config, 'agent_settings') and config.agent_settings:
            settings = config.agent_settings
        else:
            settings = config.model_cfg.model_settings # type: ignore

        self.n_obs_steps = settings['n_obs_steps']
        self.data_config_name = settings.get('data_config', 'genmanip')
        # modality configs and transforms
        embodiment_tag = EmbodimentTag(settings['embodiment_tag'])
        data_config_cls: BaseDataConfig = DATA_CONFIG_MAP[self.data_config_name]
        model_transform, observation_indices, action_indices = config.model_cfg.transform()
        modality_configs = data_config_cls.modality_config(observation_indices, action_indices)
        # self.modality_configs = data_config_cls.modality_config()
        self.transforms = data_config_cls.transform()

        self.dataset = LeRobotSingleDataset(
            dataset_path=settings['dataset_path'],
            modality_configs=modality_configs,
            transforms=self.transforms, # type: ignore
            embodiment_tag=embodiment_tag,
            video_backend='decord',
        )

        self.video_base_view = deque(maxlen=self.n_obs_steps)
        self.video_ego_view = deque(maxlen=self.n_obs_steps)
        self.ee_pos_state = deque(maxlen=self.n_obs_steps)
        self.ee_rot_state = deque(maxlen=self.n_obs_steps)
        self.gripper_state = deque(maxlen=self.n_obs_steps)

        self.language = deque(maxlen=self.n_obs_steps)

        self.action_execution_steps = 8
        self.action_history = deque(maxlen=8)
        self.action_queue = deque(maxlen=self.action_execution_steps)
        self.ema_alpha = 0

    def transform_action_back_joints(self,action):
        action = action.detach().cpu()
        joints = action[:]
        gripper = action[-1:]
        action_dict = {'action.joints': joints, 'action.gripper': gripper}
        action_dict = self.dataset.transforms.transforms[7].unapply(action_dict) # type: ignore
        joints = action_dict['action.joints']
        gripper = action_dict['action.gripper']
        action = joints.tolist() +  ([0.4, 0.4] if gripper[0] <=0 else [0.0, 0.0])
        return action

    def transform_action_back_eef(self,action):
        action = action.detach().cpu()
        ee_pos = action[:3]
        ee_rot = action[3:6]
        gripper = action[-1]
        action_dict = {'action.ee_pos': ee_pos, 'action.ee_rot': ee_rot} # gripper 不用反归一化
        action_dict = self.dataset.transforms.transforms[7].unapply(action_dict) # type: ignore
        processed_action = {
            'eef_position': action_dict['action.ee_pos'].tolist(),
            'eef_orientation': euler_to_quaternion_wxyz(action_dict['action.ee_rot']).tolist(),
            'gripper_action': -1 if gripper < 0.5 else 1
        }
        return processed_action

    def _predict_action(self, request):
        """
        predict the action based on the observation
        """
        # request[0]['franka_robot']['joints_state']['positions']
        self.video_base_view.append(request[0]['franka_robot']['sensors']['obs_camera'])
        self.video_ego_view.append(request[0]['franka_robot']['sensors']['realsense'])
        self.ee_pos_state.append(request[0]['franka_robot']['eef_pose']['local_pose'][0])
        # 将四元数转换为欧拉角
        quaternion = request[0]['franka_robot']['eef_pose']['local_pose'][1]
        euler_angles = quaternion_to_euler_wxyz(quaternion)
        self.ee_rot_state.append(euler_angles)
        self.gripper_state.append(request[0]['franka_robot']['joints_state']['positions'][-2:])
        self.language.append(request[0]['franka_robot']['instruction'])

        # if the length of the queue is less than the n_obs_steps, then we need to pad the queue
        while len(self.video_base_view) < self.n_obs_steps:
            self.video_base_view.append(self.video_base_view[-1])
            self.video_ego_view.append(self.video_ego_view[-1])
            self.ee_pos_state.append(self.ee_pos_state[-1])
            self.ee_rot_state.append(self.ee_rot_state[-1])
            self.gripper_state.append(self.gripper_state[-1])
            self.language.append(self.language[-1])

        raw_observation = {
            'video.base_view': np.array([x['rgb'] for x in self.video_base_view]),
            'video.ego_view': np.array([x['rgb'] for x in self.video_ego_view]),
            'state.ee_pos': np.array(self.ee_pos_state),
            'state.ee_rot': np.array(self.ee_rot_state),
            'state.gripper': np.array(self.gripper_state),
            'annotation.human.action.task_description': self.language
        }

        transformed_data = self.dataset.transforms(raw_observation)
        processed_data = data_collator_single(transformed_data)
        observation = {
                    'video': processed_data['observation.images'],
                    'state': processed_data['observation.state'],
                    'annotation.human.action.task_description': [f[0] for f in processed_data['language']]
                }

        actions = self.policy_model.inference(observation)
        actions = actions[0]
        return actions

    def action_execution(self, request):
        """
        Execute the action in the action queue
        """
        current_action = None
        if len(self.action_queue) == 0:
            actions = self._predict_action(request)
            for i in range(self.action_execution_steps):
                self.action_queue.append(actions[i])
        else:
            current_action = self.action_queue.popleft()
            current_action = self.transform_action_back_eef(current_action)
        return current_action

    def step(self, request: Dict[str, Any]):
        if request[0]['franka_robot']['step'] == 0:
            _ = self.reset()

        # action = actions[0][0].detach().cpu().tolist()
        # actions = self.transform_action_back(actions[0])
        # current_action = None
        # if len(self.action_history) == 0:
        #     for i in range(8):
        #         self.action_history.append(actions[i])

        #     current_action = self.action_history.popleft()
        # else:
        #     current_action = self.action_history.popleft()
        #     self.action_history.append(actions[-1])
        #     for i in range(7):
        #         self.action_history[i] = self.action_history[i] * (1-self.ema_alpha) + actions[i] * self.ema_alpha
        # current_action = actions[0]
        # current_action = self.transform_action_back_eef(current_action)
        current_action = self.action_execution(request)
        print(f'action to be executed is {current_action}')
        return [current_action]

    def reset(self):
        # empty the queues
        self.video_base_view.clear()
        self.video_ego_view.clear()
        self.ee_pos_state.clear()
        self.ee_rot_state.clear()
        self.gripper_state.clear()
        self.language.clear()
        self.action_history.clear()
        self.action_queue.clear()
        print('Reset the model......')
        return {'status': 'success'}
