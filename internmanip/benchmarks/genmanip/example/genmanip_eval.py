import copy
import os

from ..config.env_config import (  # noqa: F401
    AlohaSplitCameraEnable,
    EnvSettings,
    EpisodeInfo,
    FrankaCameraEnable,
    RayDistributionCfg,
)
from ..create_env import create_env
from ..recorder import Recorder
from .replay_episodes_agent import ReplayEpisodesAgent


class GenmanipEvaluator:
    """Genmanip manipulation task evaluator

    Functionality:
    1. Conducts evaluation through environment-agent interaction
    2. Records evaluation metrics using Recorder component
    3. Automatically calculates task success rates (SR)

    Parameter Specifications:
    :param dataset_path: (str) genmanip dataset path
    :param eval_tasks: (list[str]) list of relative task path to be evaluated under dataset_path
    :param res_save_path: (str|None) directory path for saving evaluation results
    :param is_save_img: (bool) when True, saves environment observation images
    :param **kwargs: more env setting params see `Class EnvSettings` in ../config.py
    """

    def __init__(self,
                 dataset_path,
                 eval_tasks,
                 res_save_path=None,
                 is_save_img=False,
                 metric_type="soft",
                 **kwargs):
        episode_list = self.get_all_episode_list(dataset_path, eval_tasks)
        env_settings = EnvSettings(episode_list=episode_list, **kwargs)
        _, self.env = create_env(env_settings)
        self.agent = ReplayEpisodesAgent(
            dataset_path, action_type=kwargs.get('action_type', 'joint_action')
        )
        self.recorder = Recorder(env_settings.robot_type, res_save_path, is_save_img, metric_type)
        self.res_save_path = res_save_path

    def get_all_episode_list(self, dataset_path, eval_tasks):
        if not eval_tasks:
            eval_tasks = []
            root_path = os.path.abspath(dataset_path)
            for root, _, files in os.walk(root_path):
                if 'scene.usd' in files and 'meta_info.pkl' in files:
                    task_name = os.path.relpath(os.path.dirname(root), start=root_path)
                    if task_name not in eval_tasks:
                        eval_tasks.append(task_name)

        episode_list = []
        for task_item in eval_tasks:
            task_path = os.path.join(dataset_path, task_item)
            assert os.path.exists(task_path), f'Task path does not exist: {task_path}'

            for episode_item in os.listdir(task_path):
                episode_path = os.path.join(task_path, episode_item)
                meta_info_path = os.path.join(episode_path, 'meta_info.pkl')
                scene_asset_path = os.path.join(episode_path, 'scene.usd')

                if not os.path.exists(meta_info_path) or not os.path.exists(scene_asset_path):
                    continue

                episode_info = EpisodeInfo(
                    episode_path=episode_path, task_name=task_item, episode_name=episode_item
                )

                episode_list.append(episode_info)

        return episode_list

    def eval(self):
        no_more_episode = False
        last_terminated_status = []
        env_reset_ids = []

        _, _ = self.env.reset()
        self.env.warm_up(steps=10)
        obs = self.env.get_observations()

        while True:
            all_env_action = self.agent.get_next_action(obs)
            obs, _, terminated_status, _, _ = self.env.step(
                action=[{'robot': action} for action in all_env_action]
            )

            self.recorder(obs)

            if last_terminated_status:
                env_reset_ids = [
                    idx
                    for idx in range(len(terminated_status))
                    if terminated_status[idx] and not last_terminated_status[idx]
                ]

            if env_reset_ids:
                self.recorder([obs[i] for i in env_reset_ids], finished=True)

                _, info = self.env.reset(env_ids=env_reset_ids)
                self.env.warm_up(steps=10)
                obs = self.env.get_observations()

                if not info or None in info:
                    no_more_episode = True

            if False not in terminated_status and no_more_episode:
                break

            last_terminated_status = copy.deepcopy(terminated_status)

        _ = self.recorder.calc_task_sr()
        self.env.close()


if __name__ == '__main__':
    evaluator = GenmanipEvaluator(
        dataset_path='/path/to/genmanip/dataset/root',
        eval_tasks=[],  # relative task path to be evaluated under dataset_path
        res_save_path=None,
        is_save_img=False,  # saves environment observation images (res_save_path must not be None)
        metric_type="soft",
        franka_camera_enable=FrankaCameraEnable(
            realsense=False, obs_camera=False, obs_camera_2=False
        ),
        aloha_split_camera_enable=AlohaSplitCameraEnable(
            top_camera=False, left_camera=False, right_camera=False
        ),
        robot_type='franka',  # type of robot, must be 'franka' or 'aloha_split'
        gripper_type='panda',  # type of gripper, must be 'panda' or 'robotiq'
        max_step=500,
        env_num=1,
        headless=True,
        action_type='joint_action',  # types: ["joint_action", "arm_gripper_action_1", "arm_gripper_action_2", "eef_action_1", "eef_action_2"]  # noqa: E501
        ray_distribution=None,  # RayDistributionCfg(proc_num=1, gpu_num_per_proc=1, head_address=None, working_dir=None)  # noqa: E501
    )

    evaluator.eval()
