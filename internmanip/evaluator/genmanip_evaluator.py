import os
import copy
import zipfile

from huggingface_hub import snapshot_download

from internmanip.evaluator.base import Evaluator
from internmanip.configs.evaluator.eval_cfg import EvalCfg
from internmanip.configs.env.genmanip_env import EpisodeInfo
from internmanip.benchmarks.genmanip.recorder import Recorder


class GenmanipEvaluator(Evaluator):
    def __init__(self, config: EvalCfg):
        if config.env.env_settings.episode_list is None \
            or len(config.env.env_settings.episode_list) == 0:
            config.env.env_settings.episode_list = self._get_all_episodes_setting_data(config)

        super().__init__(config)
        self.recorder = Recorder(
            config.env.env_settings.robot_type,
            config.env.env_settings.res_save_path,
            config.env.env_settings.is_save_img
        )

    def eval(self, distributed=False):
        """
        The entrypoint of the evaluation pipeline.
        """
        no_more_episode = False
        last_terminated_status = []
        env_reset_ids = []

        _, _ = self.env.reset()
        self.env.warm_up(steps=10)
        obs = self.env.get_obs()

        while True:
            all_env_action = self.agent.step(obs)
            obs, _, terminated_status, _, _ = self.env.step(all_env_action)

            self.recorder(obs)

            if last_terminated_status:
                env_reset_ids = [idx for idx in range(len(terminated_status)) if terminated_status[idx] and not last_terminated_status[idx]]

            if env_reset_ids:
                self.recorder([obs[i] for i in env_reset_ids], finished=True)

                _, info = self.env.reset(env_reset_ids)
                self.env.warm_up(steps=10)
                obs = self.env.get_obs()

                if not info or None in info:
                    no_more_episode = True

            if False not in terminated_status and no_more_episode:
                break

            last_terminated_status = copy.deepcopy(terminated_status)

        if distributed:
            return self.recorder.success_rate
        else:
            _ = self.recorder.calc_task_sr()
            self.env.close()

    @classmethod
    def _get_all_episodes_setting_data(cls, config):
        if config.env.env_settings.dataset_path is None:
            # TODO: upload the dataset to huggingface
            dataset_path = snapshot_download('InternRobotics/InternBench-M1', repo_type='dataset')
        else:
            dataset_path = config.env.env_settings.dataset_path

        if config.env.env_settings.eval_tasks:
            eval_tasks = config.env.env_settings.eval_tasks
        else:
            eval_tasks = []
            root_path = os.path.abspath(config.env.env_settings.dataset_path)
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
                episode_path=os.path.join(task_path, episode_item)
                meta_info_path = os.path.join(episode_path, 'meta_info.pkl')
                scene_asset_path = os.path.join(episode_path, 'scene.usd')

                if not os.path.exists(meta_info_path) or \
                    not os.path.exists(scene_asset_path):
                    continue

                episode_info = EpisodeInfo(
                    episode_path=episode_path,
                    task_name=task_item,
                    episode_name=episode_item
                )

                episode_list.append(episode_info)

        return episode_list

    def calc_task_sr(self, episode_sr_info=None):
        return self.recorder.calc_task_sr(success_rate=episode_sr_info)
