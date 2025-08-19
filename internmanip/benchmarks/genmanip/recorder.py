import concurrent.futures
import json
import os
from collections import defaultdict
from datetime import datetime

import cv2
import numpy as np


class Recorder:
    """Genmanip manipulation task data recorder
    Functionality:
    1. records robot observations data
    2. Tracks task success metrics

    Parameter Specifications:
    :param res_save_path: (str|None) Base directory path for results. If None, it will not be saved.
    :param is_save_img: (bool) When True, saves environment observation images.
    """

    def __init__(self,
                 robot_type='panda',
                 res_save_path=None,
                 is_save_img=False,
                 metric_type="soft"):
        self.res_save_path = None
        if res_save_path is not None:
            time_str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
            self.res_save_path = os.path.join(res_save_path, f'genmanip_eval_{time_str}')
            os.makedirs(self.res_save_path, exist_ok=True)

        self.is_save_img = is_save_img
        self.metric_type = metric_type
        if robot_type == 'panda':
            self.saved_obs_keys = ['robot_pose', 'joints_state', 'eef_pose']
        else:
            self.saved_obs_keys = ['robot_pose', 'joints_state', 'left_eef_pose', 'right_eef_pose']
        self.success_rate = defaultdict(dict)
        self.obs = defaultdict(dict)
        self.futures = []
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)

    def __call__(self, obs, finished=False):
        for task_obs in obs:
            if task_obs is None or 'robot' not in task_obs:
                continue

            self.record_success_rate(task_obs)

            if self.res_save_path is None:
                continue

            if finished:
                self.save_episode_info(task_obs)
                continue

            self.record_episode_obs(task_obs)
            if self.is_save_img:
                self.async_save_image(task_obs)

        self._check_futures()

    def async_save_image(self, task_obs):
        index = task_obs['robot']['step']
        task_name = task_obs['robot']['metric']['task_name']
        episode_name = task_obs['robot']['metric']['episode_name']
        camera_data_dict = task_obs['robot']['sensors']
        save_path = os.path.join(self.res_save_path, task_name, episode_name)

        for k, v in camera_data_dict.items():
            if 'rgb' in v and v['rgb'].size > 0:
                filepath = os.path.join(save_path, k, 'rgb', f'{str(index).zfill(5)}.png')
                future = self.executor.submit(self._save_single_image, v['rgb'], filepath, 'rgb')
                self.futures.append(future)

            if 'depth' in v and v['depth'].size > 0:
                filepath = os.path.join(save_path, k, 'depth', f'{str(index).zfill(5)}.png')
                future = self.executor.submit(
                    self._save_single_image, v['depth'], filepath, 'depth'
                )
                self.futures.append(future)

    def _save_single_image(self, image_data, filepath, image_type):
        if image_type == 'rgb':
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            bgr_image = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filepath, bgr_image)
        else:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            normalized_depth_image = cv2.normalize(
                image_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )
            cv2.imwrite(filepath, normalized_depth_image)

    def _check_futures(self):
        done, _ = concurrent.futures.wait(self.futures, timeout=0)
        for future in done:
            try:
                future.result()
            except Exception as e:
                print(f'Image save failed: {e}')
        self.futures = [f for f in self.futures if not f.done()]

    def record_success_rate(self, task_obs):
        task_name = task_obs['robot']['metric']['task_name']
        episode_name = task_obs['robot']['metric']['episode_name']

        self.success_rate[task_name][episode_name] = task_obs['robot']['metric']

    def record_episode_obs(self, task_obs):
        index = task_obs['robot']['step']
        task_name = task_obs['robot']['metric']['task_name']
        episode_name = task_obs['robot']['metric']['episode_name']

        if task_name not in self.obs:
            self.obs[task_name] = {}

        if episode_name not in self.obs[task_name]:
            self.obs[task_name][episode_name] = {}

        self.obs[task_name][episode_name][index] = {
            k: task_obs['robot'][k] for k in self.saved_obs_keys
        }

    def save_episode_info(self, task_obs):
        task_name = task_obs['robot']['metric']['task_name']
        episode_name = task_obs['robot']['metric']['episode_name']
        save_path = os.path.join(self.res_save_path, task_name, episode_name)
        os.makedirs(save_path, exist_ok=True)

        with open(os.path.join(save_path, 'episode_sr.json'), 'w', encoding='utf-8') as f:
            f.write(json.dumps(self.success_rate[task_name][episode_name]))

        with open(os.path.join(save_path, 'episode_obs.json'), 'w', encoding='utf-8') as f:
            f.write(json.dumps(self.serialize_data(self.obs[task_name][episode_name])))

        del self.obs[task_name][episode_name]

    def serialize_data(self, data):
        if isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, (list, tuple)):
            return [self.serialize_data(item) for item in data]
        elif isinstance(data, dict):
            return {key: self.serialize_data(value) for key, value in data.items()}
        else:
            return data

    def calc_task_sr(self, success_rate=None):
        if success_rate is None:
            success_rate = self.success_rate

        result = defaultdict(dict)

        for task_name in success_rate.keys():
            episodes_list = list(success_rate[task_name].values())
            if self.metric_type == "hard":
                sum_episode_sr = sum(1 if item['episode_sr'] == 1 else 0 for item in episodes_list)
            else:
                sum_episode_sr = sum(item['episode_sr'] for item in episodes_list)

            result[task_name]['episodes'] = sorted(
                episodes_list,
                key=lambda x: x['episode_sr'],
                reverse=True
            )
            result[task_name]['success_rate'] = round(sum_episode_sr / len(episodes_list), 3)

            print(
                f"\n{'='*50}\n",
                f'Task : < {task_name} >\n',
                json.dumps(result[task_name], indent=4),
                f"\n{'='*50}\n",
            )

        if self.res_save_path is not None:
            with open(os.path.join(self.res_save_path, 'result.json'), 'w', encoding='utf-8') as f:
                f.write(json.dumps(result))

        return result
