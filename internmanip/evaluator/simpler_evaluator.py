from internmanip.evaluator.base import Evaluator
from internmanip.configs import EvalCfg
from internmanip.configs.env.simpler_env import SimplerEnvSettings
from internmanip.env.simpler_env import SimplerEnv
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
from simpler_env.utils.visualization import write_video
from typing import Dict, Any, List, Union
from collections import defaultdict
from tqdm.auto import tqdm
import os
import json
from pathlib import Path
from transforms3d.euler import quat2euler
import numpy as np
from tabulate import tabulate
from datetime import datetime


class SimplerEvaluator(Evaluator):

    results: Dict[str, Any] = {}
    episodes_data: List[Any] = []
    eval_log_dir: str = f'{Path(__file__).parents[2]}/logs/eval/simpler'
    timestamp: str = datetime.now().strftime('%Y%m%d_%H%M%S')
    episodes_config_path: Union[str, List[str]] = f'{Path(__file__).absolute().parents[1]}/benchmarks/utils/SimplerEnv/google_robot/visual_matching/pick_coke_can.json'

    def __init__(self, config: EvalCfg):
        super().__init__(config)

        # eval log dir
        if self.config.logging_dir is not None:
            SimplerEvaluator.eval_log_dir = self.config.logging_dir
        # episodes config path
        if self.config.env.episodes_config_path is not None:
            # TODO: may need to check whether the data structure of the user custom episodes config path is valid
            SimplerEvaluator.episodes_config_path = self.config.env.episodes_config_path

    @classmethod
    def _get_all_episodes_setting_data(cls, episodes_config_path) -> List[Any]:
        """
        Get all episodes setting data from the given path(s).
        """
        episodes_config_paths: List[str] = [episodes_config_path] if isinstance(episodes_config_path, str) else episodes_config_path
        print(f'SimplerEnv evaluation episodes config path(s): {episodes_config_paths}')

        # get all evaluation sequences
        eval_sequences: List[Dict[str, Any]] = []
        internmanip_root_dir = str(Path(__file__).parents[2])
        for episodes_config_path in episodes_config_paths:
            with open(episodes_config_path, 'r') as f:
                content = f.read().replace('${INTERNMANIP_ROOT_DIR}', internmanip_root_dir)
                episodes_config = json.loads(content)

                for task_name, task_settings_list in episodes_config.items():
                    for task_settings in task_settings_list:
                        eval_sequences.append(task_settings)

        return eval_sequences

    @classmethod
    def _update_results(cls, result):
        for policy_setup, eval_setup_results in result.items():
            for eval_setup, task_results in eval_setup_results.items():
                for task_name, success_arr in task_results.items():
                    cls.results.setdefault(
                        policy_setup, {}
                    ).setdefault(
                        eval_setup, {}
                    ).setdefault(
                        task_name, []
                    ).extend(success_arr)

    @classmethod
    def _print_and_save_results(cls):
        overall_data = {
            policy_setup: {
                eval_setup: [[task_name, np.mean(success_arr)] for task_name, success_arr in task_results.items()]
                for eval_setup, task_results in eval_setup_results.items()
            }
            for policy_setup, eval_setup_results in cls.results.items()
        }

        for policy_setup, eval_setup_results in overall_data.items():
            for eval_setup, task_results in eval_setup_results.items():
                print(f'\n\n>>> Policy setup: {policy_setup}, Eval setup: {eval_setup} <<<')
                print(tabulate(task_results,
                               headers=['Task Name', 'Average Success Rate'],
                               tablefmt='grid'))

        cls.eval_log_dir = cls.eval_log_dir + '/' + cls.timestamp
        Path(cls.eval_log_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(cls.eval_log_dir) / 'results.json', 'w') as f:
            json.dump(overall_data, f, indent=4)
        print(f'Results saved to {Path(cls.eval_log_dir).absolute()}/results.json')

    def _get_current_results(self):
        task_data = defaultdict(list)
        for eval_setup_results in SimplerEvaluator.results.values():
            for task_results in eval_setup_results.values():
                for task_name, success_arr in task_results.items():
                    task_data[task_name].extend(success_arr)
        return {task: np.mean(arr) for task, arr in task_data.items() if arr}

    def eval(self):
        """
        The default entrypoint of the evaluation pipeline.
        """
        # get all episodes data
        SimplerEvaluator.episodes_data = self._get_all_episodes_setting_data(SimplerEvaluator.episodes_config_path)
        SimplerEvaluator.episodes_data = tqdm(SimplerEvaluator.episodes_data, position=0, leave=True)

        for episode_data in SimplerEvaluator.episodes_data:
            result = self._eval_single_episode(episode_data)
            self._update_results(result)
            SimplerEvaluator.episodes_data.set_description(
                ' | '.join([f'{task_name}: {success_rate * 100:.1f}%' for task_name, success_rate in self._get_current_results().items()]) + '\t'
            )
        self._print_and_save_results()

    def _eval_single_episode(self, episode_data: Dict[str, Any]):
        """
        Evaluate the policy for one episode(one task setting). It may contains multiple subtasks(e.g. variation of object position).
        """
        self.env: SimplerEnv
        self.env._build_env(SimplerEnvSettings(**episode_data))
        # run inference
        result = {
            self.env.env_settings.policy_setup: {
                self.env.env_settings.eval_setup: {
                    self.env.env_settings.task_name: []
                }
            }
        }
        for robot_init_x in self.env.robot_init_xs:
            for robot_init_y in self.env.robot_init_ys:
                for robot_init_quat in self.env.robot_init_quats:
                    episode_kwargs = dict(
                        robot_name=self.env.env_settings.robot,
                        task_name=self.env.env_settings.task_name,
                        env_name=self.env.env_settings.env_name,
                        scene_name=self.env.env_settings.scene_name,
                        robot_init_x=robot_init_x,
                        robot_init_y=robot_init_y,
                        robot_init_quat=robot_init_quat,
                        control_mode=self.env.control_mode,
                        additional_env_build_kwargs=self.env.env_settings.additional_env_build_kwargs,
                        rgb_overlay_path=self.env.env_settings.rgb_overlay_path,
                        control_freq=self.env.env_settings.control_freq,
                        sim_freq=self.env.env_settings.sim_freq,
                        max_episode_steps=self.env.env_settings.max_episode_steps,
                        enable_raytracing=self.env.env_settings.enable_raytracing,
                        additional_env_save_tags=self.env.env_settings.additional_env_save_tags,
                        obs_camera_name=self.env.env_settings.obs_camera_name
                    )

                    if self.env.env_settings.obj_variation_mode == 'xy':
                        for obj_init_x in self.env.obj_init_xs:
                            for obj_init_y in self.env.obj_init_ys:
                                episode_kwargs['obj_init_x'] = obj_init_x
                                episode_kwargs['obj_init_y'] = obj_init_y
                                result[
                                    self.env.env_settings.policy_setup
                                ][
                                    self.env.env_settings.eval_setup
                                ][
                                    self.env.env_settings.task_name
                                ].append(self.rollout(episode_kwargs))
                    elif self.env.env_settings.obj_variation_mode == 'episode':
                        for obj_episode_id in range(self.env.env_settings.obj_episode_range[0], self.env.env_settings.obj_episode_range[1]):
                            episode_kwargs['obj_episode_id'] = obj_episode_id
                            result[
                                self.env.env_settings.policy_setup
                            ][
                                self.env.env_settings.eval_setup
                            ][
                                self.env.env_settings.task_name
                            ].append(self.rollout(episode_kwargs))
                    else:
                        raise NotImplementedError()

        return result

    def rollout(self, episode_kwargs: Dict[str, Any]):
        obs = self.env.reset(options=episode_kwargs)
        # for long-horizon environments, we check if the current subtask is the final subtask
        is_final_subtask = self.env.is_final_subtask()

        # get default language instruction
        task_description = self.env.get_language_instruction()
        print(task_description)

        # Initialize logging
        image = get_image_from_maniskill2_obs_dict(self.env, obs, camera_name=self.env.env_settings.obs_camera_name)
        images = [image]
        predicted_actions = []
        predicted_terminated, done, truncated = False, False, False

        # Initialize model
        self.agent.reset(task_description)

        timestep = 0
        success = 'failure'

        # Step the environment
        while not (predicted_terminated or truncated):
            # step the model; "raw_action" is raw model action output; "action" is the processed action to be sent into maniskill env
            if 'eef_pos' not in obs['agent']:
                # TODO remove
                obs['agent']['eef_pos'] = np.asarray([0.,0,0,0,0,0,0, 0.]).astype(np.float32)
            raw_action, action = self.agent.step(image, task_description, eef_pos=obs['agent']['eef_pos'])
            predicted_actions.append(raw_action)
            predicted_terminated = bool(action['terminate_episode'][0] > 0)
            if predicted_terminated:
                if not is_final_subtask:
                    # advance the environment to the next subtask
                    predicted_terminated = False
                    self.env.advance_to_next_subtask()

            # step the environment
            action = np.concatenate([action['world_vector'], action['rot_axangle'], action['gripper']])
            obs, reward, done, truncated, info = self.env.step(action)

            success = 'success' if done else 'failure'
            new_task_description = self.env.get_language_instruction()
            if new_task_description != task_description:
                task_description = new_task_description
                print(task_description)
            is_final_subtask = self.env.is_final_subtask()

            print(timestep, info)

            image = get_image_from_maniskill2_obs_dict(self.env, obs, camera_name=self.env.env_settings.obs_camera_name)
            images.append(image)
            timestep += 1

        episode_stats = info.get('episode_stats', {})

        # save video
        env_save_name = episode_kwargs['env_name']
        for k, v in episode_kwargs['additional_env_build_kwargs'].items():
            env_save_name = env_save_name + f'_{k}_{v}'
        if episode_kwargs['additional_env_save_tags'] is not None:
            env_save_name = env_save_name + f"_{episode_kwargs['additional_env_save_tags']}"
        ckpt_path_basename = self.config.agent.base_model_path if self.config.agent.base_model_path[-1] != '/' else self.config.agent.base_model_path[:-1]
        ckpt_path_basename = ckpt_path_basename.split('/')[-1]
        if self.env.env_settings.obj_variation_mode == 'xy':
            video_name = f"{success}_obj_{episode_kwargs['obj_init_x']}_{episode_kwargs['obj_init_y']}"
        elif self.env.env_settings.obj_variation_mode == 'episode':
            video_name = f"{success}_obj_episode_{episode_kwargs['obj_episode_id']}"
        for k, v in episode_stats.items():
            video_name = video_name + f'_{k}_{v}'
        video_name = video_name + '.mp4'
        if episode_kwargs['rgb_overlay_path'] is not None:
            rgb_overlay_path_str = os.path.splitext(os.path.basename(episode_kwargs['rgb_overlay_path']))[0]
        else:
            rgb_overlay_path_str = 'None'
        r, p, y = quat2euler(episode_kwargs['robot_init_quat'])
        video_path = f"{ckpt_path_basename}/{episode_kwargs['scene_name']}/{episode_kwargs['control_mode']}/{env_save_name}/rob_{episode_kwargs['robot_init_x']}_{episode_kwargs['robot_init_y']}_rot_{r:.3f}_{p:.3f}_{y:.3f}_rgb_overlay_{rgb_overlay_path_str}/{video_name}"
        video_path = os.path.join(SimplerEvaluator.eval_log_dir, SimplerEvaluator.timestamp, video_path)
        write_video(video_path, images, fps=5)

        # save action trajectory
        action_path = video_path.replace('.mp4', '.png')
        action_root = os.path.dirname(action_path) + '/actions/'
        os.makedirs(action_root, exist_ok=True)
        action_path = action_root + os.path.basename(action_path)
        self.agent.visualize_epoch(predicted_actions, images, save_path=action_path)

        return success == 'success'
