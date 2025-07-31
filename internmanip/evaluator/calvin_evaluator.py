from internmanip.evaluator.base import Evaluator
from internmanip.configs.evaluator.eval_cfg import EvalCfg
from internmanip.env.calvin_env import CalvinEnv
from collections import defaultdict
import json
from pathlib import Path
from calvin_agent.evaluation.utils import (
    collect_plan,
    count_success,
    get_env_state_for_initial_condition,
    print_and_save,
)
import hydra
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from typing import List, Dict, Any
from datetime import datetime


class CalvinEvaluator(Evaluator):

    num_sequences: int = 1000
    results: List[int] = None
    episodes_data: List[Any] = []
    eval_log_dir: str = f'{Path(__file__).parents[2]}/logs/eval/calvin'
    timestamp: str = datetime.now().strftime('%Y%m%d_%H%M%S')
    episodes_config_path: str = f'{Path(__file__).absolute().parents[1]}/benchmarks/utils/calvin/eval_sequences.json'

    def __init__(self, config: EvalCfg):
        super().__init__(config)
        # task oracle
        conf_dir = f'{Path(__file__).absolute().parents[1]}/benchmarks/calvin/calvin_models/conf'
        task_cfg = OmegaConf.load(Path(conf_dir) / 'callbacks/rollout/tasks/new_playtable_tasks.yaml')
        self.task_oracle = hydra.utils.instantiate(task_cfg)
        # val annotations
        self.env: CalvinEnv
        if self.env.env_settings.diverse_inst:
            with open(f'{Path(__file__).absolute().parents[1]}/benchmarks/utils/calvin/lang_annotation_cache.json', 'r') as f:
                self.val_annotations = json.load(f)
        else:
            self.val_annotations = OmegaConf.load(Path(conf_dir) / 'annotations/new_playtable_validation.yaml')
        # eval log dir
        if self.config.logging_dir is not None:
            CalvinEvaluator.eval_log_dir = self.config.logging_dir
        # episode length
        self.episode_length = self.env.env_settings.episode_length
        # num sequences
        CalvinEvaluator.num_sequences = self.env.env_settings.num_sequences
        # episodes config path
        if self.config.env.episodes_config_path is not None and isinstance(self.config.env.episodes_config_path, str):
            # TODO: may need to check whether the data structure of the user custom episodes config path is valid
            CalvinEvaluator.episodes_config_path = self.config.env.episodes_config_path
        # plans
        self.plans = defaultdict(list)

    @classmethod
    def _get_all_episodes_setting_data(cls, episodes_config_path) -> List[Any]:
        """
        Get all episodes setting data from the given path.
        """
        with open(episodes_config_path, 'r') as f:
            eval_sequences = json.load(f)
        eval_sequences = eval_sequences[:cls.num_sequences]
        cls.results = [None] * len(eval_sequences)

        return [{'episode_i': i, 'episode_setting': sequence_data} for i, sequence_data in enumerate(eval_sequences)]

    @classmethod
    def _update_results(cls, result):
        cls.results[result['episode_i']] = result['success_counter']

    @classmethod
    def _print_and_save_results(cls):
        cls.eval_log_dir = cls.eval_log_dir + '/' + cls.timestamp
        Path(cls.eval_log_dir).mkdir(parents=True, exist_ok=True)
        print_and_save(cls.results, [episode_data['episode_setting'] for episode_data in cls.episodes_data], Path(cls.eval_log_dir), None)
        print(f'Results saved to {Path(cls.eval_log_dir).absolute()}/results.json')

    def eval(self):
        """
        The default entrypoint of the evaluation pipeline.
        """
        # get all episodes data
        CalvinEvaluator.episodes_data = self._get_all_episodes_setting_data(CalvinEvaluator.episodes_config_path)
        if not self.env.env_settings.debug:
            CalvinEvaluator.episodes_data = tqdm(CalvinEvaluator.episodes_data, position=0, leave=True)

        for episode_data in CalvinEvaluator.episodes_data:
            result = self._eval_single_episode(episode_data)
            self._update_results(result)
            CalvinEvaluator.episodes_data.set_description(
                ' '.join([f'{i + 1}/5 : {v * 100:.1f}% |' for i, v in enumerate(count_success([result for result in CalvinEvaluator.results if result is not None]))]) + '|'
            )
        self._print_and_save_results()

    def _eval_single_episode(self, episode_data: Dict[str, Any]):
        """
        Evaluate the policy for one episode. It contains multiple subtasks(a sequence of natural language instructions).
        """
        initial_state, eval_sequence = episode_data['episode_setting'][0], episode_data['episode_setting'][1]
        robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
        self.env.reset(options={'robot_obs': robot_obs, 'scene_obs': scene_obs})
        success_counter = 0

        for subtask_i, subtask in enumerate(eval_sequence):
            if self.env.env_settings.reset:
                success = self.rollout(subtask, subtask_i, episode_data['episode_i'], robot_obs=robot_obs, scene_obs=scene_obs)
            else:
                success = self.rollout(subtask, subtask_i, episode_data['episode_i'])
            if success:
                success_counter += 1
            else:
                return {'episode_i': episode_data['episode_i'], 'success_counter': success_counter}
        return {'episode_i': episode_data['episode_i'], 'success_counter': success_counter}

    def rollout(self, subtask, subtask_i, episode_i, robot_obs=None, scene_obs=None):
        """
        Run the actual rollout on one subtask (which is one natural language instruction).
        """
        planned_actions = []
        if robot_obs is not None and scene_obs is not None:
            self.env.reset({'robot_obs': robot_obs, 'scene_obs': scene_obs})
        obs = self.env.get_obs()
        # get lang annotation for subtask
        if self.env.env_settings.diverse_inst:
            lang_annotation = self.val_annotations[episode_i][subtask_i]
        else:
            lang_annotation = self.val_annotations[subtask][0]
        lang_annotation = lang_annotation.split('\n')[0]
        if '\u2019' in lang_annotation:
            lang_annotation.replace('\u2019', '\'')
        self.agent.reset()
        start_info = self.env.get_info()

        for step in range(self.episode_length):
            action = self.agent.step(obs, lang_annotation)

            if len(planned_actions) == 0:
                if action.shape == (7,):
                    planned_actions.append(action)
                else:
                    planned_actions.extend([action[i] for i in range(action.shape[0])])
            action = planned_actions.pop(0)
            obs, _, _, current_info = self.env.step(action)
            if step == 0:
                collect_plan(self.agent, self.plans, subtask)
            # check if current step solves a task
            current_task_info = self.task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
            if len(current_task_info) > 0:
                return True
        return False
