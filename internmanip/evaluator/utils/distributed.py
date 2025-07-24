from typing import List, Dict
from collections import defaultdict

import types
import json
import time
import threading
import logging
import ray
from ray.actor import ActorHandle
from datetime import datetime

from internmanip.configs import ServerCfg
from internmanip.configs import EvalCfg, DistributedCfg
from internmanip.evaluator.base import EvaluatorRegistry


def _setup_logging_func():
    global logger
    logger = logging.getLogger("ray")
    logger.setLevel(logging.DEBUG)


class EvaluatorRayActorGroup:
    """
    Evaluators grouped by ray actors for distributed evaluation.
    """
    def __init__(self, config: EvalCfg):
        self.config = config
        self.distributed_settings = config.distributed_cfg if config.distributed_cfg is not None else DistributedCfg(num_workers=2)
        self.num_workers = self.distributed_settings.num_workers
        self.actors_resources_released = True

        # init some attributes of the evaluator class(results, episodes_data, update_results, print_and_save_results, etc.)
        self.evaluator_class = EvaluatorRegistry[config.eval_type].value
        
        if self.config.eval_type == "GENMANIP":
            self.episodes_data = self.evaluator_class._get_all_episodes_setting_data(self.config)
            self.episodes_batches = self._prepare_episodes_batches()
            self.server_cfg_list = []
            for i in range(self.num_workers):
                self.server_cfg_list.append(ServerCfg(server_port=5000+i))
        else:
            if self.evaluator_class.__name__ == "CalvinEvaluator":
                self.evaluator_class.num_sequences = self.config.env.env_settings.num_sequences
            self.eval_log_dir = self.evaluator_class.eval_log_dir
            self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.episodes_config_path = self.config.env.episodes_config_path if self.config.env.episodes_config_path is not None else self.evaluator_class.episodes_config_path
            self.episodes_data = self.evaluator_class._get_all_episodes_setting_data(self.episodes_config_path)
        
            self.results = self.evaluator_class.results

        # the ray actors within one ray actor group will share the same results and episodes data pool
        self.results_updating_lock = threading.Lock()
        self.processed_episode_ids = set()
        self.episode_data_loading_lock = threading.Lock()

        # bind the methods `_update_results` and `_print_and_save_results` of the evaluator class to the EvaluatorRayActorGroup instance
        self._update_results = types.MethodType(self.evaluator_class._update_results.__func__, self)
        self._print_and_save_results = types.MethodType(self.evaluator_class._print_and_save_results.__func__, self)

        # init ray cluster
        self._init_ray_cluster()

        # init evaluator actors
        self.evaluator_ray_actors: List[ActorHandle] = []
        self._init_evaluator_actors()
        self.actors_resources_released = False

    def _init_ray_cluster(self):
        """
        Initialize Ray cluster
        """
        if not ray.is_initialized():
            if self.distributed_settings.ray_head_ip is None or self.distributed_settings.ray_head_ip == 'auto':
                ray_address = 'auto'
            else:
                ray_address = f"ray://{self.distributed_settings.ray_head_ip}:10001"

            ray_config = {
                "address": ray_address,
                "include_dashboard": self.distributed_settings.include_dashboard,
                "dashboard_port": self.distributed_settings.dashboard_port,
                # "runtime_env": {
                #     "worker_process_setup_hook": _setup_logging_func,
                # }
            }
            
            try:
                ray.init(**ray_config)
            except Exception as e:
                raise RuntimeError(f"Failed to initialize ray cluster: {e}\n"
                                   f"Please check the ray init config: {ray_config}")
            
            _setup_logging_func()
            logger.info(f"Ray initialized with config: {ray_config}")

    def _init_evaluator_actors(self):
        """
        Initialize evaluator Ray actors.
        """
        # get cluster available resources
        cluster_resources = ray.available_resources()
        available_gpus = int(cluster_resources.get("GPU", 0))
        available_cpus = int(cluster_resources.get("CPU", 0))
        
        logger.info(f"Ray cluster available resources: {cluster_resources}")
        
        # allocate resources per worker
        self.gpus_per_worker = available_gpus / self.num_workers
        self.cpus_per_worker = available_cpus / self.num_workers
        
        for i in range(self.num_workers):
            if self.config.eval_type == "GENMANIP":
                self.config.env.env_settings.episode_list = self.episodes_batches[i]
                self.config.agent.server_cfg = self.server_cfg_list[i]

            self.evaluator_ray_actors.append(
                ray.remote(self.evaluator_class).options(
                    name=f"{self.evaluator_class.__name__}_{i}",
                    num_cpus=self.cpus_per_worker,
                    num_gpus=self.gpus_per_worker,
                    runtime_env={
                        "env_vars": {
                            "HF_ENDPOINT": "https://hf-mirror.com",
                        }
                    }
                ).remote(self.config)
            )
            logger.info(f"Created evaluator actor {self.evaluator_class.__name__}_{i} with "
                        f"CPUs: {self.cpus_per_worker}, GPUs: {self.gpus_per_worker}")
        
        logger.info(f"Waiting for {self.num_workers} evaluator ray actors to be ready...")
        try:
            ray.wait([actor.ping.remote() for actor in self.evaluator_ray_actors], num_returns=self.num_workers)
            logger.info("All evaluator ray actors are ready now.")
        except Exception as e:
            raise RuntimeError(f"Failed to connect to evaluator ray actors: {e}")

    def _next_episode_data_index(self):
        with self.episode_data_loading_lock:
            for episode_i in range(len(self.episodes_data)):
                if episode_i not in self.processed_episode_ids:
                    self.processed_episode_ids.add(episode_i)
                    return episode_i
        return None
    
    def _prepare_episodes_batches(self):
        """
        For each evaluator, prepare a batch of episodes for evaluation.
        """
        episodes_batches = []
        batch_size = len(self.episodes_data) // self.num_workers
        for i in range(self.num_workers):
            if i != self.num_workers - 1:
                episodes_batches.append(self.episodes_data[i*batch_size:(i+1)*batch_size])
            else:
                episodes_batches.append(self.episodes_data[i*batch_size:len(self.episodes_data)])

        return episodes_batches

    @classmethod
    def _format_time(cls, seconds):
        days, seconds = divmod(seconds, 86400)
        hours, seconds = divmod(seconds, 3600)
        minutes, seconds = divmod(seconds, 60)
        time_parts = []
        if days > 0:
            time_parts.append(f"{int(days)}")
        if hours > 0:
            time_parts.append(f"{int(hours):02}")
        time_parts.append(f"{int(minutes):02}")
        time_parts.append(f"{int(seconds):02}")
        return ":".join(time_parts)
    
    def eval(self):
        """
        Distributed evaluation.
        """
        logger.info(f"Starting distributed evaluation with {self.num_workers} workers")
        logger.info(f"Total episodes to evaluate: {len(self.episodes_data)}")
        
        try:
            # GENMANIP based on InternUtopia backend
            # it needs preprare episodes batches for initializing the SimulatorRunner
            logger.info("=== Evaluation started ===")
            if self.config.eval_type == "GENMANIP":
                pending_episodes: List[ray.ObjectRef] = []
                res_per_work = []

                # eval
                for evaluator in self.evaluator_ray_actors:
                    result_future = evaluator.eval.remote(distributed=True)
                    pending_episodes.append(result_future)

                while len(pending_episodes) > 0:
                    # wait for tasks to complete
                    # once a task is done, it will be added to the done_episodes list and removed from the pending_episodes list
                    done_episodes, pending_episodes = ray.wait(pending_episodes,
                                                            num_returns=min(len(pending_episodes), self.num_workers),
                                                            timeout=min(len(pending_episodes), self.num_workers))
                    # process the completed tasks
                    for done_episode in done_episodes:
                        result = ray.get(done_episode)
                        res_per_work.append(result)
                
                # save result
                episode_sr_info = defaultdict(dict)
                for d in res_per_work:
                    for k, v in d.items():
                        if isinstance(v, dict):
                            episode_sr_info[k].update(v)
                        else:
                            episode_sr_info[k] = v
                
                for evaluator in self.evaluator_ray_actors:
                    result_future = evaluator.calc_task_sr.remote(episode_sr_info)
                    pending_episodes.append(result_future)

                while len(pending_episodes) > 0:
                    # wait for tasks to complete
                    # once a task is done, it will be added to the done_episodes list and removed from the pending_episodes list
                    done_episodes, pending_episodes = ray.wait(pending_episodes,
                                                            num_returns=min(len(pending_episodes), self.num_workers),
                                                            timeout=min(len(pending_episodes), self.num_workers))
                    # process the completed tasks
                    for done_episode in done_episodes:
                        result = ray.get(done_episode)
                
                logger.info(f"\n{'='*50}\n{json.dumps(result, indent=4)}\n{'='*50}\n")
            
            # Calvin and Simpler can load single episode data dynamically
            elif self.config.eval_type in ["CALVIN", "SIMPLER"]:
                # prepare the pending episodes for the first epoch
                pending_episodes: List[ray.ObjectRef] = []
                taskid_map_to_actor: Dict[ray.ObjectRef, ActorHandle] = {}
                for evaluator in self.evaluator_ray_actors:
                    episode_i = self._next_episode_data_index()
                    if episode_i is not None:
                        result_future = evaluator._eval_single_episode.remote(self.episodes_data[episode_i])
                        pending_episodes.append(result_future)
                        taskid_map_to_actor[result_future] = evaluator
                
                # start the evaluation loop
                self.start_time = time.time()
                done_episodes_count = 0
                total_episodes = len(self.episodes_data)
                while len(pending_episodes) > 0:
                    # wait for tasks to complete
                    # once a task is done, it will be added to the done_episodes list and removed from the pending_episodes list
                    done_episodes, pending_episodes = ray.wait(pending_episodes,
                                                            num_returns=min(len(pending_episodes), self.num_workers),
                                                            timeout=min(len(pending_episodes), self.num_workers))
                    # process the completed tasks
                    for done_episode in done_episodes:
                        result = ray.get(done_episode)
                        self._update_results(result)
                        evaluator = taskid_map_to_actor.pop(done_episode)

                        # assign new task to the idle evaluator
                        new_episode_i = self._next_episode_data_index()
                        if new_episode_i is not None:
                            new_result_future = evaluator._eval_single_episode.remote(self.episodes_data[new_episode_i])
                            pending_episodes.append(new_result_future)
                            taskid_map_to_actor[new_result_future] = evaluator
                    
                    if len(done_episodes) > 0:
                        done_episodes_count += len(done_episodes)
                        elapsed_time = time.time() - self.start_time
                        avg_time_per_episode = elapsed_time / done_episodes_count
                        eta = avg_time_per_episode * (total_episodes - done_episodes_count)

                        # format the time
                        elapsed_time_str = self._format_time(elapsed_time)
                        eta_str = self._format_time(eta)

                        logger.info(f"Progress: {done_episodes_count}/{total_episodes} episodes completed "
                                    f"[{done_episodes_count/total_episodes*100:.1f}%] "
                                    f"[{elapsed_time_str}<{eta_str}, {avg_time_per_episode:.2f}s/episode]")

                self._print_and_save_results()
            logger.info("=== Evaluation completed ===")
        except Exception as e:
            logger.error(f"Error occurred in distributed evaluation: {e}")
            raise e
        finally:
            self.release_resources()

    def release_resources(self):
        """
        Release resources reserved for evaluator ray actors.
        If self.distributed_settings.kill_cluster_on_exit is True(default), 
        the ray cluster will be killed after the evaluation is finished or interrupted.
        """
        if self.actors_resources_released:
            return
        
        logger.info("Releasing resources reserved for evaluator ray actors...")
        
        for evaluator in self.evaluator_ray_actors:
            try:
                ray.kill(actor=evaluator, no_restart=True)
                logger.info(f"Killed evaluator ray actor {evaluator}")
            except Exception as e:
                logger.warning(f"Failed to kill ray actor {evaluator}: {e}")
        
        self.actors_resources_released = True
        logger.info("Resources for evaluator ray actors released.")

        logger.info("Cleanup state and disconnect ray cluster...")
        ray.shutdown()
        logger.info("Ray cluster disconnected.")