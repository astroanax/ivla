import traceback
from typing import Any, Dict

from internutopia.core.scene.scene import IScene
from internutopia.core.datahub import DataHub
from internutopia.core.task import BaseTask
from internutopia.core.util import log

from ..config.task_config import ManipulationTaskCfg


@BaseTask.register('ManipulationTask')
class ManipulationTask(BaseTask):
    def __init__(self, config: ManipulationTaskCfg, scene: IScene):
        super().__init__(config, scene)

        self.steps = 0
        self.success_steps = 0
        self.flag_error = False

    def post_reset(self) -> None:
        """Calls while doing a .reset() on the world."""
        super().post_reset()
        self.set_light_intensity()

    def get_observations(self) -> Dict[str, Any]:
        """
        Returns current observations from the objects needed for the behavioral layer.

        Return:
            Dict[str, Any]: observation of robots in this task
        """
        if not self.work:
            return {}
        
        obs = {}
        for robot_name, robot in self.robots.items():
            try:
                _obs = robot.get_obs()
                if _obs:
                    obs[robot_name] = _obs
                    obs[robot_name]['instruction'] = self.config.prompt
                    obs[robot_name]['metric'] = self.metrics['manipulation_success_metric'].calc_episode_sr()
                    obs[robot_name]['step'] = self.steps
            except Exception as e:
                log.error(self.name)
                log.error(e)
                traceback.print_exc()
                self.flag_error = True
                return {}
            
        return obs

    def calculate_metrics(self) -> dict:
        metrics_res = {}
        for name, metric in self.metrics.items():
            metrics_res[name] = metric.calc()

        return metrics_res
    
    def is_done(self) -> bool:
        self.steps = self.steps + 1
        if self.metrics['manipulation_success_metric'].get_episode_sr() == 1:
            self.success_steps = self.success_steps + 1

        flag_max_step = self.steps > self.config.max_step
        flag_success = self.success_steps > self.config.max_success_step
        
        return DataHub.get_episode_finished(self.name) or flag_max_step or flag_success or self.flag_error
    
    def set_light_intensity(self):
        from omni.isaac.core.utils.prims import get_prim_at_path, find_matching_prim_paths
        
        demolight_paths = find_matching_prim_paths("/World/*/scene/obj_defaultGroundPlane/GroundPlane/DomeLight")

        for light_path in demolight_paths:
            prim = get_prim_at_path(light_path)
            intensity = prim.GetProperty('inputs:intensity')
            per_light_intensity = max(180, 1000.0/len(demolight_paths))
            intensity.Set(per_light_intensity)

    def clear_rigid_bodies(self):
        super().clear_rigid_bodies()
        self.set_light_intensity()
