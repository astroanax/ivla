from pathlib import Path
from typing import List

import numpy as np
import yaml
from internutopia.core.robot.controller import BaseController
from internutopia.core.robot.robot import BaseRobot
from internutopia.core.scene.scene import IScene

from ..config.task_config import IKAlohaSplitCuroboControllerCfg


@BaseController.register('IKAlohaSplitCuroboController')
class IKAlohaSplitCuroboController(BaseController):
    def __init__(
        self,
        config: IKAlohaSplitCuroboControllerCfg,
        robot: BaseRobot,
        scene: IScene,
    ):
        super().__init__(config, robot, scene)

        # preprocess curobo cfg
        with open(config.curobo_cfg_path, 'r') as f:
            curobo_cfg = yaml.load(f, Loader=yaml.FullLoader)

        def preprocess_curobo_cfg(curobo_cfg):
            curobo_cfg['robot_cfg']['kinematics']['usd_path'] = curobo_cfg['robot_cfg'][
                'kinematics'
            ]['usd_path'].replace('${ASSETS_DIR}', str(Path(config.curobo_cfg_path).parent))
            curobo_cfg['robot_cfg']['kinematics']['urdf_path'] = curobo_cfg['robot_cfg'][
                'kinematics'
            ]['urdf_path'].replace('${ASSETS_DIR}', str(Path(config.curobo_cfg_path).parent))
            return curobo_cfg

        # create planners
        from .curobo_planner.piper import CuroboPiperPlanner

        self._left_planner = CuroboPiperPlanner(
            preprocess_curobo_cfg(curobo_cfg),
            self.robot.prim_path,
            'left',
        )
        self._right_planner = CuroboPiperPlanner(
            preprocess_curobo_cfg(curobo_cfg),
            self.robot.prim_path,
            'right',
        )
        self._left_arm_joint_indices = [12, 14, 16, 18, 20, 22, 24, 25]
        self._right_arm_joint_indices = [13, 15, 17, 19, 21, 23, 26, 27]

    def action_to_control(self, action: List | np.ndarray):
        """
        Args:
            action (np.ndarray): n-element 1d array containing:
              0. left_eef_target_position
              1. left_eef_target_orientation
              3. right_eef_target_position
              4. right_eef_target_orientation
        """
        assert len(action) == 4, 'action must contain 4 elements'
        assert (
            self._left_planner is not None and self._right_planner is not None
        ), 'curobo planners are not initialized'

        left_eef_target_position = None if action[0] is None else np.array(action[0])
        left_eef_target_orientation = None if action[1] is None else np.array(action[1])
        right_eef_target_position = None if action[2] is None else np.array(action[2])
        right_eef_target_orientation = None if action[3] is None else np.array(action[3])
        cur_joint_positions = self.robot.articulation.get_joint_positions()
        result_left = self._left_planner.ik_single(
            (left_eef_target_position, left_eef_target_orientation),
            cur_joint_positions[self._left_arm_joint_indices[:6]],
        )
        result_right = self._right_planner.ik_single(
            (right_eef_target_position, right_eef_target_orientation),
            cur_joint_positions[self._right_arm_joint_indices[:6]],
        )
        result = np.concatenate([result_left, result_right])
        return result
