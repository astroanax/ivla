from collections import defaultdict

import copy
import numpy as np
from internutopia.core.robot.isaacsim.articulation import IsaacsimArticulation
from internutopia.core.robot.robot import BaseRobot
from internutopia.core.scene.scene import IScene

from ..config.task_config import AlohaSplitRobotCfg
from .aloha_split_actions import validate_action


@BaseRobot.register('AlohaSplit')
class AlohaSplit(BaseRobot):
    def __init__(self, config: AlohaSplitRobotCfg, scene: IScene):
        super().__init__(config, scene)

        self.prim_path = config.prim_path

        self.articulation = IsaacsimArticulation.create(
            usd_path=config.usd_path,
            prim_path=self.prim_path,
            name=config.name,
            position=config.position,
        )

        self.left_arm_joint_indices = [12, 14, 16, 18, 20, 22]
        self.right_arm_joint_indices = [13, 15, 17, 19, 21, 23]
        self.left_gripper_joint_indices = [24, 25]
        self.right_gripper_joint_indices = [26, 27]

    def post_reset(self):
        super().post_reset()
        self.articulation.set_joint_positions([-0.3], joint_indices=[7])
        self.articulation._articulation_view.set_joint_position_targets(
            [-0.3],
            joint_indices=[7]
        )
        self.init_joint_positions = self.articulation.get_joint_positions()
        self.articulation.set_gains(
            kps=[572957800.0] * 12,
            kds=[57295.78] * 12,
            joint_indices=[i for i in self.left_arm_joint_indices + self.right_arm_joint_indices],
        )

    def apply_action(self, action):
        """
        Args:
            action: inputs for controllers.
        """
        if not action:
            return

        joint_positions = copy.deepcopy(self.init_joint_positions)
        action_type, action = validate_action(action)

        joint_controller = self.controllers['joint_controller']
        ik_controller = self.controllers['arm_ik_aloha_split_curobo_controller']

        if action_type == 'arm_gripper':
            if isinstance(action.left_arm_gripper_action.gripper_action, int):
                action.left_arm_gripper_action.gripper_action = (
                    [0.05, 0.05]
                    if action.left_arm_gripper_action.gripper_action == -1
                    else [0.0, 0.0]
                )

            if isinstance(action.right_arm_gripper_action.gripper_action, int):
                action.right_arm_gripper_action.gripper_action = (
                    [0.05, 0.05]
                    if action.right_arm_gripper_action.gripper_action == -1
                    else [0.0, 0.0]
                )

            for i in range(len(self.left_arm_joint_indices)):
                joint_positions[self.left_arm_joint_indices[i]] = (
                    action.left_arm_gripper_action.arm_action[i]
                )
            for i in range(len(self.right_arm_joint_indices)):
                joint_positions[self.right_arm_joint_indices[i]] = (
                    action.right_arm_gripper_action.arm_action[i]
                )
            for i in range(len(self.left_gripper_joint_indices)):
                joint_positions[self.left_gripper_joint_indices[i]] = (
                    action.left_arm_gripper_action.gripper_action[i]
                )
            for i in range(len(self.right_gripper_joint_indices)):
                joint_positions[self.right_gripper_joint_indices[i]] = (
                    action.right_arm_gripper_action.gripper_action[i]
                )

        elif action_type == 'eef':
            if isinstance(action.left_eef_action.gripper_action, int):
                action.left_eef_action.gripper_action = (
                    [0.05, 0.05] if action.left_eef_action.gripper_action == -1 else [0.0, 0.0]
                )

            if isinstance(action.right_eef_action.gripper_action, int):
                action.right_eef_action.gripper_action = (
                    [0.05, 0.05] if action.right_eef_action.gripper_action == -1 else [0.0, 0.0]
                )

            arm_control = ik_controller.action_to_control(
                [
                    action.left_eef_action.eef_position,
                    action.left_eef_action.eef_orientation,
                    action.right_eef_action.eef_position,
                    action.right_eef_action.eef_orientation,
                ]
            )

            for i in range(len(self.left_arm_joint_indices)):
                joint_positions[self.left_arm_joint_indices[i]] = arm_control[i]
            for i in range(len(self.right_arm_joint_indices)):
                joint_positions[self.right_arm_joint_indices[i]] = arm_control[i + 8]
            for i in range(len(self.left_gripper_joint_indices)):
                joint_positions[self.left_gripper_joint_indices[i]] = (
                    action.left_eef_action.gripper_action[i]
                )
            for i in range(len(self.right_gripper_joint_indices)):
                joint_positions[self.right_gripper_joint_indices[i]] = (
                    action.right_eef_action.gripper_action[i]
                )

        control = joint_controller.action_to_control([joint_positions])
        self.articulation.apply_action(control)

    def get_obs(self) -> defaultdict:
        obs = defaultdict(dict)

        # robot_pose
        obs['robot_pose'] = self.articulation.get_pose()

        # joints_state
        obs['joints_state']['positions'] = self.articulation.get_joint_positions()
        obs['joints_state']['velocities'] = self.articulation.get_joint_velocities()

        # eef_pose
        eef_position_left, eef_orientation_left = self.controllers[
            'arm_ik_aloha_split_curobo_controller'
        ]._left_planner.fk_single(obs['joints_state']['positions'][self.left_arm_joint_indices])
        eef_position_right, eef_orientation_right = self.controllers[
            'arm_ik_aloha_split_curobo_controller'
        ]._right_planner.fk_single(obs['joints_state']['positions'][self.right_arm_joint_indices])

        obs['left_eef_pose'] = (np.array(eef_position_left), np.array(eef_orientation_left))
        obs['right_eef_pose'] = (np.array(eef_position_right), np.array(eef_orientation_right))

        # sensors
        for sensor_name, sensor_obs in self.sensors.items():
            obs['sensors'][sensor_name] = sensor_obs.get_data()

        return obs
