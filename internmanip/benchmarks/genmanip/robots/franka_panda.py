from collections import defaultdict

import numpy as np
import roboticstoolbox as rtb
from scipy.spatial.transform import Rotation as R

from internutopia.core.robot.robot import BaseRobot
from internutopia.core.scene.scene import IScene
from internutopia_extension.robots.franka import Franka

from .actions import validate_action
from ..config.task_config import FrankaPandaRobotCfg


@BaseRobot.register('FrankaPanda')
class FrankaPanda(BaseRobot):
    def __init__(self, config: FrankaPandaRobotCfg, scene: IScene):
        super().__init__(config, scene)

        self._robot_ik_base = None
        self.prim_path = config.prim_path
        self._robot_scale = np.array([1.0, 1.0, 1.0])

        self.articulation = Franka(
            prim_path=self.prim_path,
            name=config.name,
            position=config.position,
            end_effector_prim_name="franka/panda_hand",
            usd_path=config.usd_path,
            scale=self._robot_scale
        )

    def get_robot_scale(self):
        return self._robot_scale

    def get_robot_ik_base(self):
        return self._robot_ik_base
    
    def post_reset(self):
        super().post_reset()
        self._robot_ik_base = self._rigid_body_map[self.prim_path + '/franka/panda_link0']

    def apply_action(self, action):
        """
        Args:
            action: inputs for controllers.
        """
        if not action:
            return
        
        action_type, action = validate_action(action)

        joint_controller = self.controllers['joint_controller']
        gripper_controller = self.controllers['gripper_controller']
        ik_controller = self.controllers['arm_ik_controller']

        if action_type=='joint':
            control = joint_controller.action_to_control([action])
        elif action_type=='arm_gripper':
            if isinstance(action.gripper_action, list):
                control = joint_controller.action_to_control([action.arm_action+action.gripper_action])
            else:
                gripper_control = gripper_controller.action_to_control(['open' if action.gripper_action==-1 else 'close'])
                control = joint_controller.action_to_control([action.arm_action+gripper_control.joint_positions[-2:]])
        elif action_type=='eef':
            arm_control = ik_controller.action_to_control([action.eef_position, action.eef_orientation])
            if isinstance(action.gripper_action, list):
                control = joint_controller.action_to_control([arm_control.joint_positions.tolist()+action.gripper_action])
            else:
                gripper_control = gripper_controller.action_to_control(['open' if action.gripper_action==-1 else 'close'])
                control = joint_controller.action_to_control([arm_control.joint_positions.tolist()+gripper_control.joint_positions[-2:]])

        self.articulation.apply_action(control)

    def get_obs(self) -> defaultdict:
        obs = defaultdict(dict)

        # robot_pose
        obs['robot_pose'] = self.articulation.get_pose()

        # joints_state
        obs['joints_state']['positions'] = self.articulation.get_joint_positions()
        obs['joints_state']['velocities'] = self.articulation.get_joint_velocities()
        
        # eef_pose
        panda = rtb.models.Panda()
        hand_pose = panda.fkine(q=obs['joints_state']['positions'], end="panda_hand").A
        eef_position = hand_pose[:3, 3]
        eef_orientation = R.from_matrix(hand_pose[:3, :3]).as_quat()[[3, 0, 1, 2]]
        obs['eef_pose'] = (np.array(eef_position), np.array(eef_orientation))

        # controllers
        # for c_obs_name, controller_obs in self.controllers.items():
        #     obs['controllers'][c_obs_name] = controller_obs.get_obs()
        
        # sensors
        for sensor_name, sensor_obs in self.sensors.items():
            obs['sensors'][sensor_name] = sensor_obs.get_data()

        return obs
