from collections import OrderedDict

import numpy as np
from internutopia.core.robot.robot import BaseRobot
from internutopia.core.scene.scene import IScene
from internutopia.core.sensor.camera import ICamera
from internutopia.core.sensor.sensor import BaseSensor
from internutopia.core.util import log

from ..config.task_config import CameraCfg


@BaseSensor.register('Camera')
class Camera(BaseSensor):
    """
    wrap of isaac sim's Camera class
    """

    def __init__(self, config: CameraCfg, robot: BaseRobot, scene: IScene = None):
        super().__init__(config, robot, scene)

    def post_reset(self):
        self._camera = self.create_camera()

    def get_prim_path(self, robot_type, gripper_type) -> str:
        if robot_type == 'franka' and self.name == 'realsense':
            if gripper_type == 'panda':
                return self._robot.prim_path + '/franka/panda_hand/geometry/realsense'
            return self._robot.prim_path + '/robotiq/arm/panda_link8/realsense'

        if robot_type == 'franka' and self.name == 'obs_camera':
            if gripper_type == 'panda':
                return self._robot.prim_path + '/franka/obs_camera'
            return self._robot.prim_path + '/robotiq/obs_camera'

        if robot_type == 'franka' and self.name == 'obs_camera_2':
            if gripper_type == 'panda':
                return self._robot.prim_path + '/franka/obs_camera_2'
            return self._robot.prim_path + '/robotiq/obs_camera_2'

        if robot_type == 'aloha_split' and self.name == 'top_camera':
            return (
                self._robot.prim_path
                + '/aloha_split/split_aloha_mid_360_with_piper/split_aloha_mid_360_with_piper/top_camera_link/Camera'  # noqa E501
            )

        if robot_type == 'aloha_split' and self.name == 'left_camera':
            return (
                self._robot.prim_path
                + '/aloha_split/split_aloha_mid_360_with_piper/split_aloha_mid_360_with_piper/fl/link6/Camera'
            )

        if robot_type == 'aloha_split' and self.name == 'right_camera':
            return (
                self._robot.prim_path
                + '/aloha_split/split_aloha_mid_360_with_piper/split_aloha_mid_360_with_piper/fr/link6/Camera'
            )

    def create_camera(self) -> ICamera:
        """Create an isaac-sim camera object.

        Initializes the camera's resolution and prim path based on configuration.

        Returns:
            ICamera: The initialized camera object.
        """
        # Use the configured camera resolution if provided.
        prim_path = self.get_prim_path(self.config.robot_type, self.config.gripper_type)

        log.debug('camera_prim_path: ' + prim_path)
        log.debug('name            : ' + self.name)
        camera = ICamera.create(
            name=self.name,
            prim_path=prim_path,
            distance_to_image_plane=self.config.depth_obs,
            resolution=self.config.resolution,
        )
        return camera

    def get_data(self) -> OrderedDict:
        data = OrderedDict()
        rgba_data = self._camera.get_rgba()
        depth_data = self._camera.get_distance_to_image_plane()

        if isinstance(rgba_data, np.ndarray) and rgba_data.size > 0:
            data['rgb'] = rgba_data[:, :, :3]

        if isinstance(depth_data, np.ndarray) and depth_data.size > 0:
            data['depth'] = depth_data

        return data
