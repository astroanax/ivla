import numpy as np
from collections import OrderedDict

from internutopia.core.scene.scene import IScene
from internutopia.core.robot.robot import BaseRobot
from internutopia.core.sensor.sensor import BaseSensor
from internutopia.core.sensor.camera import ICamera
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

    def create_camera(self) -> ICamera:
        """Create an isaac-sim camera object.

        Initializes the camera's resolution and prim path based on configuration.

        Returns:
            ICamera: The initialized camera object.
        """
        # Use the configured camera resolution if provided.
        if self.name=='realsense':
            if self.config.gripper_type=="panda":
                prim_path = self._robot.prim_path + '/franka/panda_hand/geometry/realsense'
            else:
                prim_path = self._robot.prim_path + '/robotiq/arm/panda_link8/realsense'
        elif self.name=='obs_camera':
            if self.config.gripper_type=="panda":
                prim_path = self._robot.prim_path + '/franka/obs_camera'
            else:
                prim_path = self._robot.prim_path + '/robotiq/obs_camera'
        else:
            if self.config.gripper_type=="panda":
                prim_path = self._robot.prim_path + '/franka/obs_camera_2'
            else:
                prim_path = self._robot.prim_path + '/robotiq/obs_camera_2'
        
        log.debug('camera_prim_path: ' + prim_path)
        log.debug('name            : ' + self.name)
        camera = ICamera.create(
            name=self.name,
            prim_path=prim_path,
            distance_to_image_plane=self.config.depth_obs,
            resolution=self.config.resolution
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
