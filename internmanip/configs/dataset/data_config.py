# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod

from internmanip.dataset.base import ModalityConfig
from internmanip.dataset.transform.base import ComposedModalityTransform, ModalityTransform
from internmanip.dataset.transform.concat import ConcatTransform
from internmanip.dataset.transform.state_action import (
    # IdentityTransform,
    StateActionSinCosTransform,
    StateActionToTensor,
    StateActionTransform,
)
from internmanip.dataset.transform.video import (
    VideoColorJitter,
    VideoCrop,
    VideoResize,
    VideoToNumpy,
    VideoToTensor,
)


class BaseDataConfig(ABC):
    @abstractmethod
    def modality_config(self) -> dict[str, ModalityConfig]:
        pass

    @abstractmethod
    def transform(self) -> ModalityTransform:
        pass


###########################################################################################


class DemoDataConfig(BaseDataConfig):

    video_keys = ["video.ego_view"]
    state_keys = [
        "state.left_arm",
        "state.left_hand",
        "state.left_leg",
        "state.neck",
        "state.right_arm",
        "state.right_hand",
        "state.right_leg",
        "state.waist",
    ]
    action_keys = [
        "action.left_hand",
        "action.right_hand",
    ]
    language_keys = ["annotation.human.action.task_description", "annotation.human.validity"]
    observation_indices = [0]
    action_indices = list(range(16))

    

    def modality_config(self) -> dict[str, ModalityConfig]:
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )

        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )

        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )

        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )

        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }

        return modality_configs

    def transform(self) -> ModalityTransform:

        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            # VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)


###########################################################################################


class SweepDataConfig(BaseDataConfig):

    video_keys = ["video.base_view","video.ego_view"]
    state_keys = ["state.ee_pos","state.ee_rot"]
    action_keys = ["action.delta_ee_pos","action.delta_ee_rot", "action.gripper"]
    language_keys = ["annotation.human.action.task_description"]

    def modality_config(self, observation_indices, action_indices) -> dict[str, ModalityConfig]:
        self.action_indices = action_indices
        self.observation_indices = observation_indices
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )

        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )

        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )

        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )

        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }

        return modality_configs

    def transform(self):
        transforms = [
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.ee_pos": "mean_std",
                    "state.ee_rot": "mean_std",
                },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.delta_ee_pos": "mean_std",
                    "action.delta_ee_rot": "mean_std",
                    "action.gripper": "binary"
                }
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            )
        ]
        return transforms




class SweepDataConfigJoint(BaseDataConfig):

    video_keys = ["video.base_view","video.ego_view"]
    state_keys = ["state.joints", "state.gripper"]
    action_keys = ["action.joints", "action.gripper"]

    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(50))

    def modality_config(self, observation_indices, action_indices) -> dict[str, ModalityConfig]:
        self.action_indices = action_indices
        self.observation_indices = observation_indices
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )

        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )

        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )

        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )

        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }

        return modality_configs

    def transform(self):
        transforms = [
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            # VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            # VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.joints": "mean_std",
                    "state.gripper": "binary",
                },
            ),

            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.joints": "mean_std",
                    "action.gripper": "binary"
                }
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            )
        ]
        return transforms


###########################################################################################


class BridgeDataV2DataConfig(BaseDataConfig):

    video_keys = ["video.image_0"]
    state_keys = ["state.ee_pos","state.ee_rot","state.gripper"]
    action_keys = ["action.delta_ee_pos","action.delta_ee_rot","action.gripper"]
    language_keys = ["annotation.human.action.task_description"]

    def modality_config(self, observation_indices, action_indices) -> dict[str, ModalityConfig]:
        self.action_indices = action_indices
        self.observation_indices = observation_indices
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )

        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )

        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )

        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )

        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }

        return modality_configs

    def transform(self):
        transforms = [
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.ee_pos": "mean_std",
                    "state.ee_rot": "mean_std",
                    "state.gripper": "binary"
                },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.delta_ee_pos": "mean_std",
                    "action.delta_ee_rot": "mean_std",
                    "action.gripper": "binary"
                }
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            )
        ]
        return transforms

###########################################################################################


class WidowxMinMaxDataConfig(BaseDataConfig):
    video_keys = ["video.image_0"]
    state_keys = ["state.ee_pos","state.ee_rot","state.gripper"]
    action_keys = ["action.delta_ee_pos","action.delta_ee_rot","action.gripper"]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self, observation_indices, action_indices) -> dict[str, ModalityConfig]:
        self.action_indices = action_indices
        self.observation_indices = observation_indices
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )

        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )

        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )

        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )

        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }

        return modality_configs

    def transform(self):
        transforms = [
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            # VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            )
        ]
        return transforms



class WidowxQ99DataConfig(BaseDataConfig):
    video_keys = ["video.image_0"]
    state_keys = ["state.ee_pos","state.ee_rot","state.gripper"]
    action_keys = ["action.delta_ee_pos","action.delta_ee_rot","action.gripper"]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self, observation_indices, action_indices) -> dict[str, ModalityConfig]:
        self.action_indices = action_indices
        self.observation_indices = observation_indices
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )

        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )

        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )

        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )

        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }

        return modality_configs

    def transform(self):
        transforms = [
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            # VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            )
        ]
        return transforms



class GenManipDataJointConfig(BaseDataConfig):
    video_keys = [ "video.base_view", 'video.ego_view']
    state_keys = ['state.joints', "state.gripper"]
    action_keys = ["action.delta_joints", "action.gripper"]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self, observation_indices, action_indices) -> dict[str, ModalityConfig]:
        self.action_indices = action_indices
        self.observation_indices = observation_indices
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )

        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )

        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )

        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )


        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,

            "language": language_modality,
        }

        return modality_configs

    def transform(self):
        transforms = [
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            # VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.joints": "mean_std",
                    "state.gripper": "binary",
                },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.delta_joints": "mean_std",
                    "action.gripper": "binary",
                    
                }
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            )
        ]
        return transforms

class GenManipDataConfig(BaseDataConfig):
    video_keys = ["video.ego_view", "video.base_view", "video.base_2_view"]
    state_keys = ["state.joints","state.gripper","state.joints_vel","state.gripper_vel","state.ee_pos","state.ee_rot"]
    action_keys = ["action.gripper","action.delta_ee_pos","action.delta_ee_rot"]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self, observation_indices, action_indices) -> dict[str, ModalityConfig]:
        self.action_indices = action_indices
        self.observation_indices = observation_indices
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )

        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )

        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )

        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )


        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }

        return modality_configs

    def transform(self):
        transforms = [
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            # VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.joints": "mean_std",
                    "state.gripper": "binary",
                    "state.joints_vel": "mean_std",
                    "state.gripper_vel": "binary",
                    "state.ee_pos": "mean_std",
                    "state.ee_rot": "mean_std"
                },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.joints": "mean_std",
                    "action.gripper_w": "binary",
                    "action.gripper": "binary",
                    "action.ee_pos": "mean_std",
                    "action.ee_rot": "mean_std",
                    "action.delta_joints": "mean_std",
                    "action.delta_ee_pos": "mean_std",
                    "action.delta_ee_rot": "mean_std",
                    
                }
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            )
        ]
        return transforms

###########################################################################################


class GenManipEEFDataConfig(BaseDataConfig):
    video_keys = ["video.ego_view", "video.base_view"]
    state_keys = ["state.ee_pos","state.ee_rot","state.gripper"]
    action_keys = ["action.ee_pos","action.ee_rot","action.gripper"]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self, observation_indices, action_indices) -> dict[str, ModalityConfig]:
        self.action_indices = action_indices
        self.observation_indices = observation_indices
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )

        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )

        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )

        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )


        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }

        return modality_configs

    def transform(self):
        transforms = [
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            # VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.gripper": "binary",
                    "state.ee_pos": "mean_std",
                    "state.ee_rot": "mean_std"
                },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.gripper": "binary",
                    "action.ee_pos": "mean_std",
                    "action.ee_rot": "mean_std",
                }
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            )
        ]
        return transforms


class GenManipA2DDataConfig(BaseDataConfig):
    video_keys = ["video.head", "video.hand_left", "video.hand_right"]
    state_keys = ["state.effector_position","state.head_position","state.joint_position","state.waist_position","state.robot_orientation","state.robot_position"]
    action_keys = ["action.effector_position","action.head_position","action.joint_position","action.waist_position"]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self, observation_indices, action_indices) -> dict[str, ModalityConfig]:
        self.action_indices = action_indices
        self.observation_indices = observation_indices
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )

        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )

        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )

        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )


        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }

        return modality_configs

    def transform(self):
        transforms = [
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            # VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "mean_std" for key in self.state_keys},
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "mean_std" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            )
        ]
        return transforms

###########################################################################################

class GoogleRobotDataConfig(BaseDataConfig):

    video_keys = ["video.image"]
    state_keys = ["state.ee_pos","state.ee_rot","state.gripper"]
    action_keys = ["action.delta_ee_pos","action.delta_ee_rot", "action.gripper"]
    language_keys = ["annotation.human.action.task_description"]

    def modality_config(self, observation_indices, action_indices) -> dict[str, ModalityConfig]:
        self.action_indices = action_indices
        self.observation_indices = observation_indices
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )

        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )

        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )

        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )

        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }

        return modality_configs

    def transform(self):
        transforms = [
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.ee_pos": "mean_std",
                    "state.ee_rot": "mean_std",
                    "state.gripper": "binary"
                },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.delta_ee_pos": "mean_std",
                    "action.delta_ee_rot": "mean_std",
                    "action.gripper": "binary"
                }
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            )
        ]
        return transforms


###########################################################################################


class GoogleQ99DataConfig(BaseDataConfig):

    video_keys = ["video.image"]
    state_keys = ["state.ee_pos","state.ee_rot","state.gripper"]
    action_keys = ["action.delta_ee_pos","action.delta_ee_rot", "action.gripper"]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self, observation_indices, action_indices) -> dict[str, ModalityConfig]:
        self.action_indices = action_indices
        self.observation_indices = observation_indices
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )

        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )

        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )

        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )

        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }

        return modality_configs

    def transform(self):
        transforms = [
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            # VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=['state.gripper'],
                normalization_modes={
                    "state.gripper": "binary"
                },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.delta_ee_pos": "q99",
                    "action.delta_ee_rot": "q99",
                    "action.gripper": "binary"
                }
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            )
        ]
        return transforms
        # return ComposedModalityTransform(transforms=transforms)


###########################################################################################

class GoogleMinMaxDataConfig(BaseDataConfig):

    video_keys = ["video.image"]
    state_keys = ["state.ee_pos","state.ee_rot","state.gripper"]
    action_keys = ["action.delta_ee_pos","action.delta_ee_rot", "action.gripper"]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self, observation_indices, action_indices) -> dict[str, ModalityConfig]:
        self.action_indices = action_indices
        self.observation_indices = observation_indices
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )

        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )

        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )

        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )

        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }

        return modality_configs

    def transform(self):
        transforms = [
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            # VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            )
        ]
        return transforms
        # return ComposedModalityTransform(transforms=transforms)


###########################################################################################

class CalvinDataConfig(BaseDataConfig):
    video_keys = ["video.image_base", "video.image_wrist"]
    state_keys = ["state.ee_pos","state.ee_rot"]
    action_keys = ["action.delta_ee_pos","action.delta_ee_rot","action.gripper"]
    language_keys = ["annotation.human.action.task_description"]

    def modality_config(self, observation_indices, action_indices) -> dict[str, ModalityConfig]:
        self.action_indices = action_indices
        self.observation_indices = observation_indices
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )

        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )

        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )

        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )

        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }

        return modality_configs

    def transform(self):
        transforms = [
            VideoToTensor(apply_to=[self.video_keys[0]]),
            VideoCrop(apply_to=[self.video_keys[0]], scale=0.95),
            VideoResize(apply_to=[self.video_keys[0]], height=224, width=224, interpolation="linear"),
            VideoToTensor(apply_to=[self.video_keys[1]]),
            VideoCrop(apply_to=[self.video_keys[1]], scale=0.95),
            VideoResize(apply_to=[self.video_keys[1]], height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.ee_pos": "mean_std",
                    "state.ee_rot": "mean_std",
                },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.delta_ee_pos": "mean_std",
                    "action.delta_ee_rot": "mean_std",
                    "action.gripper": "binary"
                }
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            )
        ]
        return transforms

###########################################################################################


class Gr1ArmsOnlyDataConfig(BaseDataConfig):
    video_keys = ["video.ego_view"]
    state_keys = [
        "state.left_arm",
        "state.right_arm",
        "state.left_hand",
        "state.right_hand",
    ]
    action_keys = [
        "action.left_arm",
        "action.right_arm",
        "action.left_hand",
        "action.right_hand",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self, observation_indices, action_indices) -> dict[str, ModalityConfig]:
        self.action_indices = action_indices
        self.observation_indices = observation_indices
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )

        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )

        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )

        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )

        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }

        return modality_configs

    def transform(self) -> ModalityTransform:
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionSinCosTransform(apply_to=self.state_keys),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            )
        ]
        return ComposedModalityTransform(transforms=transforms)


###########################################################################################


class So100DataConfig(BaseDataConfig):
    video_keys = ["video.webcam"]
    state_keys = ["state.single_arm", "state.gripper"]
    action_keys = ["action.single_arm", "action.gripper"]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self, observation_indices, action_indices) -> dict[str, ModalityConfig]:
        self.action_indices = action_indices
        self.observation_indices = observation_indices
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )

        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )

        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )

        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )

        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }

        return modality_configs

    def transform(self) -> ModalityTransform:
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            )
        ]
        return ComposedModalityTransform(transforms=transforms)


###########################################################################################


class Gr1FullUpperBodyDataConfig(BaseDataConfig):
    video_keys = ["video.front_view"]
    state_keys = [
        "state.left_arm",
        "state.right_arm",
        "state.left_hand",
        "state.right_hand",
        "state.waist",
        "state.neck",
    ]
    action_keys = [
        "action.left_arm",
        "action.right_arm",
        "action.left_hand",
        "action.right_hand",
        "action.waist",
        "action.neck",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self, observation_indices, action_indices) -> dict[str, ModalityConfig]:
        self.action_indices = action_indices
        self.observation_indices = observation_indices
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            )
        ]

        return ComposedModalityTransform(transforms=transforms)


###########################################################################################


class BimanualPandaGripperDataConfig(BaseDataConfig):
    video_keys = [
        "video.right_wrist_view",
        "video.left_wrist_view",
        "video.front_view",
    ]
    state_keys = [
        "state.right_arm_eef_pos",
        "state.right_arm_eef_quat",
        "state.right_gripper_qpos",
        "state.left_arm_eef_pos",
        "state.left_arm_eef_quat",
        "state.left_gripper_qpos",
    ]
    action_keys = [
        "action.right_arm_eef_pos",
        "action.right_arm_eef_rot",
        "action.right_gripper_close",
        "action.left_arm_eef_pos",
        "action.left_arm_eef_rot",
        "action.left_gripper_close",
    ]

    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self, observation_indices, action_indices) -> dict[str, ModalityConfig]:
        self.action_indices = action_indices
        self.observation_indices = observation_indices
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.right_arm_eef_pos": "min_max",
                    "state.right_gripper_qpos": "min_max",
                    "state.left_arm_eef_pos": "min_max",
                    "state.left_gripper_qpos": "min_max",
                },
                target_rotations={
                    "state.right_arm_eef_quat": "rotation_6d",
                    "state.left_arm_eef_quat": "rotation_6d",
                },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.right_gripper_close": "binary",
                    "action.left_gripper_close": "binary",
                },
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            )
        ]

        return ComposedModalityTransform(transforms=transforms)


###########################################################################################


class BimanualPandaHandDataConfig(BaseDataConfig):
    video_keys = [
        "video.right_wrist_view",
        "video.left_wrist_view",
        "video.ego_view",
    ]
    state_keys = [
        "state.right_arm_eef_pos",
        "state.right_arm_eef_quat",
        "state.right_hand",
        "state.left_arm_eef_pos",
        "state.left_arm_eef_quat",
        "state.left_hand",
    ]
    action_keys = [
        "action.right_arm_eef_pos",
        "action.right_arm_eef_rot",
        "action.right_hand",
        "action.left_arm_eef_pos",
        "action.left_arm_eef_rot",
        "action.left_hand",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self, observation_indices, action_indices) -> dict[str, ModalityConfig]:
        self.action_indices = action_indices
        self.observation_indices = observation_indices
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.right_arm_eef_pos": "min_max",
                    "state.right_hand": "min_max",
                    "state.left_arm_eef_pos": "min_max",
                    "state.left_hand": "min_max",
                },
                target_rotations={
                    "state.right_arm_eef_quat": "rotation_6d",
                    "state.left_arm_eef_quat": "rotation_6d",
                },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.right_hand": "min_max",
                    "action.left_hand": "min_max",
                },
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            )
        ]

        return ComposedModalityTransform(transforms=transforms)


###########################################################################################


class FrankaStateDataConfig(BaseDataConfig):
    video_keys = ["video.ego_view", "video.base_view"]
    state_keys = ["state.ee_pos","state.ee_rot"]
    action_keys = ["action.delta_ee_pos","action.delta_ee_rot", "action.gripper"]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self, observation_indices, action_indices) -> dict[str, ModalityConfig]:
        self.action_indices = action_indices
        self.observation_indices = observation_indices
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )

        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )

        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )

        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )

        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }

        return modality_configs

    def transform(self):
        transforms = [
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.ee_pos": "mean_std",
                    "state.ee_rot": "mean_std",
                },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.delta_ee_pos": "mean_std",
                    "action.delta_ee_rot": "mean_std",
                    "action.gripper": "binary"
                }
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            )
        ]
        return transforms

###########################################################################################


class Gr1ArmsWaistDataConfig(Gr1ArmsOnlyDataConfig):
    video_keys = ["video.ego_view"]
    state_keys = [
        "state.left_arm",
        "state.right_arm",
        "state.left_hand",
        "state.right_hand",
        "state.waist",
    ]
    action_keys = [
        "action.left_arm",
        "action.right_arm",
        "action.left_hand",
        "action.right_hand",
        "action.waist",
    ]
    language_keys = ["annotation.human.coarse_action"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self, observation_indices, action_indices) -> dict[str, ModalityConfig]:
        self.action_indices = action_indices
        self.observation_indices = observation_indices
        return super().modality_config(observation_indices, action_indices)

    def transform(self):
        return super().transform()


###########################################################################################

DATA_CONFIG_MAP = {
    "demo": DemoDataConfig(),
    "sweep": SweepDataConfig(),
    "sweep_joint": SweepDataConfigJoint(),
    "bridgedata_v2": BridgeDataV2DataConfig(),
    "widowx_minmax": WidowxMinMaxDataConfig(),
    "widowx_q99": WidowxQ99DataConfig(),
    "genmanip_v1": GenManipDataConfig(),
    "genmanip_joint": GenManipDataJointConfig(),
    "genmanip_eef": GenManipEEFDataConfig(),
    "genmanip_a2d": GenManipA2DDataConfig(),
    "google_robot": GoogleRobotDataConfig(),
    "google_minmax": GoogleMinMaxDataConfig(),
    "google_q99": GoogleQ99DataConfig(),
    "calvin_abcd": CalvinDataConfig(),
    "gr1_arms_waist": Gr1ArmsWaistDataConfig(),
    "gr1_arms_only": Gr1ArmsOnlyDataConfig(),
    "gr1_full_upper_body": Gr1FullUpperBodyDataConfig(),
    "bimanual_panda_gripper": BimanualPandaGripperDataConfig(),
    "bimanual_panda_hand": BimanualPandaHandDataConfig(),
    "franka_gripper": FrankaStateDataConfig(),
    "so100": So100DataConfig(),
}
