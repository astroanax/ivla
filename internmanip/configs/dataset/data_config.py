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
    StateActionToTensor,
    StateActionTransform,
)
from internmanip.dataset.transform.video import (
    VideoColorJitter,
    VideoCrop,
    VideoResize,
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

class SweepDataConfig(BaseDataConfig):

    video_keys = ['video.base_view','video.ego_view']
    state_keys = ['state.ee_pos','state.ee_rot']
    action_keys = ['action.delta_ee_pos','action.delta_ee_rot', 'action.gripper']
    language_keys = ['annotation.human.action.task_description']

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
            'video': video_modality,
            'state': state_modality,
            'action': action_modality,
            'language': language_modality,
        }

        return modality_configs

    def transform(self):
        transforms = [
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation='linear'),
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
                    'state.ee_pos': 'mean_std',
                    'state.ee_rot': 'mean_std',
                },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    'action.delta_ee_pos': 'mean_std',
                    'action.delta_ee_rot': 'mean_std',
                    'action.gripper': 'binary'
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

    video_keys = ['video.image_0']
    state_keys = ['state.ee_pos','state.ee_rot','state.gripper']
    action_keys = ['action.delta_ee_pos','action.delta_ee_rot','action.gripper']
    language_keys = ['annotation.human.action.task_description']

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
            'video': video_modality,
            'state': state_modality,
            'action': action_modality,
            'language': language_modality,
        }

        return modality_configs

    def transform(self):
        transforms = [
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation='linear'),
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
                    'state.ee_pos': 'mean_std',
                    'state.ee_rot': 'mean_std',
                    'state.gripper': 'binary'
                },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    'action.delta_ee_pos': 'mean_std',
                    'action.delta_ee_rot': 'mean_std',
                    'action.gripper': 'binary'
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

class GenManipDataConfig(BaseDataConfig):
    video_keys = ['video.ego_view', 'video.base_view', 'video.base_2_view']
    state_keys = ['state.joints','state.gripper','state.joints_vel','state.gripper_vel','state.ee_pos','state.ee_rot']
    action_keys = ['action.gripper','action.delta_ee_pos','action.delta_ee_rot']
    language_keys = ['annotation.human.action.task_description']

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
            'video': video_modality,
            'state': state_modality,
            'action': action_modality,
            'language': language_modality,
        }

        return modality_configs

    def transform(self):
        transforms = [
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation='linear'),
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
                    'state.joints': 'mean_std',
                    'state.gripper': 'binary',
                    'state.joints_vel': 'mean_std',
                    'state.gripper_vel': 'binary',
                    'state.ee_pos': 'mean_std',
                    'state.ee_rot': 'mean_std'
                },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    'action.gripper': 'binary',
                    'action.delta_ee_pos': 'mean_std',
                    'action.delta_ee_rot': 'mean_std'
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

class GoogleRobotDataConfig(BaseDataConfig):
    video_keys = ['video.image']
    state_keys = ['state.ee_pos','state.ee_rot','state.gripper']
    action_keys = ['action.delta_ee_pos','action.delta_ee_rot', 'action.gripper']
    language_keys = ['annotation.human.action.task_description']

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
            'video': video_modality,
            'state': state_modality,
            'action': action_modality,
            'language': language_modality,
        }

        return modality_configs

    def transform(self):
        transforms = [
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation='linear'),
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
                    'state.ee_pos': 'mean_std',
                    'state.ee_rot': 'mean_std',
                    'state.gripper': 'binary'
                },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    'action.delta_ee_pos': 'mean_std',
                    'action.delta_ee_rot': 'mean_std',
                    'action.gripper': 'binary'
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

class CalvinDataConfig(BaseDataConfig):
    video_keys = ['video.image_base', 'video.image_wrist']
    state_keys = ['state.ee_pos','state.ee_rot']
    action_keys = ['action.delta_ee_pos','action.delta_ee_rot','action.gripper']
    language_keys = ['annotation.human.action.task_description']

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
            'video': video_modality,
            'state': state_modality,
            'action': action_modality,
            'language': language_modality,
        }

        return modality_configs

    def transform(self):
        transforms = [
            VideoToTensor(apply_to=[self.video_keys[0]]),
            VideoCrop(apply_to=[self.video_keys[0]], scale=0.95),
            VideoResize(apply_to=[self.video_keys[0]], height=224, width=224, interpolation='linear'),
            VideoToTensor(apply_to=[self.video_keys[1]]),
            VideoCrop(apply_to=[self.video_keys[1]], scale=0.95),
            VideoResize(apply_to=[self.video_keys[1]], height=224, width=224, interpolation='linear'),
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
                    'state.ee_pos': 'mean_std',
                    'state.ee_rot': 'mean_std',
                },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    'action.delta_ee_pos': 'mean_std',
                    'action.delta_ee_rot': 'mean_std',
                    'action.gripper': 'binary'
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

class So100DataConfig(BaseDataConfig):
    video_keys = ['video.webcam']
    state_keys = ['state.single_arm', 'state.gripper']
    action_keys = ['action.single_arm', 'action.gripper']
    language_keys = ['annotation.human.task_description']

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
            'video': video_modality,
            'state': state_modality,
            'action': action_modality,
            'language': language_modality,
        }

        return modality_configs

    def transform(self) -> ModalityTransform:
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation='linear'),
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
                normalization_modes={key: 'min_max' for key in self.state_keys},
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: 'min_max' for key in self.action_keys},
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

DATA_CONFIG_MAP = {
    'sweep': SweepDataConfig(),
    'bridgedata_v2': BridgeDataV2DataConfig(),
    'genmanip_v1': GenManipDataConfig(),
    'google_robot': GoogleRobotDataConfig(),
    'calvin_abcd': CalvinDataConfig(),
    'so100': So100DataConfig()
}
