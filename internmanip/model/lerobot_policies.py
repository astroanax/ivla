# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import abc
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Type, TypeVar

import draccus
from internmanip.model.hub import HubMixin
from internmanip.model.types import FeatureType, NormalizationMode, PolicyFeature
from internmanip.model.utils import auto_select_torch_device, is_amp_available, is_torch_device_available
from internmanip.trainer.optim.optimizers import OptimizerConfig
from internmanip.trainer.optim.schedulers import LRSchedulerConfig
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import CONFIG_NAME
from huggingface_hub.errors import HfHubHTTPError


# Generic variable that is either PreTrainedConfig or a subclass thereof
T = TypeVar("T", bound="PreTrainedConfig")


@dataclass
class PreTrainedConfig(draccus.ChoiceRegistry, HubMixin, abc.ABC):
    """
    Base configuration class for policy models.

    Args:
        n_obs_steps: Number of environment steps worth of observations to pass to the policy (takes the
            current step and additional steps going back).
        input_shapes: A dictionary defining the shapes of the input data for the policy.
        output_shapes: A dictionary defining the shapes of the output data for the policy.
        input_normalization_modes: A dictionary with key representing the modality and the value specifies the
            normalization mode to apply.
        output_normalization_modes: Similar dictionary as `input_normalization_modes`, but to unnormalize to
            the original scale.
    """

    n_obs_steps: int = 1
    normalization_mapping: dict[str, NormalizationMode] = field(default_factory=dict)

    input_features: dict[str, PolicyFeature] = field(default_factory=dict)
    output_features: dict[str, PolicyFeature] = field(default_factory=dict)

    device: str | None = None  # cuda | cpu | mp
    # `use_amp` determines whether to use Automatic Mixed Precision (AMP) for training and evaluation. With AMP,
    # automatic gradient scaling is used.
    use_amp: bool = False

    def __post_init__(self):
        self.pretrained_path = None
        if not self.device or not is_torch_device_available(self.device):
            auto_device = auto_select_torch_device()
            logging.warning(f"Device '{self.device}' is not available. Switching to '{auto_device}'.")
            self.device = auto_device.type

        # Automatically deactivate AMP if necessary
        if self.use_amp and not is_amp_available(self.device):
            logging.warning(
                f"Automatic Mixed Precision (amp) is not available on device '{self.device}'. Deactivating AMP."
            )
            self.use_amp = False

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)

    @property
    @abc.abstractmethod
    def observation_delta_indices(self) -> list | None:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def action_delta_indices(self) -> list | None:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def reward_delta_indices(self) -> list | None:
        raise NotImplementedError

    @abc.abstractmethod
    def get_optimizer_preset(self) -> OptimizerConfig:
        raise NotImplementedError

    @abc.abstractmethod
    def get_scheduler_preset(self) -> LRSchedulerConfig | None:
        raise NotImplementedError

    @abc.abstractmethod
    def validate_features(self) -> None:
        raise NotImplementedError

    @property
    def robot_state_feature(self) -> PolicyFeature | None:
        for _, ft in self.input_features.items():
            if ft.type is FeatureType.STATE:
                return ft
        return None

    @property
    def env_state_feature(self) -> PolicyFeature | None:
        for _, ft in self.input_features.items():
            if ft.type is FeatureType.ENV:
                return ft
        return None

    @property
    def image_features(self) -> dict[str, PolicyFeature]:
        return {key: ft for key, ft in self.input_features.items() if ft.type is FeatureType.VISUAL}

    @property
    def action_feature(self) -> PolicyFeature | None:
        for _, ft in self.output_features.items():
            if ft.type is FeatureType.ACTION:
                return ft
        return None

    def _save_pretrained(self, save_directory: Path) -> None:
        # Create a copy of the config for serialization
        config_dict = self.__dict__.copy()
        
        # Convert PolicyFeature objects to dictionaries for JSON serialization
        if hasattr(self, 'input_features'):
            config_dict['input_features'] = {
                key: feature.to_dict() if hasattr(feature, 'to_dict') else feature
                for key, feature in self.input_features.items()
            }
        
        if hasattr(self, 'output_features'):
            config_dict['output_features'] = {
                key: feature.to_dict() if hasattr(feature, 'to_dict') else feature
                for key, feature in self.output_features.items()
            }
        
        # Convert NormalizationMode enums to strings
        if hasattr(self, 'normalization_mapping'):
            config_dict['normalization_mapping'] = {
                key: value.value if hasattr(value, 'value') else value
                for key, value in self.normalization_mapping.items()
            }
        
        with open(save_directory / CONFIG_NAME, "w") as f:
            import json
            json.dump(config_dict, f, indent=4)

    def to_json_string(self,):
        return ''
    
    @classmethod
    def from_pretrained(
        cls: Type[T],
        pretrained_name_or_path: str | Path,
        *,
        force_download: bool = False,
        resume_download: bool = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        **policy_kwargs,
    ) -> T:
        model_id = str(pretrained_name_or_path)
        config_file: str | None = None
        if Path(model_id).is_dir():
            if CONFIG_NAME in os.listdir(model_id):
                config_file = os.path.join(model_id, CONFIG_NAME)
            else:
                print(f"{CONFIG_NAME} not found in {Path(model_id).resolve()}")
        else:
            try:
                config_file = hf_hub_download(
                    repo_id=model_id,
                    filename=CONFIG_NAME,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
            except HfHubHTTPError as e:
                raise FileNotFoundError(
                    f"{CONFIG_NAME} not found on the HuggingFace Hub in {model_id}"
                ) from e

        # Load the config file
        with open(config_file, 'r') as f:
            import json
            config_data = json.load(f)
        
        # Convert PolicyFeature dictionaries back to PolicyFeature objects
        if 'input_features' in config_data:
            from internmanip.model.types import PolicyFeature
            config_data['input_features'] = {
                key: PolicyFeature.from_dict(feature_data) if isinstance(feature_data, dict) else feature_data
                for key, feature_data in config_data['input_features'].items()
            }
        
        if 'output_features' in config_data:
            from internmanip.model.types import PolicyFeature
            config_data['output_features'] = {
                key: PolicyFeature.from_dict(feature_data) if isinstance(feature_data, dict) else feature_data
                for key, feature_data in config_data['output_features'].items()
            }
        
        # Convert NormalizationMode strings back to enums
        if 'normalization_mapping' in config_data:
            from internmanip.model.types import NormalizationMode
            config_data['normalization_mapping'] = {
                key: NormalizationMode(value) if isinstance(value, str) else value
                for key, value in config_data['normalization_mapping'].items()
            }
        
        # Create the config instance
        instance = cls(**config_data)
        return instance
