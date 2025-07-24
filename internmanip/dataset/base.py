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

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel, ValidationError
from torch.utils.data import Dataset
from tqdm import tqdm

from .utils.video import get_all_frames, get_frames_by_timestamps

from .embodiment_tags import EmbodimentTag
from .schema import (
    DatasetMetadata,
    DatasetStatisticalValues,
    LeRobotModalityMetadata,
    LeRobotStateActionMetadata,
)
from .transform import ComposedModalityTransform

from huggingface_hub import hf_hub_download, list_repo_files
import os

LE_ROBOT_MODALITY_FILENAME = "meta/modality.json"
LE_ROBOT_EPISODE_FILENAME = "meta/episodes.jsonl"
LE_ROBOT_TASKS_FILENAME = "meta/tasks.jsonl"
LE_ROBOT_INFO_FILENAME = "meta/info.json"
LE_ROBOT_STATS_FILENAME = "meta/stats.json"
LE_ROBOT_DATA_FILENAME = "data/*/*.parquet"

def aggregate_feature_stats(stats_ft_list: list[dict[str, dict]]) -> dict[str, dict[str, np.ndarray]]:
    """Aggregates stats for a single feature."""

    means = np.stack([s.mean for s in stats_ft_list])
    variances = np.stack([s.std ** 2 for s in stats_ft_list])
    counts = np.stack([s.count for s in stats_ft_list])
    total_count = counts.sum(axis=0)

    # Prepare weighted mean by matching number of dimensions
    while counts.ndim < means.ndim:
        counts = np.expand_dims(counts, axis=-1)

    # Compute the weighted mean
    weighted_means = means * counts
    total_mean = weighted_means.sum(axis=0) / total_count

    # Compute the variance using the parallel algorithm
    delta_means = means - total_mean
    weighted_variances = (variances + delta_means**2) * counts
    total_variance = weighted_variances.sum(axis=0) / total_count
    q01 = stats.norm.ppf(0.01, loc=total_mean, scale=np.sqrt(total_variance),)
    q99 = stats.norm.ppf(0.99, loc=total_mean, scale=np.sqrt(total_variance),)
    return {
        "min": np.min(np.stack([s.min for s in stats_ft_list]), axis=0),
        "max": np.max(np.stack([s.max for s in stats_ft_list]), axis=0),
        "mean": total_mean,
        "std": np.sqrt(total_variance),
        "count": total_count,
        "q01": q01,
        "q99": q99,
    }

def aggregate_stats(stats_list: list[DatasetMetadata]) -> dict[str, dict[str, np.ndarray]]:
    """Aggregate stats from multiple compute_stats outputs into a single set of stats.

    The final stats will have the union of all data keys from each of the stats dicts.

    For instance:
    - new_min = min(min_dataset_0, min_dataset_1, ...)
    - new_max = max(max_dataset_0, max_dataset_1, ...)
    - new_mean = (mean of all data, weighted by counts)
    - new_std = (std of all data)
    """
    embodiment_stats = {}
    for stats in stats_list:
        if not isinstance(stats, DatasetMetadata):
            raise ValueError(f"Expected DatasetMetadata, got {type(stats)}")
        if stats.embodiment_tag not in embodiment_stats.keys():
            embodiment_stats[stats.embodiment_tag] = [stats.statistics]
        else:
            embodiment_stats[stats.embodiment_tag].append(stats.statistics)
    aggregated_stats={}
    for tag,stats in embodiment_stats.items():
        state_keys = [key for key in stats[0].state.keys()]
        action_keys = [key for key in stats[0].action.keys()]
        states = {key:[] for key in state_keys}
        actions = {key:[] for key in action_keys}
        for stat in stats:
            if not isinstance(stat, DatasetStatistics):
                raise ValueError(f"Expected DatasetStatistics, got {type(stat)} for tag {tag}")
            for key in state_keys:
                states[key].append(stat.state[key])
            for key in action_keys:
                actions[key].append(stat.action[key])

        state_stats = {}
        for key in state_keys:
            temp = aggregate_feature_stats(states[key])
            state_stats[key] = DatasetStatisticalValues(**temp)
        action_stats = {}
        for key in action_keys:
            temp = aggregate_feature_stats(actions[key])
            action_stats[key] = DatasetStatisticalValues(**temp)
        
        aggregated_stats[tag.value] = DatasetStatistics(state=state_stats,action=action_stats)

    return aggregated_stats

def calculate_dataset_statistics(parquet_paths: list[Path]) -> dict:
    """Calculate the dataset statistics of all columns for a list of parquet files."""
    # Dataset statistics
    all_low_dim_data_list = []
    # Collect all the data
    for parquet_path in tqdm(
        sorted(list(parquet_paths)),
        desc="Collecting all parquet files...",
    ):
        # Load the parquet file
        parquet_data = pd.read_parquet(parquet_path)
        parquet_data = parquet_data
        all_low_dim_data_list.append(parquet_data)
    all_low_dim_data = pd.concat(all_low_dim_data_list, axis=0)
    # Compute dataset statistics
    dataset_statistics = {}
    for le_modality in all_low_dim_data.columns:
        print(f"Computing statistics for {le_modality}...")
        np_data = np.vstack(
            [np.asarray(x, dtype=np.float32) for x in all_low_dim_data[le_modality]]
        )
        dataset_statistics[le_modality] = {
            "mean": np.mean(np_data, axis=0).tolist(),
            "std": np.std(np_data, axis=0).tolist(),
            "min": np.min(np_data, axis=0).tolist(),
            "max": np.max(np_data, axis=0).tolist(),
            "q01": np.quantile(np_data, 0.01, axis=0).tolist(),
            "q99": np.quantile(np_data, 0.99, axis=0).tolist(),
        }
    return dataset_statistics


class ModalityConfig(BaseModel):
    """Configuration for a modality."""

    delta_indices: list[int]
    """Delta indices to sample relative to the current index. The returned data will correspond to the original data at a sampled base index + delta indices."""
    modality_keys: list[str]
    """The keys to load for the modality in the dataset."""


class LeRobotSingleDataset(Dataset):
    """
    Base dataset class for LeRobot that supports sharding.
    """

    def __init__(
        self,
        dataset_path: Path | str,
        modality_configs: dict[str, ModalityConfig],
        embodiment_tag: str | EmbodimentTag,
        video_backend: str = "decord",
        tolerance_s: float = 1e-4,
        video_backend_kwargs: dict | None = None,
        transforms: ComposedModalityTransform | None = None,
        augsteps: int = 10,
    ):
        """
        Initialize the dataset.

        Args:
            dataset_path (Path | str): The path to the dataset.
            modality_configs (dict[str, ModalityConfig]): The configuration for each modality. The keys are the modality names, and the values are the modality configurations.
                See `ModalityConfig` for more details.
            video_backend (str): Backend for video reading.
            video_backend_kwargs (dict): Keyword arguments for the video backend when initializing the video reader.
            tolerance_s (float, optional): Tolerance in seconds used to ensure data timestamps are actually in sync with the fps value. 
                It is used at the init of the dataset to make sure that each timestamps is separated to the next by 1/fps +/- tolerance_s. 
                This also applies to frames decoded from video files. It is also used to check that `delta_timestamps` (when provided) are multiples of 1/fps. Defaults to 1e-4.
            transforms (ComposedModalityTransform): The transforms to apply to the dataset.
            embodiment_tag (EmbodimentTag): Overload the embodiment tag for the dataset. e.g. define it as "new_embodiment"
            augsteps (int): The number of steps to augment. If 0, no augmentation is applied.
        """
        # first check if the path directory exists
        # if not Path(dataset_path).exists():
        #     raise FileNotFoundError(f"Dataset path {dataset_path} does not exist")

        hf_cache_dir = Path(os.getenv("HF_DATASETS_CACHE", Path.home() / ".cache" / "huggingface" / "datasets"))
        if Path(dataset_path).exists():
            pass 
        else:
            cached_dirs = list(hf_cache_dir.rglob(f"{dataset_path.replace('/', '__')}*"))
            if cached_dirs:
                dataset_path = str(cached_dirs[0])
            else:
                repo_id = dataset_path
                all_files = list_repo_files(repo_id=repo_id, repo_type="dataset")
                local_dir = hf_cache_dir / repo_id.replace("/", "__")

                for file in all_files:
                    hf_hub_download(
                        repo_id=repo_id,
                        filename=file,
                        repo_type="dataset",
                        local_dir=local_dir,
                        local_dir_use_symlinks=False,
                    )
                dataset_path = str(local_dir)

        self.tolerance_s = tolerance_s
        self.modality_configs = modality_configs
        self.video_backend = video_backend
        self.video_backend_kwargs = video_backend_kwargs if video_backend_kwargs is not None else {}
        self.augstep_ = augsteps>0
        self.augsteps = augsteps
        if transforms is None:
            self.transforms = ComposedModalityTransform(transforms=[])
        elif isinstance(transforms, list):
            self.transforms = ComposedModalityTransform(transforms=transforms)
        else:
            self.transforms = transforms

        self._dataset_path = Path(dataset_path)
        self._dataset_name = self._dataset_path.name
        if isinstance(embodiment_tag, EmbodimentTag):
            self.tag = embodiment_tag.value
        else:
            self.tag = embodiment_tag

        self.language_data = {}
        self._metadata = self._get_metadata(EmbodimentTag(self.tag))
        self._delta_indices = self._get_delta_indices()

        # LeRobot-specific config
        self._lerobot_modality_meta = self._get_lerobot_modality_meta()
        self._lerobot_info_meta = self._get_lerobot_info_meta()
        self._data_path_pattern = self._get_data_path_pattern()
        self._video_path_pattern = self._get_video_path_pattern()
        self._chunk_size = self._get_chunk_size()
        self._tasks = self._get_tasks()
        self.curr_traj_data = None
        self.curr_traj_id = None
        

        self._trajectory_ids, self._trajectory_lengths = self._get_trajectories()
        self._all_steps = self._get_all_steps()
        self._modality_keys = self._get_modality_keys()
        self.set_transforms_metadata(self.metadata)
        self.set_epoch(0)
        print(f"Initialized dataset {self.dataset_name} with {embodiment_tag}")

        # Check if the dataset is valid
        self._check_integrity()


    @property
    def dataset_path(self) -> Path:
        """The path to the dataset that contains the METADATA_FILENAME file."""
        return self._dataset_path

    @property
    def metadata(self) -> DatasetMetadata:
        """The metadata for the dataset, loaded from metadata.json in the dataset directory"""
        return self._metadata

    @property
    def trajectory_ids(self) -> np.ndarray:
        """The trajectory IDs in the dataset, stored as a 1D numpy array of strings."""
        return self._trajectory_ids

    @property
    def trajectory_lengths(self) -> np.ndarray:
        """The trajectory lengths in the dataset, stored as a 1D numpy array of integers.
        The order of the lengths is the same as the order of the trajectory IDs.
        """
        return self._trajectory_lengths

    @property
    def all_steps(self) -> list[tuple[int, int]]:
        """The trajectory IDs and base indices for all steps in the dataset.
        Example:
            self.trajectory_ids: [0, 1, 2]
            self.trajectory_lengths: [3, 2, 4]
            return: [
                ("traj_0", 0), ("traj_0", 1), ("traj_0", 2),
                ("traj_1", 0), ("traj_1", 1),
                ("traj_2", 0), ("traj_2", 1), ("traj_2", 2), ("traj_2", 3)
            ]
        """
        return self._all_steps

    @property
    def modality_keys(self) -> dict:
        """The modality keys for the dataset. The keys are the modality names, and the values are the keys for each modality.

        Example: {
            "video": ["video.image_side_0", "video.image_side_1"],
            "state": ["state.eef_position", "state.eef_rotation"],
            "action": ["action.eef_position", "action.eef_rotation"],
            "language": ["language.human.task"],
            "timestamp": ["timestamp"],
            "reward": ["reward"],
        }
        """
        return self._modality_keys

    @property
    def delta_indices(self) -> dict[str, np.ndarray]:
        """The delta indices for the dataset. The keys are the modality.key, and the values are the delta indices for each modality.key."""
        return self._delta_indices

    @property
    def dataset_name(self) -> str:
        """The name of the dataset."""
        return self._dataset_name

    @property
    def lerobot_modality_meta(self) -> LeRobotModalityMetadata:
        """The metadata for the LeRobot dataset."""
        return self._lerobot_modality_meta

    @property
    def lerobot_info_meta(self) -> dict:
        """The metadata for the LeRobot dataset."""
        return self._lerobot_info_meta

    @property
    def data_path_pattern(self) -> str:
        """The path pattern for the LeRobot dataset."""
        return self._data_path_pattern

    @property
    def video_path_pattern(self) -> str:
        """The path pattern for the LeRobot dataset."""
        return self._video_path_pattern

    @property
    def chunk_size(self) -> int:
        """The chunk size for the LeRobot dataset."""
        return self._chunk_size

    @property
    def tasks(self) -> pd.DataFrame:
        """The tasks for the dataset."""
        return self._tasks

    def _get_metadata(self, embodiment_tag: EmbodimentTag) -> DatasetMetadata:
        """Get the metadata for the dataset.

        Returns:
            dict: The metadata for the dataset.
        """

        # 1. Modality metadata
        modality_meta_path = self.dataset_path / LE_ROBOT_MODALITY_FILENAME
        assert (
            modality_meta_path.exists()
        ), f"Please provide a {LE_ROBOT_MODALITY_FILENAME} file in {self.dataset_path}"

        # 1.1. State and action modalities
        simplified_modality_meta: dict[str, dict] = {}
        with open(modality_meta_path, "r") as f:
            le_modality_meta = LeRobotModalityMetadata.model_validate(json.load(f))
        for modality in ["state", "action"]:
            simplified_modality_meta[modality] = {}
            le_state_action_meta: dict[str, LeRobotStateActionMetadata] = getattr(
                le_modality_meta, modality
            )
            for subkey in le_state_action_meta:
                state_action_dtype = np.dtype(le_state_action_meta[subkey].dtype)
                if np.issubdtype(state_action_dtype, np.floating):
                    continuous = True
                else:
                    continuous = False
                simplified_modality_meta[modality][subkey] = {
                    "absolute": le_state_action_meta[subkey].absolute,
                    "rotation_type": le_state_action_meta[subkey].rotation_type,
                    "continuous": continuous,
                }

        # 1.2. Video modalities
        le_info_path = self.dataset_path / LE_ROBOT_INFO_FILENAME
        assert (
            le_info_path.exists()
        ), f"Please provide a {LE_ROBOT_INFO_FILENAME} file in {self.dataset_path}"
        with open(le_info_path, "r") as f:
            le_info = json.load(f)
        simplified_modality_meta["video"] = {}
        for new_key in le_modality_meta.video:
            original_key = le_modality_meta.video[new_key].original_key
            if original_key is None:
                original_key = new_key
            le_video_meta = le_info["features"][original_key]
            height = le_video_meta["shape"][le_video_meta["names"].index("height")]
            width = le_video_meta["shape"][le_video_meta["names"].index("width")]
            # NOTE(FH): different lerobot dataset versions have different keys for the number of channels and fps

            try:
                channels = le_video_meta["shape"][le_video_meta["names"].index("channel")]
            except:
                try:
                    channels = le_video_meta["shape"][le_video_meta["names"].index("channels")]
                except:
                    channels = le_video_meta["shape"][le_video_meta["names"].index("rgb")]
  
            try:
                fps = le_video_meta["video_info"]["video.fps"]
            except:
                fps = le_video_meta["info"]["video.fps"]

            simplified_modality_meta["video"][new_key] = {
                "resolution": [width, height],
                "channels": channels,
                "fps": fps,
            }

        # 2. Dataset statistics
        stats_path = self.dataset_path / LE_ROBOT_STATS_FILENAME
        try:
            with open(stats_path, "r") as f:
                le_statistics_raw = json.load(f)
            le_statistics = {}
            for key, stat in le_statistics_raw.items():
                le_statistics[key] = {k: np.array(v) for k, v in stat.items()}
            for stat in le_statistics.values():
                DatasetStatisticalValues.model_validate(stat)
        except (FileNotFoundError, ValidationError) as e:
            print(f"Failed to load dataset statistics: {e}")
            print(f"Calculating dataset statistics for {self.dataset_name}")
            # Get all parquet files in the dataset paths
            parquet_files = list((self.dataset_path).glob(LE_ROBOT_DATA_FILENAME))
            le_statistics = calculate_dataset_statistics(parquet_files)
            with open(stats_path, "w") as f:
                json.dump(le_statistics, f, indent=4)
        dataset_statistics = {}
        for our_modality in ["state", "action"]:
            dataset_statistics[our_modality] = {}
            for subkey in simplified_modality_meta[our_modality]:
                dataset_statistics[our_modality][subkey] = {}
                state_action_meta = le_modality_meta.get_key_meta(f"{our_modality}.{subkey}")
                assert isinstance(state_action_meta, LeRobotStateActionMetadata)
                le_modality = state_action_meta.original_key if our_modality!=state_action_meta.original_key else state_action_meta.original_key + "." + subkey
                for stat_name in le_statistics[le_modality]:
                    stat = np.array(le_statistics[le_modality][stat_name])
                    dataset_statistics[our_modality][subkey][stat_name] = stat
                    simplified_modality_meta[our_modality][subkey]["shape"] = stat.shape

        # 3. Full dataset metadata
        metadata = DatasetMetadata(
            statistics=dataset_statistics,  # type: ignore
            modalities=simplified_modality_meta,  # type: ignore
            embodiment_tag=embodiment_tag,
        )

        return metadata

    def _get_trajectories(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the trajectories in the dataset."""
        # Get trajectory lengths, IDs, and whitelist from dataset metadata
        episode_path = self.dataset_path / LE_ROBOT_EPISODE_FILENAME
        episode_metadata = []
        with open(episode_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    episode_metadata.append(json.loads(line))
        trajectory_ids = []
        trajectory_lengths = []
        for episode in episode_metadata:
            trajectory_ids.append(episode["episode_index"])
            trajectory_lengths.append(episode["length"])
            if "tasks" in episode.keys():
                self.language_data[episode["episode_index"]] = episode["tasks"]
        return np.array(trajectory_ids), np.array(trajectory_lengths)

    def _get_all_steps(self) -> list[tuple[int, int]]:
        """Get the trajectory IDs and base indices for all steps in the dataset.

        Returns:
            list[tuple[str, int]]: A list of (trajectory_id, base_index) tuples.

        Example:
            self.trajectory_ids: [0, 1, 2]
            self.trajectory_lengths: [3, 2, 4]
            return: [
                ("traj_0", 0), ("traj_0", 1), ("traj_0", 2),
                ("traj_1", 0), ("traj_1", 1),
                ("traj_2", 0), ("traj_2", 1), ("traj_2", 2), ("traj_2", 3)
            ]
        """
        all_steps: list[tuple[int, int]] = []
        for trajectory_id, trajectory_length in zip(self.trajectory_ids, self.trajectory_lengths):
            for base_index in range(trajectory_length):
                all_steps.append((trajectory_id, base_index))
            if self.augstep_:
                chunk_index = self.get_episode_chunk(trajectory_id) # id//chunk-size
                parquet_path = self.dataset_path / self.data_path_pattern.format(
                    episode_chunk=chunk_index, episode_index=trajectory_id
                )
                assert parquet_path.exists(), f"Parquet file not found at {parquet_path}"
                data = pd.read_parquet(parquet_path)
                le_state_or_action_cfg = getattr(self.lerobot_modality_meta, 'action')
                change_indices = set()
                values = []
                for key in self._get_modality_keys()['action']: # iteration
                    subkey = key.split('.')[1]
                    if 'gripper' in subkey:
                        le_key = le_state_or_action_cfg[subkey].original_key if 'action'!=le_state_or_action_cfg[subkey].original_key else le_state_or_action_cfg[subkey].original_key + "." + subkey
                        value = data[le_key].to_numpy().tolist()
                        values.append(value)
                if values != []:
                    for i in range(len(values[0])-2):
                        flag = [(values[j][i] == values[j][i + 1]).all() and (values[j][i + 1] == values[j][i + 2]).all() for j in range(len(values))] # window_size 3
                        if False in flag:
                            change_indices.update((i,i+1,i+2))
                    for change_index in change_indices:
                        for i in range(self.augsteps):
                            all_steps.append((trajectory_id, change_index))
                else:
                    print(f"No action-gripper data found for trajectory {trajectory_id} in {parquet_path}. Skipping augmentation.")
        return all_steps

    def _get_modality_keys(self) -> dict:
        """Get the modality keys for the dataset.
        The keys are the modality names, and the values are the keys for each modality.
        See property `modality_keys` for the expected format.
        """
        modality_keys = defaultdict(list)
        for modality, config in self.modality_configs.items():
            modality_keys[modality] = config.modality_keys
        return modality_keys

    def _get_delta_indices(self) -> dict[str, np.ndarray]:
        """Restructure the delta indices to use modality.key as keys instead of just the modalities."""
        delta_indices: dict[str, np.ndarray] = {}
        for config in self.modality_configs.values():
            for key in config.modality_keys:
                delta_indices[key] = np.array(config.delta_indices)
        return delta_indices

    def _get_lerobot_modality_meta(self) -> LeRobotModalityMetadata:
        """Get the metadata for the LeRobot dataset."""
        modality_meta_path = self.dataset_path / LE_ROBOT_MODALITY_FILENAME
        assert (
            modality_meta_path.exists()
        ), f"Please provide a {LE_ROBOT_MODALITY_FILENAME} file in {self.dataset_path}"
        with open(modality_meta_path, "r") as f:
            modality_meta = LeRobotModalityMetadata.model_validate(json.load(f))
        return modality_meta

    def _get_lerobot_info_meta(self) -> dict:
        """Get the metadata for the LeRobot dataset."""
        info_meta_path = self.dataset_path / LE_ROBOT_INFO_FILENAME
        with open(info_meta_path, "r") as f:
            info_meta = json.load(f)
        return info_meta

    def _get_data_path_pattern(self) -> str:
        """Get the data path pattern for the LeRobot dataset."""
        return self.lerobot_info_meta["data_path"]

    def _get_video_path_pattern(self) -> str:
        """Get the video path pattern for the LeRobot dataset."""
        return self.lerobot_info_meta["video_path"]

    def _get_chunk_size(self) -> int:
        """Get the chunk size for the LeRobot dataset."""
        return self.lerobot_info_meta["chunks_size"]

    def _get_tasks(self) -> pd.DataFrame:
        """Get the tasks for the dataset."""
        tasks_path = self.dataset_path / LE_ROBOT_TASKS_FILENAME
        with open(tasks_path, "r") as f:
            tasks = [json.loads(line) for line in f]
        df = pd.DataFrame(tasks)
        return df.set_index("task_index")

    def _check_integrity(self):
        """Use the config to check if the keys are valid and detect silent data corruption."""
        ERROR_MSG_HEADER = f"Error occurred in initializing dataset {self.dataset_name}:\n"

        for modality_config in self.modality_configs.values():
            for key in modality_config.modality_keys:
                if key == "lapa_action" or key == "dream_actions":
                    continue  # no need for any metadata for lapa actions because it comes normalized
                # Check if the key is valid
                if "annotation" in key:
                    continue # no need fot thye check of annotations
                try:
                    self.lerobot_modality_meta.get_key_meta(key)
                except Exception as e:
                    raise ValueError(
                        ERROR_MSG_HEADER + f"Unable to find key {key} in modality metadata:\n{e}"
                    )

    def set_transforms_metadata(self, metadata: DatasetMetadata):
        """Set the metadata for the transforms. This is useful for transforms that need to know the metadata, such as the normalization values."""
        self.transforms.set_metadata(metadata)

    def set_epoch(self, epoch: int):
        """Set the epoch for the dataset.

        Args:
            epoch (int): The epoch to set.
        """
        self.epoch = epoch

    def __len__(self) -> int:
        """Get the total number of data points in the dataset.

        Returns:
            int: the total number of data points in the dataset.
        """
        return len(self.all_steps)

    def __str__(self) -> str:
        """Get the description of the dataset."""
        return f"{self.dataset_name} ({len(self)} steps)"

    def __getitem__(self, index: int) -> dict:
        """Get the data for a single step in a trajectory.

        Args:
            index (int): The index of the step to get.

        Returns:
            dict: The data for the step.
        """
        trajectory_id, base_index = self.all_steps[index]
            
        return self.transforms(self.get_step_data(trajectory_id, base_index))

    def get_step_data(self, trajectory_id: int, base_index: int) -> dict:
        """Get the RAW data for a single step in a trajectory. No transforms are applied.

        Args:
            trajectory_id (int): The name of the trajectory.
            base_index (int): The base step index in the trajectory.

        Returns:
            dict: The RAW data for the step.

        Example return:
            {
                "video": {
                    "video.image_side_0": [B, T, H, W, C],
                    "video.image_side_1": [B, T, H, W, C],
                },
                "state": {
                    "state.eef_position": [B, T, state_dim],
                    "state.eef_rotation": [B, T, state_dim],
                },
                "action": {
                    "action.eef_position": [B, T, action_dim],
                    "action.eef_rotation": [B, T, action_dim],
                },
            }
        """
        data = {}
        # Get the data for all modalities
        self.curr_traj_data = self.get_trajectory_data(trajectory_id)
        for modality in self.modality_keys:
            # Get the data corresponding to each key in the modality
            for key in self.modality_keys[modality]:
                data[key] = self.get_data_by_modality(trajectory_id, modality, key, base_index)
        
        try:
            step_indices = self.delta_indices[self.modality_keys['action'][0]] + base_index
            # step_indices = self.delta_indices['action.joints'] + base_index
            # Get the trajectory index
            trajectory_index = self.get_trajectory_index(trajectory_id)
            # Get the maximum length of the trajectory
            max_length = self.trajectory_lengths[trajectory_index]
            action_pad = step_indices >= max_length
            data['action_pad'] = action_pad
        except:
            pass
        return data

    def get_trajectory_data(self, trajectory_id: int) -> pd.DataFrame:
        """Get the data for a trajectory."""
        if self.curr_traj_id == trajectory_id and self.curr_traj_data is not None:
            return self.curr_traj_data
        else:
            chunk_index = self.get_episode_chunk(trajectory_id)
            parquet_path = self.dataset_path / self.data_path_pattern.format(
                episode_chunk=chunk_index, episode_index=trajectory_id
            )
            assert parquet_path.exists(), f"Parquet file not found at {parquet_path}"
            return pd.read_parquet(parquet_path)

    def get_trajectory_index(self, trajectory_id: int) -> int:
        """Get the index of the trajectory in the dataset by the trajectory ID.
        This is useful when you need to get the trajectory length or sampling weight corresponding to the trajectory ID.

        Args:
            trajectory_id (str): The ID of the trajectory.

        Returns:
            int: The index of the trajectory in the dataset.
        """
        trajectory_indices = np.where(self.trajectory_ids == trajectory_id)[0]
        if len(trajectory_indices) != 1:
            raise ValueError(
                f"Error finding trajectory index for {trajectory_id}, found {trajectory_indices=}"
            )
        return trajectory_indices[0]

    def get_episode_chunk(self, ep_index: int) -> int:
        """Get the chunk index for an episode index."""
        return ep_index // self.chunk_size

    def retrieve_data_and_pad(
        self,
        array: np.ndarray,
        step_indices: np.ndarray,
        max_length: int,
        padding_strategy: str = "first_last",
    ) -> np.ndarray:
        """Retrieve the data from the dataset and pad it if necessary.
        Args:
            array (np.ndarray): The array to retrieve the data from.
            step_indices (np.ndarray): The step indices to retrieve the data for.
            max_length (int): The maximum length of the data.
            padding_strategy (str): The padding strategy, either "first" or "last".
        """
        # Get the padding indices
        front_padding_indices = step_indices < 0
        end_padding_indices = step_indices >= max_length
        padding_positions = np.logical_or(front_padding_indices, end_padding_indices)
        # Retrieve the data with the non-padding indices
        # If there exists some padding, Given T step_indices, the shape of the retrieved data will be (T', ...) where T' < T
        raw_data = array[step_indices[~padding_positions]]
        assert isinstance(raw_data, np.ndarray), f"{type(raw_data)=}"
        # This is the shape of the output, (T, ...)
        if raw_data.ndim == 1:
            expected_shape = (len(step_indices),)
        else:
            expected_shape = (len(step_indices), *array.shape[1:])

        # Pad the data
        output = np.zeros(expected_shape)
        # Assign the non-padded data
        output[~padding_positions] = raw_data
        # If there exists some padding, pad the data
        if padding_positions.any():
            if padding_strategy == "first_last":
                # Use first / last step data to pad
                front_padding_data = array[0]
                end_padding_data = array[-1]
                output[front_padding_indices] = front_padding_data
                output[end_padding_indices] = end_padding_data
            elif padding_strategy == "zero":
                # Use zero padding
                output[padding_positions] = 0
            else:
                raise ValueError(f"Invalid padding strategy: {padding_strategy}")
        return output

    def get_video_path(self, trajectory_id: int, key: str) -> Path:
        chunk_index = self.get_episode_chunk(trajectory_id)
        original_key = self.lerobot_modality_meta.video[key].original_key
        if original_key is None:
            original_key = key
        video_filename = self.video_path_pattern.format(
            episode_chunk=chunk_index, episode_index=trajectory_id, video_key=original_key
        )
        return self.dataset_path / video_filename

    def get_video(
        self,
        trajectory_id: int,
        key: str,
        base_index: int,
    ) -> np.ndarray:
        """Get the video frames for a trajectory by a base index.

        Args:
            dataset (BaseSingleDataset): The dataset to retrieve the data from.
            trajectory_id (str): The ID of the trajectory.
            key (str): The key of the video.
            base_index (int): The base index of the trajectory.

        Returns:
            np.ndarray: The video frames for the trajectory and frame indices. Shape: (T, H, W, C)
        """
        # Get the step indices
        step_indices = self.delta_indices[key] + base_index
        # print(f"{step_indices=}")
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Ensure the indices are within the valid range
        # This is equivalent to padding the video with extra frames at the beginning and end
        step_indices = np.maximum(step_indices, 0)
        step_indices = np.minimum(step_indices, self.trajectory_lengths[trajectory_index] - 1)
        assert key.startswith("video."), f"Video key must start with 'video.', got {key}"
        # Get the sub-key
        key = key.replace("video.", "")
        video_path = self.get_video_path(trajectory_id, key)
        # Get the action/state timestamps for each frame in the video
        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        assert "timestamp" in self.curr_traj_data.columns, f"No timestamp found in {trajectory_id=}"
        timestamp: np.ndarray = self.curr_traj_data["timestamp"].to_numpy()
        # Get the corresponding video timestamps from the step indices
        video_timestamp = timestamp[step_indices]

        return get_frames_by_timestamps(
            video_path.as_posix(),
            video_timestamp,
            video_backend=self.video_backend,
            tolerance_s = self.tolerance_s,
            video_backend_kwargs=self.video_backend_kwargs,
        )

    def get_state_or_action(
        self,
        trajectory_id: int,
        modality: str,
        key: str,
        base_index: int,
    ) -> np.ndarray:
        """Get the state or action data for a trajectory by a base index.
        If the step indices are out of range, pad with the data:
            if the data is stored in absolute format, pad with the first or last step data;
            otherwise, pad with zero.

        Args:
            dataset (BaseSingleDataset): The dataset to retrieve the data from.
            trajectory_id (int): The ID of the trajectory.
            modality (str): The modality of the data.
            key (str): The key of the data.
            base_index (int): The base index of the trajectory.

        Returns:
            np.ndarray: The data for the trajectory and step indices.
        """
        # Get the step indices
        step_indices = self.delta_indices[key] + base_index
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Get the maximum length of the trajectory
        max_length = self.trajectory_lengths[trajectory_index]
        assert key.startswith(modality + "."), f"{key} must start with {modality + '.'}, got {key}"
        # Get the sub-key, e.g. state.joint_angles -> joint_angles
        key = key.replace(modality + ".", "")
        # Get the lerobot key
        le_state_or_action_cfg = getattr(self.lerobot_modality_meta, modality)
        le_key = le_state_or_action_cfg[key].original_key if modality!=le_state_or_action_cfg[key].original_key else le_state_or_action_cfg[key].original_key + "." + key
        # Get the data array, shape: (T, D)
        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        assert le_key in self.curr_traj_data.columns, f"No {le_key} found in {trajectory_id=}"
        data_array: np.ndarray = np.stack(self.curr_traj_data[le_key])  # type: ignore
        assert data_array.ndim == 2, f"Expected 2D array, got {data_array.shape} array of {le_key} in {trajectory_id=}"
        # Get the state or action configuration
        state_or_action_cfg = getattr(self.metadata.modalities, modality)[key]

        # Pad the data
        return self.retrieve_data_and_pad(
            array=data_array,
            step_indices=step_indices,
            max_length=max_length,
            padding_strategy="first_last" if state_or_action_cfg.absolute else "zero",
        )

    def get_language(
        self,
        trajectory_id: int,
        key: str,
        base_index: int,
    ) -> list[str]:
        """Get the language annotation data for a trajectory by step indices.

        Args:
            dataset (BaseSingleDataset): The dataset to retrieve the data from.
            trajectory_id (int): The ID of the trajectory.
            key (str): The key of the annotation.
            base_index (int): The base index of the trajectory.

        Returns:
            list[str]: The annotation data for the trajectory and step indices. If no matching data is found, return empty strings.
        """
        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        # Get the step indices
        step_indices = self.delta_indices[key] + base_index
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Get the maximum length of the trajectory
        max_length = self.trajectory_lengths[trajectory_index]
        # Get the end times corresponding to the closest indices
        step_indices = np.maximum(step_indices, 0)
        step_indices = np.minimum(step_indices, max_length - 1)
        # Get the annotations
        task_indices: list[int] = []
        assert key.startswith(
            "annotation."
        ), f"Language key must start with 'annotation.', got {key}"
        subkey = key.replace("annotation.", "")
        annotation_meta = self.lerobot_modality_meta.annotation
        if annotation_meta is not None:
            assert annotation_meta is not None, f"Annotation metadata is None for {subkey}"
            assert (
                subkey in annotation_meta
            ), f"Annotation key {subkey} not found in metadata, available annotation keys: {annotation_meta.keys()}"
            subkey_meta = annotation_meta[subkey]
            original_key = subkey_meta.original_key
            if original_key is None:
                original_key = key
            for i in range(len(step_indices)):
                task_indices.append(self.curr_traj_data[original_key][step_indices[i]].item())
            return self.tasks.loc[task_indices]["task"].tolist()
        else:
            assert self.language_data is not {}, f"Annotation metadata is None for {subkey}"
            language_data = []
            for i in range(len(step_indices)):
                language_data.append(self.language_data[trajectory_id][0])
            return language_data

    def get_data_by_modality(
        self,
        trajectory_id: int,
        modality: str,
        key: str,
        base_index: int,
    ):
        """Get the data corresponding to the modality for a trajectory by a base index.
        This method will call the corresponding helper method based on the modality.
        See the helper methods for more details.
        NOTE: For the language modality, the data is padded with empty strings if no matching data is found.

        Args:
            dataset (BaseSingleDataset): The dataset to retrieve the data from.
            trajectory_id (int): The ID of the trajectory.
            modality (str): The modality of the data.
            key (str): The key of the data.
            base_index (int): The base index of the trajectory.
        """
        if modality == "video":
            return self.get_video(trajectory_id, key, base_index)
        elif modality == "state" or modality == "action":
            return self.get_state_or_action(trajectory_id, modality, key, base_index)
        elif modality == "language":
            return self.get_language(trajectory_id, key, base_index)
        else:
            raise ValueError(f"Invalid modality: {modality}")

    def get_input_output_features(self, data_config: str) -> tuple[dict, dict]:
        """Get input and output features for model configuration.
        
        Returns:
            tuple[dict, dict]: A tuple containing (input_features, output_features).
                - input_features: Dict mapping observation keys to PolicyFeature objects
                - output_features: Dict mapping action keys to PolicyFeature objects
                
        Example:
            input_features = {
                "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(14,)),
                "observation.images.base_view": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
            }
            output_features = {
                "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
            }
        """
        from internmanip.model.types import PolicyFeature, FeatureType
        
        # Handle sweep_joint data_config with hardcoded features
        if data_config == "sweep_joint":
            input_features = {
                'video.ego_view': PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
                'video.base_view': PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
                'observation.state': PolicyFeature(type=FeatureType.STATE, shape=(8,))
            }
            output_features = {
                'action': PolicyFeature(type=FeatureType.ACTION, shape=(8,))
            }
            return input_features, output_features
        elif data_config == "genmanip":
            input_features = {
                'video.ego_view': PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
                'video.base_view': PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
                'observation.state': PolicyFeature(type=FeatureType.STATE, shape=(8,))
            }
            output_features = {
                'action': PolicyFeature(type=FeatureType.ACTION, shape=(7,))
            }
            return input_features, output_features
        elif data_config in ["google", "google_minmax", "google_q99"]:
            input_features = {
                'video.image': PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
                'observation.state': PolicyFeature(type=FeatureType.STATE, shape=(7,))
            }
            output_features = {
                'action': PolicyFeature(type=FeatureType.ACTION, shape=(7,))
            }
            return input_features, output_features

        elif data_config in ["widowx", "widowx_minmax"]:
            input_features = {
                'video.image_0': PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
                'video.image_1': PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
                'video.image_2': PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
                'video.image_3': PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
                'observation.state': PolicyFeature(type=FeatureType.STATE, shape=(7,))
            }
            output_features = {
                'action': PolicyFeature(type=FeatureType.ACTION, shape=(7,))
            }
            return input_features, output_features
        elif data_config == "calvin":
            input_features = {
                'video.image_base': PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
                'video.image_wrist': PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
                'observation.state': PolicyFeature(type=FeatureType.STATE, shape=(7,))
            }
            output_features = {
                'action': PolicyFeature(type=FeatureType.ACTION, shape=(7,))
            }
            return input_features, output_features
        else:
            raise NotImplementedError(f"Data config {data_config} not implemented")

class MultiLeRobotDataset(torch.utils.data.Dataset):
    """A dataset consisting of multiple underlying `LeRobotDataset`s.

    The underlying `LeRobotDataset`s are effectively concatenated, and this class adopts much of the API
    structure of `LeRobotDataset`.
    """

    def __init__(
        self,
        repo_ids: list[str],
        root: Path | str,
        modality_configs: dict[str, ModalityConfig],
        embodiment_tags: list[str] | list[EmbodimentTag],
        transforms: ComposedModalityTransform,
        video_backend: str,
        augsteps: int = 10,
    ):
        super().__init__()
        self.repo_ids = repo_ids
        self.root = Path(root)
        # Construct the underlying datasets passing everything but `transform` and `delta_timestamps` which
        # are handled by this class.
        self._datasets = [
            DiskDataset(
                dataset_path=self.root / repo_id,
                modality_configs = modality_configs,
                embodiment_tag=embodiment_tag,
                transforms=transforms,
                video_backend=video_backend,
                augsteps=augsteps,
            )
            for repo_id,embodiment_tag in zip(repo_ids,embodiment_tags)
        ]

        # with multiple robots of different ranges. Instead we should have one normalization per robot.
        self.metadata = aggregate_stats([dataset.metadata for dataset in self._datasets])
        # Reallocate
        for dataset in self._datasets:
            tag = dataset.metadata.embodiment_tag
            dataset.set_transforms_metadata(
                DatasetMetadata(
                    statistics=self.metadata[tag.value],
                    modalities=dataset.metadata.modalities,
                    embodiment_tag=tag,
                    )
                )
    @property
    def repo_id_to_index(self):
        """Return a mapping from dataset repo_id to a dataset index automatically created by this class.

        This index is incorporated as a data key in the dictionary returned by `__getitem__`.
        """
        return {repo_id: i for i, repo_id in enumerate(self.repo_ids)}

    @property
    def repo_index_to_id(self):
        """Return the inverse mapping if repo_id_to_index."""
        return {v: k for k, v in self.repo_id_to_index}


    @property
    def num_frames(self) -> int:
        """Number of samples/frames."""
        return sum(d.num_frames for d in self._datasets)

    @property
    def num_episodes(self) -> int:
        """Number of episodes."""
        return sum(d.num_episodes for d in self._datasets)


    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds.")
        # Determine which dataset to get an item from based on the index.
        start_idx = 0
        dataset_idx = 0
        for dataset in self._datasets:
            if idx >= start_idx + dataset.num_frames:
                start_idx += dataset.num_frames
                dataset_idx += 1
                continue
            break
        else:
            raise AssertionError("We expect the loop to break out as long as the index is within bounds.")
        item = self._datasets[dataset_idx][idx - start_idx]
        item["dataset_index"] = torch.tensor(dataset_idx)

        return item
