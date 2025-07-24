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
import torch
import torchvision
import av
import cv2
import decord
import numpy as np
import os
# from torchcodec.decoders import VideoDecoder

def get_frames_by_indices(
    video_path: str,
    indices: list[int] | np.ndarray,
    video_backend: str = "decord",
    video_backend_kwargs: dict = {},
) -> np.ndarray:
    if video_backend == "decord":
        vr = decord.VideoReader(video_path, **video_backend_kwargs)
        frames = vr.get_batch(indices)
        return frames.asnumpy()
    elif video_backend == "opencv":
        frames = []
        cap = cv2.VideoCapture(video_path, **video_backend_kwargs)
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                raise ValueError(f"Unable to read frame at index {idx}")
            frames.append(frame)
        cap.release()
        frames = np.array(frames)
        return frames
    else:
        raise NotImplementedError


def get_frames_by_timestamps(
    video_path: str,
    timestamps: list[float] | np.ndarray,
    tolerance_s: float,
    video_backend: str = "decord",
    video_backend_kwargs: dict = {},
) -> np.ndarray:
    """Get frames from a video at specified timestamps.
    Args:
        video_path (str): Path to the video file.
        timestamps (list[int] | np.ndarray): Timestamps to retrieve frames for, in seconds.
        video_backend (str, optional): Video backend to use. Defaults to "decord".
    Returns:
        np.ndarray: Frames at the specified timestamps.
    """
    if video_backend == "decord":
        assert os.path.exists(video_path), f"{video_path} not found!"
        vr = decord.VideoReader(video_path, **video_backend_kwargs)
        num_frames = len(vr)
        # Retrieve the timestamps for each frame in the video
        frame_ts: np.ndarray = vr.get_frame_timestamp(range(num_frames))
        # Map each requested timestamp to the closest frame index
        # Only take the first element of the frame_ts array which corresponds to start_seconds
        indices = np.abs(frame_ts[:, :1] - timestamps).argmin(axis=0)
        frames = vr.get_batch(indices)
        return frames.asnumpy()
    elif video_backend == "opencv":
        # Open the video file
        cap = cv2.VideoCapture(video_path, **video_backend_kwargs)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")
        # Retrieve the total number of frames
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Calculate timestamps for each frame
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_ts = np.arange(num_frames) / fps
        frame_ts = frame_ts[:, np.newaxis]  # Reshape to (num_frames, 1) for broadcasting
        # Map each requested timestamp to the closest frame index
        indices = np.abs(frame_ts - timestamps).argmin(axis=0)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                raise ValueError(f"Unable to read frame at index {idx}")
            frames.append(frame)
        cap.release()
        frames = np.array(frames)
        return frames
    elif video_backend == "torchvision_av":
        # set backend
        torchvision.set_video_backend("pyav") # pyav doesnt support accuracte seek
        # set a video stream reader
        reader = torchvision.io.VideoReader(video_path, "video")
        # set the first and last requested timestamps
        # Note: previous timestamps are usually loaded, since we need to access the previous key frame
        first_ts = timestamps[0]
        last_ts = timestamps[-1]
        # access closest key frame of the first requested frame
        # Note: closest key frame timestamp is usally smaller than `first_ts` (e.g. key frame can be the first frame of the video)
        # for details on what `seek` is doing see: https://pyav.basswood-io.com/docs/stable/api/container.html?highlight=inputcontainer#av.container.InputContainer.seek
        reader.seek(first_ts, keyframes_only=True)
        # load all frames until last requested frame
        loaded_frames = []
        loaded_ts = []
        for frame in reader:
            current_ts = frame["pts"]
            loaded_frames.append(frame["data"])
            loaded_ts.append(current_ts)
            if current_ts >= last_ts:
                break

        reader.container.close()
        reader = None
        query_ts = torch.tensor(timestamps, dtype=torch.float32)
        loaded_ts = torch.tensor(loaded_ts, dtype=torch.float32)

        # compute distances between each query timestamp and timestamps of all loaded frames
        dist = torch.cdist(query_ts[:, None], loaded_ts[:, None], p=1)
        min_, argmin_ = dist.min(1)

        is_within_tol = min_ < tolerance_s
        assert is_within_tol.all(), (
            f"One or several query timestamps unexpectedly violate the tolerance ({min_[~is_within_tol]} > {tolerance_s=})."
            "It means that the closest frame that can be loaded from the video is too far away in time."
            "This might be due to synchronization issues with timestamps during data collection."
            "To be safe, we advise to ignore this item during training."
            f"\nqueried timestamps: {query_ts}"
            f"\nloaded timestamps: {loaded_ts}"
            f"\nvideo: {video_path}"
            f"\nbackend: {video_backend}"
        )

        # get closest frames to the query timestamps
        closest_frames = torch.stack([loaded_frames[idx] for idx in argmin_])
        closest_ts = loaded_ts[argmin_]
        frames = np.array(closest_frames)
        assert len(timestamps) == len(closest_frames)
        return frames.transpose(0, 2, 3, 1)
    elif video_backend == "torchcodec":
        # initialize video decoder
        decoder = VideoDecoder(video_path, device="cpu", seek_mode="approximate")
        loaded_frames = []
        loaded_ts = []
        # get metadata for frame information
        metadata = decoder.metadata
        average_fps = metadata.average_fps

        # convert timestamps to frame indices
        frame_indices = [round(ts * average_fps) for ts in timestamps]

        # retrieve frames based on indices
        frames_batch = decoder.get_frames_at(indices=frame_indices)

        for frame, pts in zip(frames_batch.data, frames_batch.pts_seconds, strict=False):
            loaded_frames.append(frame)
            loaded_ts.append(pts.item())

        query_ts = np.array(timestamps)
        loaded_ts = np.array(loaded_ts)

        # compute distances between each query timestamp and loaded timestamps
        dist_np = np.abs(query_ts[:, None] - loaded_ts[None, :])
        min_ = np.min(dist_np, axis=1)
        argmin_ = np.argmin(dist_np, axis=1)

        # get closest frames to the query timestamps
        closest_frames = np.stack([loaded_frames[idx] for idx in argmin_])
        closest_ts = loaded_ts[argmin_]

        closest_frames = closest_frames.astype(np.uint8)

        return closest_frames.transpose(0, 2, 3, 1)
    else:
        raise NotImplementedError


def get_all_frames(
    video_path: str,
    video_backend: str = "decord",
    video_backend_kwargs: dict = {},
    resize_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """Get all frames from a video.
    Args:
        video_path (str): Path to the video file.
        video_backend (str, optional): Video backend to use. Defaults to "decord".
        video_backend_kwargs (dict, optional): Keyword arguments for the video backend.
        resize_size (tuple[int, int], optional): Resize size for the frames. Defaults to None.
    """
    if video_backend == "decord":
        vr = decord.VideoReader(video_path, **video_backend_kwargs)
        frames = vr.get_batch(range(len(vr))).asnumpy()
    elif video_backend == "pyav":
        container = av.open(video_path)
        frames = []
        for frame in container.decode(video=0):
            frame = frame.to_ndarray(format="rgb24")
            frames.append(frame)
        frames = np.array(frames)
    elif video_backend == "torchvision_av":
        # set backend and reader
        torchvision.set_video_backend("pyav")
        reader = torchvision.io.VideoReader(video_path, "video")
        frames = []
        for frame in reader:
            frames.append(frame["data"])
        frames = np.array(frames)
        frames = frames.transpose(0, 2, 3, 1)
    else:
        raise NotImplementedError(f"Video backend {video_backend} not implemented")
    # resize frames if specified
    if resize_size is not None:
        frames = [cv2.resize(frame, resize_size) for frame in frames]
        frames = np.array(frames)
    return frames
