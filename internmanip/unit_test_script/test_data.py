import os
import matplotlib.pyplot as plt

from internmanip.dataset.base import LeRobotSingleDataset, ModalityConfig
from internmanip.dataset.schema import EmbodimentTag
from internmanip.dataset.transform.base import ComposedModalityTransform
from internmanip.dataset.transform.video import VideoToTensor, VideoCrop, VideoResize, VideoColorJitter, VideoToNumpy
from internmanip.dataset.transform.state_action import StateActionToTensor, StateActionTransform
from internmanip.dataset.transform.concat import ConcatTransform

# 数据路径
REPO_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(REPO_PATH, "demo_data/robot_sim.PickNPlace")
print("Loading dataset... from", DATA_PATH)

# modality 配置
modality_configs = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["video.ego_view"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "state.left_arm",
            "state.left_hand",
            "state.left_leg",
            "state.neck",
            "state.right_arm",
            "state.right_hand",
            "state.right_leg",
            "state.waist",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "action.left_hand",
            "action.right_hand",
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.action.task_description", "annotation.human.validity"],
    ),
}

embodiment_tag = EmbodimentTag.GR1

dataset = LeRobotSingleDataset(DATA_PATH, modality_configs, embodiment_tag=embodiment_tag)

print("\n"*2)
print("="*100)
print(f"{' Humanoid Dataset ':=^100}")
print("="*100)

# 打印第7个样本
resp = dataset[7]
print("第7个样本类型:", type(resp))
print("包含字段:", resp.keys())
for k, v in resp.items():
    if hasattr(v, 'shape'):
        print(f"{k}: shape={v.shape}")
    else:
        print(f"{k}: type={type(v)}")

# 可视化部分图片
images_list = []
for i in range(100):
    if i % 10 == 0:
        resp = dataset[i]
        img = resp["video.ego_view"][0]
        images_list.append(img)

fig, axs = plt.subplots(2, 5, figsize=(20, 10))
for i, ax in enumerate(axs.flat):
    ax.imshow(images_list[i])
    ax.axis("off")
    ax.set_title(f"Image {i}")
plt.tight_layout()
plt.show()

# transform 配置
video_modality = modality_configs["video"]
state_modality = modality_configs["state"]
action_modality = modality_configs["action"]

to_apply_transforms = ComposedModalityTransform(
    transforms=[
        VideoToTensor(apply_to=video_modality.modality_keys),
        VideoCrop(apply_to=video_modality.modality_keys, scale=0.95),
        VideoResize(apply_to=video_modality.modality_keys, height=224, width=224, interpolation="linear"),
        VideoColorJitter(apply_to=video_modality.modality_keys, brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08),
        VideoToNumpy(apply_to=video_modality.modality_keys),
        StateActionToTensor(apply_to=state_modality.modality_keys),
        StateActionTransform(apply_to=state_modality.modality_keys, normalization_modes={key: "min_max" for key in state_modality.modality_keys}),
        StateActionToTensor(apply_to=action_modality.modality_keys),
        StateActionTransform(apply_to=action_modality.modality_keys, normalization_modes={key: "min_max" for key in action_modality.modality_keys}),
        ConcatTransform(
            video_concat_order=video_modality.modality_keys,
            state_concat_order=state_modality.modality_keys,
            action_concat_order=action_modality.modality_keys,
        ),
    ]
)

dataset = LeRobotSingleDataset(
    DATA_PATH,
    modality_configs,
    transforms=to_apply_transforms,
    embodiment_tag=embodiment_tag
)

# 再次打印第7个样本
resp = dataset[7]
print("应用transform后的第7个样本:")
print("包含字段:", resp.keys())
for k, v in resp.items():
    if hasattr(v, 'shape'):
        print(f"{k}: shape={v.shape}")
    else:
        print(f"{k}: type={type(v)}") 