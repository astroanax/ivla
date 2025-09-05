#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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

"""
π0: A Vision-Language-Action Flow Model for General Robot Control

[Paper](https://www.physicalintelligence.company/download/pi0.pdf)
[Jax code](https://github.com/Physical-Intelligence/openpi)

Designed by Physical Intelligence. Ported from Jax by Hugging Face.

Install pi0 extra dependencies:
```bash
pip install -e ".[pi0]"
```

Example of finetuning the pi0 pretrained model (`pi0_base` in `openpi`):
```bash
python lerobot/scripts/train.py \
--policy.path=lerobot/pi0 \
--dataset.repo_id=danaaubakirova/koch_test
```

Example of finetuning the pi0 neural network with PaliGemma and expert Gemma
pretrained with VLM default parameters before pi0 finetuning:
```bash
python lerobot/scripts/train.py \
--policy.type=pi0 \
--dataset.repo_id=danaaubakirova/koch_test
```

Example of using the pi0 pretrained model outside LeRobot training framework:
```python
policy = Pi0Policy.from_pretrained("lerobot/pi0")
```

"""

import math
from typing import Callable, Iterator, Any, Dict, List, Optional
from collections import deque
import os
from pathlib import Path

from transformers.image_processing_base import BatchFeature
from transformers import AutoConfig, AutoModel

from internmanip.model.utils import get_safe_dtype
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from transformers import AutoTokenizer
from transformers.data.data_collator import DataCollatorMixin

from internmanip.model.basemodel.pi0.paligemma_with_expert import PaliGemmaWithExpertConfig, PaliGemmaWithExpertModel
from internmanip.configs.model.pi0_cfg import PI0Config
from internmanip.model.basemodel.base import BasePolicyModel
from internmanip.model.basemodel.transforms.pi0 import collator_pi0
from internmanip.model.data_collator_registry import DataCollatorRegistry


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device='cpu'
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f'dimension ({dimension}) must be divisible by 2')

    if time.ndim != 1:
        raise ValueError('The time tensor is expected to be of shape `(batch_size, )`.')

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    # Only detach if we're in inference mode (no_grad context)
    if not torch.is_grad_enabled():
        pos_emb = pos_emb.detach()
    return pos_emb


def sample_beta(alpha, beta, bsize, device):
    gamma1 = torch.empty((bsize,), device=device).uniform_(0, 1).pow(1 / alpha)
    gamma2 = torch.empty((bsize,), device=device).uniform_(0, 1).pow(1 / beta)
    result = gamma1 / (gamma1 + gamma2)
    # Only detach if we're in inference mode (no_grad context)
    if not torch.is_grad_enabled():
        result = result.detach()
    return result


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks
    return att_2d_masks


def resize_with_pad(img, width, height, pad_value=-1):
    # assume no-op when width height fits already
    if img.ndim != 4:
        raise ValueError(f'(b,c,h,w) expected, but {img.shape}')

    cur_height, cur_width = img.shape[2:]

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_img = F.interpolate(
        img, size=(resized_height, resized_width), mode='bilinear', align_corners=False
    )

    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))

    # pad on left and top of image
    padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
    return padded_img


def pad_vector(vector, new_dim):
    """Can be (batch_size x sequence_length x features_dimension)
    or (batch_size x features_dimension)
    """
    if vector.shape[-1] == new_dim:
        return vector
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
    new_vector[..., :current_dim] = vector
    return new_vector



DataCollatorRegistry.register_fn(PI0Config.model_type, collator_pi0)


class PI0Policy(BasePolicyModel):
    """Wrapper class around PI0FlowMatching model to train and run inference within LeRobot."""

    config_class = PI0Config
    name = 'pi0'

    def __init__(
        self,
        config: PI0Config,
        local_model_path: str = None,
        # dataset_stats: dict[str, dict[str, Tensor]] | None = None,
        **kwargs
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """
        nn.Module.__init__(self)

        super().__init__(config)

        self.local_model_path = local_model_path
        config.validate_features()
        self.config = config

        self.language_tokenizer = AutoTokenizer.from_pretrained(self.config.pretrained_model_path)
        self.model = PI0FlowMatching(config)

        self.reset()
        self._keys_to_ignore_on_save = None
        if self.config.load_vlm_pretrained_weights:
            self.model.paligemma_with_expert.load_pretrained_weights()

    def reset(self):
        """This should be called whenever the environment is reset."""
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

    def get_optim_params(self) -> dict:
        return self.parameters()


    def inference(self, batch, **kwargs):
        """
        batch['video'].shape B 1 * N * 3 * H * W  # same as forward
        batch['state'].shape B * 1 * 7
        batch['annotation.human.action.task_description'] [[pick up]]
        """
        # inputs shape:
        # inputs['video.base_view'] = N * 3 * H * W   pixel range from [0.0, 1.0], maintain original image size
        # inputs['video.ego_view']: : N * 3 * H * W  # for google robot, windowx, this should be zero
        # inputs['state.joints']: N * 50 * 3
        # inputs['state.gripper']: N * 50 * 1
        # inputs['action.joints']: N * 50 * 3
        # inputs['action.gripper']: N * 50 * 1
        # inputs['action_pad']: N * 50* 1 bool
        # inputs['task'] : N * 1 str

        actions = self.select_action(batch, noise=None)
        return BatchFeature(data={'action_pred': actions})


    def calc_loss(self, *args, **kwargs):
        """
        Calculate the loss.
        """
        pass


    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """
        self.eval()
        batch['video'] = (batch['video'] - 0.5) * 2


        if batch['video'].shape[2] == 1:
            device = batch['video'].device
            bsize = batch['video'].shape[0]
            mask = torch.zeros(bsize, dtype=torch.bool, device=device)
            mask_true = torch.ones(bsize, dtype=torch.bool, device=device)
            img_masks = [mask_true, mask]
            images = [batch['video'][:,0,0], torch.zeros_like(batch['video'][:,0,0])]
        else:
            device = batch['video'].device
            bsize = batch['video'].shape[0]
            mask_true = torch.ones(bsize, dtype=torch.bool, device=device)
            img_masks = [mask_true, mask_true]
            images = [batch['video'][:,0,0], batch['video'][:,0,1]]

        state = self.prepare_state(batch)
        lang_tokens, lang_masks = self.prepare_language(batch)

        # state = state.to(torch.bfloat16)

        actions = self.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, noise=noise
        )

        return actions


    def forward(self, batch: dict[str, Tensor], noise=None, time=None) -> tuple[Tensor, dict[str, Tensor]]:
        """Do a full training forward pass to compute the loss"""
        # pi0 specific transform
        batch['video'] = (batch['video'] - 0.5) * 2

        if batch['video'].shape[2] == 1:
            device = batch['video'].device
            bsize = batch['video'].shape[0]
            mask = torch.zeros(bsize, dtype=torch.bool, device=device)
            mask_true = torch.ones(bsize, dtype=torch.bool, device=device)
            img_masks = [mask_true, mask]
            images = [batch['video'][:,0,0], torch.zeros_like(batch['video'][:,0,0])]
        else:
            device = batch['video'].device
            bsize = batch['video'].shape[0]
            mask_true = torch.ones(bsize, dtype=torch.bool, device=device)
            img_masks = [mask_true, mask_true]
            images = [batch['video'][:,0,0], batch['video'][:,0,1]]

    #         state_keys = ["state.joints", "state.gripper"]
    # action_keys = ["action.joints", "action.gripper"]
        # batch['state'] = torch.cat([batch['state.joints'], batch['state.gripper']], dim=-1).to(torch.float32)
        # batch.pop('state.joints')
        # batch.pop('state.gripper')


        # batch['action'] = torch.cat([batch['action.joints'], batch['action.gripper']], dim=-1).to(torch.float32)
        # batch.pop('action.joints')
        # batch.pop('action.gripper')

        state = self.prepare_state(batch)
        lang_tokens, lang_masks = self.prepare_language(batch)
        actions = self.prepare_action(batch)

        loss_dict = {}
        state = state.to(torch.bfloat16)
        actions = actions.to(torch.bfloat16)
        batch['action_pad'] = torch.from_numpy(batch['action_pad']).to(state.device)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=True):
            losses = self.model.forward(images, img_masks, lang_tokens, lang_masks, state, actions, noise, time)
        loss_dict['losses_after_forward'] = losses.clone()



        # Remove padding
        losses = losses[:, :, : 7]
        loss_dict['losses_after_rm_padding'] = losses.clone()

        loss = losses.mean()

        loss_dict['l2_loss'] = loss.item()
        # loss_dict['loss'] = (losses[:, :, :batch['action'].shape[-1]] * ~batch['action_pad'][...,None]).mean()
        loss_dict['loss'] = losses[:, :, :batch['action'].shape[-1]].mean()

        return loss_dict


    def prepare_language(self, batch) -> tuple[Tensor, Tensor]:
        """Tokenize the text input"""
        device = batch['state'].device
        tasks = batch['annotation.human.action.task_description']

        # PaliGemma prompt has to end with a new line
        tasks = [task if type(task) == list else [task] for task in tasks] #convert to a list
        tasks = [task[0] if task[0].endswith('\n') else f'{task[0]}\n' for task in tasks]

        tokenized_prompt = self.language_tokenizer.__call__(
            tasks,
            padding='max_length',
            padding_side='right',
            max_length=self.config.tokenizer_max_length,
            return_tensors='pt',
        )
        lang_tokens = tokenized_prompt['input_ids'].to(device=device)
        lang_masks = tokenized_prompt['attention_mask'].to(device=device, dtype=torch.bool)

        return lang_tokens, lang_masks



    def prepare_state(self, batch):
        """Pad state"""
        state = pad_vector(batch['state'], self.config.max_state_dim)
        return state

    def prepare_action(self, batch):
        """Pad action"""
        actions = pad_vector(batch['action'], self.config.max_action_dim)
        return actions


    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str,
                          **kwargs):
        """
        The policy is set in evaluation mode by default using `policy.eval()` (dropout modules are
        deactivated). To train it, you should first set it back in training mode with `policy.train()`.
        """
        tune_visual = kwargs.pop('tune_visual', True)
        tune_llm = kwargs.pop('tune_llm', False)
        tune_projector = kwargs.pop('tune_projector', True)
        tune_diffusion_model = kwargs.pop('tune_diffusion_model', True)
        tokenizer_max_length = kwargs.pop('tokenizer_max_length', 64)
        policy = super().from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )
        

        policy.config.chunk_size = 4  # initial with default chunk size
        policy.config.n_action_steps = 4 # initial with default chunk size
        # policy.config.max_state_dim = 7
        # policy.config.max_action_dim = 7
        
        # policy = PI0Policy(policy.config)  this strategy is able to train lerobot pi0 from scratch

        if tune_visual:
            for p in policy.model.paligemma_with_expert.paligemma.vision_tower.parameters():
                p.requires_grad=True
        if tune_projector:
            for p in policy.model.paligemma_with_expert.paligemma.multi_modal_projector.parameters():
                p.requires_grad=True
        policy.config.tokenizer_max_length = tokenizer_max_length
        policy.model.paligemma_with_expert.paligemma.language_model.lm_head.weight.requires_grad = False
        policy.model.paligemma_with_expert.gemma_expert.lm_head.weight.requires_grad = False
        policy.model.paligemma_with_expert.paligemma.language_model.model.embed_tokens.weight.requires_grad = False

        policy.model.paligemma_with_expert.paligemma.language_model.model.norm.weight.requires_grad = False
        policy.model.paligemma_with_expert.paligemma.language_model.model.layers[17].post_attention_layernorm.weight.requires_grad = False
        policy.model.paligemma_with_expert.paligemma.language_model.model.layers[17].mlp.down_proj.weight.requires_grad = False
        policy.model.paligemma_with_expert.paligemma.language_model.model.layers[17].mlp.up_proj.weight.requires_grad = False
        policy.model.paligemma_with_expert.paligemma.language_model.model.layers[17].mlp.gate_proj.weight.requires_grad = False
        policy.model.paligemma_with_expert.paligemma.language_model.model.layers[17].self_attn.o_proj.weight.requires_grad = False
        # policy.model.paligemma_with_expert.paligemma.language_model.model.embed_tokens.weight.requires_grad = False
        # policy.to(config.device)
        # policy.eval()

        # policy.model.paligemma_with_expert.load_pretrained_weights()
        return policy




# register
AutoConfig.register('pi0', PI0Config)
AutoModel.register(PI0Config, PI0Policy)
class PI0FlowMatching(nn.Module):
    """
    π0: A Vision-Language-Action Flow Model for General Robot Control

    [Paper](https://www.physicalintelligence.company/download/pi0.pdf)
    [Jax code](https://github.com/Physical-Intelligence/openpi)

    Designed by Physical Intelligence. Ported from Jax by Hugging Face.
    ┌──────────────────────────────┐
    │               actions        │
    │               ▲              │
    │              ┌┴─────┐        │
    │  kv cache    │Gemma │        │
    │  ┌──────────►│Expert│        │
    │  │           │      │        │
    │ ┌┴────────┐  │x 10  │        │
    │ │         │  └▲──▲──┘        │
    │ │PaliGemma│   │  │           │
    │ │         │   │  robot state │
    │ │         │   noise          │
    │ └▲──▲─────┘                  │
    │  │  │                        │
    │  │  image(s)                 │
    │  language tokens             │
    └──────────────────────────────┘
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        paligemma_with_export_config = PaliGemmaWithExpertConfig(
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            train_expert_only=self.config.train_expert_only,
            pretrained_model_path=self.config.pretrained_model_path,
            attention_implementation=self.config.attention_implementation,
        )
        self.paligemma_with_expert = PaliGemmaWithExpertModel(paligemma_with_export_config)

        # Projections are float32
        self.state_proj = nn.Linear(self.config.max_state_dim, self.config.proj_width)

        # self.config.max_state_dim = 7
        # self.config.max_action_dim = 7
        self.action_in_proj = nn.Linear(self.config.max_action_dim, self.config.proj_width)
        self.action_out_proj = nn.Linear(self.config.proj_width, self.config.max_action_dim)


        self.action_time_mlp_in = nn.Linear(self.config.proj_width * 2, self.config.proj_width)
        self.action_time_mlp_out = nn.Linear(self.config.proj_width, self.config.proj_width)

        self.set_requires_grad()
        self.flow_sampling = 'beta'
        self.flow_sig_min = 0.001
        if self.flow_sampling == "beta":
            flow_alpha = 1.5
            flow_beta = 1
            self.flow_t_max = 1 - 0.001
            self.flow_beta_dist = torch.distributions.Beta(flow_alpha, flow_beta)        

        
        

    def set_requires_grad(self):
        for params in self.state_proj.parameters():
            params.requires_grad = self.config.train_state_proj

    def sample_noise(self, shape, device):
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
            requires_grad=False
        )
        # Only detach if we're in inference mode (no_grad context)
        if not torch.is_grad_enabled():
            noise = noise.detach()
        return noise

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        time = time.to(dtype=torch.float32, device=device)
        # Only detach if we're in inference mode (no_grad context)
        if not torch.is_grad_enabled():
            time = time.detach()
        return time

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        """
        # TODO: avoid list in python and torch.cat ; prefer pre-allocation with torch.empty
        embs = []
        pad_masks = []
        att_masks = []

        # Get batch size from first image for later use
        bsize = images[0].shape[0] if images else lang_tokens.shape[0]

        # TODO: remove for loop
        for (
            img,
            img_mask,
        ) in zip(images, img_masks, strict=False):
            img_emb = self.paligemma_with_expert.embed_image(img)
            img_emb = img_emb.to(dtype=torch.bfloat16)

            # Normalize image embeddings
            img_emb_dim = img_emb.shape[-1]
            scaling_factor = torch.tensor(img_emb_dim**0.5, dtype=img_emb.dtype, device=img_emb.device)
            # Only detach if we're in inference mode (no_grad context)
            if not torch.is_grad_enabled():
                scaling_factor = scaling_factor.detach()
            img_emb = img_emb * scaling_factor

            _, num_img_embs = img_emb.shape[:2]
            img_mask = img_mask[:, None].expand(bsize, num_img_embs)

            embs.append(img_emb)
            pad_masks.append(img_mask)

            # Create attention masks so that image tokens attend to each other
            att_masks += [0] * num_img_embs

        lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)

        # Normalize language embeddings
        lang_emb_dim = lang_emb.shape[-1]
        scaling_factor = math.sqrt(lang_emb_dim)
        # Only detach if we're in inference mode (no_grad context)
        if not torch.is_grad_enabled():
            scaling_factor = torch.tensor(scaling_factor, dtype=lang_emb.dtype, device=lang_emb.device).detach()
        else:
            scaling_factor = torch.tensor(scaling_factor, dtype=lang_emb.dtype, device=lang_emb.device)
        lang_emb = lang_emb * scaling_factor

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        # Embed state
        state_emb = self.state_proj(state)
        state_emb = state_emb.to(dtype=torch.bfloat16)
        embs.append(state_emb[:, None, :])
        bsize = state_emb.shape[0]
        dtype = state_emb.dtype
        device = state_emb.device

        state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)

        # Set attention masks so that image and language inputs do not attend to state or actions
        att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.config.proj_width, min_period=4e-3, max_period=4.0, device=device
        )
        time_emb = time_emb.type(dtype=dtype)
        # Only detach if we're in inference mode (no_grad context)
        if not torch.is_grad_enabled():
            time_emb = time_emb.detach()

        # Fuse timestep + action information using an MLP
        action_emb = self.action_in_proj(noisy_actions)

        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)

        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)  # swish == silu
        action_time_emb = self.action_time_mlp_out(action_time_emb)

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.n_action_steps - 1))



        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def sample_fm_time(self, bsz: int) -> torch.FloatTensor:
        if self.flow_sampling == "uniform":  # uniform between 0 and 1
            """https://github.com/gle-bellier/flow-matching/blob/main/Flow_Matching.ipynb"""
            eps = 1e-5
            t = (torch.rand(1) + torch.arange(bsz) / bsz) % (1 - eps)
        elif self.flow_sampling == "beta":  # from pi0 paper
            z = self.flow_beta_dist.sample((bsz,))
            t = self.flow_t_max * (1 - z)  # flip and shift
        return t

    def psi_t(
        self,
        x: torch.FloatTensor,
        x1: torch.FloatTensor,
        t: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Conditional Flow"""
        t = t[:, None, None]  # (B, 1, 1)
        return (1 - (1 - self.flow_sig_min) * t) * x + t * x1
    

    def forward(
        self, images, img_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None
    ) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        # if time is None:
            # time = self.sample_time(actions.shape[0], actions.device)
        if time is None:
            t = self.sample_fm_time(len(lang_tokens)).to(images[0].dtype).to(state.device)
            time = t
        x0 = torch.randn_like(actions, device=t.device, dtype=t.dtype)
        x1 = actions
        # psi_t = self.psi_t(x0, x1, t)

        noise[...,7:] = 0.
        # PiZero-style flow matching: psi_t and target d_psi
        time_expanded = time[:, None, None]
        # psi_t = (1 - (1 - sigma_min) * t) * x0 + t * x1
        sigma_min = self.flow_sig_min
        psi_t = (1 - (1 - sigma_min) * time_expanded) * noise + time_expanded * actions
        # target velocity d_psi = x1 - (1 - sigma_min) * x0
        d_psi = actions - (1 - sigma_min) * noise

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)

        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(state[:, 0], psi_t, time)
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        # position_ids = torch.cumsum(pad_masks, dim=1) # paligama start from 1000

        (_, suffix_out), _ = self.paligemma_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )


        suffix_out = suffix_out[:, -self.config.n_action_steps :]
        # Original openpi code, upcast attention output
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)

        losses = F.mse_loss(v_t, d_psi, reduction='none')
        return losses

    def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state, noise=None) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = state.shape[0]
        device = state.device

        if noise is None:
            actions_shape = (bsize, self.config.n_action_steps, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1


        # Compute image and language key value cache
        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )

        dt = 1.0 / self.config.num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        noise[...,7:] = 0.
        x_t = noise
        time = torch.tensor(0.0, dtype=torch.float32, device=device)
        
        # Store original dtypes to restore later
        original_state_proj_dtype = self.state_proj.weight.dtype
        original_action_in_proj_dtype = self.action_in_proj.weight.dtype
        original_action_time_mlp_in_dtype = self.action_time_mlp_in.weight.dtype
        original_action_time_mlp_out_dtype = self.action_time_mlp_out.weight.dtype
        
        # Temporarily convert to float32 for inference
        self.state_proj = self.state_proj.to(torch.float32)
        self.action_in_proj = self.action_in_proj.to(torch.float32)
        self.action_time_mlp_in = self.action_time_mlp_in.to(torch.float32)
        self.action_time_mlp_out = self.action_time_mlp_out.to(torch.float32)
        
        with torch.autocast('cuda', dtype=torch.float32, enabled=True):
            # PiZero-style forward Euler integration from t=0 to 1
            for _ in range(self.config.num_steps):
                expanded_time = time.expand(bsize)
                x_t[...,7:] = 0.
                v_t = self.denoise_step(
                    state,
                    prefix_pad_masks,
                    past_key_values,
                    x_t,
                    expanded_time,
                )
                x_t += dt * v_t
                time += dt
        
        # Restore original dtypes
        self.state_proj = self.state_proj.to(original_state_proj_dtype)
        self.action_in_proj = self.action_in_proj.to(original_action_in_proj_dtype)
        self.action_time_mlp_in = self.action_time_mlp_in.to(original_action_time_mlp_in_dtype)
        self.action_time_mlp_out = self.action_time_mlp_out.to(original_action_time_mlp_out_dtype)
        
        return x_t

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        state = state[:, 0]
        state = state.to(torch.bfloat16)
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1


        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
        )
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.n_action_steps :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        return v_t
