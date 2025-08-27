"""
Wrapper around the joint model (mixtures). Siglip from PaliGemma, action-time encoder, proprio encoder, action decoder. Flow matching training

Generates causal masking for the mixtures

Potentially customized to add/remove mixtures, e.g., remove proprio or add another vision module

This code is from openpizero.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple
import einops
import os

from transformers.image_processing_base import BatchFeature
from internmanip.configs.model.openpi0_cfg import OpenPI0Config
from internmanip.model.basemodel.base import BasePolicyModel
from internmanip.model.basemodel.openpi0.joint_model import JointModel
from internmanip.model.basemodel.openpi0.kv_cache import KVCache
from internmanip.model.basemodel.openpi0.modules import ActionEncoder, SinusoidalPosEmb
from internmanip.model.basemodel.openpi0.paligemma.siglip import PaliGemmaMultiModalProjector, SiglipVisionModel
from internmanip.model.basemodel.openpi0.processing import VLAProcessor
from internmanip.model.data_collator_registry import DataCollatorRegistry
from internmanip.model.basemodel.transforms.pi0 import collator_pi0
from transformers import AutoConfig, AutoModel
from transformers import AutoTokenizer
import torch
from torch import nn
import numpy as np
from huggingface_hub import snapshot_download
# from src.model.kv_cache import KVCache
# from src.model.vla.modules import (
#     ActionEncoder,
#     SinusoidalPosEmb,
# )
log = logging.getLogger(__name__)

DataCollatorRegistry.register_fn(OpenPI0Config.model_type, collator_pi0)


class NoSyncBase:
    def no_sync(self):
        if self.use_ddp:
            # If DDP is used, call the actual `no_sync` method
            return torch.nn.parallel.DistributedDataParallel.no_sync(self)
        else:
            # Otherwise, return the dummy context manager
            class DummyContext:
                def __enter__(self):
                    pass

                def __exit__(self, exc_type, exc_value, traceback):
                    pass

            return DummyContext()



class PiZero(BasePolicyModel, NoSyncBase):
    name = 'openpi0'
    config_class = OpenPI0Config

    def __init__(self, config: OpenPI0Config, 
        local_model_path: str = None,
        # dataset_stats: dict[str, dict[str, Tensor]] | None = None,
        **kwargs):
        use_ddp = False
        super().__init__()
        self.config = config
        self.use_ddp = use_ddp  # used in NoSyncBase
        self.vocab_size = config.vocab_size
        config.pad_token_id = 0
        self.pad_token_id = config.pad_token_id
        self.image_token_index = config.image_token_index

        self.use_lm_head = False

        self.max_image_text_tokens = config.max_image_text_tokens
        self.num_proprio_tokens = config.cond_steps
        self.num_action_tokens = config.horizon_steps
        self.total_num_tokens = (
            self.max_image_text_tokens
            + self.num_proprio_tokens
            + self.num_action_tokens
        )

        self.image_text_hidden_size = config.get_mixture_config().vlm.hidden_size
        self.proprio_hidden_size = config.get_mixture_config().proprio.hidden_size
        self.action_hidden_size = config.get_mixture_config().action.hidden_size

        # Action parameterization
        self.num_inference_steps = config.num_inference_steps
        self.horizon_steps = config.horizon_steps
        self.action_dim = config.action_dim
        self.proprio_dim = config.proprio_dim
        self.final_action_clip_value = config.final_action_clip_value
        self.flow_sig_min = 0.001

        print(config)
        # import ipdb;ipdb.set_trace()
        # text input only
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            self.image_text_hidden_size,
            self.pad_token_id,
        )  # 0.527B parameters

        # Vision
        
        # from src.model.paligemma.siglip import SiglipVisionModel
        self.vision_tower = SiglipVisionModel(config.get_vision_config())
        

        self.multi_modal_projector = PaliGemmaMultiModalProjector(config.get_vision_projector_config())

        # Mixtures
        self.joint_model = JointModel(config.get_joint_config())

        # Action, proprio, time encoders
        self.action_expert_adaptive_mode = config.action_expert_adaptive_mode
        if config.action_expert_adaptive_mode:  # adaLN or adaLN-Zero
            self.action_encoder = ActionEncoder(
                self.action_dim,
                self.action_hidden_size,
                time_cond=False,
            )
            self.time_embedding = SinusoidalPosEmb(
                config.time_hidden_size, config.time_max_period
            )
        else:  # matching pi0
            self.action_encoder = ActionEncoder(
                self.action_dim,
                self.action_hidden_size,
                time_cond=True,
            )
            self.time_embedding = SinusoidalPosEmb(
                self.action_hidden_size, config.time_max_period
            )
        self.proprio_encoder = nn.Linear(
            self.proprio_dim,
            self.proprio_hidden_size,
        )

        # Action decoder
        self.action_decoder = nn.Linear(
            self.action_hidden_size,
            self.action_dim,
        )

        # optional text output
        if self.use_lm_head:
            self.lm_head = nn.Linear(
                self.image_text_hidden_size,
                self.vocab_size,
                bias=False,
            )
            self.lm_head.weight = self.embed_tokens.weight  # tie weights
            
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.pretrained_model_path, padding_side="right"
        )
        self.processor = VLAProcessor(
            self.tokenizer,
            num_image_tokens=config.get_vision_config().num_image_tokens,
            max_seq_len=config.max_seq_len,
            tokenizer_padding=config.tokenizer_padding,
        )
        self.flow_sampling = config.flow_sampling
        if config.flow_sampling == "beta":
            flow_alpha = 1.5
            flow_beta = 1
            self.flow_t_max = 1 - 0.001
            self.flow_beta_dist = torch.distributions.Beta(flow_alpha, flow_beta)


        # self.joint_model.mixtures.proprio.norm.weight.requires_grad = False
        self.joint_model.mixtures.proprio.layers[17].post_attention_layernorm.weight.requires_grad = False
        self.joint_model.mixtures.proprio.layers[17].mlp.down_proj.weight.requires_grad = False
        self.joint_model.mixtures.proprio.layers[17].mlp.up_proj.weight.requires_grad = False
        self.joint_model.mixtures.proprio.layers[17].mlp.gate_proj.weight.requires_grad = False
        self.joint_model.mixtures.proprio.layers[17].self_attn.o_proj.weight.requires_grad = False
        self.joint_model.mixtures.vlm.layers[17].post_attention_layernorm.weight.requires_grad = False
        self.joint_model.mixtures.vlm.layers[17].mlp.down_proj.weight.requires_grad = False
        self.joint_model.mixtures.vlm.layers[17].mlp.up_proj.weight.requires_grad = False
        self.joint_model.mixtures.vlm.layers[17].mlp.gate_proj.weight.requires_grad = False
        self.joint_model.mixtures.vlm.layers[17].self_attn.o_proj.weight.requires_grad = False
        self.embed_tokens.weight.requires_grad = False
        

        self.tie_action_proprio_weights()
        self.load_pretrained_weights()
        
        


    def preprocess_batch(self, batch, split_mask: bool, sample_fm_time: bool):
        # TODO(allenzren): support multi-image / proprio history
        # pls support it next

        if batch['video'].dtype == torch.uint8:
            images = (batch['video'][:, 0].permute(0,1, 3, 4, 2)).to(torch.uint8).cpu()
        else:
            images = (batch['video'][:, 0].permute(0,1, 3, 4, 2) * 255).to(torch.uint8).cpu()
        proprios = batch["state"]  # B 1 7
        actions = batch.pop("action", None)  # remove the time dimension
        texts = [
            text[0] if type(text) == list else text for text in batch["annotation.human.action.task_description"]
        ]
        images = einops.rearrange(images, "B T H W C -> B (T C) H W")  # remove cond_steps dimension
        model_inputs = self.processor(text=texts, images=images)

        # build causal mask and position ids for action
        causal_mask, vlm_position_ids, proprio_position_ids, action_position_ids = (self.build_causal_mask_and_position_ids(model_inputs["attention_mask"], self.dtype))

        inputs = {
            "input_ids": model_inputs["input_ids"],
            "pixel_values": model_inputs["pixel_values"].to(self.dtype),
            "vlm_position_ids": vlm_position_ids,
            "proprio_position_ids": proprio_position_ids,
            "action_position_ids": action_position_ids,
            "proprios": proprios.to(self.dtype),
            "actions": actions.to(self.dtype) if actions is not None else None,
        }
        if split_mask:
            image_text_proprio_mask, action_mask = (
                self.split_full_mask_into_submasks(causal_mask)
            )
            inputs["image_text_proprio_mask"] = image_text_proprio_mask
            inputs["action_mask"] = action_mask
        else:
            inputs["causal_mask"] = causal_mask

        # sample flow matching timesteps
        if sample_fm_time:
            inputs["t"] = self.sample_fm_time(len(texts)).to(self.dtype)

        inputs = {k: v.to(self.device) if v is not None else v for k, v in inputs.items()}
        return inputs

    def sample_fm_time(self, bsz: int) -> torch.FloatTensor:
        if self.flow_sampling == "uniform":  # uniform between 0 and 1
            """https://github.com/gle-bellier/flow-matching/blob/main/Flow_Matching.ipynb"""
            eps = 1e-5
            t = (torch.rand(1) + torch.arange(bsz) / bsz) % (1 - eps)
        elif self.flow_sampling == "beta":  # from pi0 paper
            z = self.flow_beta_dist.sample((bsz,))
            t = self.flow_t_max * (1 - z)  # flip and shift
        return t

    @property
    def action_expert_parameters(self):
        return (
            list(self.action_encoder.parameters())
            + list(self.action_decoder.parameters())
            + list(self.proprio_encoder.parameters())
            + list(self.joint_model.mixtures["action"].parameters())
        )  # note: action and proprio share weights

    @property
    def trainable_vlm_parameters(self):
        return (
            list(self.vision_tower.parameters())
            + list(self.multi_modal_projector.parameters())
            + self.trainable_gemma_parameters
        )

    @property
    def lora_trainable_vlm_parameters(self):
        params = []
        for name, param in self.vision_tower.named_parameters():
            if "lora_" in name:
                params.append(param)
        for name, param in self.multi_modal_projector.named_parameters():
            if "lora_" in name:
                params.append(param)
        params.extend(self.trainable_lora_gemma_parameters)
        return params

    @property
    def trainable_gemma_parameters(self):
        gemma_parameters = []
        for name, param in self.joint_model.mixtures["vlm"].named_parameters():
            if not self._check_gemma_unused_parameter_by_name(name):
                gemma_parameters.append(param)
        return gemma_parameters

    @property
    def trainable_lora_gemma_parameters(self):
        gemma_parameters = []
        for name, param in self.joint_model.mixtures["vlm"].named_parameters():
            if not self._check_gemma_unused_parameter_by_name(name):
                if "lora_" in name:
                    gemma_parameters.append(param)
        return gemma_parameters


    def load_pretrained_weights(self):
        """vision, projector, lm from paligemma"""
        import glob

        from safetensors import safe_open

        # google/paligemma-3b-pt-224
        # load tensors from files
        if not Path(self.config.pretrained_model_path).exists():
            snapshot_path = snapshot_download(
                repo_id=self.config.pretrained_model_path,
                # cache_dir=config.model_kwargs['HF_cache_dir'],
                local_files_only=True,
            )
            self.config.pretrained_model_path = snapshot_path
        
        safetensors_files = glob.glob(
            os.path.join(self.config.pretrained_model_path, "*.safetensors")
        )
        
        tensors = {}
        for safetensors_file in safetensors_files:
            with safe_open(safetensors_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    tensors[key] = f.get_tensor(key)

        # load embed tokens
        embed_tokens_state_dict = self.embed_tokens.state_dict()
        for k, v in tensors.items():
            if "embed_tokens" in k:
                new_key = k.replace("language_model.model.embed_tokens.", "")
                embed_tokens_state_dict[new_key] = v
        self.embed_tokens.load_state_dict(embed_tokens_state_dict, strict=True)
        print("Loaded pre-trained weights for embed tokens")

        # load vision tower --- "vision_tower.vision_model" -> "vision_model"
        vision_tower_state_dict = self.vision_tower.state_dict()
        for k, v in tensors.items():
            if "vision_tower" in k:
                new_key = k.replace("vision_tower.", "")
                vision_tower_state_dict[new_key] = v
        self.vision_tower.load_state_dict(vision_tower_state_dict, strict=True)
        print("Loaded pre-trained weights for vision tower")

        # load projector --- "multi_modal_projector.linear" -> "linear"
        multi_modal_projector_state_dict = self.multi_modal_projector.state_dict()
        for k, v in tensors.items():
            if "multi_modal_projector" in k:
                new_key = k.replace("multi_modal_projector.", "")
                multi_modal_projector_state_dict[new_key] = v
        self.multi_modal_projector.load_state_dict(
            multi_modal_projector_state_dict, strict=True
        )
        print("Loaded pre-trained weights for projector")

        # load lm --- do not change any lora weights
        joint_model_state_dict = self.joint_model.state_dict()
        lora_keys = []
        for key in (
            joint_model_state_dict.keys()
        ):  # avoid RuntimeError: OrderedDict mutated during iteration
            if "lora_" in key:
                lora_keys.append(key)
        for key in lora_keys:
            del joint_model_state_dict[key]
        for k, v in tensors.items():
            if "language_model.model" in k:
                new_key = k.replace("language_model.model.", "mixtures.vlm.")
                joint_model_state_dict[new_key] = v
        self.joint_model.load_state_dict(joint_model_state_dict, strict=False)
        print("Loaded pre-trained weights for lm part of the joint model")

    def _check_gemma_unused_parameter_by_name(self, name: str) -> bool:
        """no need to train vlm parameters after attention of last layer"""
        last_hidden_layer_index = self.joint_model.num_hidden_layers - 1
        if (
            f"{last_hidden_layer_index}.post" in name
            or f"{last_hidden_layer_index}.mlp" in name
            or f"{last_hidden_layer_index}.self_attn.o_proj" in name
            or f"{last_hidden_layer_index}.self_attn.v_proj" in name
        ):  # final norm is not initialized
            return True
        return False

    def freeze_non_lora_weights_in_vlm(self):
        """Keep all bias frozen"""
        for name, param in self.vision_tower.named_parameters():
            param.requires_grad = True if "lora_" in name else False
        print("Froze non-lora weights in vision tower")

        for name, param in self.multi_modal_projector.named_parameters():
            param.requires_grad = True if "lora_" in name else False
        print("Froze non-lora weights in projector")

        for name, param in self.joint_model.mixtures["vlm"].named_parameters():
            if not self._check_gemma_unused_parameter_by_name(name):
                param.requires_grad = True if "lora_" in name else False
        print("Froze non-lora weights in lm part of the joint model")

    def freeze_unused_weights(self):
        """text embedding and part of last layer of vlm, including lora"""
        self.embed_tokens.weight.requires_grad = False
        for name, param in self.joint_model.mixtures["vlm"].named_parameters():
            if self._check_gemma_unused_parameter_by_name(name):
                param.requires_grad = False

    def freeze_all_weights(self):
        for _, param in self.named_parameters():
            param.requires_grad = False

    def tie_action_proprio_weights(self):
        """technically more than just tying weights"""
        self.joint_model.mixtures["proprio"] = self.joint_model.mixtures["action"]

    def build_text_cache(self):
        return KVCache()

    # ---------- Input preparation ----------#

    def build_causal_mask_and_position_ids(
        self, attention_mask: torch.Tensor, dtype: torch.dtype
    ) -> Tuple[torch.FloatTensor]:
        """
        block attention --- padding for unused text tokens

                 img/text img/text img/text (padding) proprio action action
        img/text    x        x        x
        img/text    x        x        x
        img/text    x        x        x
        (padding)
        proprio     x        x        x                 x
        action      x        x        x                 x       x      x
        action      x        x        x                 x       x      x
        """
        bsz = attention_mask.size(0)
        proprio_start = self.max_image_text_tokens
        proprio_end = self.max_image_text_tokens + self.num_proprio_tokens
        action_start = proprio_end
        image_text_token_cnts = torch.sum(attention_mask, dim=1)
        causal_mask = torch.full(
            (bsz, self.total_num_tokens, self.total_num_tokens),
            torch.finfo(dtype).min,
            dtype=dtype,
        )  # smallest value, avoid using inf for softmax nan issues with padding
        for idx, cnt in enumerate(image_text_token_cnts):
            causal_mask[idx, :cnt, :cnt] = 0  # image/text attend to itself
            causal_mask[idx, proprio_start:, :cnt] = (
                0  # proprio/action attend to image/text
            )
        causal_mask[:, proprio_start:proprio_end, proprio_start:proprio_end] = (
            0  # proprio attend to itself
        )
        causal_mask[:, action_start:, proprio_start:] = (
            0  # action attend to itself and proprio
        )

        # add the head dimension
        # [Batch_Size, Q_Len, KV_Len] -> [Batch_Size, Num_Heads_Q, Q_Len, KV_Len]
        causal_mask = causal_mask.unsqueeze(1)

        # position ids for each blocks --- start at 1
        vlm_position_ids = torch.arange(1, self.max_image_text_tokens + 1).repeat(
            bsz, 1
        )
        proprio_position_ids = torch.arange(1, self.num_proprio_tokens + 1).repeat(
            bsz, 1
        )
        action_position_ids = torch.arange(
            self.num_proprio_tokens + 1,
            self.num_proprio_tokens + self.num_action_tokens + 1,
        ).repeat(bsz, 1)
        # since proprio and action share the same mixture weights, makes sense to use [1 (proprio), 2 (action), 3 (action), ...] instead of [1 (proprio), 1 (action), 2 (action), ...]
        return causal_mask, vlm_position_ids, proprio_position_ids, action_position_ids

    def split_full_mask_into_submasks(
        self, causal_mask: torch.FloatTensor
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """split into ones for paligemma and action"""
        image_text_proprio_mask = causal_mask[
            ...,
            : self.max_image_text_tokens + self.num_proprio_tokens,
            : self.max_image_text_tokens + self.num_proprio_tokens,
        ]
        action_mask = causal_mask[..., -self.num_action_tokens :, :]
        return image_text_proprio_mask, action_mask

    def build_causal_mask_and_position_ids_for_text(
        self,
        q_len: int,
        attention_mask: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        dtype, device = attention_mask.dtype, attention_mask.device

        if kv_cache is None or kv_cache.num_items() == 0:
            # do not mask any token, because we're in the prefill phase
            # assume no padding
            causal_mask = torch.full((bsz, q_len, q_len), 0, dtype=dtype, device=device)
        else:
            assert q_len == 1, "Using KV cache so should only use one single token"
            kv_len = kv_cache.num_items() + q_len
            # also in this case we don't need to mask anything, since each query should be able to attend all previous tokens.
            # this only works when we have no padding
            causal_mask = torch.full(
                (bsz, q_len, kv_len), 0, dtype=dtype, device=device
            )

        # add the head dimension
        # [Batch_Size, Q_Len, KV_Len] -> [Batch_Size, Num_Heads_Q, Q_Len, KV_Len]
        causal_mask = causal_mask.unsqueeze(1)

        if kv_cache is not None and kv_cache.num_items() > 0:
            # use the last location
            position_ids = attention_mask.cumsum(-1)[:, -1:]
        else:
            # create position_ids based on the size of the attention_mask
            # for padded tokens, use number 1
            position_ids = (attention_mask.cumsum(-1)).masked_fill_(
                (attention_mask == 0), 1
            )
        return causal_mask, position_ids

    # ---------- Inference ----------#

    def _forward_siglip_and_text_embedding(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
    ) -> torch.FloatTensor:
        dtype, device = pixel_values.dtype, pixel_values.device

        # text embedding
        # [Batch_Size, Seq_Len, Hidden_Size]
        inputs_embeds = self.embed_tokens(input_ids)

        # image features from siglip and projector
        # [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Hidden_Size]
        selected_image_feature = self.vision_tower(pixel_values)
        image_features = self.multi_modal_projector(selected_image_feature)

        # normalize the image features
        _, _, embed_dim = image_features.shape
        bsz, seq_len = input_ids.shape
        scaled_image_features = image_features / (self.image_text_hidden_size**0.5)

        # import ipdb;ipdb.set_trace()
        # put embedding together - image, text, padding
        final_embedding = torch.full((bsz, seq_len, embed_dim), self.config.pad_token_id, dtype=scaled_image_features.dtype, device=device)

        # [Batch_Size, Seq_Len]
        text_mask = (input_ids != self.image_token_index) & (
            input_ids != self.pad_token_id
        )
        image_mask = input_ids == self.image_token_index
        final_embedding[text_mask] = inputs_embeds[text_mask].to(final_embedding.dtype)
        for i in range(bsz):
            image_indices = image_mask[i].nonzero(as_tuple=True)[0]
            num_image_tokens = len(image_indices)
            final_embedding[i, image_indices] = scaled_image_features[
                i, :num_image_tokens
            ]
        return final_embedding

    def inference(
        self,
        batch,
    ) -> torch.FloatTensor:
        # import ipdb;ipdb.set_trace()        
        inputs = self.preprocess_batch(batch, split_mask=True, sample_fm_time=False)


        input_ids = inputs['input_ids']
        pixel_values = inputs['pixel_values']
        # causal_mask = inputs['causal_mask']
        vlm_position_ids = inputs['vlm_position_ids']
        proprio_position_ids = inputs['proprio_position_ids']
        action_position_ids = inputs['action_position_ids']
        proprios = inputs['proprios']
        # actions = inputs['actions']
        # t = inputs['t']
        # import ipdb;ipdb.set_trace()
        
        dtype, device = pixel_values.dtype, pixel_values.device
        bsz = pixel_values.size(0)

        kv_caches = self.joint_model.build_mixture_caches()

        # merge the text tokens and the image tokens
        inputs_embeds = self._forward_siglip_and_text_embedding(input_ids, pixel_values)

        # proprio
        proprio_embeds = self.proprio_encoder(proprios)

        # forward pass thru the vlm and proprio, cache the kv
        _, kv_caches = self.joint_model(
            attention_mask=inputs['image_text_proprio_mask'],
            position_ids_all={
                "vlm": vlm_position_ids,
                "proprio": proprio_position_ids,
            },
            embeds_all={
                "vlm": inputs_embeds,
                "proprio": proprio_embeds,
            },
            kv_caches=kv_caches,
            return_caches=True,
        )
        # import ipdb;ipdb.set_trace()
        # sample pure action noise
        action = torch.randn(
            (bsz, self.horizon_steps, self.action_dim), device=device, dtype=dtype
        )

        # forward euler integration --- using kv caches of vlm and proprio
        delta_t = 1.0 / self.num_inference_steps
        t = torch.zeros(bsz, device=device, dtype=dtype)
        for _ in range(self.num_inference_steps):
            # encode action and time into embedding
            time_cond = self.time_embedding(t)
            # [Batch_Size, Horizon_Steps, Embed_Dim]
            # import ipdb;ipdb.set_trace()
            if self.action_expert_adaptive_mode:
                action_embeds = self.action_encoder(action)
            else:
                action_embeds = self.action_encoder(action, time_cond)
            # [Batch_Size, Horizon_Steps, Embed_Dim]
            action_embeds = self.joint_model(
                attention_mask=inputs['action_mask'],
                position_ids_all={"action": action_position_ids},
                embeds_all={"action": action_embeds},
                time_cond=time_cond,
                kv_caches=kv_caches,
                cache_mode="append_non_active",  # use caches from other mixtures, i.e., vlm and proprio
            )["action"]
            # print(action_embeds)

            # decode action: [Batch_Size, Horizon_Steps, Action_Dim]
            action_vel = self.action_decoder(action_embeds)
            action += delta_t * action_vel
            t += delta_t

        # clamp final output if specified
        if self.final_action_clip_value is not None:
            action = torch.clamp(
                action,
                -self.final_action_clip_value,
                self.final_action_clip_value,
            )
        # import ipdb;ipdb.set_trace()
        # import numpy as np
        # from PIL import Image
        # Image.fromarray((((pixel_values[0]+1)/2).cpu().to(torch.float32).permute(1,2,0).numpy() * 255).astype(np.uint8)).save('train.png')
        return BatchFeature(data={'action_pred': action})

    def infer_action_naive(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        causal_mask: torch.FloatTensor,
        vlm_position_ids: torch.LongTensor,
        proprio_position_ids: torch.LongTensor,
        action_position_ids: torch.LongTensor,
        proprios: torch.FloatTensor,
    ) -> torch.FloatTensor:
        dtype, device = pixel_values.dtype, pixel_values.device
        bsz = pixel_values.size(0)

        kv_caches = self.joint_model.build_mixture_caches()

        # merge the text tokens and the image tokens
        inputs_embeds = self._forward_siglip_and_text_embedding(input_ids, pixel_values)

        # encode proprio
        proprio_embeds = self.proprio_encoder(proprios)

        # sample pure action noise
        action = torch.randn(
            (bsz, self.horizon_steps, self.action_dim), device=device, dtype=dtype
        )

        # forward euler integration --- run vlm in each step, which is unnecessary
        delta_t = 1.0 / self.num_inference_steps
        t = torch.zeros(bsz, device=device, dtype=dtype)
        for _ in range(self.num_inference_steps):
            # encode action and time into embedding
            time_cond = self.time_embedding(t)
            # [Batch_Size, Horizon_Steps, Embed_Dim]
            if self.action_expert_adaptive_mode:
                action_embeds = self.action_encoder(action)
            else:
                action_embeds = self.action_encoder(action, time_cond)
            action_embeds = self.joint_model(
                attention_mask=causal_mask,
                position_ids_all={
                    "vlm": vlm_position_ids,
                    "proprio": proprio_position_ids,
                    "action": action_position_ids,
                },
                embeds_all={
                    "vlm": inputs_embeds.clone(),  # clone needed due to modified in-place
                    "proprio": proprio_embeds.clone(),
                    "action": action_embeds,
                },
                time_cond=time_cond,
                kv_caches=kv_caches,
                cache_mode="no_append",  # no new tokens
            )["action"]
            # decode action: [Batch_Size, Horizon_Steps, Action_Dim]
            action_vel = self.action_decoder(action_embeds)
            action += delta_t * action_vel
            t += delta_t

        # clamp final output if specified
        if self.final_action_clip_value is not None:
            action = torch.clamp(
                action,
                -self.final_action_clip_value,
                self.final_action_clip_value,
            )
        return action

    def infer_text(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        attention_mask: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple:
        q_len = input_ids.size(1)

        # text tokens + image tokens
        inputs_embeds = self._forward_siglip_and_text_embedding(input_ids, pixel_values)

        # build causal mask and position ids for text
        (
            causal_mask,
            position_ids,
        ) = self.build_causal_mask_and_position_ids_for_text(
            q_len, attention_mask, kv_cache
        )

        hidden_states = self.joint_model(
            attention_mask=causal_mask,
            position_ids_all={"vlm": position_ids},
            embeds_all={"vlm": inputs_embeds},
            kv_caches={"vlm": kv_cache},
            cache_mode="append",  # new tokens for the active mixture
            final_layer_post_attn_skip_names=[],  # do not skip vlm last layer
        )["vlm"]
        logits = self.lm_head(hidden_states)
        output = {
            "logits": logits,
        }
        if kv_cache is not None:
            output["kv_cache"] = kv_cache
        return output

    # ---------- Flow matching training ----------#

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
        self,
        batch,

    ) -> torch.FloatTensor:

        
        inputs = self.preprocess_batch(batch, split_mask=False, sample_fm_time=True)

        input_ids = inputs['input_ids']
        pixel_values = inputs['pixel_values']
        causal_mask = inputs['causal_mask']
        vlm_position_ids = inputs['vlm_position_ids']
        proprio_position_ids = inputs['proprio_position_ids']
        action_position_ids = inputs['action_position_ids']
        proprios = inputs['proprios']
        actions = inputs['actions']
        t = inputs['t']
        
        """flow matching loss for action prediction, no use of kv cache"""
        # noisy action
        # [Batch_Size, Horizon_Steps, Action_Dim]
        x0 = torch.randn_like(actions, device=t.device, dtype=t.dtype)
        x1 = actions
        psi_t = self.psi_t(x0, x1, t)

        # text tokens + image tokens
        inputs_embeds = self._forward_siglip_and_text_embedding(input_ids, pixel_values)

        # proprio
        proprio_embeds = self.proprio_encoder(proprios)

        # inference with noisy action
        # [Batch_Size, Embed_Dim]
        time_cond = self.time_embedding(t)
        # [Batch_Size, Horizon_Steps, Embed_Dim]
        if self.action_expert_adaptive_mode:
            action_embeds = self.action_encoder(psi_t)
        else:
            action_embeds = self.action_encoder(psi_t, time_cond)
        action_embeds = self.joint_model(
            attention_mask=causal_mask,
            position_ids_all={
                "vlm": vlm_position_ids,
                "proprio": proprio_position_ids,
                "action": action_position_ids,
            },
            embeds_all={
                "vlm": inputs_embeds,
                "proprio": proprio_embeds,
                "action": action_embeds,
            },
            time_cond=time_cond,
            kv_caches={},  # no caching during training
        )["action"]

        # [Batch_Size, Horizon_Steps, Action_Dim]
        v_psi = self.action_decoder(action_embeds)

        # compare to true velocity
        d_psi = x1 - (1 - self.flow_sig_min) * x0
        loss = torch.mean((v_psi - d_psi) ** 2)
        return {'loss': loss}


class PiZeroInference(PiZero):
    def forward(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        image_text_proprio_mask: torch.FloatTensor,
        action_mask: torch.FloatTensor,
        vlm_position_ids: torch.LongTensor,
        proprio_position_ids: torch.LongTensor,
        action_position_ids: torch.LongTensor,
        proprios: torch.FloatTensor,
    ) -> torch.FloatTensor:
        return super().inference(
            input_ids,
            pixel_values,
            image_text_proprio_mask,
            action_mask,
            vlm_position_ids,
            proprio_position_ids,
            action_position_ids,
            proprios,
        )


AutoConfig.register("openpi0", OpenPI0Config)
AutoModel.register(OpenPI0Config, PiZero)


if __name__ == "__main__":
    # DEBUG CODE
    pass
    