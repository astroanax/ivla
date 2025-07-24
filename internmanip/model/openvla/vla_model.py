from internmanip.model.basemodel.base import BasePolicyModel
from transformers import AutoModelForVision2Seq, AutoProcessor, PretrainedConfig
import torch
from torch import nn
from typing import Optional


class VLAModel(BasePolicyModel):
    def __init__(
        self, 
        model_name_or_path: Optional[str] = None, 
        **kwargs
    ):
        nn.Module.__init__(self)

        if model_name_or_path is None:
            model_name_or_path = "openvla/openvla-7b"
        self.cuda_device_id = kwargs.get("device_id", None)
        if self.cuda_device_id is None or self.cuda_device_id >= torch.cuda.device_count():
            self.cuda_device_id = 0
        self.cuda_device = torch.device(f"cuda:{self.cuda_device_id}")
        self.processor = AutoProcessor.from_pretrained(
            model_name_or_path, 
            trust_remote_code=True
        )
        self.vla = AutoModelForVision2Seq.from_pretrained(
            model_name_or_path,
            attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(self.cuda_device)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        return cls(pretrained_model_name_or_path, **kwargs)