from enum import Enum
from transformers import PretrainedConfig, PreTrainedModel
from typing import Optional
import logging
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError, RepositoryNotFoundError
import os
import packaging
import safetensors
from safetensors.torch import load_model as load_model_as_safetensor



class BasePolicyModel(PreTrainedModel):

    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        *args,
        **kwargs
    ):
        if config is None:
            config = PretrainedConfig()
        super().__init__(config, *args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError('Not implemented in base policy model class')


    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        try:
            # NOTE(YL) This downloads the model to the local cache and returns the local path to the model
            # download the model, saved in ~/.cache/huggingface/hub/, return the path of folder
            cache_dir = kwargs.pop('HF_cache_dir')
            local_model_path = snapshot_download(pretrained_model_name_or_path, repo_type='model', cache_dir=cache_dir)
            print(
                f'Model found in the huggingface hub. Loading from path: {pretrained_model_name_or_path}'
            )
            # HFValidationError, RepositoryNotFoundError
        except (HFValidationError, RepositoryNotFoundError):
            print(
                f'Model not found or avail in the huggingface hub. Loading from local path: {pretrained_model_name_or_path}'
            )
            local_model_path = pretrained_model_name_or_path
        try:
            print(local_model_path)
            pretrained_model = super().from_pretrained(local_model_path, local_model_path=local_model_path, **kwargs)
        except ValueError:
            # add metadata to the model config
            from safetensors.torch import load_file, save_file
            files = os.listdir(local_model_path)
            for file in files:
                if file.endswith('.safetensors'):
                    model_file = os.path.join(local_model_path, file)
                    tensors = load_file(model_file, device='cpu')
                    save_file(tensors, model_file, metadata={'format': 'pt'})
            pretrained_model = super().from_pretrained(local_model_path, local_model_path=local_model_path, **kwargs)

        return pretrained_model

    @classmethod
    def _load_as_safetensor(cls, model, model_file: str, map_location: str, strict: bool):
        """
        This is for loading the full version of lerobot/pi0 model.
        Commonly, previous load style will miss the language embedding weights.
        """

        if packaging.version.parse(safetensors.__version__) < packaging.version.parse('0.4.3'):
            load_model_as_safetensor(model, model_file, strict=strict)
            if map_location != 'cpu':
                logging.warning(
                    "Loading model weights on other devices than 'cpu' is not supported natively in your version of safetensors."
                    " This means that the model is loaded on 'cpu' first and then copied to the device."
                    ' This leads to a slower loading time.'
                    ' Please update safetensors to version 0.4.3 or above for improved performance.'
                )
                model.to(map_location)
        else:

            from safetensors import safe_open
            model_file_ = model_file.replace('model.safetensors', 'model-00001-of-00003.safetensors')
            aa = safe_open(model_file_, framework='pt', device='cpu')

            print(safetensors.torch.load_model(model, model_file, strict=strict, device=map_location))
            print('emb weight', model.model.paligemma_with_expert.paligemma.language_model.model.embed_tokens.load_state_dict({'weight': aa.get_tensor('paligemma.language_model.model.embed_tokens.weight')}))
        return model
