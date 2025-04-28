import torch
from .cubing_momentum import Cubing

class IdentityMap(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_resampler_type": None}


def build_cubing(model_args, vision_dim, vision_toks_len,embed_dim=256, window_size=3, lm_dim=4096, **kwargs):
    cubing_type = getattr(model_args, "mm_cubing", None)
    mm_use_thumbnail = getattr(model_args, "mm_use_thumbnail", True)
    forward_n_layers = getattr(model_args, 'cubing_vit_forward_n_layers', -1)
    
    if cubing_type != 'identity': # Default
        return Cubing(cubing_type, vision_dim, vision_toks_len,embed_dim,window_size, mm_use_thumbnail, forward_n_layers, lm_dim, **kwargs)
    else:
        return IdentityMap()
