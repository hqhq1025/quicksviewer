from functools import partial
import numpy as np

import torch
from torch import nn
from torch.nn.init import trunc_normal_

def get_3d_sincos_pos_embed(embed_dim, image_size):
    """
    image_size: image_size or (n_images, image_height, image_width)
    return:
    pos_embed: [n_images, image_height, image_height, embed_dim]
    """
    if isinstance(image_size, int):
        image_size = [image_size] * 3
    else: # 3d
        grid_t_size, grid_h_size, grid_w_size = image_size[0], image_size[1], image_size[2]

    grid_t = torch.arange(grid_t_size, dtype=torch.float32)
    grid_h = torch.arange(grid_h_size, dtype=torch.float32)
    grid_w = torch.arange(grid_w_size, dtype=torch.float32)
    grid = torch.meshgrid(grid_t, grid_w, grid_h) # (.shape=[t,w,h])
    grid = torch.stack(grid, dim=0)

    pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def get_3d_sincos_pos_embed_from_grid(embed_dim, grid):
    """ The embeddings parts for time-height-width will be [3/8, 3/8, 2/8].
    """
    assert embed_dim % 8 == 0
    # use 2/8 of dimensions to encode grid_t
    emb_t = get_1d_sincos_pos_embed_from_grid_new(embed_dim // 8 * 2, grid[0])  # (T, H, W, D/8*2)
    # use 3/8 of dimensions to encode grid_h and grid_w
    emb_h = get_1d_sincos_pos_embed_from_grid_new(embed_dim // 8 * 3, grid[1])  # (T, H, W, D/8*3)
    emb_w = get_1d_sincos_pos_embed_from_grid_new(embed_dim // 8 * 3, grid[2])  # (T, H, W, D/8*3)

    emb = torch.cat([emb_t, emb_h, emb_w], dim=-1)  # (T, H, W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid_new(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (T, H, W)
    out: (T, H, W, D)
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    out = torch.einsum('thw,d->thwd', pos, omega)  # (T, H, W, D/2), outer product

    emb_sin = torch.sin(out)  # (T, H, W, D/2)
    emb_cos = torch.cos(out)  # (T, H, W, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=-1)  # (T, H, W, D)
    return emb


class Resampler(nn.Module):
    """
    A 2D perceiver-resampler network with one cross attention layers by
       given learnable queries and 2d sincos pos_emb
    Outputs:
        A tensor with the shape of (batch_size, num_queries, embed_dim)
    """
    def __init__(self, model_args, **kwargs):
        super().__init__()
        self.num_queries = getattr(model_args, 'num_queries', kwargs.get('num_queries', 64))
        self.kv_dim = getattr(model_args, 'mm_resampler_visiondim', kwargs.get('mm_resampler_visiondim', None))
        self.embed_dim = getattr(model_args, 'mm_resampler_embeddim', kwargs.get('mm_resampler_embeddim'))
        self.num_heads = getattr(model_args, 'num_heads', kwargs.get('num_heads', None))
        self.adaptive = getattr(model_args, 'adaptive', kwargs.get('adaptive', True))
        self.max_size = getattr(model_args, 'max_size', kwargs.get('max_size', (300, 24, 24,)))

        self.num_heads = self.embed_dim//128 if not self.num_heads else self.num_heads

        self.query = nn.Parameter(torch.zeros(self.num_queries, self.embed_dim))

        if self.kv_dim is not None and self.kv_dim != self.embed_dim:
            self.kv_proj = nn.Linear(self.kv_dim, self.embed_dim, bias=False)
        else:
            self.kv_proj = nn.Identity()

        self.attn = nn.MultiheadAttention(self.embed_dim, self.num_heads)
        self.ln_q = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.ln_kv = nn.LayerNorm(self.embed_dim, eps=1e-6)

        self.ln_post = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.proj = nn.Parameter((self.embed_dim ** -0.5) * torch.randn(self.embed_dim, self.embed_dim))

        self._set_3d_pos_cache(self.max_size)
        self.apply(self._init_weights)

    def _set_3d_pos_cache(self, max_size, device='cpu'):
        pos_embed = get_3d_sincos_pos_embed(self.embed_dim, max_size).to(device)
        self.register_buffer("pos_embed", pos_embed, persistent=False)

    def _adjust_pos_cache(self, max_thw_sizes, device):
        if max_thw_sizes[0] > self.max_size[0] or max_thw_sizes[1] > self.max_size[1] or max_thw_sizes[2] > self.max_size[2]:
            self.max_size = [max(max_thw_sizes[0], self.max_size[0]), max(max_thw_sizes[1], self.max_size[1]), max(max_thw_sizes[2], self.max_size[2])]
            self._set_3d_pos_cache(self.max_size, device)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, image_features: torch.tensor, tgt_size_range: list, *args, **kwargs):
        """ Forward resampler using 3D positional encoding. Pass right-open ranges [[t_star,t_end),[w_start,w_end),[h_start,h_end)] for 'tgt_size_range' if input videos, elsewise pass [[w_start,w_end),[h_start,h_end)] for input images.
        """
        tgt_size_range = [[0,_] if isinstance(_,int) else _ for _ in tgt_size_range] # convert to range
        if len(tgt_size_range) == 2:
            tgt_size_range = [[0,1], tgt_size_range[0], tgt_size_range[1]]
        elif len(tgt_size_range) == 3: # iunput videos
            image_features = image_features.view(image_features.shape[0], -1, image_features.shape[-1])
        B, L, D = image_features.shape

        tgt_sizes_range = torch.tensor(tgt_size_range, device=image_features.device, dtype=torch.int32).unsqueeze(0).repeat(B,1,1)
        tgt_sizes = torch.tensor([_[1]-_[0] for _ in tgt_size_range], device=image_features.device, dtype=torch.int32).unsqueeze(0).repeat(B,1)

        x = image_features
        assert x.shape[0] == tgt_sizes_range.shape[0]
        bs = x.shape[0]

        device = x.device
        dtype = x.dtype

        patch_len = tgt_sizes[:, 0] * tgt_sizes[:, 1] * tgt_sizes[:, 2]

        self._adjust_pos_cache([_[1] for _ in tgt_size_range], device=device) # -1 for right-open

        max_patch_len = torch.max(patch_len) if patch_len.nelement() !=0 else 0
        key_padding_mask = torch.zeros((bs, max_patch_len), dtype=torch.bool, device=device)

        pos_embed = []
        for i in range(bs):
            tgt_t, tgt_h, tgt_w = tgt_sizes[i]
            range_t, range_h, range_w = tgt_sizes_range[i]
            pos_embed.append(self.pos_embed[range_t[0]:range_t[1], range_h[0]:range_h[1], range_w[0]:range_w[1], :].reshape((tgt_t * tgt_h * tgt_w, -1)).to(dtype))  # n_images * patches * D
            key_padding_mask[i, patch_len[i]:] = True

        x = self.kv_proj(x)  # B * L * D
        x = self.ln_kv(x).permute(1, 0, 2)  # L * B * D

        q = self.ln_q(self.query)  # Q * D

        if pos_embed!=[]:
            pos_embed = torch.nn.utils.rnn.pad_sequence(pos_embed, batch_first=True, padding_value=0.0).permute(1, 0, 2)  # BLD => L * B * D
        else:
            pos_embed, key_padding_mask = torch.zeros_like(x, device=x.device), None
        pos_embed = pos_embed.to(x.device)
        out = self.attn(
            self._repeat(q, bs),  # Q * B * D
            x + pos_embed,  # L * B * D +  L * B * D
            x,
            key_padding_mask=key_padding_mask)[0]
        #  out: Q * B * D
        x = out.permute(1, 0, 2)  # B * Q * D

        x = self.ln_post(x)
        x = x @ self.proj
        return x

    def _repeat(self, query, N: int):
        return query.unsqueeze(1).repeat(1, N, 1)

    @property
    def config(self):
        return {
            "mm_resampler_type": "qformer",
            "num_queries": self.num_queries,
            "kv_dim" : self.kv_dim,
            "embed_dim" : self.embed_dim,
            "num_heads" : self.num_heads,
            # "norm_layer" : self.norm_layer,
            "adaptive" : self.adaptive,
            "max_size" : self.max_size,
        }

    @property
    def hidden_size(self):
        return self.embed_dim



if __name__ == '__main__':
    pos_embed = get_3d_sincos_pos_embed(embed_dim=4096, image_size=(3600,24,24))
    print(pos_embed)