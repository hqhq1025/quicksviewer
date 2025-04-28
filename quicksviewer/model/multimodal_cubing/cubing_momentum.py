""" Using Gumnbel-Softmax to group cubes in a derivable way.
https://arxiv.org/pdf/1611.01144
https://github.com/YongfeiYan/Gumbel_Softmax_VAE
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.init import trunc_normal_
from quicksviewer.train.sequence_parallel import get_pg_manager


def find_segments(A):
    segments = []
    pre, cur = 0, 0
    while cur < len(A):
        if cur==len(A)-1 or (A[cur+1]!=0):
            segments.append((pre, cur+1)) # add tuple [closed-left open-right)
            pre = cur+1
        cur += 1
    return segments

def sample_gumbel(shape, eps=1e-20, dtype=torch.bfloat16):
    U = torch.rand(shape, dtype=dtype)
    U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature, lr_gumbel):
    y = logits + sample_gumbel(logits.size(), dtype=logits.dtype) * lr_gumbel
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature, topk=1, lr_gumbel=0.1):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    shape = logits.shape
    y = gumbel_softmax_sample(logits, temperature, lr_gumbel) # (1, N, 2)
    # here we take the second dimension as the prob which is same as https://github.com/irwinherrmann/stochastic-gates/blob/master/models/seq_with_gumbel.py
    y = y[:, :, 1]
    
    _, ind = y.topk(k=topk, dim=-1) # Qiji: changed to N-hot
    y_hard = torch.zeros_like(y)
    y_hard.scatter_(1, ind, 1)
    y_hard = (y_hard - y).detach() + y
    return y, y_hard


class Cubing(nn.Module):
    def __init__(self,
                cubing_type,
                vision_dim,
                vision_toks_len,
                embed_dim=256,
                window_size=3,
                mm_use_thumbnail=True,
                forward_n_layers=-1,
                lm_dim = 4096,
                **kwargs) -> None:
        super().__init__(**kwargs)

        self.cubing_type = cubing_type
        self.vision_dim = vision_dim
        self.embed_dim = vision_dim if embed_dim is None else embed_dim
        self.mm_use_thumbnail = mm_use_thumbnail
        self.forward_n_layers = forward_n_layers

        self.agg_frame_fn = nn.Sequential(
            nn.Linear(self.vision_dim, self.vision_dim),
        )

        self.proj_fn = nn.Sequential(
            nn.LayerNorm(self.vision_dim),
            nn.Linear(self.vision_dim, self.vision_dim),
            nn.GELU(),
            nn.Linear(self.vision_dim, 2),
            )

        self.thumbnail_fn = nn.Sequential(
            nn.AvgPool2d(kernel_size=(9,1), stride=(9,1)),
            nn.Linear(self.vision_dim, lm_dim)
        )

        self.PROCESS_GROUP_MANAGER = get_pg_manager()
        if self.PROCESS_GROUP_MANAGER is None:
            self.sp_degree = -1
            self.sp_rank = -1
        else:
            self.sp_group = self.PROCESS_GROUP_MANAGER.sp_pg
            self.sp_degree = self.PROCESS_GROUP_MANAGER.sp_degree
            self.sp_rank = self.PROCESS_GROUP_MANAGER.sp_rank
        

    def forward(self, vision_tower, resampler, images, tgt_size, videos_bound, temperature=0.5, FPQ=5, lr_gumbel=0.1):
        """
          Params:
            @FPQ: the average number of frames per cube.
            @lr_gumbel: the weight of the gumbel_noise, for annealing.
        """
        bs = len(images)

        # debug_feats = None
        cube_bound = [] # [(bounds of cubes for video 1), ...]
        pre_e = 0
        list_feats = []
        list_z = []
        for i, (vs, ve) in enumerate(videos_bound):
            if vs > pre_e:
                pre_img_feats = vision_tower(images[pre_e: vs])
                pre_img_feats = resampler(pre_img_feats, tgt_size)
                list_feats.append(pre_img_feats)

            video = vision_tower(images[vs: ve], self.forward_n_layers) # Use first n-layers
            bf = len(video)

            # Momentum
            vid_feats_momentum = [video[1] - video[0]]
            alpha = 0.8
            for ii in range(2,bf):
                vid_feats_momentum.append(alpha*(video[ii]-video[ii-1]) + (1-alpha)*(vid_feats_momentum[-1]))
            vid_feats = torch.stack(vid_feats_momentum, dim=0) # (bf-1, 576, 1024)

            # Aggregate tokens in a frame
            # a. version-1: simply mean
            # vid_feats = vid_feats.mean(dim=1)
            # b. version-2: projection
            vid_feats = self.agg_frame_fn(vid_feats) # Before or After for project vit feats
            vid_feats = vid_feats.mean(dim=1)

            z = self.proj_fn(vid_feats) # (bf-1, 2)
            list_z.append(z)

            num_cubes = max(round(bf/FPQ)-1, 1) # -1 to exclude beginning
            # print(f"#### [lr_gumbel = {lr_gumbel}]")
            z, z_hard = gumbel_softmax(z.unsqueeze(0), temperature, topk=num_cubes, lr_gumbel=lr_gumbel)
            z, z_hard = z.squeeze(0), z_hard.squeeze(0)

            # Force adding 1 to beginning
            pad_z, pad_z_hard = torch.ones(1,dtype=z.dtype,device=z.device), torch.ones(1,dtype=z_hard.dtype,device=z_hard.device)
            z, z_hard = torch.cat([pad_z,z]), torch.cat([pad_z_hard, z_hard]) # (bf, )

            # Decide the cubes boundaries based on Gumbel-Sampling
            bounds = find_segments(z_hard)
            # print(f"********\n Cube bounds: {bounds}")
            cube_bound.append(bounds)

            vid_feats = []
            for ii,(s,e) in enumerate(bounds):
                feat = video[s:e]
                feat = resampler(feat.unsqueeze(0), tgt_size_range=[[s,e], [0,tgt_size[0]], [0,tgt_size[1]]]) # feat: (1, bf*576, d) ## Change this for using 3D resampler
                vid_feats.append(feat)
            vid_feats = torch.cat(vid_feats, 0)

            if self.mm_use_thumbnail:
                # Get boundaries-specific features for implicit supervision the boundaries
                if self.forward_n_layers >= 0:
                    indexs = torch.nonzero(z_hard).view(-1)
                    video_thumb = video[indexs]
                    video_thumb = vision_tower(forward_n_layers=self.forward_n_layers, forward_nth_embeds = video_thumb)
                    video = video.scatter(0, indexs.view(-1,1,1).repeat(1,*video.shape[1:]), video_thumb)
                bound_feats = video * z_hard.unsqueeze(1).unsqueeze(1) / z_hard.sum().detach().item() # video.shape = (128,576,4096)

                bound_feats = torch.sum(bound_feats, dim=0, keepdim=True)
                # SUPPORT for sequence_parallel ! Only add thumbnail feature at the first chunk of sequence_parallel
                if self.sp_degree > 0:
                    dist.barrier(group=self.sp_group)
                    dist.all_reduce(bound_feats, op=dist.ReduceOp.SUM, group=self.sp_group)
                    bound_feats = bound_feats / self.sp_degree
                bound_feats = self.thumbnail_fn(bound_feats)
                vid_feats = torch.cat([bound_feats, vid_feats], 0) # (bf+1, 64, 4096)
                vid_feats = vid_feats[int(self.sp_rank>0):] # Add thumbnail only at the first chunk

            list_feats.append(vid_feats)
            pre_e = ve

        if pre_e != bs:
            post_img_feats = vision_tower(images[pre_e:])
            post_img_feats = resampler(post_img_feats, tgt_size)
            list_feats.append(post_img_feats)
        
        image_features = torch.vstack(list_feats)

        if self.mm_use_thumbnail:
            # Update with thumbnail only on the first chunk
            if self.sp_degree <=1 or (self.sp_degree >1 and self.sp_rank==0):
                videos_bound = [(vb[0]+i, vb[1]+i+1) for i,vb in enumerate(videos_bound)]
                cube_bound = [[(0,1)]+[(cb[0]+1, cb[1]+1) for cb in vcb] for vcb in cube_bound] # (0,0) for thumbnail seg
        return image_features, videos_bound, cube_bound, list_z

    @property
    def config(self):
        return {
            "cubing_type": self.cubing_type,
            "mm_use_thumbnail": self.mm_use_thumbnail
        }

    @property
    def hidden_size(self):
        return self.embed_dim




