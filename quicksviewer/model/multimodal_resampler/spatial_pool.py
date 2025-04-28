import torch
import torch.nn as nn
import math


class SpatialPool(nn.Module):
    def __init__(self, model_args, mm_resampler_embeddim, mm_resampler_visiondim):
        super().__init__()

        self.mode = model_args.mm_spatial_pool_mode
        self.stride = model_args.mm_spatial_pool_stride
        self.out_channels = getattr(model_args, "mm_spatial_pool_out_channels", mm_resampler_visiondim)

        if self.mode == "average":
            self.pool = nn.AvgPool2d(kernel_size=self.stride, stride=self.stride)
        elif self.mode == "max":
            self.pool = nn.MaxPool2d(kernel_size=self.stride, stride=self.stride)
        elif self.mode == "conv":
            self.pool = nn.Conv2d(in_channels=mm_resampler_visiondim, out_channels=self.out_channels, kernel_size=self.stride, stride=self.stride)
        else:
            raise ValueError(f"Unknown pooling mode: {self.pool}.")

    def forward(self, image_features, tgt_size, *args, **kwargs):
        B, _, F = image_features.shape

        image_features_spatial = image_features.view(B, tgt_size[0], tgt_size[1], F).permute(0, 3, 1, 2)
        image_features_spatial_pool = self.pool(image_features_spatial)

        return image_features_spatial_pool.flatten(2).transpose(1, 2).contiguous()

    @property
    def config(self):
        return {
            "mm_resampler_type": "spatial_pool",
            "mm_spatial_pool_stride": self.stride,
            "mm_spatial_pool_mode": self.mode,
            "mm_spatial_pool_out_channels": self.out_channels,
        }

    @property
    def hidden_size(self):
        return self.out_channels
