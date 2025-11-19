import torch
import torch.nn as nn
import numpy as np
from functools import reduce
import math

'''
basic pooling / linear interpolation
'''

class Downsampling(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.up = nn.AvgPool2d(kernel_size=5, stride=4, padding=2)
    
    def forward(self, x):
        return self.up(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

class Upsampling(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
    
    def forward(self, x):
        result = torch.nn.functional.interpolate(x.permute(0, 3, 1, 2), size=self.size, mode='bilinear', align_corners=False)
        
        return result.permute(0, 2, 3, 1)
    
'''
Pixel Shuffle and Unshuffle
'''

class Downsample(nn.Module):
    def __init__(self, n_feat, keep_dim=False):
        super(Downsample, self).__init__()

        if keep_dim:
            n_feat_out = n_feat // 4
        else:
            n_feat_out = n_feat // 2

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat_out, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class DownsampleV2(nn.Module):
    def __init__(self, n_feat):
        super(DownsampleV2, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 4, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat, keep_dim=False):
        super(Upsample, self).__init__()

        if keep_dim:
            n_feat_out = n_feat * 4
        else:
            n_feat_out = n_feat * 2

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat_out, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

class UpsampleV2(nn.Module):
    def __init__(self, n_feat):
        super(UpsampleV2, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 4, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))
    def forward(self, x):
        return self.body(x)

class LearnableSamplingMatrix(nn.Module):
    """
    Learnable dense-to-coarse sampler that records the sampling matrix for reuse
    during upsampling.
    """

    def __init__(self, reduction: int = 2):
        super().__init__()
        self.reduction = reduction
        self.register_parameter('sampling_matrix', None)
        self.dense_shape = None
        self.coarse_shape = None

    def _initialize_if_needed(self, h: int, w: int, device: torch.device, dtype: torch.dtype) -> None:
        if self.sampling_matrix is not None:
            if (h, w) != self.dense_shape:
                raise ValueError(
                    f"LearnableSamplingMatrix expected spatial shape {self.dense_shape} but received {(h, w)}"
                )
            return

        if h % self.reduction != 0 or w % self.reduction != 0:
            raise ValueError(
                f"Input spatial shape ({h}, {w}) must be divisible by reduction factor {self.reduction}"
            )

        coarse_h = h // self.reduction
        coarse_w = w // self.reduction
        init = torch.randn(coarse_h * coarse_w, h * w, device=device, dtype=dtype) * (1.0 / math.sqrt(h * w))
        self.sampling_matrix = nn.Parameter(init)
        self.dense_shape = (h, w)
        self.coarse_shape = (coarse_h, coarse_w)

    def _normalized_weights(self) -> torch.Tensor:
        return torch.softmax(self.sampling_matrix, dim=-1)

    def downsample(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        self._initialize_if_needed(h, w, x.device, x.dtype)
        weights = self._normalized_weights()  # Nc x Nd
        flat = x.reshape(b, c, -1)
        coarse = torch.einsum('bcn,kn->bck', flat, weights)
        return coarse.reshape(b, c, *self.coarse_shape)

    def upsample(self, x: torch.Tensor) -> torch.Tensor:
        if self.sampling_matrix is None:
            raise RuntimeError("Sampling matrix has not been initialized. Call downsample before upsample.")
        b, c, h, w = x.shape
        if (h, w) != self.coarse_shape:
            raise ValueError(
                f"LearnableSamplingMatrix expected coarse shape {self.coarse_shape} but received {(h, w)}"
            )
        weights = self._normalized_weights()  # Nc x Nd
        dense = torch.einsum('bck,nk->bcn', x.reshape(b, c, -1), weights.transpose(0, 1))
        return dense.reshape(b, c, *self.dense_shape)

class LearnablePointDownsample(nn.Module):
    """
    Channel aware downsampling that first projects channels and then applies a
    learnable dense-to-coarse sampler.
    """

    def __init__(self, in_channels: int, keep_dim: bool = False, sampler: LearnableSamplingMatrix = None, reduction: int = 2):
        super().__init__()
        out_channels = in_channels if keep_dim else in_channels * 2
        self.channel_proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GELU()
        )
        self.sampler = sampler if sampler is not None else LearnableSamplingMatrix(reduction=reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_proj(x)
        return self.sampler.downsample(x)

class LearnablePointUpsample(nn.Module):
    """
    Upsampling counterpart that reuses the sampling matrix learned during the
    downsampling stage.
    """

    def __init__(self, in_channels: int, keep_dim: bool = False, sampler: LearnableSamplingMatrix = None, reduction: int = 2):
        super().__init__()
        if not keep_dim and in_channels % 2 != 0:
            raise ValueError("in_channels must be divisible by 2 when keep_dim is False.")
        out_channels = in_channels if keep_dim else in_channels // 2
        self.sampler = sampler if sampler is not None else LearnableSamplingMatrix(reduction=reduction)
        self.channel_proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.sampler.upsample(x)
        return self.channel_proj(x)
