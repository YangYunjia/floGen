import torch
import torch.nn as nn
import numpy as np
from functools import reduce
from abc import abstractmethod
import math

from flowvae.base_model.mlp import mlp
from flowvae.base_model.utils import IntpConv
from .attention import Attention

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

class ReusableSamplingCore(nn.Module):

    def __init__(self, in_channels: int):
        super().__init__()
    
    @abstractmethod
    def downsample(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def upsample(self, x: torch.Tensor) -> torch.Tensor:
        pass

class ReusableDownsample(nn.Module):
    """
    Use the Reusable sampling core for downsampling, and record sampling weights
    """

    def __init__(self, sampler: ReusableSamplingCore, in_channels: int, keep_dim: bool = False):

        super().__init__()
        out_channels = in_channels if keep_dim else in_channels * 2
        self.channel_proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GELU()
        )
        self.__setattr__("sampler", sampler)    # to avoid multiple register of sampler

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_proj(x)
        return self.sampler.downsample(x)

class ReusableUpsample(nn.Module):
    """
    Upsampling counterpart that reuses the sampling matrix learned during the
    downsampling stage.
    """

    def __init__(self, sampler: ReusableSamplingCore, in_channels: int, keep_dim: bool = False):
        super().__init__()
        if not keep_dim and in_channels % 2 != 0:
            raise ValueError("in_channels must be divisible by 2 when keep_dim is False.")
        out_channels = in_channels if keep_dim else in_channels // 2
        self.channel_proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GELU()
        )
        self.__setattr__("sampler", sampler)    # to avoid multiple register of sampler

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.sampler.upsample(x)
        return self.channel_proj(x)

class LearnableSamplingMatrix(ReusableSamplingCore):
    """
    Learnable dense-to-coarse sampler that records the sampling matrix for reuse
    during upsampling.
    """

    def __init__(self, in_channels: int, reduction: int = 2):
        super().__init__(in_channels)
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
        self.sampling_matrix = nn.Parameter(init)   # same for every shape
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

class AttentionPointSampler(ReusableSamplingCore):
    """
    Dynamic sampler that aggregates dense features into sparse representatives
    through attention weights that depend on the current input.
    """

    def __init__(self, in_channels: int, reduction: int = 2, heads: int = 4, temperature: float = 0.05):
        super().__init__(in_channels)
        if reduction < 1:
            raise ValueError("Reduction factor must be >= 1.")
        self.reduction = reduction
        self.temperature = temperature
        hidden_dim = max(in_channels // 2, 1)
        self.selector = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        self.heads = heads
        self.token_attn = Attention(dim=in_channels, heads=self.heads, dim_head=in_channels // self.heads, dropout=0.)
        self.cached_weights = None
        self.cached_shape = None

    def _split_windows(self, x: torch.Tensor):
        b, c, h, w = x.shape
        if h % self.reduction != 0 or w % self.reduction != 0:
            raise ValueError(
                f"Input spatial shape ({h}, {w}) must be divisible by reduction factor {self.reduction}"
            )
        h_coarse = h // self.reduction
        w_coarse = w // self.reduction
        window_size = self.reduction * self.reduction
        windows = x.reshape(b, c, h_coarse, self.reduction, w_coarse, self.reduction)
        windows = windows.permute(0, 2, 4, 3, 5, 1).reshape(b, h_coarse, w_coarse, window_size, c)
        return windows, h_coarse, w_coarse, window_size

    def downsample(self, x: torch.Tensor) -> torch.Tensor:
        windows, h_coarse, w_coarse, window_size = self._split_windows(x)
        scores = self.selector(windows).squeeze(-1)  # B Hc Wc L
        weights = torch.softmax(scores / self.temperature, dim=3).unsqueeze(-1)  # B Hc Wc L 1
        pooled = torch.sum(weights * windows, dim=3)  # B Hc Wc C
        tokens = pooled.reshape(x.shape[0], h_coarse * w_coarse, -1)
        attn_out = self.token_attn(tokens, None)
        tokens = tokens + attn_out
        coarse = tokens.reshape(x.shape[0], h_coarse, w_coarse, -1).permute(0, 3, 1, 2)

        self.cached_weights = weights
        self.cached_shape = (x.shape[2], x.shape[3])
        return coarse

    def upsample(self, x: torch.Tensor) -> torch.Tensor:
        if self.cached_weights is None or self.cached_shape is None:
            raise RuntimeError("Sampler cache is empty. Call `downsample` before `upsample`.")
        b, c, h_coarse, w_coarse = x.shape
        weights = self.cached_weights
        if weights.shape[1] != h_coarse or weights.shape[2] != w_coarse:
            raise ValueError("Cached weights do not match the coarse resolution.")
        reduction = self.reduction
        h, w = self.cached_shape

        coarse = x.permute(0, 2, 3, 1).unsqueeze(3)  # B Hc Wc 1 C
        window_tokens = weights * coarse  # B Hc Wc L C
        window_tokens = window_tokens.reshape(b, h_coarse, w_coarse, reduction, reduction, c)
        dense = window_tokens.permute(0, 5, 1, 3, 2, 4).reshape(b, c, h, w)

        self.cached_weights = None
        self.cached_shape = None
        return dense


class LowRankSampler(ReusableSamplingCore):
    """
    Cross attention without explicit Nc x N matrices via rank-limited token
    projections along dense and coarse dimensions.
    """

    def __init__(self, in_channels: int, reduction: int = 2, rank: int = 64, temperature: float = 0.05):
        super().__init__(in_channels)
        if reduction < 1:
            raise ValueError("Reduction factor must be >= 1.")
        self.channels = in_channels
        self.reduction = reduction
        self.temperature = temperature
        self.rank = rank
        hidden_dim = max(in_channels // 2, 1)
        self.selector = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        self.dense_mapper = nn.Linear(in_channels, rank)
        self.coarse_mapper = nn.Linear(in_channels, rank)
        self.cached_dense_weights = None
        self.cached_coarse_weights = None
        self.cached_shape = None
        self.cached_coarse_shape = None

    def _split_windows(self, x: torch.Tensor):
        b, c, h, w = x.shape
        if h % self.reduction != 0 or w % self.reduction != 0:
            raise ValueError(
                f"Input spatial shape ({h}, {w}) must be divisible by reduction factor {self.reduction}"
            )
        h_coarse = h // self.reduction
        w_coarse = w // self.reduction
        window_size = self.reduction * self.reduction
        windows = x.reshape(b, c, h_coarse, self.reduction, w_coarse, self.reduction)
        windows = windows.permute(0, 2, 4, 3, 5, 1).reshape(b, h_coarse, w_coarse, window_size, c)
        return windows, h_coarse, w_coarse, window_size

    def downsample(self, x: torch.Tensor) -> torch.Tensor:
        windows, h_coarse, w_coarse, window_size = self._split_windows(x)
        scores = self.selector(windows).squeeze(-1)
        weights = torch.softmax(scores / self.temperature, dim=3).unsqueeze(-1)
        coarse_seed = torch.sum(weights * windows, dim=3)  # B Hc Wc C
        dense_tokens = x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, self.channels)  # B N C
        coarse_tokens = coarse_seed.reshape(x.shape[0], -1, self.channels)  # B Nc C

        dense_logits = self.dense_mapper(dense_tokens)  # B N r
        dense_weights = torch.softmax(dense_logits.transpose(1, 2), dim=2)  # B r N
        basis = torch.bmm(dense_weights, dense_tokens)  # B r C

        coarse_logits = self.coarse_mapper(coarse_tokens)  # B Nc r
        coarse_weights = torch.softmax(coarse_logits.transpose(1, 2), dim=2)  # B r Nc
        coarse_out = torch.bmm(coarse_weights.transpose(1, 2), basis)  # B Nc C
        coarse = coarse_out.reshape(x.shape[0], h_coarse, w_coarse, self.channels).permute(0, 3, 1, 2)

        self.cached_dense_weights = dense_weights
        self.cached_coarse_weights = coarse_weights
        self.cached_shape = (x.shape[2], x.shape[3])
        self.cached_coarse_shape = (h_coarse, w_coarse)
        return coarse

    def upsample(self, x: torch.Tensor) -> torch.Tensor:
        if self.cached_dense_weights is None or self.cached_coarse_weights is None \
           or self.cached_shape is None or self.cached_coarse_shape is None:
            raise RuntimeError("Sampler cache is empty. Call `downsample` before `upsample`.")

        h, w = self.cached_shape
        h_coarse, w_coarse = self.cached_coarse_shape
        if x.shape[2] != h_coarse or x.shape[3] != w_coarse:
            raise ValueError("Input coarse feature spatial shape does not match cached shape.")

        coarse_tokens = x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, self.channels)  # B Nc C
        basis = torch.bmm(self.cached_coarse_weights, coarse_tokens)  # B r C
        dense_tokens = torch.bmm(self.cached_dense_weights.transpose(1, 2), basis)  # B N C
        dense = dense_tokens.reshape(x.shape[0], h, w, self.channels).permute(0, 3, 1, 2)

        self.cached_dense_weights = None
        self.cached_coarse_weights = None
        self.cached_shape = None
        self.cached_coarse_shape = None
        return dense

'''
class CrossAttentionSampler(ReusableSamplingCore):
    """
    Build coarse tokens and perform cross attention between coarse queries and
    dense keys/values. The resulting attention map acts as a global downsampling
    matrix that can be reused to upsample.
    """

    def __init__(self, channels: int, reduction: int = 2, heads: int = 4, temperature: float = 0.05):
        super().__init__()
        if reduction < 1:
            raise ValueError("Reduction factor must be >= 1.")
        self.channels = channels
        self.reduction = reduction
        self.temperature = temperature
        hidden_dim = max(channels // 2, 1)
        self.selector = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        self.heads = heads
        self.dim_head = channels // self.heads
        self.scale = self.dim_head ** -0.5
        # self.attention = Attention(hidden_dim, self.heads, self.dim_head, dropout=0.)
        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)
        self.reconstruct_proj = nn.Linear(channels, channels)
        self.cached_attn = None
        self.cached_shape = None
        self.cached_coarse_shape = None

    def _split_windows(self, x: torch.Tensor):
        b, c, h, w = x.shape
        if h % self.reduction != 0 or w % self.reduction != 0:
            raise ValueError(
                f"Input spatial shape ({h}, {w}) must be divisible by reduction factor {self.reduction}"
            )
        h_coarse = h // self.reduction
        w_coarse = w // self.reduction
        window_size = self.reduction * self.reduction
        windows = x.reshape(b, c, h_coarse, self.reduction, w_coarse, self.reduction)
        windows = windows.permute(0, 2, 4, 3, 5, 1).reshape(b, h_coarse, w_coarse, window_size, c)
        return windows, h_coarse, w_coarse, window_size

    def downsample(self, x: torch.Tensor) -> torch.Tensor:
        windows, h_coarse, w_coarse, window_size = self._split_windows(x)
        scores = self.selector(windows).squeeze(-1)
        weights = torch.softmax(scores / self.temperature, dim=3).unsqueeze(-1)
        coarse_seed = torch.sum(weights * windows, dim=3)  # B Hc Wc C
        tokens = coarse_seed.reshape(x.shape[0], h_coarse * w_coarse, self.channels)
        dense_tokens = x.reshape(x.shape[0], self.channels, -1).permute(0, 2, 1)  # B N C

        q = self.q_proj(tokens)
        k = self.k_proj(dense_tokens)
        v = self.v_proj(dense_tokens)

        q = q.reshape(x.shape[0], -1, self.heads, self.dim_head).permute(0, 2, 1, 3)
        k = k.reshape(x.shape[0], -1, self.heads, self.dim_head).permute(0, 2, 1, 3)
        v = v.reshape(x.shape[0], -1, self.heads, self.dim_head).permute(0, 2, 1, 3)

        attn = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) * self.scale, dim=-1)
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(x.shape[0], h_coarse * w_coarse, self.channels)
        out = self.out_proj(out)
        coarse = out.reshape(x.shape[0], h_coarse, w_coarse, self.channels).permute(0, 3, 1, 2)

        self.cached_attn = (q, k, v)  # 
        self.cached_shape = (x.shape[2], x.shape[3])
        self.cached_coarse_shape = (h_coarse, w_coarse)
        return coarse

    def upsample(self, x: torch.Tensor) -> torch.Tensor:
        if self.cached_attn is None or self.cached_shape is None or self.cached_coarse_shape is None:
            raise RuntimeError("Sampler cache is empty. Call `downsample` before `upsample`.")

        h, w = self.cached_shape
        h_coarse, w_coarse = self.cached_coarse_shape
        if x.shape[2] != h_coarse or x.shape[3] != w_coarse:
            raise ValueError("Input coarse feature spatial shape does not match cached shape.")

        tokens = x.permute(0, 2, 3, 1).reshape(x.shape[0], h_coarse * w_coarse, self.channels)
        tokens = self.reconstruct_proj(tokens)

        tokens = tokens.reshape(x.shape[0], -1, self.heads, self.dim_head).permute(0, 2, 1, 3)

        q, k, v = self.cached_attn
        attn = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) * self.scale, dim=-1)
        dense = torch.matmul(attn.transpose(-1, -2), tokens)

        dense = dense.permute(0, 2, 1, 3).reshape(x.shape[0], h * w, self.channels)
        dense = dense.permute(0, 2, 1).reshape(x.shape[0], self.channels, h, w)

        self.cached_attn = None
        self.cached_shape = None
        self.cached_coarse_shape = None
        return dense
'''