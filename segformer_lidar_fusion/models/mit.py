"""Mix Transformer (MiT) encoder — B0 variant.

Implements the hierarchical vision transformer backbone from
*SegFormer: Simple and Efficient Design for Semantic Segmentation with
Transformers* (Xie et al., NeurIPS 2021).

Key design choices
-------------------
* **Overlapping Patch Embedding** — uses stride < kernel so neighbouring
  patches share pixels, preserving local continuity.
* **Efficient Self-Attention** — spatial-reduction attention (SRA) reduces the
  K/V spatial resolution by ``sr_ratio``, cutting compute from O(N²) to
  O(N²/R²).
* **Mix-FFN** — replaces positional encoding with a 3×3 depth-wise
  convolution inside the feed-forward network, giving implicit positional
  information while remaining resolution-agnostic.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class OverlapPatchEmbed(nn.Module):
    """Overlapping patch embedding via strided convolution."""

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 32,
        patch_size: int = 7,
        stride: int = 4,
    ) -> None:
        super().__init__()
        padding = patch_size // 2
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=stride, padding=padding,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        x = self.proj(x)                       # (B, C, H, W)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)       # (B, N, C)
        x = self.norm(x)
        return x, H, W


class EfficientSelfAttention(nn.Module):
    """Multi-head self-attention with spatial reduction (SRA)."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 1,
        sr_ratio: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.sr_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        q = (
            self.q(x)
            .reshape(B, N, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )

        if self.sr_ratio > 1:
            x_sr = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_sr = self.sr(x_sr).reshape(B, C, -1).permute(0, 2, 1)
            x_sr = self.sr_norm(x_sr)
        else:
            x_sr = x

        kv = (
            self.kv(x_sr)
            .reshape(B, -1, 2, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MixFFN(nn.Module):
    """Feed-forward with 3×3 depth-wise conv for implicit position info."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_features = hidden_features or in_features * 4
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = nn.Conv2d(
            hidden_features, hidden_features,
            kernel_size=3, padding=1, groups=hidden_features,
        )
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = self.fc1(x)
        B, N, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Single transformer block: LayerNorm → SRA → LayerNorm → MixFFN."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        sr_ratio: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAttention(
            dim, num_heads=num_heads, sr_ratio=sr_ratio,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MixFFN(dim, int(dim * mlp_ratio), drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class DropPath(nn.Module):
    """Stochastic depth (drop path) regularisation."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = torch.rand(x.shape[0], 1, 1, device=x.device) >= self.drop_prob
        return x / (1.0 - self.drop_prob) * keep


# ---------------------------------------------------------------------------
# MiT-B0 backbone
# ---------------------------------------------------------------------------

class MixTransformerB0(nn.Module):
    """Mix Vision Transformer — B0 variant.

    Produces four feature maps at 1/4, 1/8, 1/16, 1/32 of input resolution
    with channel dimensions [32, 64, 160, 256].
    """

    # B0 hyper-parameters (from the paper)
    EMBED_DIMS: list[int]  = [32, 64, 160, 256]
    NUM_HEADS:  list[int]  = [1, 2, 5, 8]
    DEPTHS:     list[int]  = [2, 2, 2, 2]
    SR_RATIOS:  list[int]  = [8, 4, 2, 1]
    PATCH_SIZES: list[int] = [7, 3, 3, 3]
    STRIDES:    list[int]  = [4, 2, 2, 2]
    MLP_RATIO:  float      = 4.0

    def __init__(
        self,
        in_channels: int = 3,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.channels = list(self.EMBED_DIMS)

        total_depth = sum(self.DEPTHS)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]

        cur = 0
        for i in range(4):
            # Patch embedding
            patch_embed = OverlapPatchEmbed(
                in_channels=in_channels if i == 0 else self.EMBED_DIMS[i - 1],
                embed_dim=self.EMBED_DIMS[i],
                patch_size=self.PATCH_SIZES[i],
                stride=self.STRIDES[i],
            )
            # Transformer blocks
            blocks = nn.ModuleList([
                TransformerBlock(
                    dim=self.EMBED_DIMS[i],
                    num_heads=self.NUM_HEADS[i],
                    sr_ratio=self.SR_RATIOS[i],
                    mlp_ratio=self.MLP_RATIO,
                    drop=drop_rate,
                    drop_path=dpr[cur + j],
                )
                for j in range(self.DEPTHS[i])
            ])
            norm = nn.LayerNorm(self.EMBED_DIMS[i])

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"blocks{i + 1}", blocks)
            setattr(self, f"norm{i + 1}", norm)
            cur += self.DEPTHS[i]

        self.apply(self._init_weights)

    # ------------------------------------------------------------------
    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            nn.init.normal_(m.weight, 0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return multi-scale features [stage1 … stage4]."""
        outputs: list[torch.Tensor] = []
        for i in range(4):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            blocks = getattr(self, f"blocks{i + 1}")
            norm = getattr(self, f"norm{i + 1}")

            x, H, W = patch_embed(x)
            for blk in blocks:
                x = blk(x, H, W)
            x = norm(x)
            x = x.reshape(x.shape[0], H, W, -1).permute(0, 3, 1, 2)
            outputs.append(x)
        return outputs
