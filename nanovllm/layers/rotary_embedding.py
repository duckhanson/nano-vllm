import math
from functools import lru_cache
import torch
from torch import nn


def _llama3_inv_freq(inv_freq: torch.Tensor, rope_scaling: dict) -> torch.Tensor:
    """Llama 3.1-style RoPE frequency adjustment (long-context extension).
    Mirrors transformers.modeling_rope_utils._compute_llama3_parameters —
    reimplemented here since nano-vLLM's RotaryEmbedding has no scaling support."""
    factor = rope_scaling["factor"]
    low_freq_factor = rope_scaling["low_freq_factor"]
    high_freq_factor = rope_scaling["high_freq_factor"]
    old_context_len = rope_scaling["original_max_position_embeddings"]

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    wavelen = 2 * math.pi / inv_freq

    inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed_inv_freq = smooth_factor * inv_freq_llama / factor + (1 - smooth_factor) * inv_freq_llama
    is_medium_freq = ~(wavelen < high_freq_wavelen) & ~(wavelen > low_freq_wavelen)
    return torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        rope_scaling: dict | None = None,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        if rope_scaling is not None and rope_scaling.get("rope_type") == "llama3":
            inv_freq = _llama3_inv_freq(inv_freq, rope_scaling)
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


@lru_cache(1)
def _get_rope_cached(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling_items: tuple | None,
):
    rope_scaling = dict(rope_scaling_items) if rope_scaling_items is not None else None
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base, rope_scaling)
    return rotary_emb


def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
):
    rope_scaling_items = tuple(sorted(rope_scaling.items())) if rope_scaling is not None else None
    return _get_rope_cached(head_size, rotary_dim, max_position, base, rope_scaling_items)
