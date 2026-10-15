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
    smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
    is_medium_freq = ~(wavelen < high_freq_wavelen) & ~(wavelen > low_freq_wavelen)
    return torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)


def _yarn_inv_freq(inv_freq: torch.Tensor, rope_scaling: dict, base: float) -> tuple[torch.Tensor, float]:
    """YaRN's frequencies and attention factor (Qwen3's long-context setting).
    Mirrors transformers.modeling_rope_utils._compute_yarn_parameters for the
    keys Qwen3 uses; the attention factor scales cos and sin."""
    for key in ("attention_factor", "mscale", "mscale_all_dim"):
        if rope_scaling.get(key) is not None:
            raise NotImplementedError(f"YaRN {key} is not supported")
    factor = rope_scaling["factor"]
    old_context_len = rope_scaling["original_max_position_embeddings"]
    beta_fast = rope_scaling.get("beta_fast") or 32
    beta_slow = rope_scaling.get("beta_slow") or 1
    dim = 2 * inv_freq.numel()

    def correction_dim(rotations):
        return dim * math.log(old_context_len / (rotations * 2 * math.pi)) / (2 * math.log(base))

    low, high = correction_dim(beta_fast), correction_dim(beta_slow)
    if rope_scaling.get("truncate", True):
        low, high = math.floor(low), math.ceil(high)
    low, high = max(low, 0), min(high, dim - 1)
    if low == high:
        high += 0.001
    ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low)).clamp(0, 1)
    extrapolation = 1 - ramp  # Share of each frequency left as trained.
    inv_freq = inv_freq / factor * (1 - extrapolation) + inv_freq * extrapolation
    attention_factor = 0.1 * math.log(factor) + 1.0 if factor > 1 else 1.0
    return inv_freq, attention_factor


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
        attention_factor = 1.0
        rope_type = (rope_scaling or {}).get("rope_type", "default")
        if rope_type == "llama3":
            inv_freq = _llama3_inv_freq(inv_freq, rope_scaling)
        elif rope_type == "yarn":
            inv_freq, attention_factor = _yarn_inv_freq(inv_freq, rope_scaling, base)
        elif rope_type != "default":
            raise NotImplementedError(f"rope_type {rope_type!r}")
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos() * attention_factor
        sin = freqs.sin() * attention_factor
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
