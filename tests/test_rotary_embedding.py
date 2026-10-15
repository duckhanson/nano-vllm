"""RoPE scaling in RotaryEmbedding against transformers' own parameters (CPU only)."""

import pytest
import torch
from transformers import Qwen3Config
from transformers import modeling_rope_utils

from nanovllm.layers.rotary_embedding import RotaryEmbedding

HEAD_DIM, POSITIONS, THETA = 128, 4096, 1_000_000.0
# Qwen3's documented YaRN setting (factor 4 over the 32K it was trained on).
YARN = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768, "rope_theta": THETA}


def _reference_cache(rope_parameters):
    config = Qwen3Config(head_dim=HEAD_DIM, max_position_embeddings=POSITIONS, rope_parameters=rope_parameters)
    inv_freq, attention_factor = modeling_rope_utils.ROPE_INIT_FUNCTIONS[rope_parameters["rope_type"]](config) \
        if rope_parameters["rope_type"] != "default" else (
            1.0 / THETA ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float) / HEAD_DIM), 1.0)
    freqs = torch.outer(torch.arange(POSITIONS, dtype=torch.float), inv_freq)
    return torch.cat((freqs.cos() * attention_factor, freqs.sin() * attention_factor), dim=-1).unsqueeze(1)


def test_yarn_matches_transformers_frequencies_and_attention_factor():
    rope = RotaryEmbedding(HEAD_DIM, HEAD_DIM, POSITIONS, THETA, YARN)
    torch.testing.assert_close(rope.cos_sin_cache, _reference_cache(YARN))
    # 0.1 ln 4 + 1: cos and sin carry YaRN's attention factor.
    assert rope.cos_sin_cache[0, 0, 0].item() == pytest.approx(1.1386, abs=1e-4)


def test_default_rope_type_is_unscaled():
    default = {"rope_type": "default", "rope_theta": THETA}
    torch.testing.assert_close(RotaryEmbedding(HEAD_DIM, HEAD_DIM, POSITIONS, THETA, default).cos_sin_cache,
                               _reference_cache(default))
    torch.testing.assert_close(RotaryEmbedding(HEAD_DIM, HEAD_DIM, POSITIONS, THETA).cos_sin_cache,
                               _reference_cache(default))


def test_unknown_rope_type_raises():
    with pytest.raises(NotImplementedError):
        RotaryEmbedding(HEAD_DIM, HEAD_DIM, POSITIONS, THETA, {"rope_type": "longrope", "rope_theta": THETA})
