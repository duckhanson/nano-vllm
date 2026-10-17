"""RoPE scaling in RotaryEmbedding against transformers' own parameters (CPU only)."""

import pytest
import torch
from transformers import Phi3Config
from transformers import Qwen3Config
from transformers import modeling_rope_utils
from transformers.models.phi3 import modeling_phi3

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
        RotaryEmbedding(HEAD_DIM, HEAD_DIM, POSITIONS, THETA, {"rope_type": "dynamic", "rope_theta": THETA})


# Phi-4-mini's shape: 3/4 of each head rotates; LongRoPE over a shorter pretrained length.
ROTARY_DIM, ORIGINAL = 96, 512
LONGROPE = {"rope_type": "longrope", "rope_theta": 10000.0, "partial_rotary_factor": 0.75,
            "original_max_position_embeddings": ORIGINAL,
            "long_factor": [1.0 + 0.1 * i for i in range(ROTARY_DIM // 2)],
            "short_factor": [1.0] * (ROTARY_DIM // 2)}


def _phi3_config():
    # Phi3Config keeps its own original_max_position_embeddings (default 4096) over rope_parameters'.
    return Phi3Config(hidden_size=2 * HEAD_DIM, num_attention_heads=2, max_position_embeddings=POSITIONS,
                      original_max_position_embeddings=ORIGINAL, rope_parameters=dict(LONGROPE))


def test_longrope_matches_transformers_long_factor_and_attention_factor():
    # transformers' long factor is the one it uses past original_max_position_embeddings.
    inv_freq, attention_factor = modeling_rope_utils.ROPE_INIT_FUNCTIONS["longrope"](
        _phi3_config(), seq_len=ORIGINAL + 1)
    freqs = torch.outer(torch.arange(POSITIONS, dtype=torch.float), inv_freq)
    want = torch.cat((freqs.cos() * attention_factor, freqs.sin() * attention_factor), dim=-1).unsqueeze(1)
    rope = RotaryEmbedding(HEAD_DIM, ROTARY_DIM, POSITIONS, LONGROPE["rope_theta"], LONGROPE)
    torch.testing.assert_close(rope.cos_sin_cache, want)
    # sqrt(1 + ln 8 / ln 512): the factor is max_position_embeddings over the pretrained length.
    assert rope.cos_sin_cache[0, 0, 0].item() == pytest.approx(1.1547, abs=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="forward is torch.compile'd; CPU inductor needs Python.h")
def test_partial_rotary_rotates_only_the_first_rotary_dim_channels_as_transformers():
    rope = RotaryEmbedding(HEAD_DIM, ROTARY_DIM, POSITIONS, LONGROPE["rope_theta"], LONGROPE).cuda()
    positions = torch.tensor([0, 7, 600, POSITIONS - 1], device="cuda")
    q, k = torch.randn(4, 2, HEAD_DIM, device="cuda"), torch.randn(4, 1, HEAD_DIM, device="cuda")
    got_q, got_k = rope(positions, q, k)
    cos, sin = rope.cos_sin_cache[positions, 0].chunk(2, dim=-1)
    # transformers' layout: [batch, heads, tokens, head_dim], cos and sin repeated over both halves.
    want_q, want_k = modeling_phi3.apply_rotary_pos_emb(
        q.transpose(0, 1)[None], k.transpose(0, 1)[None], torch.cat((cos, cos), -1)[None], torch.cat((sin, sin), -1)[None])
    torch.testing.assert_close(got_q, want_q[0].transpose(0, 1))
    torch.testing.assert_close(got_k, want_k[0].transpose(0, 1))
