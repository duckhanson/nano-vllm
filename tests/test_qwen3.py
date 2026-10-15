"""Qwen3's layers are built from its config: AWQ, QK-norm and RoPE parameters (CPU only)."""

import tempfile

import pytest
import torch
import torch.distributed as dist
from transformers import Qwen3Config

from nanovllm.layers.rotary_embedding import RotaryEmbedding
from nanovllm.models import qwen3

AWQ = {"quant_method": "awq", "bits": 4, "group_size": 128, "zero_point": True, "version": "gemm"}
YARN = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768, "rope_theta": 1_000_000.0}


@pytest.fixture(scope="module", autouse=True)
def _single_rank_group():
    if not dist.is_initialized():
        store = tempfile.NamedTemporaryFile(delete=False).name
        dist.init_process_group("gloo", init_method=f"file://{store}", rank=0, world_size=1)
    yield


def _config(**kwargs):
    # Qwen3-8B's head shape, shrunk elsewhere.
    return Qwen3Config(hidden_size=256, intermediate_size=256, num_attention_heads=2, num_key_value_heads=1,
                       head_dim=128, num_hidden_layers=1, max_position_embeddings=512,
                       rope_parameters=kwargs.pop("rope_parameters", YARN), **kwargs)


def test_awq_checkpoint_quantizes_all_four_linears():
    layer = qwen3.Qwen3DecoderLayer(_config(quantization_config=AWQ))
    for linear in (layer.self_attn.qkv_proj, layer.self_attn.o_proj, layer.mlp.gate_up_proj, layer.mlp.down_proj):
        assert linear.weight is None and linear.qweight is not None
    plain = qwen3.Qwen3DecoderLayer(_config())
    assert plain.self_attn.qkv_proj.weight is not None


@pytest.mark.parametrize("bias", [False, True])
def test_qk_norm_is_built_with_or_without_attention_bias(bias):
    # transformers' Qwen3Attention always has q_norm and k_norm; attention_bias only adds the qkv bias.
    attn = qwen3.Qwen3DecoderLayer(_config(attention_bias=bias)).self_attn
    assert isinstance(attn.q_norm, torch.nn.Module) and isinstance(attn.k_norm, torch.nn.Module)
    assert (attn.qkv_proj.bias is not None) == bias


def test_rope_takes_theta_and_yarn_from_rope_parameters():
    rope = qwen3.Qwen3DecoderLayer(_config()).self_attn.rotary_emb
    want = RotaryEmbedding(128, 128, 512, YARN["rope_theta"], YARN)
    torch.testing.assert_close(rope.cos_sin_cache, want.cos_sin_cache)


def test_missing_rope_theta_raises_instead_of_defaulting():
    config = _config(rope_parameters={"rope_type": "default", "rope_theta": 1.0})
    del config.rope_parameters["rope_theta"]
    with pytest.raises(KeyError):
        qwen3.Qwen3DecoderLayer(config)
