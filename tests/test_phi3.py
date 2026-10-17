"""Phi-4-mini's layers are built from its config and load its fused checkpoint (CPU only)."""

import tempfile

import pytest
import torch
import torch.distributed as dist
import transformers
from transformers import Phi3Config

from nanovllm.engine import model_runner
from nanovllm.layers.rotary_embedding import RotaryEmbedding
from nanovllm.models import phi3
from nanovllm.utils import loader

LONGROPE = {"rope_type": "longrope", "rope_theta": 10000.0, "partial_rotary_factor": 0.75,
            "original_max_position_embeddings": 128,
            "long_factor": [1.0 + 0.05 * i for i in range(48)], "short_factor": [1.0] * 48}


@pytest.fixture(scope="module", autouse=True)
def _single_rank_group():
    if not dist.is_initialized():
        store = tempfile.NamedTemporaryFile(delete=False).name
        dist.init_process_group("gloo", init_method=f"file://{store}", rank=0, world_size=1)
    yield


def _config():
    # Phi-4-mini's head shape (head_dim 128, 3 query heads per KV head, 96 rotary channels), shrunk elsewhere.
    return Phi3Config(hidden_size=384, intermediate_size=256, num_attention_heads=3, num_key_value_heads=1,
                      num_hidden_layers=2, max_position_embeddings=512, original_max_position_embeddings=128,
                      vocab_size=64, tie_word_embeddings=True, architectures=["Phi3ForCausalLM"],
                      pad_token_id=0, bos_token_id=0, eos_token_id=0, rope_parameters=dict(LONGROPE))


def test_rope_rotates_three_quarters_of_each_head_with_longrope():
    attn = phi3.Phi3DecoderLayer(_config()).self_attn
    assert attn.head_dim == 128 and attn.rotary_emb.rotary_dim == 96
    want = RotaryEmbedding(128, 96, 512, LONGROPE["rope_theta"], LONGROPE)
    torch.testing.assert_close(attn.rotary_emb.cos_sin_cache, want.cos_sin_cache)


def test_loads_transformers_fused_checkpoint_and_ties_lm_head():
    config = _config()
    torch.manual_seed(0)
    hf = transformers.Phi3ForCausalLM(config)
    with tempfile.TemporaryDirectory() as path:
        hf.save_pretrained(path)
        model = phi3.Phi3ForCausalLM(config)
        loader.load_model(model, path)
    ours = dict(model.named_parameters())
    for name, tensor in hf.state_dict().items():
        if name != "lm_head.weight":  # Tied: saved once, as the embedding.
            torch.testing.assert_close(ours[name], tensor, msg=name)
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()


def test_model_runner_knows_phi3():
    assert model_runner.get_model_cls(_config()) is phi3.Phi3ForCausalLM
