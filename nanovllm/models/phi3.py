import torch
from torch import nn
import torch.distributed as dist
from transformers import Phi3Config

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import ReplicatedLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


class Phi3Attention(nn.Module):

    def __init__(self, config: Phi3Config) -> None:
        super().__init__()
        # The checkpoint's qkv_proj is already fused, so it loads whole: one rank only.
        assert dist.get_world_size() == 1
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", None) or config.hidden_size // self.num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.qkv_proj = ReplicatedLinear(config.hidden_size, self.q_size + 2 * self.kv_size, bias=False)
        self.o_proj = RowParallelLinear(self.q_size, config.hidden_size, bias=False)
        rope_parameters = config.rope_parameters
        # No default theta: a missing one is a config error, not 10000 (ADR 0007).
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=int(self.head_dim * rope_parameters.get("partial_rotary_factor", 1.0)),
            max_position=config.max_position_embeddings,
            base=rope_parameters["rope_theta"],
            rope_scaling=rope_parameters,
        )
        self.attn = Attention(self.num_heads, self.head_dim, self.head_dim ** -0.5, self.num_kv_heads)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        q, k, v = self.qkv_proj(hidden_states).split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q.view(-1, self.num_heads, self.head_dim),
                               k.view(-1, self.num_kv_heads, self.head_dim))
        o = self.attn(q, k, v.view(-1, self.num_kv_heads, self.head_dim))
        return self.o_proj(o.flatten(1, -1))


class Phi3MLP(nn.Module):

    def __init__(self, config: Phi3Config) -> None:
        super().__init__()
        # Fused [gate, up] in the checkpoint, the order SiluAndMul takes.
        self.gate_up_proj = ReplicatedLinear(config.hidden_size, 2 * config.intermediate_size, bias=False)
        self.down_proj = RowParallelLinear(config.intermediate_size, config.hidden_size, bias=False)
        assert config.hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_up_proj(x)))


class Phi3DecoderLayer(nn.Module):

    def __init__(self, config: Phi3Config) -> None:
        super().__init__()
        self.self_attn = Phi3Attention(config)
        self.mlp = Phi3MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Phi3Model(nn.Module):

    def __init__(self, config: Phi3Config) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Phi3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Phi3ForCausalLM(nn.Module):

    def __init__(self, config: Phi3Config) -> None:
        super().__init__()
        self.model = Phi3Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)
