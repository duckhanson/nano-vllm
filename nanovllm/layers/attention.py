import torch
import torch.nn.functional as F
from torch import nn
import triton
import triton.language as tl

from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


def _gather_from_blocks(cache: torch.Tensor, block_table: torch.Tensor, block_size: int, length: int) -> torch.Tensor:
    """cache: (num_blocks, block_size, num_kv_heads, head_dim); returns (length, num_kv_heads, head_dim)."""
    n_blocks = (length + block_size - 1) // block_size
    blocks = cache[block_table[:n_blocks]]    # (n_blocks, block_size, num_kv_heads, head_dim)
    return blocks.reshape(-1, *cache.shape[-2:])[:length]


def _repeat_kv_heads(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """x: (kv_heads, seqlen, head_dim) -> (kv_heads * n_rep, seqlen, head_dim), GQA broadcast."""
    if n_rep == 1:
        return x
    return x.repeat_interleave(n_rep, dim=0)


class Attention(nn.Module):
    """ponytail: attention math is SDPA (torch.nn.functional.scaled_dot_product_attention),
    not flash-attn — flash-attn has no aarch64/Jetson wheel and from-source build was judged
    too slow/risky for M0 (see spec/2026-07-06-figure1-implementation-plan-design.md M0 notes).
    Per-sequence Python loop over the packed batch: not CUDA-graph-safe (uses .item()), so this
    only works with enforce_eager=True. No perf requirement for Figure 1 M0/M1 baseline correctness."""

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.n_rep = num_heads // num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def _prefill(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, context) -> torch.Tensor:
        block_size = self.k_cache.shape[1] if self.k_cache.numel() else 0
        n_seqs = context.cu_seqlens_q.numel() - 1
        outputs = []
        for i in range(n_seqs):
            qs, qe = context.cu_seqlens_q[i].item(), context.cu_seqlens_q[i + 1].item()
            ks, ke = context.cu_seqlens_k[i].item(), context.cu_seqlens_k[i + 1].item()
            seqlen_q, seqlen_k = qe - qs, ke - ks
            qi = q[qs:qe].transpose(0, 1)    # (num_heads, seqlen_q, head_dim)
            if context.block_tables is not None:    # prefix cache: read k/v back from the paged cache
                ki = _gather_from_blocks(self.k_cache, context.block_tables[i], block_size, seqlen_k).transpose(0, 1)
                vi = _gather_from_blocks(self.v_cache, context.block_tables[i], block_size, seqlen_k).transpose(0, 1)
            else:
                ki = k[ks:ke].transpose(0, 1)    # (num_kv_heads, seqlen_k, head_dim)
                vi = v[ks:ke].transpose(0, 1)
            ki, vi = _repeat_kv_heads(ki, self.n_rep), _repeat_kv_heads(vi, self.n_rep)
            if seqlen_q == seqlen_k:
                oi = F.scaled_dot_product_attention(qi.unsqueeze(0), ki.unsqueeze(0), vi.unsqueeze(0),
                                                     scale=self.scale, is_causal=True)
            else:
                offset = seqlen_k - seqlen_q
                q_idx = torch.arange(seqlen_q, device=q.device).unsqueeze(1) + offset
                k_idx = torch.arange(seqlen_k, device=q.device).unsqueeze(0)
                attn_mask = k_idx <= q_idx
                oi = F.scaled_dot_product_attention(qi.unsqueeze(0), ki.unsqueeze(0), vi.unsqueeze(0),
                                                     attn_mask=attn_mask, scale=self.scale)
            outputs.append(oi.squeeze(0).transpose(0, 1))    # (seqlen_q, num_heads, head_dim)
        return torch.cat(outputs, dim=0)

    def _decode(self, q: torch.Tensor, context) -> torch.Tensor:
        block_size = self.k_cache.shape[1]
        bs = q.shape[0]
        outputs = []
        for i in range(bs):
            clen = context.context_lens[i].item()
            ki = _gather_from_blocks(self.k_cache, context.block_tables[i], block_size, clen).transpose(0, 1)
            vi = _gather_from_blocks(self.v_cache, context.block_tables[i], block_size, clen).transpose(0, 1)
            ki, vi = _repeat_kv_heads(ki, self.n_rep), _repeat_kv_heads(vi, self.n_rep)
            qi = q[i].unsqueeze(1)    # (num_heads, 1, head_dim)
            oi = F.scaled_dot_product_attention(qi.unsqueeze(0), ki.unsqueeze(0), vi.unsqueeze(0), scale=self.scale)
            outputs.append(oi.squeeze(0).squeeze(1))    # (num_heads, head_dim)
        return torch.stack(outputs, dim=0)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            return self._prefill(q, k, v, context)
        else:    # decode
            return self._decode(q, context)
