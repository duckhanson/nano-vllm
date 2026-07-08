import torch
from torch import nn
import triton
import triton.language as tl
import flashinfer

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


_WORKSPACE_BYTES = 128 * 1024 * 1024
_prefill_wrapper = None
_decode_wrapper = None
_ragged_prefill_wrapper = None


def _get_prefill_wrapper(device):
    global _prefill_wrapper
    if _prefill_wrapper is None:
        buf = torch.empty(_WORKSPACE_BYTES, dtype=torch.uint8, device=device)
        _prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(buf, kv_layout="NHD")
    return _prefill_wrapper


def _get_decode_wrapper(device):
    global _decode_wrapper
    if _decode_wrapper is None:
        buf = torch.empty(_WORKSPACE_BYTES, dtype=torch.uint8, device=device)
        _decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(buf, kv_layout="NHD")
    return _decode_wrapper


def _get_ragged_prefill_wrapper(device):
    global _ragged_prefill_wrapper
    if _ragged_prefill_wrapper is None:
        buf = torch.empty(_WORKSPACE_BYTES, dtype=torch.uint8, device=device)
        _ragged_prefill_wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(buf, kv_layout="NHD")
    return _ragged_prefill_wrapper


def _paged_kv_meta(block_tables: torch.Tensor, kv_lens: torch.Tensor, block_size: int):
    """block_tables: (num_seqs, max_blocks) padded with -1, real block ids first (see
    ModelRunner.prepare_block_tables). Returns FlashInfer's paged-kv CSR triple."""
    num_pages = (kv_lens + block_size - 1) // block_size
    indptr = torch.zeros(kv_lens.numel() + 1, dtype=torch.int32, device=kv_lens.device)
    indptr[1:] = torch.cumsum(num_pages, dim=0)
    last_page_len = (kv_lens - (num_pages - 1) * block_size).to(torch.int32)
    indices = block_tables[block_tables >= 0].to(torch.int32)
    return indptr, indices, last_page_len


class Attention(nn.Module):
    """Attention math is FlashInfer's paged-KV-cache wrappers, reading/writing
    nano-vLLM's existing (num_blocks, block_size, num_kv_heads, head_dim) cache
    directly (it matches FlashInfer's NHD paged layout, no conversion needed).
    Only the warmup path (before allocate_kv_cache, no cache tensors yet) has no
    page table to read from; it uses the ragged (non-paged) prefill wrapper on
    q/k/v directly instead. GQA broadcast and causal masking (including the
    seqlen_q != seqlen_k prefix-cache case) are handled internally by FlashInfer.
    enforce_eager=True only: plan()/run() are host-synchronous, not CUDA-graph-safe."""

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
        self.k_cache = self.v_cache = torch.tensor([])

    def _prefill(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, context) -> torch.Tensor:
        if context.block_tables is None:    # warmup: no cache allocated yet
            wrapper = _get_ragged_prefill_wrapper(q.device)
            wrapper.plan(
                context.cu_seqlens_q, context.cu_seqlens_k,
                num_qo_heads=self.num_heads, num_kv_heads=self.num_kv_heads,
                head_dim_qk=self.head_dim, causal=True, sm_scale=self.scale,
                q_data_type=q.dtype, kv_data_type=k.dtype,
            )
            return wrapper.run(q, k, v)

        block_size = self.k_cache.shape[1]
        kv_lens = context.cu_seqlens_k[1:] - context.cu_seqlens_k[:-1]
        indptr, indices, last_page_len = _paged_kv_meta(context.block_tables, kv_lens, block_size)
        wrapper = _get_prefill_wrapper(q.device)
        wrapper.plan(
            context.cu_seqlens_q, indptr, indices, last_page_len,
            num_qo_heads=self.num_heads, num_kv_heads=self.num_kv_heads,
            head_dim_qk=self.head_dim, page_size=block_size, causal=True, sm_scale=self.scale,
            q_data_type=q.dtype, kv_data_type=self.k_cache.dtype,
        )
        return wrapper.run(q, (self.k_cache, self.v_cache))

    def _decode(self, q: torch.Tensor, context) -> torch.Tensor:
        block_size = self.k_cache.shape[1]
        indptr, indices, last_page_len = _paged_kv_meta(context.block_tables, context.context_lens, block_size)
        wrapper = _get_decode_wrapper(q.device)
        wrapper.plan(
            indptr, indices, last_page_len,
            num_qo_heads=self.num_heads, num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim, page_size=block_size, sm_scale=self.scale,
            q_data_type=q.dtype, kv_data_type=self.k_cache.dtype,
        )
        return wrapper.run(q, (self.k_cache, self.v_cache))

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            return self._prefill(q, k, v, context)
        else:    # decode
            return self._decode(q, context)
