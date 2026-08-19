from dataclasses import dataclass
from typing import Any
import torch


@dataclass(slots=True)
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    # Decode-mode only: the already-plan()ned FlashInfer wrapper every layer's
    # Attention._decode() should call .run() on this step (see
    # model_runner.py's plan_decode() call sites). None during prefill.
    decode_wrapper: Any | None = None
    # pass_offload/ADR-0031: is_prefill=True mode only. The already-plan()ned
    # CUDA-graph-enabled paged-prefill/append wrapper every layer's
    # Attention.forward() should call .run() on this step, for the mixed
    # decode+draft+verify batch (see model_runner.py's plan_mixed_extend()
    # call site). None for a genuine (non-mixed) prefill, in which case
    # Attention._prefill() plans its own wrapper eagerly per layer as before.
    extend_wrapper: Any | None = None
    # pass_offload/ADR-0033 memory layout redesign: two-level attention fix
    # for a row whose extension chunk shares a physical block with another
    # branch's chunk at a nonzero offset (see Attention._apply_chunk_fix()).
    # All four default None -- a round with zero chunk-fix rows leaves them
    # None and every existing call site (decode, mixed-extend, genuine
    # prefill, warmup) is unaffected. chunk_long_wrapper: the already-
    # plan()ned non-causal paged wrapper over each fixed-up row's own long
    # (real-block) region. chunk_short_wrapper: the already-plan()ned
    # causal ragged wrapper over each fixed-up row's own chunk region.
    # chunk_mask: bool, one entry per token in this round's flat q/output
    # tensor -- True for a token whose reach touches a shared, nonzero-
    # offset chunk and therefore needs its output overwritten by the fix.
    # chunk_slot_mapping: int32, RAGGED flat (not [num_rows, CHUNK_SIZE]
    # dense) -- exactly chunk_short_wrapper's own planned kv layout: each
    # fixed-up row contributes only its own valid_len physical slots (no
    # dummy padding), concatenated in row order, so
    # flat_k[context.chunk_slot_mapping] is directly usable as
    # chunk_short_wrapper.run()'s own kv argument with no further
    # reshaping/masking needed.
    chunk_long_wrapper: Any | None = None
    chunk_short_wrapper: Any | None = None
    chunk_mask: torch.Tensor | None = None
    chunk_slot_mapping: torch.Tensor | None = None
    # Plain Python bool, computed on the CPU side while chunk_mask is still
    # a Python list (before it becomes a CUDA tensor) -- real-hardware
    # regression found (2026-08-19): `context.chunk_mask.any()` on a CUDA
    # tensor forces a device-to-host sync, and Attention._apply_chunk_fix()
    # called it once per transformer layer (every round, even when no row
    # needs a fix) -- easily enough sync overhead per round to starve
    # propose_batch() of the few real draft rounds this codebase's tight
    # inter-swap-in-wave timing budget depends on (see
    # test_branch_out_verify_scheduling_soundness.py's own docstring on how
    # tight that budget already is). has_chunk_fix replaces every
    # `chunk_mask.any()` check -- sync-free by construction.
    has_chunk_fix: bool = False
    # Transient planning inputs for plan_chunk_long()/plan_chunk_short(),
    # computed by prepare_decode()/prepare_mixed_extend() and consumed by
    # ModelRunner.run_model() right before the model's forward pass --
    # exactly the same "prepare-computed, run_model-consumed" shape as
    # block_tables/context_lens themselves above, just scoped to the long/
    # chunk-region planning rather than the primary pass's own.
    # chunk_long_block_tables/chunk_long_context_lens: dense [num_rows,
    # max_len]/[num_rows], each row's own long-region block_row/context_len
    # (falls back to that row's own real, full values when it has no chunk
    # fix this round -- never read in that case, since chunk_mask is False
    # for all of its tokens, but must still be well-formed for plan_chunk_
    # long()'s own page-table math). chunk_kv_indptr: int32 [num_rows+1],
    # ragged KV indptr for chunk_short_wrapper's own plan -- one row's own
    # valid_len (0 for a row with no chunk fix this round). chunk_qo_indptr:
    # int32 [num_rows+1], ragged QUERY indptr for the same wrapper -- one
    # row's own MASKED POSITION COUNT (real-hardware regression, 2026-08-19:
    # this is NOT the same as valid_len in general -- a decode/draft row
    # has exactly one query position, masked as a whole whenever a chunk
    # fix applies regardless of how long the chunk's own content is, so its
    # qo_len is 1 while its kv_len (valid_len) can be up to CHUNK_SIZE; the
    # two only coincide for a verify/branch-out row's ragged multi-position
    # span, see Attention._apply_chunk_fix()'s own docstring).
    chunk_long_block_tables: torch.Tensor | None = None
    chunk_long_context_lens: torch.Tensor | None = None
    chunk_kv_indptr: torch.Tensor | None = None
    chunk_qo_indptr: torch.Tensor | None = None

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None, decode_wrapper=None, extend_wrapper=None, chunk_long_wrapper=None, chunk_short_wrapper=None, chunk_mask=None, chunk_slot_mapping=None, chunk_long_block_tables=None, chunk_long_context_lens=None, chunk_kv_indptr=None, chunk_qo_indptr=None, has_chunk_fix=False):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables, decode_wrapper, extend_wrapper, chunk_long_wrapper, chunk_short_wrapper, chunk_mask, chunk_slot_mapping, has_chunk_fix, chunk_long_block_tables, chunk_long_context_lens, chunk_kv_indptr, chunk_qo_indptr)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
