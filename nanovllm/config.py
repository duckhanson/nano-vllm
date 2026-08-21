import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    # pass_offload (2026-08-19): CUDA-graph row-count capacity is a distinct
    # concept from max_num_seqs -- max_num_seqs bounds concurrent *jobs*
    # (nanovllm/engine/scheduler.py's own admission loops), while this bounds
    # concurrent *rows* in one forward pass. These coincided under the
    # historical one-row-per-job invariant, which is why
    # ModelRunner.capture_cudagraph() used to derive its row capacity
    # directly from max_num_seqs -- pass_offload's branch-out-verify breaks
    # that invariant (a job can contribute its own draft/verify row AND an
    # independent branch-out-verify checkpoint row in the same round, plus
    # each live fork adds another row), so row capacity needs its own knob.
    # None (default) preserves the exact previous behavior (derive from
    # max_num_seqs) for every existing single-branch caller; set explicitly
    # per experiment once real branch-count headroom is needed.
    max_graph_rows: int | None = None
    # pass_offload (2026-08-20, direction 3 of ADR-0033's mixed-extend
    # eager-fallback investigation): the mixed decode+draft+verify+branch-out
    # extend path's own row-count ceiling, decoupled from max_graph_rows/
    # max_bs the same way max_graph_rows was decoupled from max_num_seqs --
    # branch-out verify's checkpoint/fork rows can push a round's row count
    # past max_bs even when max_bs itself is sized correctly for ordinary
    # per-job decode/draft/verify rows, and forcing that round to fall back
    # to eager execution (rather than replaying a captured graph) changes its
    # timing shape, not just its safety -- undesirable for a benchmark whose
    # own gantt needs graph-replay-representative timing. None (default)
    # preserves the exact previous behavior (mixed-extend graphs captured at
    # a single row-count ceiling == max_bs, same as before this field
    # existed) for every existing caller.
    max_extend_rows: int | None = None
    # pass_offload/ADR-0036: attention's paged-read granularity, decoupled
    # from kvcache_block_size (the real KV cache's allocation/I/O/prefix-hash
    # unit, unchanged). A page id is block_id * (kvcache_block_size //
    # kv_page_size) + intra_block_page_index -- a pure re-numbering of
    # existing physical blocks, no new storage or allocation. None (default)
    # preserves the exact previous behavior (page size == kvcache_block_size,
    # i.e. today's ordinary paged attention) for every caller that doesn't
    # set it. This is also pass_offload's own branch/checkpoint chunk size
    # (ADR-0035's CHUNK_SIZE) -- not an independent knob that happens to
    # match: a chunk must start on a page boundary and fit within one page
    # for ADR-0036's "every page is self-contained, no cross-page offset"
    # property to hold, so the two concerns share this single value by
    # construction, not by convention.
    kv_page_size: int | None = None

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        if self.kv_page_size is not None:
            assert self.kvcache_block_size % self.kv_page_size == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
