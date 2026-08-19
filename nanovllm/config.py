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

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
