import pickle
import flashinfer
import torch
import torch.distributed as dist
import torch.nn.functional as F
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.models.llama import LlamaForCausalLM
from nanovllm.layers.attention import WORKSPACE_BYTES, plan_decode, plan_mixed_extend
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model
from nanovllm.utils.mapped_kv_cache import allocate_mapped_kv_cache


MODEL_REGISTRY = {
    "Qwen3ForCausalLM": Qwen3ForCausalLM,
    "LlamaForCausalLM": LlamaForCausalLM,
}


def get_model_cls(hf_config):
    for arch in hf_config.architectures:
        if arch in MODEL_REGISTRY:
            return MODEL_REGISTRY[arch]
    raise ValueError(f"Unsupported model architectures: {hf_config.architectures}")


def _bucket_ladder(ceiling: int) -> list[int]:
    """Fine-grained low end (1, 2, 4, 8), then 16-wide steps up to `ceiling`,
    with `ceiling` itself always included -- a bucket lookup for a real value
    == ceiling always has a candidate (2026-08-05 real-hardware fix, see
    capture_cudagraph()'s own decode-bucket comment for the StopIteration
    this avoids). Shared by decode's row-count buckets and the mixed-extend
    path's token-count buckets (pass_offload/ADR-0031 follow-up) -- same
    ladder shape, different axis."""
    return sorted(
        set(b for b in (1, 2, 4, 8) if b <= ceiling)
        | {ceiling}
        | set(range(16, ceiling + 1, 16))
    )


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        self.model = get_model_cls(hf_config)(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool, self.graph_decode_wrappers
            del self.graphs_extend, self.graph_extend_wrappers
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        # Built BEFORE the free-memory measurement below, so its 128MB
        # workspace allocation is actually counted against num_kvcache_blocks'
        # budget -- constructing it after (as an earlier version of this
        # method did) silently left num_kvcache_blocks over-sized relative to
        # what was really still free, since the measurement it's based on
        # would predate this allocation (2026-07-25 real-hardware finding:
        # this was enough to make a tight-margin real workload -- bm13's
        # sequential-real-PG19-groups precompute -- run out of free blocks
        # with `IndexError: pop from an empty deque`, once enforce_eager=False
        # also started reserving graph-pool memory on top of this).
        self._decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            torch.empty(WORKSPACE_BYTES, dtype=torch.uint8, device="cuda"), kv_layout="NHD",
        )
        # pass_offload/ADR-0031: eager singleton for the mixed decode+draft+
        # verify batch -- same convention as _decode_wrapper above, built
        # here (not lazily, unlike attention.py's own _get_prefill_wrapper())
        # so its 128MB workspace is also counted against the budget
        # measurement below, same reasoning as _decode_wrapper's own comment.
        self._extend_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            torch.empty(WORKSPACE_BYTES, dtype=torch.uint8, device="cuda"), kv_layout="NHD",
        )
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        kv_cache_shape = (2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        self.kv_cache, self.kv_cache_cpu_view, self.kv_cache_shm_name = allocate_mapped_kv_cache(kv_cache_shape, hf_config.dtype)
        layer_id = 0
        attn_module = None
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                attn_module = module
                layer_id += 1

        # Every layer's Attention is constructed with the same num_heads/
        # num_kv_heads/head_dim/scale (see e.g. LlamaAttention.__init__) --
        # decode planning only needs one copy of these, read from whichever
        # layer's module the loop above last touched (2026-07-25: this is
        # what makes plan_decode() callable once per decode step instead of
        # once per layer -- see model_runner's run_model()/capture_cudagraph()
        # and attention.py's own docstring).
        self._decode_plan_kwargs = dict(
            num_qo_heads=attn_module.num_heads, num_kv_heads=attn_module.num_kv_heads,
            head_dim=attn_module.head_dim, sm_scale=attn_module.scale,
            q_dtype=hf_config.dtype, kv_dtype=hf_config.dtype,
        )

    def _block_table_row(self, seq: Sequence) -> list[int]:
        # pass_offload/ADR-0030: a seq drafting this round (see
        # prepare_decode()'s own comment) carries its own compacted,
        # ready-set block row instead of its real (possibly not-yet-fully-
        # resident) block_table -- duck-typed, this file never imports the
        # attribute's type.
        # pass_offload/ADR-0031: a seq verifying this round instead carries
        # its own full block_table + extension blocks via
        # `_verify_context_override` -- checked first since a row is never
        # both drafting and verifying in the same round.
        verify_override = getattr(seq, "_verify_context_override", None)
        if verify_override is not None:
            return verify_override.block_row
        override = getattr(seq, "_draft_context_override", None)
        return override.block_row if override is not None else seq.block_table

    def prepare_block_tables(self, seqs: list[Sequence]):
        rows = [self._block_table_row(seq) for seq in seqs]
        max_len = max(len(row) for row in rows)
        block_tables = [row + [-1] * (max_len - len(row)) for row in rows]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        if seqs[0].block_table:    # empty only during warmup_model(), before any block allocation
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        # pass_offload/ADR-0030: a seq may carry a `_draft_context_override`
        # (input_token/position/slot/context_len/block_row) for this one
        # call -- e.g. a still-resuming job drafting off its own compacted,
        # ready-set KV view instead of its real (not-yet-fully-resident)
        # one, sharing this same batched decode call with genuinely running
        # seqs. This file has no concept of "draft" itself -- it only ever
        # reads whichever of these two shapes a seq presents; see run()'s
        # own _sample() for the matching split on the sampling side (a
        # draft row must never share a real seq's sampler call -- see that
        # method's own docstring for why).
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            override = getattr(seq, "_draft_context_override", None)
            if override is None:
                input_ids.append(seq.last_token)
                positions.append(len(seq) - 1)
                context_lens.append(len(seq))
                slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)
            else:
                input_ids.append(override.input_token)
                positions.append(override.position)
                context_lens.append(override.context_len)
                slot_mapping.append(override.slot)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_mixed_extend(self, seqs: list[Sequence]):
        """pass_offload/ADR-0031: builds one ragged batch spanning three row
        shapes -- real decode (query_len=1, full KV, no override), draft
        continuation (query_len=1, sparse ready-set KV,
        `_draft_context_override`), and verify (query_len=N+1, full
        KV+extension, `_verify_context_override`) -- for FlashInfer's
        paged-prefill/append wrapper. Generalizes prepare_decode()'s per-row
        override duck-typing (decode vs draft) and prepare_prefill()'s
        ragged cu_seqlens_q flattening (verify's own multi-token span) into
        one. This file has no concept of *why* a row is one shape or
        another -- see proposer.py/offload_scheduler.py for that. `run()`
        decides to call this instead of prepare_decode()/prepare_prefill()
        purely by duck-typing whether any seq in `seqs` carries a
        `_verify_context_override` (ADR-0031's own resolution: no explicit
        mode parameter threaded through step()/call())."""
        input_ids = []
        positions = []
        slot_mapping = []
        cu_seqlens_q = [0]
        context_lens = []
        for seq in seqs:
            verify = getattr(seq, "_verify_context_override", None)
            if verify is not None:
                input_ids.extend(verify.input_tokens)
                positions.extend(verify.positions)
                slot_mapping.extend(verify.slot_mapping)
                cu_seqlens_q.append(cu_seqlens_q[-1] + len(verify.input_tokens))
                context_lens.append(verify.num_cached_tokens + len(verify.input_tokens))
                continue
            draft = getattr(seq, "_draft_context_override", None)
            if draft is not None:
                input_ids.append(draft.input_token)
                positions.append(draft.position)
                slot_mapping.append(draft.slot)
                context_lens.append(draft.context_len)
            else:
                input_ids.append(seq.last_token)
                positions.append(len(seq) - 1)
                slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)
                context_lens.append(len(seq))
            cu_seqlens_q.append(cu_seqlens_q[-1] + 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(True, cu_seqlens_q=cu_seqlens_q, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool, is_mixed_extend: bool = False):
        if is_mixed_extend:
            context = get_context()
            total_tokens = input_ids.size(0)
            if self.enforce_eager or total_tokens > self.graph_extend_tokens[-1]:
                # pass_offload/ADR-0031: eager fallback -- enforce_eager, or a
                # round whose total flattened query-token count exceeds even
                # this path's largest captured bucket (mirrors the decode
                # branch's own `input_ids.size(0) > 512` eager fallback
                # below). One plan() per round (not per layer), mirroring
                # plan_decode()'s existing hoist -- see plan_mixed_extend()'s
                # own docstring.
                plan_mixed_extend(
                    self._extend_wrapper, context.cu_seqlens_q, context.block_tables, context.context_lens,
                    self.block_size, **self._decode_plan_kwargs,
                )
                context.extend_wrapper = self._extend_wrapper
                hidden = self.model(input_ids, positions)
            else:
                # pass_offload/ADR-0031 follow-up: CUDA-graph bucketed by
                # total flattened query-token count, not row count (a verify
                # row's own N has no fixed ceiling the way row count does --
                # ADR-0031 decision 5). Row count is fixed at this path's own
                # `max_bs` (capture_cudagraph()) across every bucket; real
                # rows (this round's own seqs order) come first, padding rows
                # after get zero query length + zero KV context.
                bucket_t = next(x for x in self.graph_extend_tokens if x >= total_tokens)
                graph = self.graphs_extend[bucket_t]
                extend_wrapper = self.graph_extend_wrappers[bucket_t]
                graph_vars = self.graph_extend_vars
                num_rows = context.block_tables.size(0)

                graph_vars["input_ids"][:total_tokens] = input_ids
                graph_vars["positions"][:total_tokens] = positions
                # -1 ("no real token here") so store_kvcache_kernel's own
                # `if slot == -1: return` no-ops every padding token instead
                # of corrupting real KV-cache memory (mirrors the decode
                # branch's identical slot_mapping.fill_(-1) below).
                graph_vars["slot_mapping"].fill_(-1)
                graph_vars["slot_mapping"][:total_tokens] = context.slot_mapping
                graph_vars["qo_indptr"].fill_(total_tokens)
                graph_vars["qo_indptr"][:num_rows + 1] = context.cu_seqlens_q
                graph_vars["context_lens"].zero_()
                graph_vars["context_lens"][:num_rows] = context.context_lens
                graph_vars["block_tables"].fill_(-1)
                graph_vars["block_tables"][:num_rows, :context.block_tables.size(1)] = context.block_tables
                plan_mixed_extend(
                    extend_wrapper, graph_vars["qo_indptr"], graph_vars["block_tables"], graph_vars["context_lens"],
                    self.block_size, **self._decode_plan_kwargs,
                )
                context.extend_wrapper = extend_wrapper
                graph.replay()
                hidden = graph_vars["outputs"][:total_tokens]
            # Bypasses compute_logits()/ParallelLMHead's prefill-mode
            # last-token-only slicing -- wrong here, since verify rows need
            # every position's logits, not just each row's last. Safe under
            # tp_size==1 (this project's own config; see proposer.py's
            # verify_batch(), which already uses this identical bypass
            # today, being absorbed into this shared path).
            return F.linear(hidden, self.model.lm_head.weight)
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            if not is_prefill:
                # One plan() per decode step (not per layer -- see
                # attention.py's plan_decode() docstring); every layer's
                # Attention._decode() reads this same freshly-planned
                # wrapper off context.decode_wrapper.
                context = get_context()
                plan_decode(
                    self._decode_wrapper, context.block_tables, context.context_lens,
                    self.block_size, **self._decode_plan_kwargs,
                )
                context.decode_wrapper = self._decode_wrapper
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            bucket_bs = next(x for x in self.graph_bs if x >= bs)
            graph = self.graphs[bucket_bs]
            decode_wrapper = self.graph_decode_wrappers[bucket_bs]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            # Padding rows (bs..bucket_bs-1) must read as zero pages, not
            # whatever a previous, possibly-larger call left behind -- -1 is
            # this codebase's own "no block here" convention (see
            # prepare_block_tables()); plan_decode() below reads the FULL
            # bucket_bs-wide tensor (not just [:bs]), since the wrapper's
            # FlashInfer-fixed batch size is bucket_bs, so stale positive
            # block ids in the unused rows would otherwise get silently
            # counted as real pages by _paged_kv_meta()'s `>= 0` filter.
            graph_vars["block_tables"].fill_(-1)
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            # Sliced to :bucket_bs, not the full max_bs-wide graph_vars --
            # the wrapper's own FlashInfer-fixed batch size is bucket_bs.
            plan_decode(
                decode_wrapper, graph_vars["block_tables"][:bucket_bs], graph_vars["context_lens"][:bucket_bs],
                self.block_size, **self._decode_plan_kwargs,
            )
            context.decode_wrapper = decode_wrapper
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def _sample(self, seqs: list[Sequence], logits: torch.Tensor) -> list[list[int]]:
        # pass_offload/ADR-0030/0031: rows carrying a `_draft_context_override`
        # or `_verify_context_override` must be argmax'd directly off
        # `logits`, never through self.sampler() -- routing every row through
        # one shared sampler call regardless would change that call's batch
        # shape/size, consuming a different number of random draws from the
        # same RNG stream even for the genuinely running rows in the same
        # call. That would silently break their bit-exact match against a
        # non-fused (vanilla) round using the same seed -- the soundness
        # invariant self-spec's own real-hardware gates (SM5/SM9/SM10) depend
        # on. Slicing the running rows out first keeps their own sampler()
        # call identical in shape/order to what a round with no draft/verify
        # rows at all would have produced. Return shape generalizes to one
        # list of token ids per row (ADR-0031 decision 6): length 1 for
        # decode/draft rows, length N+1 for a verify row's own whole span.
        verify_idx = [i for i, seq in enumerate(seqs) if getattr(seq, "_verify_context_override", None) is not None]
        if not verify_idx:
            # No verify row this round -- logits rows are 1:1 with `seqs`
            # indices (every row is query_len=1); ADR-0030's own decode/draft
            # split, unchanged.
            draft_idx = [i for i, seq in enumerate(seqs) if getattr(seq, "_draft_context_override", None) is not None]
            if not draft_idx:
                temperatures = self.prepare_sample(seqs)
                flat = self.sampler(logits, temperatures).tolist()
            else:
                draft_set = set(draft_idx)
                decode_idx = [i for i in range(len(seqs)) if i not in draft_set]
                flat = [0] * len(seqs)
                if decode_idx:
                    decode_seqs = [seqs[i] for i in decode_idx]
                    temperatures = self.prepare_sample(decode_seqs)
                    decode_tokens = self.sampler(logits[decode_idx], temperatures).tolist()
                    for i, token_id in zip(decode_idx, decode_tokens):
                        flat[i] = token_id
                draft_tokens = logits[draft_idx].float().argmax(dim=-1).tolist()
                for i, token_id in zip(draft_idx, draft_tokens):
                    flat[i] = token_id
            return [[t] for t in flat]

        # pass_offload/ADR-0031: at least one verify row (query_len=N+1) is
        # present -- logits rows are no longer 1:1 with `seqs` indices, so
        # every row (decode/draft/verify alike) is sliced by its own
        # (start, end) span within the flattened logits, recomputed here the
        # same way prepare_mixed_extend() built cu_seqlens_q, in seqs order.
        verify_set = set(verify_idx)
        spans = []
        cum = 0
        for seq in seqs:
            verify = getattr(seq, "_verify_context_override", None)
            n = len(verify.input_tokens) if verify is not None else 1
            spans.append((cum, cum + n))
            cum += n

        decode_rows = [
            i for i in range(len(seqs))
            if i not in verify_set and getattr(seqs[i], "_draft_context_override", None) is None
        ]
        token_ids: list[list[int]] = [[] for _ in seqs]

        if decode_rows:
            decode_seqs = [seqs[i] for i in decode_rows]
            temperatures = self.prepare_sample(decode_seqs)
            decode_logit_idx = [spans[i][0] for i in decode_rows]
            decode_tokens = self.sampler(logits[decode_logit_idx], temperatures).tolist()
            for i, token_id in zip(decode_rows, decode_tokens):
                token_ids[i] = [token_id]

        for i, seq in enumerate(seqs):
            if i in verify_set:
                start, end = spans[i]
                token_ids[i] = [int(t) for t in logits[start:end].float().argmax(dim=-1).tolist()]
            elif getattr(seq, "_draft_context_override", None) is not None:
                start, _ = spans[i]
                token_ids[i] = [int(logits[start].float().argmax(dim=-1))]
        return token_ids

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[list[int]]:
        # pass_offload/ADR-0031: dispatch to the mixed decode+draft+verify
        # path purely by duck-typing whether any seq carries a
        # `_verify_context_override` -- no explicit mode parameter threaded
        # through step()/call() (this ADR's own resolution). A round with no
        # verify row falls through unchanged to the existing ADR-0030
        # decode+draft-only path.
        if any(getattr(seq, "_verify_context_override", None) is not None for seq in seqs):
            input_ids, positions = self.prepare_mixed_extend(seqs)
            logits = self.run_model(input_ids, positions, True, is_mixed_extend=True)
        else:
            input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
            logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self._sample(seqs, logits) if self.rank == 0 else None
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        # -1 ("no block here", this codebase's own convention -- see
        # prepare_block_tables()), not 0 -- 0 is a valid block id, and
        # _paged_kv_meta()'s `>= 0` filter would otherwise silently count
        # every unused row's leftover entries as real pages once
        # plan_decode() starts reading this tensor for real (2026-07-25;
        # previously never mattered, since nothing ever called
        # _paged_kv_meta()/plan() against this dummy warmup/capture data
        # before FlashInfer's plan()/run() split existed here -- see
        # run_model()'s own graph branch for the equivalent replay-time fix).
        block_tables = torch.full((max_bs, max_num_blocks), -1, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        # `max_bs` itself is always included -- [1, 2, 4, 8] + range(16, ...)
        # alone leaves a gap for any max_bs in [9, 15] (nothing >= 9 and <
        # 16), which run_model()'s own `next(x for x in self.graph_bs if x >=
        # bs)` bucket lookup then has no candidate for once a real batch size
        # lands there -- StopIteration (2026-08-05 real-hardware finding,
        # pass_offload's own bs=6 sweep, max_num_seqs=12). The fixed prefix
        # must itself be filtered to <= max_bs (2026-08-05 follow-up,
        # max_num_seqs=4 crash): every buffer above (input_ids, positions,
        # ...) is only max_bs wide, so a bucket bigger than max_bs would
        # slice past the real batch and hand FlashInfer's paged-decode
        # wrapper a row count that doesn't match what it was constructed
        # for -- "batch size ... mismatches the batch size set during
        # initialization" from flashinfer/decode.py's own plan().
        self.graph_bs = _bucket_ladder(max_bs)
        self.graphs = {}
        self.graph_pool = None
        self.graph_decode_wrappers = {}

        # One shared 128MB scratch buffer across every bucket's wrapper --
        # matches the eager singleton's own sizing (allocate_kv_cache()) --
        # only the tiny indptr/indices/last_page_len buffers need to be
        # per-bucket (FlashInfer's CUDAGraphBatchDecodeWithPagedKVCacheWrapper
        # fixes batch_size to whatever paged_kv_last_page_len_buffer's own
        # length is, for that wrapper instance's whole lifetime).
        decode_workspace_buffer = torch.empty(WORKSPACE_BYTES, dtype=torch.uint8, device="cuda")

        for bs in reversed(self.graph_bs):
            indptr_buffer = torch.zeros(bs + 1, dtype=torch.int32, device="cuda")
            indices_buffer = torch.zeros(bs * max_num_blocks, dtype=torch.int32, device="cuda")
            last_page_len_buffer = torch.zeros(bs, dtype=torch.int32, device="cuda")
            decode_wrapper = flashinfer.CUDAGraphBatchDecodeWithPagedKVCacheWrapper(
                decode_workspace_buffer, indptr_buffer, indices_buffer, last_page_len_buffer, kv_layout="NHD",
            )
            self.graph_decode_wrappers[bs] = decode_wrapper

            graph = torch.cuda.CUDAGraph()
            set_context(
                False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs],
                block_tables=block_tables[:bs], decode_wrapper=decode_wrapper,
            )
            # plan() itself is never captured (FlashInfer disallows it, see
            # attention.py's docstring) -- called once here, eagerly, before
            # both the warmup and capture forward passes below, which reuse
            # this same planned state (identical dummy data either way, so
            # nothing needs re-planning between them).
            plan_decode(decode_wrapper, block_tables[:bs], context_lens[:bs], self.block_size, **self._decode_plan_kwargs)
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        # pass_offload/ADR-0031 follow-up: CUDA-graph buckets for the mixed
        # decode+draft+verify extend path, bucketed by total flattened
        # query-token count (a verify row's own N has no fixed ceiling the
        # way row count does -- ADR-0031 decision 5), not row count. Row
        # count is fixed at `max_bs` across every one of these buckets (same
        # value decode's own buckets use above); only the token-width
        # buffers vary per bucket. `max_bs * 8` is a deliberately modest
        # ceiling, not `config.max_num_batched_tokens` directly -- a real
        # round's total tokens is `max_bs` decode/draft rows (1 token each)
        # plus a typically-small verify window, not a full-batch prefill
        # flood; capping the ladder there keeps bucket count (and this
        # one-time capture cost, one forward pass per bucket) in the same
        # order of magnitude as decode's own ladder above, instead of
        # `range(16, 16384, 16)`'s ~1000 buckets. Still bounded above by
        # config's own hard cap; anything past this ceiling falls back to
        # the existing eager path in run_model(), same safety net decode's
        # own `input_ids.size(0) > 512` fallback already relies on.
        extend_ceiling = min(config.max_num_batched_tokens, max_bs * 8)
        self.graph_extend_tokens = _bucket_ladder(extend_ceiling)
        self.graphs_extend = {}
        self.graph_extend_wrappers = {}

        extend_workspace_buffer = torch.empty(WORKSPACE_BYTES, dtype=torch.uint8, device="cuda")
        extend_input_ids = torch.zeros(extend_ceiling, dtype=torch.int64)
        extend_positions = torch.zeros(extend_ceiling, dtype=torch.int64)
        extend_slot_mapping = torch.zeros(extend_ceiling, dtype=torch.int32)
        extend_outputs = torch.zeros(extend_ceiling, hf_config.hidden_size)
        extend_qo_indptr = torch.zeros(max_bs + 1, dtype=torch.int32, device="cuda")
        # Separate from decode's own context_lens/block_tables above (not
        # reused) -- keeps each graph family's buffer ownership unambiguous,
        # even though decode never runs concurrently with this path.
        extend_context_lens = torch.zeros(max_bs, dtype=torch.int32)
        extend_block_tables = torch.full((max_bs, max_num_blocks), -1, dtype=torch.int32)

        for t in reversed(self.graph_extend_tokens):
            qo_indptr_buf = torch.zeros(max_bs + 1, dtype=torch.int32, device="cuda")
            kv_indptr_buf = torch.zeros(max_bs + 1, dtype=torch.int32, device="cuda")
            kv_indices_buf = torch.zeros(max_bs * max_num_blocks, dtype=torch.int32, device="cuda")
            kv_last_page_len_buf = torch.zeros(max_bs, dtype=torch.int32, device="cuda")
            extend_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                extend_workspace_buffer, kv_layout="NHD", use_cuda_graph=True,
                qo_indptr_buf=qo_indptr_buf, paged_kv_indptr_buf=kv_indptr_buf,
                paged_kv_indices_buf=kv_indices_buf, paged_kv_last_page_len_buf=kv_last_page_len_buf,
            )
            self.graph_extend_wrappers[t] = extend_wrapper

            # Dummy capture data: row 0 alone carries the whole bucket's
            # query-token capacity `t` (qo_indptr flat at `t` for every
            # other row -- zero query length), with a real, valid block
            # range so plan()'s indices stay self-consistent with its own
            # indptr. This fixes FlashInfer's `_max_total_num_rows` at `t`
            # for this wrapper's whole lifetime (confirmed via a real-
            # hardware probe, see docs/adr/0031...); later real plan()
            # calls may use any total <= t. Every other row is genuinely
            # zero-context (last_page_len=0, block row all -1), exercising
            # the same padding shape run_model()'s graph branch uses below.
            n_blocks = (t + self.block_size - 1) // self.block_size
            extend_context_lens.zero_()
            extend_context_lens[0] = t
            extend_block_tables.fill_(-1)
            extend_block_tables[0, :n_blocks] = torch.arange(n_blocks, dtype=torch.int32, device="cuda")
            extend_qo_indptr.fill_(t)
            extend_qo_indptr[0] = 0

            graph = torch.cuda.CUDAGraph()
            set_context(
                True, cu_seqlens_q=extend_qo_indptr, slot_mapping=extend_slot_mapping[:t],
                context_lens=extend_context_lens, block_tables=extend_block_tables,
                extend_wrapper=extend_wrapper,
            )
            plan_mixed_extend(
                extend_wrapper, extend_qo_indptr, extend_block_tables, extend_context_lens,
                self.block_size, **self._decode_plan_kwargs,
            )
            extend_outputs[:t] = self.model(extend_input_ids[:t], extend_positions[:t])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                extend_outputs[:t] = self.model(extend_input_ids[:t], extend_positions[:t])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs_extend[t] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_extend_vars = dict(
            input_ids=extend_input_ids,
            positions=extend_positions,
            slot_mapping=extend_slot_mapping,
            qo_indptr=extend_qo_indptr,
            context_lens=extend_context_lens,
            block_tables=extend_block_tables,
            outputs=extend_outputs,
        )

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
