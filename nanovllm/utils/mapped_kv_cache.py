"""Level 4 zero-copy KV cache (see the parent project's DECISIONS.md
2026-07-24 Level-4 spike entries): allocates the model's kv_cache tensor as
CUDA-mapped pinned host memory (cudaHostAlloc(..., cudaHostAllocMapped))
instead of ordinary device memory, so the parent project's background
swap-in disk reads can write directly into it -- no H2D copy, no scatter
kernel, no GPU-side contention with foreground decode/propose/verify --
while the attention kernels still read it as an ordinary CUDA tensor.
cudaMallocManaged was considered and rejected: this Jetson reports
concurrentManagedAccess=0, which would require a whole-device
cudaDeviceSynchronize() around every host-side touch. Validated before this
integration: memory visibility (700/700 trials, no explicit CUDA fence
needed between a CPU write and a GPU kernel read), performance (1.66-1.86x
faster per block than the read+scatter path it replaces, and structurally
insensitive to concurrent GPU load since the read touches no CUDA API at
all), and FlashInfer numeric equivalence (bit-exact, 0/5 mismatches using
this project's own real BatchDecodeWithPagedKVCacheWrapper).

model_runner.py is this module's only caller -- the pass_offload experiment
package (a layer built on top of nano-vLLM, not the other way around) never
imports this module directly; it just reads ModelRunner.kv_cache_cpu_view
as a plain attribute.
"""
import ctypes
import os

import torch
from torch.utils.cpp_extension import CUDA_HOME, load

_EXT_SOURCE = os.path.join(os.path.dirname(__file__), "mapped_kv_cache_ext.cpp")
_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = load(
            name="nanovllm_mapped_kv_cache",
            sources=[_EXT_SOURCE],
            extra_include_paths=[os.path.join(CUDA_HOME, "include")],
            extra_ldflags=[f"-L{os.path.join(CUDA_HOME, 'lib64')}", "-lcudart"],
            verbose=False,
        )
    return _ext


def allocate_mapped_kv_cache(shape: tuple, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (cuda_view, cpu_view): two tensors of `shape`/`dtype` sharing
    one cudaHostAllocMapped allocation.

    cuda_view is tagged device='cuda' (torch::from_blob over the CUDA
    device pointer) -- what the model's attention kernels read/write,
    exactly like a normal device tensor (see this module's own docstring
    for the bit-exact equivalence this rests on).

    cpu_view is a plain CPU tensor over the SAME physical memory -- what
    the parent project's disk I/O reads/writes directly, with no separate
    scatter step and no extra synchronization needed before the GPU side
    sees it.

    Never explicitly freed -- allocated once per engine construction and
    lives for the process's lifetime, same convention as the plain
    torch.empty(device='cuda') kv_cache tensor this replaces (also never
    explicitly freed)."""
    ext = _get_ext()
    nbytes = dtype.itemsize
    for s in shape:
        nbytes *= s
    host_ptr = ext.alloc_mapped(nbytes)
    ctypes_buf = (ctypes.c_uint8 * nbytes).from_address(host_ptr)
    cpu_view = torch.frombuffer(ctypes_buf, dtype=torch.uint8).view(dtype).view(*shape)
    cuda_view = ext.wrap_mapped_as_cuda(host_ptr, list(shape), dtype)
    return cuda_view, cpu_view
