"""Level 4 zero-copy KV cache (see the parent project's DECISIONS.md
2026-07-24 Level-4 spike entries), shared-memory backed: allocates the
model's kv_cache tensor over POSIX shared memory
(multiprocessing.shared_memory.SharedMemory), registered for CUDA
visibility via cudaHostRegister(..., cudaHostRegisterMapped) instead of
cudaHostAlloc allocating its own fresh memory -- so the parent project's
background swap-in disk reads can write directly into it -- no H2D copy, no
scatter kernel, no GPU-side contention with foreground decode/propose/verify
-- while the attention kernels still read it as an ordinary CUDA tensor.
cudaMallocManaged was considered and rejected: this Jetson reports
concurrentManagedAccess=0, which would require a whole-device
cudaDeviceSynchronize() around every host-side touch.

The switch from cudaHostAlloc (originally adopted, ADR-0022) to shared
memory + cudaHostRegister is what enables a second, spawn()ed process
(the parent project's IO worker process) to attach to the SAME physical
buffer by shm name and read/write it directly with zero CUDA calls of its
own -- cudaHostAlloc memory has no such cross-process story (CUDA's own IPC
handles only cover cudaMalloc device memory). Real-hardware validated in a
standalone spike (logs/2026-08-03.md): a spawned process's writes are
visible through the CUDA device pointer, and the parent process's own
subsequent CUDA calls are unaffected (unlike fork(), which the parent's CUDA
context does not survive). ADR-0022's own memory-visibility/performance/
FlashInfer-bit-exact validation was re-run against this mechanism before
integration (see the parent project's docs for the re-validation record).

model_runner.py is this module's only caller -- the pass_offload experiment
package (a layer built on top of nano-vLLM, not the other way around) never
imports this module directly; it just reads ModelRunner.kv_cache_cpu_view
as a plain attribute.
"""
import ctypes
import os
from multiprocessing import shared_memory

import torch
from torch.utils.cpp_extension import CUDA_HOME, load

_EXT_SOURCE = os.path.join(os.path.dirname(__file__), "mapped_kv_cache_ext.cpp")
_ext = None

# Kept alive for the process's lifetime -- a SharedMemory object's mapping
# is torn down when the object itself is garbage-collected, so letting the
# only reference be the local variable inside allocate_mapped_kv_cache()
# would unmap the segment out from under cpu_view/cuda_view the moment this
# function returns.
_shm_handles: list[shared_memory.SharedMemory] = []


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


def allocate_mapped_kv_cache(shape: tuple, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Returns (cuda_view, cpu_view, shm_name): two tensors of `shape`/`dtype`
    sharing one POSIX shared-memory segment, plus that segment's name -- a
    second process attaches to the same physical memory with
    shared_memory.SharedMemory(name=shm_name, create=False).

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
    shm = shared_memory.SharedMemory(create=True, size=nbytes)
    _shm_handles.append(shm)
    # A temporary ctypes view over shm.buf only to get the buffer's address
    # for the CUDA calls below -- the address is the mapping's, stable for
    # as long as `shm` itself is alive, not tied to this wrapper object's own
    # lifetime, so it's fine to let it go out of scope right after (deleted
    # explicitly, matching the spike's own fix for the
    # "cannot close exported pointers exist" error otherwise raised later
    # by shm.close()).
    ctypes_buf = (ctypes.c_uint8 * nbytes).from_buffer(shm.buf)
    ptr = ctypes.addressof(ctypes_buf)
    ext.register_mapped(ptr, nbytes)
    cpu_view = torch.frombuffer(shm.buf, dtype=torch.uint8).view(dtype).view(*shape)
    cuda_view = ext.wrap_registered_as_cuda(ptr, list(shape), dtype)
    del ctypes_buf
    return cuda_view, cpu_view, shm.name
