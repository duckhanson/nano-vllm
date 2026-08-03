// Level 4 zero-copy KV cache, shared-memory backed: cudaHostRegister(...,
// cudaHostRegisterMapped) over an existing POSIX shared-memory mapping
// (Python side allocates it via multiprocessing.shared_memory.SharedMemory,
// see mapped_kv_cache.py), instead of cudaHostAlloc allocating its own fresh
// memory. The difference from the original cudaHostAlloc design (ADR-0022)
// is what it enables, not raw performance: a second, spawn()ed process can
// attach to the same shm segment by name and read/write it directly with
// zero CUDA calls of its own -- cudaHostAlloc memory has no cross-process
// story (CUDA's own IPC handles only cover cudaMalloc device memory).
// Real-hardware validated in a standalone spike (logs/2026-08-03.md): a
// spawned process's writes are visible through the CUDA device pointer, and
// the parent process's own subsequent CUDA calls are unaffected (unlike
// fork(), which the parent's CUDA context does not survive).
//
// No kernel code needed -- pure CUDA Runtime API + ATen calls, so this
// compiles as ordinary C++ (no nvcc kernel compilation), linked against
// libcudart (torch.utils.cpp_extension.load already links libtorch).
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <string>

void register_mapped(int64_t ptr, int64_t nbytes) {
  cudaError_t err = cudaHostRegister(reinterpret_cast<void *>(ptr),
                                      (size_t)nbytes, cudaHostRegisterMapped);
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("cudaHostRegister failed: ") +
                              cudaGetErrorString(err));
  }
}

// torch.frombuffer (Python side) only ever produces a CPU tensor -- the
// Python buffer protocol is a CPU-memory concept, so it can't tag a tensor
// device='cuda' even though the bytes are physically GPU-reachable via
// cudaHostGetDevicePointer. Kernels (FlashInfer's included) branch on
// tensor.is_cuda/tensor.device, so the CUDA-side view needs a tensor
// actually tagged device='cuda' -- torch::from_blob with a CUDA
// TensorOptions and the *device* pointer does that, over the same physical
// memory the CPU view (built in Python, see mapped_kv_cache.py) addresses.
// No deleter passed -- lifetime is process-scoped (this allocation happens
// once per engine construction and is never freed early, same convention as
// the kv_cache tensor it replaces).
torch::Tensor wrap_registered_as_cuda(int64_t host_ptr,
                                       std::vector<int64_t> sizes,
                                       torch::ScalarType dtype) {
  void *dev_ptr = nullptr;
  cudaError_t err = cudaHostGetDevicePointer(
      &dev_ptr, reinterpret_cast<void *>(host_ptr), 0);
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("cudaHostGetDevicePointer failed: ") +
                              cudaGetErrorString(err));
  }
  auto options = torch::TensorOptions().dtype(dtype).device(torch::kCUDA);
  return torch::from_blob(dev_ptr, sizes, options);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("register_mapped", &register_mapped);
  m.def("wrap_registered_as_cuda", &wrap_registered_as_cuda);
}
