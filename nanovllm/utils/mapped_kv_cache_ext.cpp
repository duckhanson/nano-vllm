// Level 4 zero-copy KV cache: raw cudaHostAlloc(..., cudaHostAllocMapped)
// allocator, plus a torch::from_blob wrapper over the CUDA device pointer.
// See the parent project's DECISIONS.md 2026-07-24 Level-4 spike entries --
// this mechanism (not cudaMallocManaged; this device's
// concurrentManagedAccess=0 ruled that out) was validated there: memory
// visibility (700/700 trials, no explicit CUDA fence needed), performance
// (1.66-1.86x faster than read+scatter), and FlashInfer numeric equivalence
// (bit-exact, 0/5 mismatches) all confirmed before this real integration.
//
// No kernel code needed -- pure CUDA Runtime API + ATen calls, so this
// compiles as ordinary C++ (no nvcc kernel compilation), linked against
// libcudart (torch.utils.cpp_extension.load already links libtorch).
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <string>

int64_t alloc_mapped(int64_t nbytes) {
  void *ptr = nullptr;
  cudaError_t err = cudaHostAlloc(&ptr, (size_t)nbytes, cudaHostAllocMapped);
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("cudaHostAlloc failed: ") +
                              cudaGetErrorString(err));
  }
  return reinterpret_cast<int64_t>(ptr);
}

void free_mapped(int64_t host_ptr) {
  cudaFreeHost(reinterpret_cast<void *>(host_ptr));
}

// torch.frombuffer (Python side, via ctypes) only ever produces a CPU
// tensor -- the Python buffer protocol is a CPU-memory concept, so it can't
// tag a tensor device='cuda' even though the bytes are physically
// GPU-reachable via cudaHostGetDevicePointer. Kernels (FlashInfer's
// included) branch on tensor.is_cuda/tensor.device, so the CUDA-side view
// needs a tensor actually tagged device='cuda' -- torch::from_blob with a
// CUDA TensorOptions and the *device* pointer does that, over the same
// physical memory the CPU view (built in Python, see mapped_kv_cache.py)
// addresses. No deleter passed -- lifetime is process-scoped (this
// allocation happens once per engine construction and is never freed
// early, same convention as the kv_cache tensor it replaces).
torch::Tensor wrap_mapped_as_cuda(int64_t host_ptr, std::vector<int64_t> sizes,
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
  m.def("alloc_mapped", &alloc_mapped);
  m.def("free_mapped", &free_mapped);
  m.def("wrap_mapped_as_cuda", &wrap_mapped_as_cuda);
}
