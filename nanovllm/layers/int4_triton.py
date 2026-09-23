"""INT4 (AWQ-style, group 128, asymmetric) GEMM in Triton for M = 1-32 rows.

Activations may be fp16 or bf16; scales and bias are stored in the same dtype.

Weights are packed 8 nibbles per int32 along K, interleaved within each
128-row group so the kernel can unpack each nibble plane straight into a K=16
tensor-core dot without a register reshape. An AWQ checkpoint (packed along N
in its own nibble order) therefore needs a one-time repack at load. The zero
point is folded into a per-group bias, w = q * scale + bias.
"""

import dataclasses

import torch
import triton
import triton.language as tl

GROUP_SIZE = 128


@dataclasses.dataclass(frozen=True)
class QuantizedWeight:
    """An INT4 weight of logical shape [K, N] (y = x @ w).

    Attributes:
        qweight: int32 [K // 8, N]; nibble j of row 16 * g + r holds
            q[128 * g + 16 * j + r].
        scales: [K // GROUP_SIZE, N] in the activation dtype.
        bias: [K // GROUP_SIZE, N] in the activation dtype, -zero * scale.
    """

    qweight: torch.Tensor
    scales: torch.Tensor
    bias: torch.Tensor


def pack_along_k(q: torch.Tensor) -> torch.Tensor:
    """Packs int32 values in [0, 16) of shape [K, N] into int32 [K // 8, N]."""
    k, n = q.shape
    q = q.reshape(k // GROUP_SIZE, 8, 16, n)
    packed = torch.zeros(k // GROUP_SIZE, 16, n, dtype=torch.int32, device=q.device)
    for j in range(8):
        packed |= q[:, j] << (4 * j)
    return packed.reshape(k // 8, n)


def quantized_weight(q: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor,
                     dtype: torch.dtype) -> QuantizedWeight:
    """Builds a QuantizedWeight from unpacked q [K, N] with w = (q - zero) * scale."""
    bias = (-zeros.float() * scales.float()).to(dtype)
    return QuantizedWeight(pack_along_k(q), scales.to(dtype).contiguous(),
                           bias.contiguous())


# Nibble i of an AutoAWQ GEMM int32 holds column order[i] with order =
# [0, 2, 4, 6, 1, 3, 5, 7]; this is its inverse, indexed by column.
_AWQ_REVERSE_ORDER = (0, 4, 1, 5, 2, 6, 3, 7)


def _awq_unpack(packed: torch.Tensor) -> torch.Tensor:
    """Unpacks AutoAWQ GEMM int32 [R, C // 8] into int32 [R, C] in [0, 16)."""
    shifts = torch.tensor([4 * i for i in _AWQ_REVERSE_ORDER],
                          dtype=torch.int32, device=packed.device)
    values = (packed[:, :, None] >> shifts) & 0xF
    return values.reshape(packed.shape[0], -1)


def from_awq(qweight: torch.Tensor, qzeros: torch.Tensor, scales: torch.Tensor,
             dtype: torch.dtype) -> QuantizedWeight:
    """Repacks one AutoAWQ GEMM linear layer (group 128, zero point).

    Args:
        qweight: int32 [K, N // 8].
        qzeros: int32 [K // GROUP_SIZE, N // 8].
        scales: fp16 [K // GROUP_SIZE, N].
        dtype: activation dtype the layer will run in.
    """
    return quantized_weight(_awq_unpack(qweight), scales, _awq_unpack(qzeros), dtype)


def dequantize(weight: QuantizedWeight) -> torch.Tensor:
    """Returns w [K, N] in the weight's dtype, for row counts the kernel does not cover."""
    k8, n = weight.qweight.shape
    shifts = torch.arange(8, dtype=torch.int32, device=weight.qweight.device) * 4
    packed = weight.qweight.reshape(k8 // 16, 1, 16, n)
    q = (packed >> shifts[None, :, None, None]) & 0xF  # [groups, 8, 16, N]
    q = q.reshape(k8 // 16, GROUP_SIZE, n).float()
    w = q * weight.scales.float()[:, None, :] + weight.bias.float()[:, None, :]
    return w.reshape(k8 * 8, n).to(weight.scales.dtype)


def _configs():
    return [
        triton.Config({"BLOCK_N": block_n}, num_warps=warps, num_stages=stages)
        for block_n in (64, 128, 256)
        for warps in (2, 4, 8)
        for stages in (2, 3, 4)
    ]


# Keyed on BLOCK_M, not M, so every M in a bucket reuses one tuning result and
# CUDA graph capture never triggers a fresh autotune; results persist on disk.
@triton.autotune(configs=_configs(), key=["BLOCK_M", "N", "K", "IS_FP16"],
                 cache_results=True)
@triton.jit
def _int4_matmul_kernel(x_ptr, qw_ptr, scale_ptr, bias_ptr, out_ptr, M, N, K,
                        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                        GROUP: tl.constexpr, IS_FP16: tl.constexpr):
    pid_n = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_16 = tl.arange(0, 16)
    mask_m = offs_m < M

    if IS_FP16:
        DTYPE: tl.constexpr = tl.float16
        MAGIC_BITS: tl.constexpr = 0x6400
        MAGIC: tl.constexpr = 1024.0
    else:
        DTYPE: tl.constexpr = tl.bfloat16
        MAGIC_BITS: tl.constexpr = 0x4300
        MAGIC: tl.constexpr = 128.0

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for g in range(0, K // GROUP):
        packed = tl.load(qw_ptr + (g * 16 + offs_16)[:, None] * N + offs_n[None, :])
        part = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        x_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for j in tl.static_range(8):
            x = tl.load(
                x_ptr + offs_m[:, None] * K + (g * GROUP + j * 16 + offs_16)[None, :],
                mask=mask_m[:, None], other=0.0)
            # OR-ing a nibble into the mantissa of MAGIC (fp16 1024.0 / bf16
            # 128.0) gives MAGIC + q exactly, skipping an int-to-float convert.
            q = ((packed >> (4 * j)) & 0xF) | MAGIC_BITS
            part = tl.dot(x, q.to(tl.int16).to(DTYPE, bitcast=True), part)
            x_sum += tl.sum(x.to(tl.float32), axis=1)
        # x @ ((MAGIC + q) * s + b') = part * s + sum(x) * b', b' = b - MAGIC * s.
        s = tl.load(scale_ptr + g * N + offs_n).to(tl.float32)
        b = tl.load(bias_ptr + g * N + offs_n).to(tl.float32)
        acc += part * s[None, :] + x_sum[:, None] * (b - MAGIC * s)[None, :]

    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :],
             acc.to(DTYPE), mask=mask_m[:, None])


def matmul(x: torch.Tensor, weight: QuantizedWeight) -> torch.Tensor:
    """Returns x @ w for x of shape [M, K] with M <= 32, in x's dtype."""
    m, k = x.shape
    n = weight.qweight.shape[1]
    if m > 32:
        raise ValueError(f"M={m} > 32 is out of this kernel's range")
    if x.dtype != weight.scales.dtype:
        raise ValueError(f"x is {x.dtype} but the weight was built for {weight.scales.dtype}")
    out = torch.empty(m, n, device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_N"]),)
    _int4_matmul_kernel[grid](
        x.contiguous(), weight.qweight, weight.scales, weight.bias, out, m, n, k,
        BLOCK_M=16 if m <= 16 else 32, GROUP=GROUP_SIZE,
        IS_FP16=x.dtype == torch.float16)
    return out
