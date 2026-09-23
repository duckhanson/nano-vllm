"""Correctness of the Triton INT4 GEMM against a dequantized reference."""

import pytest
import torch

from nanovllm.layers import int4_triton

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _random_weight(k, n, dtype):
    q = torch.randint(0, 16, (k, n), dtype=torch.int32, device="cuda")
    groups = k // int4_triton.GROUP_SIZE
    scales = torch.rand(groups, n, device="cuda", dtype=dtype) * 0.01 + 0.001
    zeros = torch.randint(0, 16, (groups, n), device="cuda").to(dtype)
    w_ref = (q.float() - zeros.float().repeat_interleave(128, 0)) * (
        scales.float().repeat_interleave(128, 0))
    return int4_triton.quantized_weight(q, scales, zeros, dtype), w_ref


def _rel_err(out, ref):
    return ((out.float() - ref).abs().max() / ref.abs().max()).item()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("m", [1, 3, 8, 16, 32])
@pytest.mark.parametrize("n,k", [(256, 512), (4096, 14336)])
def test_matmul_matches_dequantized_reference(dtype, m, n, k):
    torch.manual_seed(0)
    weight, w_ref = _random_weight(k, n, dtype)
    x = torch.randn(m, k, device="cuda", dtype=dtype)

    out = int4_triton.matmul(x, weight)

    assert out.shape == (m, n)
    assert out.dtype == dtype
    assert _rel_err(out, x.float() @ w_ref) < 1e-2


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_dequantize_matches_reference(dtype):
    torch.manual_seed(0)
    weight, w_ref = _random_weight(512, 256, dtype)

    w = int4_triton.dequantize(weight)

    assert w.shape == (512, 256)
    assert w.dtype == dtype
    assert _rel_err(w, w_ref) < 1e-2


def _awq_pack(values):
    """Packs int32 [R, C] along columns in AutoAWQ's GEMM nibble order."""
    order = [0, 2, 4, 6, 1, 3, 5, 7]
    r, c = values.shape
    values = values.reshape(r, c // 8, 8)
    packed = torch.zeros(r, c // 8, dtype=torch.int32, device=values.device)
    for i, col in enumerate(order):
        packed |= values[:, :, col] << (4 * i)
    return packed


def test_from_awq_matches_awq_dequantization():
    torch.manual_seed(0)
    k, n = 512, 256
    q = torch.randint(0, 16, (k, n), dtype=torch.int32, device="cuda")
    zeros = torch.randint(0, 16, (k // 128, n), dtype=torch.int32, device="cuda")
    scales = torch.rand(k // 128, n, device="cuda", dtype=torch.float16) * 0.01 + 0.001
    x = torch.randn(4, k, device="cuda", dtype=torch.float16)

    weight = int4_triton.from_awq(_awq_pack(q), _awq_pack(zeros), scales, torch.float16)
    out = int4_triton.matmul(x, weight)

    w_ref = (q - zeros.repeat_interleave(128, 0)).float() * (
        scales.float().repeat_interleave(128, 0))
    assert _rel_err(out, x.float() @ w_ref) < 1e-2
