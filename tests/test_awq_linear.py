"""AWQ loading and forward of the merged linear layers used by Llama."""

import tempfile

import pytest
import torch
import torch.distributed as dist

from nanovllm.layers import linear

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

_AWQ_ORDER = [0, 2, 4, 6, 1, 3, 5, 7]


@pytest.fixture(scope="module", autouse=True)
def _single_rank_group():
    if not dist.is_initialized():
        store = tempfile.NamedTemporaryFile(delete=False).name
        dist.init_process_group("gloo", init_method=f"file://{store}", rank=0, world_size=1)
    yield


def _awq_pack(values):
    r, c = values.shape
    values = values.reshape(r, c // 8, 8)
    packed = torch.zeros(r, c // 8, dtype=torch.int32, device=values.device)
    for i, col in enumerate(_AWQ_ORDER):
        packed |= values[:, :, col] << (4 * i)
    return packed


def _random_awq(k, n):
    """Returns (qweight, qzeros, scales) in AutoAWQ layout and w [N, K] in fp32."""
    q = torch.randint(0, 16, (k, n), dtype=torch.int32, device="cuda")
    zeros = torch.randint(0, 16, (k // 128, n), dtype=torch.int32, device="cuda")
    scales = torch.rand(k // 128, n, device="cuda", dtype=torch.float16) * 0.01 + 0.001
    w = (q - zeros.repeat_interleave(128, 0)).float() * scales.float().repeat_interleave(128, 0)
    return (_awq_pack(q), _awq_pack(zeros), scales), w.t()


def _load(layer, shards):
    """Loads {shard_id: (qweight, qzeros, scales)} through the params' weight_loaders."""
    for shard_id, tensors in shards.items():
        for name, tensor in zip(("qweight", "qzeros", "scales"), tensors):
            param = getattr(layer, name)
            if shard_id is None:
                param.weight_loader(param, tensor)
            else:
                param.weight_loader(param, tensor, shard_id)
    layer.process_weights_after_loading()


def _check(layer, w, k):
    for m in (4, 64, 2048):  # Triton kernel, its 32-row chunks, and the dequantize path.
        x = torch.randn(m, k, device="cuda", dtype=torch.float16)
        ref = x.float() @ w.t()
        out = layer(x)
        assert out.dtype == torch.float16
        assert ((out.float() - ref).abs().max() / ref.abs().max()) < 1e-2


@pytest.fixture
def fp16_cuda_default():
    dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float16)
    torch.set_default_device("cuda")
    yield
    torch.set_default_device("cpu")
    torch.set_default_dtype(dtype)


def test_qkv_awq(fp16_cuda_default):
    k, head, heads, kv_heads = 512, 64, 8, 2
    layer = linear.QKVParallelLinear(k, head, heads, kv_heads, quantized=True)
    shards = {}
    ws = []
    for shard_id, n in (("q", heads * head), ("k", kv_heads * head), ("v", kv_heads * head)):
        shards[shard_id], w = _random_awq(k, n)
        ws.append(w)
    _load(layer, shards)
    _check(layer, torch.cat(ws), k)


def test_gate_up_awq(fp16_cuda_default):
    k, n = 512, 384
    layer = linear.MergedColumnParallelLinear(k, [n, n], quantized=True)
    (gate, w_gate), (up, w_up) = _random_awq(k, n), _random_awq(k, n)
    _load(layer, {0: gate, 1: up})
    _check(layer, torch.cat([w_gate, w_up]), k)


def test_row_awq(fp16_cuda_default):
    k, n = 1024, 256
    layer = linear.RowParallelLinear(k, n, quantized=True)
    tensors, w = _random_awq(k, n)
    _load(layer, {None: tensors})
    _check(layer, w, k)
    assert not hasattr(layer, "qweight")  # Raw AWQ tensors are freed after repacking.


def test_rows_past_32_run_the_int4_kernel_in_32_row_chunks(fp16_cuda_default):
    """Verify with a tree (up to ~128 rows) must not fall to the per-call dequantize path (65 ms per layer)."""
    k, n = 1024, 256
    layer = linear.RowParallelLinear(k, n, quantized=True)
    tensors, _ = _random_awq(k, n)
    _load(layer, {None: tensors})
    x = torch.randn(128, k, device="cuda", dtype=torch.float16)
    out = layer(x)
    for i in range(0, 128, 32):
        assert torch.equal(out[i:i + 32], layer(x[i:i + 32]))
    assert linear.CHUNKED_INT4_MAX_ROWS >= 128
