import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.layers import int4_triton

# Up to here, 32-row INT4 kernel calls beat dequantizing the whole layer each call (63 ms per layer on
# the Orin; they cross near 1400 rows, tree_verify 2026-10-13); prefill's thousands of rows dequantize.
CHUNKED_INT4_MAX_ROWS = 1024


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
        quantized: bool = False,
    ):
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        self.output_size = output_size
        self.int4 = None
        if quantized:
            # AutoAWQ GEMM layout, repacked by process_weights_after_loading().
            if self.tp_size != 1:
                raise NotImplementedError("AWQ layers support tensor_parallel_size=1 only")
            self.act_dtype = torch.get_default_dtype()
            groups = input_size // int4_triton.GROUP_SIZE
            self.qweight = nn.Parameter(torch.empty(input_size, output_size // 8, dtype=torch.int32), requires_grad=False)
            self.qzeros = nn.Parameter(torch.empty(groups, output_size // 8, dtype=torch.int32), requires_grad=False)
            self.scales = nn.Parameter(torch.empty(groups, output_size, dtype=torch.float16), requires_grad=False)
            for param in (self.qweight, self.qzeros, self.scales):
                param.weight_loader = self.awq_weight_loader
            self.register_parameter("weight", None)
        else:
            self.weight = nn.Parameter(torch.empty(output_size, input_size))
            self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def output_shard(self, loaded_shard_id) -> tuple[int, int]:
        """Returns (offset, size) along the output features for a checkpoint shard."""
        return 0, self.output_size

    def awq_weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id=None):
        offset, size = self.output_shard(loaded_shard_id)
        pack = self.output_size // param.size(1)  # 8 for qweight/qzeros, 1 for scales.
        param.data.narrow(1, offset // pack, size // pack).copy_(loaded_weight)

    def process_weights_after_loading(self):
        if self.weight is not None:
            return
        self.int4 = int4_triton.from_awq(self.qweight.data, self.qzeros.data, self.scales.data, self.act_dtype)
        del self.qweight, self.qzeros, self.scales
        # Autotune both row buckets now: a first call inside CUDA graph capture would fail.
        for m in (16, 32):
            int4_triton.matmul(torch.zeros(m, self.int4.qweight.size(0) * 8, dtype=self.act_dtype, device="cuda"), self.int4)

    def linear(self, x: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        if self.int4 is None:
            return F.linear(x, self.weight, bias)
        x2 = x.reshape(-1, x.size(-1))
        if x2.size(0) <= 32:
            y = int4_triton.matmul(x2, self.int4)
        elif x2.size(0) <= CHUNKED_INT4_MAX_ROWS:
            y = torch.cat([int4_triton.matmul(x2[i:i + 32], self.int4) for i in range(0, x2.size(0), 32)])
        else:
            y = x2 @ int4_triton.dequantize(self.int4)
        if bias is not None:
            y = y + bias
        return y.reshape(*x.shape[:-1], y.size(-1))


class ReplicatedLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        quantized: bool = False,
    ):
        super().__init__(input_size, output_size, bias, quantized=quantized)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x, self.bias)


class ColumnParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        quantized: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(input_size, divide(output_size, tp_size), bias, 0, quantized)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
        quantized: bool = False,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias, quantized)

    def output_shard(self, loaded_shard_id: int) -> tuple[int, int]:
        return sum(self.output_sizes[:loaded_shard_id]), self.output_sizes[loaded_shard_id]

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
        quantized: bool = False,
    ):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias, quantized)

    def output_shard(self, loaded_shard_id: str) -> tuple[int, int]:
        q_size = self.num_heads * self.head_size
        kv_size = self.num_kv_heads * self.head_size
        return {"q": (0, q_size), "k": (q_size, kv_size), "v": (q_size + kv_size, kv_size)}[loaded_shard_id]

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        quantized: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(divide(input_size, tp_size), output_size, bias, 1, quantized)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        if param_data.ndim == 1:
            param_data.copy_(loaded_weight)
            return
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.linear(x, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y
