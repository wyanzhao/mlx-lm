# Copyright © 2023-2024 Apple Inc.

import math
import os
from functools import partial

import mlx.core as mx
import mlx.nn as nn

from .activations import swiglu


def _gather_sort(x, indices):
    *_, M = indices.shape
    indices = indices.flatten()
    order = mx.argsort(indices)
    inv_order = mx.argsort(order)
    return x.flatten(0, -3)[order // M], indices[order], inv_order


def _scatter_unsort(x, inv_order, shape=None):
    x = x[inv_order]
    if shape is not None:
        x = mx.unflatten(x, 0, shape)
    return x


def _combine_enabled():
    return os.environ.get("MLX_MOE_COMBINE", "1") != "0"


# Fused MoE tail: replaces _scatter_unsort + (y * scores) + sum(axis=-2)
# (three kernels, two [R, D]-sized intermediates round-tripping DRAM) with a
# single permutation-gather: out[t] = sum_j scores[t, j] * y[inv[t*K + j]].
# The multiply and the K-wise accumulation are performed sequentially in the
# input dtype to match the elementwise mul + sum reduction of the unfused
# path. MLX_MOE_COMBINE=0 restores the unfused ops.
_combine_source = """
    uint t = threadgroup_position_in_grid.x;
    uint lid = thread_position_in_threadgroup.x;

    for (uint d0 = lid * 8; d0 < D; d0 += 256 * 8) {
      // Match the unfused ops bit-for-bit: the elementwise multiply rounds
      // its product to T and the sum accumulates sequentially in T. The
      // roundings are done with explicit integer bit ops so fast-math
      // cannot contract them into fmas.
      float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
      for (uint j = 0; j < K; j++) {
        const uint src = inv[t * K + j];
        const float s = float(scores[t * K + j]);
        const device T* row = y + size_t(src) * D + d0;
        for (uint e = 0; e < 8; e++) {
          if (d0 + e < D) {
            acc[e] = round_T(acc[e] + round_T(float(row[e]) * s));
          }
        }
      }
      device T* orow = out + size_t(t) * D + d0;
      for (uint e = 0; e < 8; e++) {
        if (d0 + e < D) {
          orow[e] = static_cast<T>(acc[e]);
        }
      }
    }
"""

_combine_header = """
// Round a float to the T grid (round-to-nearest-even), via integer bit ops
// so that fast-math cannot elide the intermediate rounding.
template <typename U>
inline float round_T_impl(float x);

template <>
inline float round_T_impl<bfloat16_t>(float x) {
  uint u = as_type<uint>(x);
  u += 0x7fffu + ((u >> 16) & 1u);
  u &= 0xffff0000u;
  return as_type<float>(u);
}

template <>
inline float round_T_impl<half>(float x) {
  return float(half(x));
}

template <>
inline float round_T_impl<float>(float x) {
  return x;
}
"""

_combine_kernel = mx.fast.metal_kernel(
    name="switch_combine",
    input_names=["y", "inv", "scores"],
    output_names=["out"],
    header=_combine_header,
    source="#define round_T round_T_impl<T>\n" + _combine_source,
)


def _switch_combine(y, inv_order, scores):
    """y: sorted rows [R, ..., D]; inv_order: [R]; scores: [..., T, K]."""
    *batch, T, K = scores.shape
    D = y.shape[-1]
    rows = T
    for b in batch:
        rows *= b
    y2 = y.reshape(-1, D)
    (out,) = _combine_kernel(
        inputs=[y2, inv_order, scores.reshape(-1).astype(y2.dtype)],
        template=[("T", y2.dtype), ("D", D), ("K", K)],
        grid=(256 * rows, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows * D,)],
        output_dtypes=[y2.dtype],
    )
    return out.reshape(*batch, T, D)


class QuantizedSwitchLinear(nn.Module):
    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        num_experts: int,
        bias: bool = True,
        group_size: int = 64,
        bits: int = 4,
        mode: str = "affine",
    ):
        super().__init__()

        scale = math.sqrt(1 / input_dims)
        self.weight, self.scales, *biases = mx.quantize(
            mx.random.uniform(
                low=-scale,
                high=scale,
                shape=(num_experts, output_dims, input_dims),
            ),
            group_size=group_size,
            bits=bits,
            mode=mode,
        )
        self.biases = biases[0] if biases else None

        if bias:
            self.bias = mx.zeros((num_experts, output_dims))

        self.group_size = group_size
        self.bits = bits
        self.mode = mode

        # Freeze this model's parameters
        self.freeze()

    @property
    def input_dims(self):
        return self.scales.shape[2] * self.group_size

    @property
    def output_dims(self):
        return self.weight.shape[1]

    @property
    def num_experts(self):
        return self.weight.shape[0]

    def __call__(self, x, indices, sorted_indices=False):
        x = mx.gather_qmm(
            x,
            self["weight"],
            self["scales"],
            self.get("biases"),
            rhs_indices=indices,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
            sorted_indices=sorted_indices,
        )
        if "bias" in self:
            x = x + mx.expand_dims(self["bias"][indices], -2)
        return x


class SwitchLinear(nn.Module):
    def __init__(
        self, input_dims: int, output_dims: int, num_experts: int, bias: bool = True
    ):
        super().__init__()
        scale = math.sqrt(1 / input_dims)
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(num_experts, output_dims, input_dims),
        )

        if bias:
            self.bias = mx.zeros((num_experts, output_dims))

    @property
    def input_dims(self):
        return self.weight.shape[2]

    @property
    def output_dims(self):
        return self.weight.shape[1]

    @property
    def num_experts(self):
        return self.weight.shape[0]

    def __call__(self, x, indices, sorted_indices=False):
        x = mx.gather_mm(
            x,
            self["weight"].swapaxes(-1, -2),
            rhs_indices=indices,
            sorted_indices=sorted_indices,
        )
        if "bias" in self:
            x = x + mx.expand_dims(self["bias"][indices], -2)
        return x

    def to_quantized(self, group_size: int = 64, bits: int = 4, mode: str = "affine"):
        num_experts, output_dims, input_dims = self.weight.shape
        ql = QuantizedSwitchLinear(
            input_dims,
            output_dims,
            num_experts,
            False,
            group_size,
            bits,
            mode=mode,
        )
        ql.weight, ql.scales, *biases = mx.quantize(
            self.weight, group_size, bits, mode=mode
        )
        ql.biases = biases[0] if biases else None

        if "bias" in self:
            ql.bias = self.bias
        return ql


class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, x, gate):
        return swiglu(gate, x)


class SwitchGLU(nn.Module):
    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        num_experts: int,
        activation=SwiGLU(),
        bias: bool = False,
    ):
        super().__init__()

        self.gate_proj = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.up_proj = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.down_proj = SwitchLinear(hidden_dims, input_dims, num_experts, bias=bias)
        self.activation = activation

    def __call__(self, x, indices, scores=None) -> mx.array:
        x = mx.expand_dims(x, (-2, -3))

        # When we have many tokens, then sort them to make sure that the access
        # of different experts is in order.
        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        if self.training:
            idx = mx.stop_gradient(idx)
        x_up = self.up_proj(x, idx, sorted_indices=do_sort)
        x_gate = self.gate_proj(x, idx, sorted_indices=do_sort)
        x = self.down_proj(
            self.activation(x_up, x_gate),
            idx,
            sorted_indices=do_sort,
        )

        # With routing scores provided, fuse unsort + weighting + top-k sum
        # into one gather kernel instead of materializing two [R, D]
        # intermediates.
        if (
            scores is not None
            and do_sort
            and not self.training
            and _combine_enabled()
        ):
            return _switch_combine(x, inv_order, scores)

        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)

        x = x.squeeze(-2)
        if scores is not None:
            x = (x * scores[..., None]).sum(axis=-2)
        return x


class SwitchMLP(nn.Module):
    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        num_experts: int,
        activation=nn.GELU(approx="precise"),
        bias: bool = False,
    ):
        super().__init__()

        self.fc1 = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.fc2 = SwitchLinear(hidden_dims, input_dims, num_experts, bias=bias)
        self.activation = activation

    def __call__(self, x, indices) -> mx.array:
        x = mx.expand_dims(x, (-2, -3))

        # When we have many tokens, then sort them to make sure that the access
        # of different experts is in order.
        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        if self.training:
            idx = mx.stop_gradient(idx)
        x = self.fc1(x, idx, sorted_indices=do_sort)
        x = self.activation(x)
        x = self.fc2(x, idx, sorted_indices=do_sort)

        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)

        return x.squeeze(-2)
