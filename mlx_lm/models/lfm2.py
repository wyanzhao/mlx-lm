# Copyright © 2025 Apple Inc.
import os
from dataclasses import dataclass
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn

# Merge decode-side projection launches (q/k/v -> qkv, w1/w3 -> w13).
# MLX_LFM2_FUSED_PROJ=0 restores the upstream per-projection layout exactly.
# Read at module construction; __call__ branches on the instance's layout so
# differently-configured instances coexist in one process (in-process A/B).
def _fused_proj():
    return os.environ.get("MLX_LFM2_FUSED_PROJ", "1") != "0"


# Fuse the decode-step ShortConv glue (split, B*x gate, state shift, 3-tap
# depthwise conv, C*y gate) into one kernel. MLX_LFM2_CONVFUSE=0 restores the
# upstream op-by-op path exactly.
def _conv_fuse():
    return os.environ.get("MLX_LFM2_CONVFUSE", "1") != "0"

# Roundings between the replaced kernels are explicit (RNE via integer bit ops
# for bfloat) so fast-math cannot contract them; the conv accumulates in float
# in ascending tap order like mlx's depthwise_conv_1d kernel it replaces.
_conv_step_header = """
template <typename U>
inline float round_U(float x);

template <>
inline float round_U<bfloat16_t>(float x) {
  uint u = as_type<uint>(x);
  u += 0x7fffu + ((u >> 16) & 1u);
  u &= 0xffff0000u;
  return as_type<float>(u);
}

template <>
inline float round_U<half>(float x) {
  return float(half(x));
}

template <>
inline float round_U<float>(float x) {
  return x;
}
"""

_conv_step_source = """
    // one thread per (batch, channel); bcx: [B, 3C], state: [B, KS-1, C],
    // w: [C, KS]; y: [B, C], state_out: [B, KS-1, C]
    uint gid = thread_position_in_grid.x;
    const uint b = gid / C;
    const uint c = gid % C;
    const device T* row = bcx + b * 3 * C;

    // B*x gate, rounded exactly like the standalone multiply kernel.
    const float bv = float(row[c]);
    const float xv = float(row[2 * C + c]);
    const float bx = round_U<T>(bv * xv);

    // Ascending-tap float accumulation, matching depthwise_conv_1d.
    float acc = 0.0f;
    const device T* srow = state + (b * (KS - 1)) * C;
    for (uint j = 0; j + 1 < KS; j++) {
      acc += float(srow[j * C + c]) * float(w[c * KS + j]);
    }
    acc += bx * float(w[c * KS + (KS - 1)]);
    const float conv = round_U<T>(acc);

    // C*y gate.
    const float cv = float(row[C + c]);
    y[b * C + c] = static_cast<T>(round_U<T>(cv * conv));

    device T* orow = state_out + (b * (KS - 1)) * C;
    for (uint j = 0; j + 2 < KS; j++) {
      orow[j * C + c] = srow[(j + 1) * C + c];
    }
    orow[(KS - 2) * C + c] = static_cast<T>(bx);
"""

_conv_step_kernel = (
    mx.fast.metal_kernel(
        name="lfm2_conv_step",
        input_names=["bcx", "state", "w"],
        output_names=["y", "state_out"],
        header=_conv_step_header,
        source=_conv_step_source,
    )
    if mx.metal.is_available()
    else None
)


def _fused_conv_step(BCx, state, weight):
    """BCx: [B, 1, 3C]; state: [B, KS-1, C]; weight: conv weight [C, KS, 1]."""
    B = BCx.shape[0]
    C, KS = weight.shape[0], weight.shape[1]
    y, state_out = _conv_step_kernel(
        inputs=[BCx.reshape(B, 3 * C), state, weight.reshape(C, KS)],
        template=[("T", BCx.dtype), ("C", C), ("KS", KS)],
        grid=(B * C, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(B, C), (B, KS - 1, C)],
        output_dtypes=[BCx.dtype, BCx.dtype],
    )
    return y.reshape(B, 1, C), state_out


# Fuse the attn decode-step pre-processing (per-head q/k RMSNorm + RoPE, 4
# kernels/layer) into one. MLX_LFM2_QKROPE_FUSE=0 restores the op-by-op path.
def _qkrope_fuse():
    return os.environ.get("MLX_LFM2_QKROPE_FUSE", "1") != "0"


# Numerics replicate the upstream kernels exactly: rms_norm's axis-64 path
# (16 active lanes x 4 sequential squares, 32-wide simd_sum,
# precise::rsqrt(acc/64+eps), w * T(x*inv) multiplied in T) and
# rope_single's non-traditional forward (exp2(-d*log2 base), fast::cos/sin,
# float rotate, cast T on store); normed values cross to the rope phase
# through T-typed threadgroup memory as they would through DRAM.
_qkrope_source = """
    // one simdgroup per head; heads [0,NQ) = q, [NQ, NQ+NKV) = k
    // consts: [0]=log2_base, [1]=eps, [2]=offset (exact float below 2^24)
    uint sg = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint head = threadgroup_position_in_grid.x * SG_PER_TG + sg;
    if (head >= NQ + NKV) return;

    const bool is_q = head < NQ;
    const device T* x = qkv + (is_q ? head * HD : NQ * HD + (head - NQ) * HD);
    const device T* w = is_q ? qw : kw;

    threadgroup T buf[SG_PER_TG][HD];

    float acc = 0.0f;
    float tx[4];
    if (lane < HD / 4) {
      for (int i = 0; i < 4; i++) {
        tx[i] = (float)x[lane * 4 + i];
        acc += tx[i] * tx[i];
      }
    }
    acc = simd_sum(acc);
    float inv = metal::precise::rsqrt(acc / (float)HD + consts[1]);
    if (lane < HD / 4) {
      for (int i = 0; i < 4; i++) {
        buf[sg][lane * 4 + i] = w[lane * 4 + i] * static_cast<T>(tx[i] * inv);
      }
    }
    simdgroup_barrier(metal::mem_flags::mem_threadgroup);

    float x1 = (float)buf[sg][lane];
    float x2 = (float)buf[sg][lane + HD / 2];
    float d = static_cast<float>(lane) / static_cast<float>(HD / 2);
    float inv_freq = metal::exp2(-d * consts[0]);
    float theta = consts[2] * inv_freq;
    float costheta = metal::fast::cos(theta);
    float sintheta = metal::fast::sin(theta);
    float rx1 = x1 * costheta - x2 * sintheta;
    float rx2 = x1 * sintheta + x2 * costheta;

    device T* o = is_q ? (q_out + head * HD) : (k_out + (head - NQ) * HD);
    o[lane] = static_cast<T>(rx1);
    o[lane + HD / 2] = static_cast<T>(rx2);
"""

_qkrope_kernel = (
    mx.fast.metal_kernel(
        name="lfm2_qknorm_rope",
        input_names=["qkv", "qw", "kw", "consts"],
        output_names=["q_out", "k_out"],
        source=_qkrope_source,
    )
    if mx.metal.is_available()
    else None
)

_QKROPE_SG_PER_TG = 8


def _fused_qknorm_rope(qkv, q_w, k_w, offset, base, eps, n_q, n_kv, hd):
    """qkv: [1, 1, (n_q+2*n_kv)*hd] decode projections -> rope'd
    q [1, n_q, 1, hd], k [1, n_kv, 1, hd] (sdpa-ready); v untouched."""
    import math

    consts = mx.array([math.log2(base), eps, float(offset)], dtype=mx.float32)
    n_heads = n_q + n_kv
    n_tg = (n_heads + _QKROPE_SG_PER_TG - 1) // _QKROPE_SG_PER_TG
    q, k = _qkrope_kernel(
        inputs=[qkv.reshape(-1), q_w, k_w, consts],
        template=[
            ("T", qkv.dtype),
            ("HD", hd),
            ("NQ", n_q),
            ("NKV", n_kv),
            ("SG_PER_TG", _QKROPE_SG_PER_TG),
        ],
        grid=(n_tg * _QKROPE_SG_PER_TG * 32, 1, 1),
        threadgroup=(_QKROPE_SG_PER_TG * 32, 1, 1),
        output_shapes=[(n_q * hd,), (n_kv * hd,)],
        output_dtypes=[qkv.dtype, qkv.dtype],
    )
    return q.reshape(1, n_q, 1, hd), k.reshape(1, n_kv, 1, hd)


def _row_slices(proj, x, bounds):
    """Per-projection matmuls against row slices of a merged (quantized) linear.

    Row-slicing keeps each projection's kernel invocation identical to the
    unmerged layout (bitwise; the single merged matmul is only bitwise at M==1,
    where the row-parallel qmv path is shape-independent).
    """
    if isinstance(proj, nn.QuantizedLinear):
        return [
            mx.quantized_matmul(
                x,
                proj.weight[s:e],
                proj.scales[s:e],
                proj.biases[s:e],
                transpose=True,
                group_size=proj.group_size,
                bits=proj.bits,
                mode=proj.mode,
            )
            for s, e in bounds
        ]
    return [x @ proj.weight[s:e].T for s, e in bounds]

from .activations import swiglu
from .base import (
    BaseModelArgs,
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from .cache import ArraysCache, KVCache


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int
    norm_eps: float
    conv_bias: bool
    conv_L_cache: int
    block_dim: int
    block_ff_dim: int
    block_multiple_of: int
    block_ffn_dim_multiplier: float
    block_auto_adjust_ff_dim: bool
    rope_theta: float = 1000000.0
    rope_parameters: Optional[dict] = None
    full_attn_idxs: Optional[List[int]] = None
    layer_types: Optional[List[str]] = None

    def __post_init__(self):
        if self.rope_parameters is not None and "rope_theta" in self.rope_parameters:
            self.rope_theta = self.rope_parameters["rope_theta"]
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.full_attn_idxs is None:
            self.full_attn_idxs = [
                i
                for i, layer_type in enumerate(self.layer_types)
                if layer_type == "full_attention"
            ]


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads

        self.head_dim = head_dim = args.hidden_size // n_heads

        self.scale = head_dim**-0.5

        self.q_layernorm = nn.RMSNorm(head_dim, eps=args.norm_eps)
        self.k_layernorm = nn.RMSNorm(head_dim, eps=args.norm_eps)

        if _fused_proj():
            self.qkv_proj = nn.Linear(
                dim, (n_heads + 2 * n_kv_heads) * head_dim, bias=False
            )
        else:
            self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
            self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
            self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        self.rope = nn.RoPE(
            self.head_dim,
            base=args.rope_theta,
            traditional=False,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        if hasattr(self, "qkv_proj"):
            if B * L == 1:
                if (
                    _qkrope_fuse()
                    and _qkrope_kernel is not None
                    and cache is not None
                    and self.head_dim == 64
                ):
                    qkv = self.qkv_proj(x)
                    queries, keys = _fused_qknorm_rope(
                        qkv,
                        self.q_layernorm.weight,
                        self.k_layernorm.weight,
                        cache.offset,
                        self.rope.base,
                        self.q_layernorm.eps,
                        self.n_heads,
                        self.n_kv_heads,
                        self.head_dim,
                    )
                    values = qkv[..., -self.n_kv_heads * self.head_dim :].reshape(
                        B, L, self.n_kv_heads, self.head_dim
                    ).transpose(0, 2, 1, 3)
                    keys, values = cache.update_and_fetch(keys, values)
                    output = scaled_dot_product_attention(
                        queries, keys, values, cache=cache, mask=mask, scale=self.scale
                    )
                    output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
                    return self.out_proj(output)
                qkv = self.qkv_proj(x).reshape(
                    B, L, self.n_heads + 2 * self.n_kv_heads, self.head_dim
                )
                queries, keys, values = mx.split(
                    qkv, [self.n_heads, self.n_heads + self.n_kv_heads], axis=2
                )
            else:
                nq = self.n_heads * self.head_dim
                nkv = self.n_kv_heads * self.head_dim
                queries, keys, values = _row_slices(
                    self.qkv_proj,
                    x,
                    [(0, nq), (nq, nq + nkv), (nq + nkv, nq + 2 * nkv)],
                )
                queries = queries.reshape(B, L, self.n_heads, -1)
                keys = keys.reshape(B, L, self.n_kv_heads, -1)
                values = values.reshape(B, L, self.n_kv_heads, -1)
        else:
            queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)
            queries = queries.reshape(B, L, self.n_heads, -1)
            keys = keys.reshape(B, L, self.n_kv_heads, -1)
            values = values.reshape(B, L, self.n_kv_heads, -1)

        queries = self.q_layernorm(queries).transpose(0, 2, 1, 3)
        keys = self.k_layernorm(keys).transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, mask=mask, scale=self.scale
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.out_proj(output)


class ShortConv(nn.Module):
    def __init__(
        self,
        args: ModelArgs,
        layer_idx: int,
    ):
        super().__init__()
        self.args = args
        self.layer_idx = layer_idx
        self.L_cache = args.conv_L_cache
        self.bias = args.conv_bias

        self.conv = nn.Conv1d(
            in_channels=args.hidden_size,
            out_channels=args.hidden_size,
            kernel_size=self.L_cache,
            groups=args.hidden_size,
            bias=self.bias,
        )
        self.in_proj = nn.Linear(args.hidden_size, 3 * args.hidden_size, bias=self.bias)
        self.out_proj = nn.Linear(args.hidden_size, args.hidden_size, bias=self.bias)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ):
        if (
            _conv_fuse()
            and _conv_step_kernel is not None
            and not self.bias
            and cache is not None
            and cache.lengths is None
            and mask is None
            and x.shape[1] == 1
        ):
            BCx = self.in_proj(x)
            state = cache[0]
            if state is None:
                state = mx.zeros(
                    (BCx.shape[0], self.L_cache - 1, self.args.hidden_size),
                    dtype=BCx.dtype,
                )
            y, cache[0] = _fused_conv_step(BCx, state, self.conv.weight)
            cache.advance(1)
            return self.out_proj(y)

        BCx = self.in_proj(x)
        B, C, x = mx.split(BCx, 3, axis=-1)
        Bx = B * x
        if mask is not None:
            Bx = mx.where(mask[..., None], Bx, 0)

        if cache is not None:
            if cache[0] is None:
                state = mx.zeros(
                    (Bx.shape[0], self.L_cache - 1, self.args.hidden_size),
                    dtype=Bx.dtype,
                )
            else:
                state = cache[0]
            Bx = mx.concatenate([state, Bx], axis=1)
            n_keep = self.L_cache - 1
            t = x.shape[1]
            if cache.lengths is not None:
                ends = mx.clip(cache.lengths, 0, t)
                positions = (ends[:, None] + mx.arange(n_keep))[..., None]
                cache[0] = mx.take_along_axis(Bx, positions, axis=1)
            else:
                cache[0] = Bx[:, -n_keep:, :]
            cache.advance(t)
        else:
            Bx = mx.pad(Bx, [(0, 0), (self.L_cache - 1, 0), (0, 0)])

        conv_out = self.conv(Bx)

        y = C * conv_out
        return self.out_proj(y)


class MLP(nn.Module):
    def __init__(
        self,
        dim: int,
        ff_dim: int,
        multiple_of: int,
        auto_adjust_ff_dim: bool,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        if auto_adjust_ff_dim:
            ff_dim = int(2 * ff_dim / 3)
            if ffn_dim_multiplier is not None:
                ff_dim = int(ffn_dim_multiplier * ff_dim)
            ff_dim = multiple_of * ((ff_dim + multiple_of - 1) // multiple_of)

        self.ff_dim = ff_dim
        if _fused_proj():
            self.w13 = nn.Linear(dim, 2 * ff_dim, bias=False)
        else:
            self.w1 = nn.Linear(dim, ff_dim, bias=False)
            self.w3 = nn.Linear(dim, ff_dim, bias=False)
        self.w2 = nn.Linear(ff_dim, dim, bias=False)

    def __call__(self, x) -> mx.array:
        if hasattr(self, "w13"):
            if x.size == x.shape[-1]:
                w1x, w3x = mx.split(self.w13(x), 2, axis=-1)
            else:
                ff = self.ff_dim
                w1x, w3x = _row_slices(self.w13, x, [(0, ff), (ff, 2 * ff)])
            return self.w2(swiglu(w1x, w3x))
        return self.w2(swiglu(self.w1(x), self.w3(x)))


class Lfm2DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.is_attention_layer = layer_idx in args.full_attn_idxs

        if self.is_attention_layer:
            self.self_attn = Attention(args)
        else:
            self.conv = ShortConv(args, layer_idx)
        self.feed_forward = MLP(
            dim=args.block_dim,
            ff_dim=args.block_ff_dim,
            multiple_of=args.block_multiple_of,
            auto_adjust_ff_dim=args.block_auto_adjust_ff_dim,
            ffn_dim_multiplier=args.block_ffn_dim_multiplier,
        )

        self.operator_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_eps)
        self.ffn_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:

        if self.is_attention_layer:
            r = self.self_attn(self.operator_norm(x), mask=mask, cache=cache)
        else:
            r = self.conv(
                self.operator_norm(x),
                mask=mask,
                cache=cache,
            )
        h = x + r
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Lfm2Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            Lfm2DecoderLayer(args, layer_idx=i) for i in range(args.num_hidden_layers)
        ]

        self.embedding_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_eps)

        self.fa_idx = args.full_attn_idxs[0]
        self.conv_idx = 0
        for i in range(args.num_hidden_layers):
            if i in args.full_attn_idxs:
                self.conv_idx += 1
            else:
                break

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        attn_mask = create_attention_mask(h, cache[self.fa_idx])
        conv_mask = create_ssm_mask(h, cache[self.conv_idx])

        for layer, c in zip(self.layers, cache):
            mask = attn_mask if layer.is_attention_layer else conv_mask
            h = layer(h, mask, cache=c)

        return self.embedding_norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Lfm2Model(args)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        out = self.model(inputs, cache, input_embeddings)
        return self.model.embed_tokens.as_linear(out)

    def sanitize(self, weights):
        sanitized_weights = {}
        for name, param in weights.items():
            if "conv.weight" in name:
                if param.shape[-1] > param.shape[1]:
                    param = param.transpose(0, 2, 1)

            sanitized_weights[name] = param

        if any("qkv_proj" in k for k, _ in self.parameters().items()) or hasattr(
            self.model.layers[self.model.fa_idx].self_attn, "qkv_proj"
        ):
            sanitized_weights = self._merge_projections(sanitized_weights)
        return sanitized_weights

    @staticmethod
    def _merge_projections(weights):
        # Checkpoints store per-projection tensors; the merged modules take
        # their row-wise (N-dim) concatenation, which leaves every output row's
        # dot product unchanged. Quantized scales/biases concatenate the same
        # way as the packed weights.
        merges = {}
        for name in weights:
            for src, dst, order in (
                ("q_proj", "qkv_proj", ("q_proj", "k_proj", "v_proj")),
                ("w1", "w13", ("w1", "w3")),
            ):
                head, sep, tail = name.rpartition(f".{src}.")
                if sep:
                    merges[(head, dst, tail)] = order
        merged = dict(weights)
        for (head, dst, tail), order in merges.items():
            parts = [merged.pop(f"{head}.{src}.{tail}") for src in order]
            merged[f"{head}.{dst}.{tail}"] = mx.concatenate(parts, axis=0)
        return merged

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [
            KVCache() if l.is_attention_layer else ArraysCache(size=1)
            for l in self.layers
        ]
