"""Benchmark fused MoE forward thunks for ipex_mxfp4 vs oneDNN backends.

Mirrors the production xpu_fused_moe path in vllm_xpu_kernels/fused_moe_interface.py
and reports per-thunk timing so we can see exactly which thunks each optimization
vector eats:

  (A) PR #5134 post-ops (oneDNN-only) -> 'act' thunk into G1 epilogue
  (B) Tier-1 fused-quant SYCL kernel  -> 'pre_quant_g1' / 'pre_quant_g2'

ipex_mxfp4 is timed through the monolithic IPEX GatedMLPMOE module used by
XpuIpexMxfp4MoEMethod. IPEX does not expose the module's internal stages here,
so it reports only whole_moe_wall. oneDNN lanes keep the expanded equivalent
MoE block timing with per-stage thunks plus whole_moe_wall.

Usage:
  ZE_AFFINITY_MASK=2 python benchmark/benchmark_fused_moe_thunks.py --mode decode
  ZE_AFFINITY_MASK=2 python benchmark/benchmark_fused_moe_thunks.py --mode prefill
  ZE_AFFINITY_MASK=2 python benchmark/benchmark_fused_moe_thunks.py --mode decode --backends ipex_mxfp4 onednn_w4a16

Backends:
  - ipex_mxfp4:    IPEX MXFP4 monolithic XpuIpexMxfp4MoEMethod/GatedMLPMOE path
  - onednn_w4a16:  VLLM_XPU_GROUPED_GEMM_BACKEND=onednn, w4a16 path through xpu_fused_moe
  - onednn_w4a8:   VLLM_XPU_GROUPED_GEMM_BACKEND=onednn + VLLM_XPU_USE_W4A8=1, w4a8 path

oneDNN lanes execute the equivalent 7-thunk MoE block (remap + G1 + activation +
G2 + gather). IPEX reports the single monolithic GatedMLPMOE wall time.

Per-thunk regions reported (in execution order):
  - remap         _moe_C.remap_hidden_states  (permute tokens to expert layout)
  - pre_quant_g1  _dynamic_per_token_quant_int8 (w4a8 only; 5 Python torch ops)
  - gemm_g1       G1 grouped GEMM (gate_up_proj)
  - act           silu_and_mul / swigluoai_and_mul (separate kernel)
  - pre_quant_g2  _dynamic_per_token_quant_int8 (w4a8 only; 5 Python torch ops)
  - gemm_g2       G2 grouped GEMM (down_proj)
  - gather        _moe_C.moe_gather (un-permute + topk reduction)
"""

import argparse
import os
import statistics
import time
from contextlib import contextmanager

import numpy as np
import torch

torch.manual_seed(0)
np.random.seed(0)

DEVICE = "xpu"

TP1_GEMM1 = {"K": 2944, "N": 6144, "group_size": 128}
TP1_GEMM2 = {"K": 3072, "N": 2880, "group_size": 128}
E_DEFAULT = 128
TOPK_DEFAULT = 4

REAL_EXPERT_TOKENS = [
    0, 130, 3, 21, 6, 3, 0, 2, 10, 73, 52, 77, 9, 17, 5, 0,
    78, 186, 18, 55, 0, 23, 3, 10, 0, 0, 23, 916, 26, 1, 149, 0,
    42, 47, 52, 686, 2212, 22, 678, 79, 2487, 5, 0, 49, 44, 25, 19, 0,
    0, 63, 543, 0, 16, 12, 6, 40, 1, 0, 49, 0, 20, 88, 135, 0,
    3, 40, 2, 5, 0, 61, 37, 6, 28, 6, 0, 66, 0, 26, 23, 1005,
    25, 13, 10, 215, 132, 20, 46, 0, 0, 0, 27, 0, 13, 4, 17, 86,
    5, 0, 59, 0, 53, 21, 15, 250, 59, 1, 39, 46, 44, 7, 68, 0,
    14, 27, 28, 3, 26, 11, 48, 155, 6, 1, 23, 0, 0, 5, 10, 74,
]
assert len(REAL_EXPERT_TOKENS) == 128


def scale_tokens_to_M(tokens, M):
    arr = np.array(tokens, dtype=np.float64)
    scaled = np.round(arr / arr.sum() * M).astype(int)
    diff = M - scaled.sum()
    if diff != 0:
        scaled[0] += diff
    return scaled.tolist()


def build_topk_ids(token_counts, top_k, num_rows, num_experts=None):
    rows_total = num_rows * top_k
    flat = []
    for e_id, cnt in enumerate(token_counts):
        eid = e_id if num_experts is None else e_id % num_experts
        flat.extend([eid] * cnt)
    if len(flat) > rows_total:
        flat = flat[:rows_total]
    elif len(flat) < rows_total:
        flat.extend([0] * (rows_total - len(flat)))
    np.random.shuffle(flat)
    return torch.tensor(flat, dtype=torch.int64, device=DEVICE).reshape(
        num_rows, top_k)


def build_router_logits(token_counts, top_k, num_rows, num_experts):
    topk_ids = build_topk_ids(token_counts, top_k, num_rows,
                              num_experts=num_experts).cpu().numpy()
    for row in range(num_rows):
        seen = set()
        for col in range(top_k):
            eid = int(topk_ids[row, col])
            if eid not in seen:
                seen.add(eid)
                continue
            for replacement in range(num_experts):
                if replacement not in seen:
                    topk_ids[row, col] = replacement
                    seen.add(replacement)
                    break

    router_logits = torch.full((num_rows, num_experts), -20.0,
                               dtype=torch.float32, device=DEVICE)
    scores = torch.linspace(4.0, 1.0, top_k, dtype=torch.float32, device=DEVICE)
    ids = torch.tensor(topk_ids, dtype=torch.int64, device=DEVICE)
    router_logits.scatter_(1, ids, scores.expand(num_rows, top_k))
    return router_logits


# ---------------------------------------------------------------------------
# Instrumented xpu_fused_moe (mirrors production but with record_function regions)
# ---------------------------------------------------------------------------

import vllm_xpu_kernels._xpu_C  # noqa
import vllm_xpu_kernels._C      # noqa
import vllm_xpu_kernels._moe_C  # noqa


def _dynamic_per_token_quant_int8(x):
    fused = getattr(torch.ops._C, "dynamic_per_token_quant_int8_asym", None)
    if fused is not None and x.dtype in (torch.bfloat16, torch.float16):
        d = x.shape[-1]
        x_flat = x.reshape(-1, d).contiguous()
        n = x_flat.shape[0]
        q = torch.empty_like(x_flat, dtype=torch.uint8)
        s = torch.empty(n, dtype=x.dtype, device=x.device)
        z = torch.empty(n, dtype=torch.uint8, device=x.device)
        fused(q, s, z, x_flat)
        return (
            q.reshape(x.shape),
            s.reshape(x.shape[:-1] + (1,)),
            z.reshape(x.shape[:-1] + (1,)),
        )

    flat = x.reshape(-1, x.shape[-1])
    min_val = flat.to(torch.float32).min(dim=-1)[0].unsqueeze(-1)
    max_val = flat.to(torch.float32).max(dim=-1)[0].unsqueeze(-1)
    scale = ((max_val - min_val) / 255.0).clamp(min=1e-10)
    zero_point = torch.clamp(torch.round(-min_val / scale), 0,
                             255).to(torch.uint8)
    quantized = torch.clamp(
        torch.round(flat.to(torch.float32) / scale +
                    zero_point.to(torch.float32)),
        0, 255,
    ).to(torch.uint8)
    return (
        quantized.reshape(x.shape),
        scale.reshape(x.shape[:-1] + (1,)).to(x.dtype),
        zero_point.reshape(x.shape[:-1] + (1,)),
    )


# Per-thunk wall-clock timings (host+gpu serial via per-call sync), µs.
THUNK_TIMES = {}
THUNK_TIMING_ENABLED = True


@contextmanager
def thunk(name):
    """Time a thunk synchronously: sync-before, time, sync-after."""
    if not THUNK_TIMING_ENABLED:
        yield
        return
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    yield
    torch.xpu.synchronize()
    dt_us = (time.perf_counter() - t0) * 1e6
    THUNK_TIMES.setdefault(name, []).append(dt_us)


def fused_moe_instrumented(
    hidden_states, w13, w13_scales, w13_bias,
    w2, w2_scales, w2_bias,
    topk_weights, topk_ids,
    num_experts, n_experts_per_token, activation,
    is_int4, is_mxfp4, output, backend_label,
    rows_for_experts_ipex=None,
    w13_ipex=None, w13_scales_ipex=None,
    w2_ipex=None, w2_scales_ipex=None,
    fuse_act_quant=True,
):
    """Production-shape MoE forward with thunk-region timing."""
    using_ipex = backend_label == "ipex_mxfp4"
    backend_env = os.environ.get("VLLM_XPU_GROUPED_GEMM_BACKEND", "").strip().lower()
    using_onednn = is_int4 and backend_env == "onednn"
    using_w4a8 = using_onednn and os.environ.get("VLLM_XPU_USE_W4A8", "0") == "1"

    if not is_int4 and not is_mxfp4:
        inter_size = list(w13.shape)[-1] // 2
    elif using_ipex:
        inter_size = list(w13_ipex.shape)[-1] // 2
    else:
        inter_size = list(w13.shape)[-2] // 2

    hidden_size = int(hidden_states.shape[1])
    num_rows = int(hidden_states.shape[0])
    num_moe_inputs = n_experts_per_token * num_rows

    gemm1_output_dtype = hidden_states.dtype
    gemm1_scales = w13_scales if (is_int4 or is_mxfp4) else None
    gemm2_scales = w2_scales if (is_int4 or is_mxfp4) else None

    expert_first_token_offset = torch.zeros(
        (num_experts + 1,), dtype=torch.int64, device=DEVICE)
    unpermuted_row_to_permuted_row = torch.empty(
        (num_rows, n_experts_per_token), dtype=torch.int32, device=DEVICE)

    _remap_quant_op = getattr(torch.ops._moe_C,
                              "remap_and_quant_hidden_states_int8", None)
    use_fused_remap_quant = (using_w4a8 and _remap_quant_op is not None)

    if use_fused_remap_quant:
        A_q = torch.empty((num_moe_inputs, hidden_size),
                          dtype=torch.uint8, device=DEVICE)
        A_scale = torch.empty(num_moe_inputs, dtype=hidden_states.dtype,
                              device=DEVICE)
        A_zp = torch.empty(num_moe_inputs, dtype=torch.uint8, device=DEVICE)
        with thunk("remap_quant_g1"):
            _remap_quant_op(
                hidden_states=hidden_states,
                remapped_q=A_q,
                remapped_scale=A_scale,
                remapped_zp=A_zp,
                expert_map=None,
                expert_first_token_offset=expert_first_token_offset,
                unpermuted_row_to_permuted_row=unpermuted_row_to_permuted_row,
                topk_ids=topk_ids,
                total_experts_num=num_experts,
                local_experts_num=num_experts)
    else:
        remapped_hidden_states = torch.empty(
            (num_moe_inputs, hidden_size),
            dtype=hidden_states.dtype, device=DEVICE)
        with thunk("remap"):
            torch.ops._moe_C.remap_hidden_states(
                hidden_states=hidden_states,
                hidden_states_scales=None,
                remapped_hidden_states=remapped_hidden_states,
                remapped_hidden_states_scales=None,
                expert_map=None,
                expert_first_token_offset=expert_first_token_offset,
                unpermuted_row_to_permuted_row=unpermuted_row_to_permuted_row,
                topk_ids=topk_ids,
                total_experts_num=num_experts,
                local_experts_num=num_experts)

    w4a8_expert_first_token_offset = (
        expert_first_token_offset.to(torch.int32) if using_w4a8 else None
    )

    max_expert_size = (num_moe_inputs + n_experts_per_token - 1) // n_experts_per_token

    if using_ipex:
        import torch.xpu as xpu
        import intel_extension_for_pytorch  # noqa: F401
        with thunk("gemm_g1"):
            gemm1_output = xpu.moe_gemm(
                remapped_hidden_states, w13_ipex, rows_for_experts_ipex,
                num_experts,
                matrix_b_scale_inv=w13_scales_ipex, bias=w13_bias,
                is_mxfp4=True)
    elif using_w4a8:
        if not use_fused_remap_quant:
            with thunk("pre_quant_g1"):
                A_q, A_scale, A_zp = _dynamic_per_token_quant_int8(remapped_hidden_states)
        gemm1_output = torch.empty(
            (num_moe_inputs, 2 * inter_size),
            dtype=gemm1_output_dtype, device=DEVICE)
        with thunk("gemm_g1"):
            torch.ops._xpu_C.onednn_grouped_gemm_w4a8(
                A_q, A_scale.reshape(-1), A_zp.reshape(-1),
                w13, gemm1_scales, w13_bias,
                gemm1_output, w4a8_expert_first_token_offset,
                2 * inter_size, hidden_size, num_experts, max_expert_size)
    else:
        gemm1_output = torch.empty(
            (num_moe_inputs, 2 * inter_size),
            dtype=gemm1_output_dtype, device=DEVICE)
        with thunk("gemm_g1"):
            torch.ops._xpu_C.grouped_gemm_interface(
                ptr_A=remapped_hidden_states, ptr_B=w13,
                ptr_scales=gemm1_scales, ptr_bias=w13_bias,
                ptr_D=gemm1_output,
                expert_first_token_offset=expert_first_token_offset,
                N=2 * inter_size, K=hidden_size, num_experts=num_experts,
                is_B_int4=is_int4, is_B_mxfp4=is_mxfp4,
                max_expert_size=max_expert_size)

    inter_size_scale = 2 if activation == "relu2_no_mul" else 1

    can_fuse_act_quant = (
        fuse_act_quant and using_w4a8 and activation == "swigluoai"
        and inter_size_scale == 1
        and getattr(torch.ops._C, "swigluoai_and_mul_quant_int8_asym", None)
        is not None)

    if can_fuse_act_quant:
        A_q2 = torch.empty(
            (num_moe_inputs, inter_size),
            dtype=torch.uint8, device=DEVICE)
        A_scale2 = torch.empty(
            num_moe_inputs, dtype=hidden_states.dtype, device=DEVICE)
        A_zp2 = torch.empty(
            num_moe_inputs, dtype=torch.uint8, device=DEVICE)
        with thunk("act_quant_fused"):
            torch.ops._C.swigluoai_and_mul_quant_int8_asym(
                A_q2, A_scale2, A_zp2, gemm1_output, 1.702, 7.0)
    else:
        act_output = torch.empty(
            (num_moe_inputs, inter_size * inter_size_scale),
            dtype=gemm1_output.dtype, device=DEVICE)
        with thunk("act"):
            if activation == "silu":
                torch.ops._C.silu_and_mul(act_output, gemm1_output)
            elif activation == "swigluoai":
                torch.ops._C.swigluoai_and_mul(act_output, gemm1_output, 1.702,
                                               7.0)
            else:
                raise ValueError(f"Unsupported activation: {activation}")
        input_A = act_output.contiguous()

    if using_ipex:
        with thunk("gemm_g2"):
            gemm2_output = xpu.moe_gemm(
                input_A, w2_ipex, rows_for_experts_ipex, num_experts,
                matrix_b_scale_inv=w2_scales_ipex, bias=w2_bias,
                is_mxfp4=True)
    elif using_w4a8:
        if not can_fuse_act_quant:
            with thunk("pre_quant_g2"):
                A_q2, A_scale2, A_zp2 = _dynamic_per_token_quant_int8(input_A)
        gemm2_output = torch.empty(
            (num_moe_inputs, hidden_size),
            dtype=hidden_states.dtype, device=DEVICE)
        with thunk("gemm_g2"):
            torch.ops._xpu_C.onednn_grouped_gemm_w4a8(
                A_q2, A_scale2.reshape(-1), A_zp2.reshape(-1),
                w2, gemm2_scales, w2_bias,
                gemm2_output, w4a8_expert_first_token_offset,
                hidden_size, inter_size * inter_size_scale,
                num_experts, max_expert_size)
    else:
        gemm2_output = torch.empty(
            (num_moe_inputs, hidden_size),
            dtype=hidden_states.dtype, device=DEVICE)
        with thunk("gemm_g2"):
            torch.ops._xpu_C.grouped_gemm_interface(
                ptr_A=input_A, ptr_B=w2,
                ptr_scales=gemm2_scales, ptr_bias=w2_bias,
                ptr_D=gemm2_output,
                expert_first_token_offset=expert_first_token_offset,
                N=hidden_size, K=inter_size * inter_size_scale,
                num_experts=num_experts,
                is_B_int4=is_int4, is_B_mxfp4=is_mxfp4,
                max_expert_size=max_expert_size)

    with thunk("gather"):
        torch.ops._moe_C.moe_gather(
            output, gemm2_output, topk_weights,
            unpermuted_row_to_permuted_row, expert_first_token_offset,
            num_experts)

    return output


# ---------------------------------------------------------------------------
# Backend setup
# ---------------------------------------------------------------------------

BACKENDS = ["ipex_mxfp4", "onednn_w4a16", "onednn_w4a8"]


def setup_backend(name, num_experts, hidden_size, inter_size, group_size,
                  num_rows, top_k, activation, has_bias):
    is_int4 = "w4a16" in name or "w4a8" in name
    is_mxfp4 = "mxfp4" in name

    if name == "ipex_mxfp4":
        env = {"VLLM_XPU_GROUPED_GEMM_BACKEND": "ipex", "VLLM_XPU_USE_W4A8": "0"}
    elif name == "onednn_w4a16":
        env = {"VLLM_XPU_GROUPED_GEMM_BACKEND": "onednn", "VLLM_XPU_USE_W4A8": "0"}
    elif name == "onednn_w4a8":
        env = {"VLLM_XPU_GROUPED_GEMM_BACKEND": "onednn", "VLLM_XPU_USE_W4A8": "1"}
    else:
        raise ValueError(f"Unknown backend {name}")

    hidden_states = (torch.randn(num_rows, hidden_size, dtype=torch.bfloat16,
                                 device=DEVICE) * 0.1).contiguous()

    token_counts = scale_tokens_to_M(REAL_EXPERT_TOKENS, num_rows * top_k)
    router_logits = build_router_logits(token_counts, top_k, num_rows, num_experts)
    full_weights = torch.softmax(router_logits, dim=-1)
    topk_weights, topk_ids = torch.topk(full_weights, top_k, dim=-1)
    topk_weights = (topk_weights / topk_weights.sum(dim=-1, keepdim=True)).contiguous()
    topk_ids = topk_ids.contiguous()

    w13 = ((torch.randint(0, 256, (num_experts, 2 * inter_size, hidden_size // 2),
                          dtype=torch.uint8, device=DEVICE)) ^ 0x88).contiguous()
    w2 = ((torch.randint(0, 256, (num_experts, hidden_size, inter_size // 2),
                         dtype=torch.uint8, device=DEVICE)) ^ 0x88).contiguous()
    w13_scales = (torch.rand(num_experts, hidden_size // group_size,
                             2 * inter_size, dtype=torch.bfloat16,
                             device=DEVICE) * 0.5 + 0.01).contiguous()
    w2_scales = (torch.rand(num_experts, inter_size // group_size,
                            hidden_size, dtype=torch.bfloat16,
                            device=DEVICE) * 0.5 + 0.01).contiguous()

    w13_ipex = w13_scales_ipex = None
    w2_ipex = w2_scales_ipex = None
    ipex_fusion = None
    rows_for_experts_ipex = None
    if name == "ipex_mxfp4":
        import intel_extension_for_pytorch as ipex

        ipex_group_size = 32
        w13_ipex = torch.randint(
            0, 256,
            (num_experts, 2 * inter_size, hidden_size // 2),
            dtype=torch.uint8, device=DEVICE).contiguous()
        w13_scales_ipex = torch.randint(
            0, 256,
            (num_experts, 2 * inter_size, hidden_size // ipex_group_size),
            dtype=torch.uint8, device=DEVICE).contiguous()
        w2_ipex = torch.randint(
            0, 256,
            (num_experts, hidden_size, inter_size // 2),
            dtype=torch.uint8, device=DEVICE).contiguous()
        w2_scales_ipex = torch.randint(
            0, 256,
            (num_experts, hidden_size, inter_size // ipex_group_size),
            dtype=torch.uint8, device=DEVICE).contiguous()
        counts = torch.tensor(token_counts, dtype=torch.int32, device=DEVICE)
        rows_for_experts_ipex = counts.contiguous()

    if has_bias:
        w13_bias = (torch.randn(num_experts, 2 * inter_size,
                                dtype=torch.bfloat16, device=DEVICE) * 0.01
                    ).contiguous()
        w2_bias = (torch.randn(num_experts, hidden_size,
                               dtype=torch.bfloat16, device=DEVICE) * 0.01
                   ).contiguous()
    else:
        w13_bias = None
        w2_bias = None

    if name == "ipex_mxfp4":
        ipex_fusion = ipex.llm.modules.GatedMLPMOE(
            w13_ipex.view(torch.int32),
            w2_ipex.view(torch.int32),
            w1_scale_inv=w13_scales_ipex,
            w2_scale_inv=w2_scales_ipex,
            w13_bias=w13_bias,
            w2_bias=w2_bias,
            is_mxfp4=True,
        )

    output = torch.empty_like(hidden_states)

    return env, {
        "hidden_states": hidden_states,
        "w13": w13, "w13_scales": w13_scales, "w13_bias": w13_bias,
        "w2": w2, "w2_scales": w2_scales, "w2_bias": w2_bias,
        "topk_weights": topk_weights, "topk_ids": topk_ids,
        "router_logits": router_logits,
        "num_experts": num_experts,
        "n_experts_per_token": top_k,
        "activation": activation,
        "is_int4": is_int4, "is_mxfp4": is_mxfp4,
        "output": output,
        "rows_for_experts_ipex": rows_for_experts_ipex,
        "w13_ipex": w13_ipex, "w13_scales_ipex": w13_scales_ipex,
        "w2_ipex": w2_ipex, "w2_scales_ipex": w2_scales_ipex,
        "ipex_fusion": ipex_fusion,
    }


def apply_env(env_dict):
    saved = {}
    for k, v in env_dict.items():
        saved[k] = os.environ.get(k)
        if v == "":
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    return saved


def restore_env(saved):
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def run_ipex_monolithic(kwargs):
    activation = "swiglu_oai" if kwargs["activation"] == "swigluoai" else kwargs["activation"]
    return kwargs["ipex_fusion"](
        kwargs["hidden_states"],
        False,
        kwargs["n_experts_per_token"],
        kwargs["router_logits"],
        True,
        None,
        None,
        activation=activation,
    )


# ---------------------------------------------------------------------------
# Bench harness
# ---------------------------------------------------------------------------

def bench_backend(name, kwargs, warmup, iters, fuse_act_quant=True):
    global THUNK_TIMING_ENABLED
    if name == "ipex_mxfp4":
        whole_times = []
        for _ in range(warmup):
            run_ipex_monolithic(kwargs)
        for _ in range(iters):
            torch.xpu.synchronize()
            t0 = time.perf_counter()
            run_ipex_monolithic(kwargs)
            torch.xpu.synchronize()
            whole_times.append((time.perf_counter() - t0) * 1e6)
        sorted_whole = sorted(whole_times)
        return {"whole_moe_wall": sorted_whole[len(sorted_whole) // 2]}

    THUNK_TIMES.clear()
    fused_kwargs = {k: v for k, v in kwargs.items()
                    if k not in ("router_logits", "ipex_fusion")}

    THUNK_TIMING_ENABLED = True
    for _ in range(warmup):
        fused_moe_instrumented(backend_label=name,
                               fuse_act_quant=fuse_act_quant, **fused_kwargs)
    THUNK_TIMES.clear()

    for _ in range(iters):
        fused_moe_instrumented(backend_label=name,
                               fuse_act_quant=fuse_act_quant, **fused_kwargs)

    medians = {}
    for region, times in THUNK_TIMES.items():
        sorted_t = sorted(times)
        medians[region] = sorted_t[len(sorted_t) // 2]

    whole_times = []
    THUNK_TIMING_ENABLED = False
    try:
        for _ in range(warmup):
            fused_moe_instrumented(backend_label=name,
                                   fuse_act_quant=fuse_act_quant, **fused_kwargs)
        for _ in range(iters):
            torch.xpu.synchronize()
            t0 = time.perf_counter()
            fused_moe_instrumented(backend_label=name,
                                   fuse_act_quant=fuse_act_quant, **fused_kwargs)
            torch.xpu.synchronize()
            whole_times.append((time.perf_counter() - t0) * 1e6)
    finally:
        THUNK_TIMING_ENABLED = True
    sorted_whole = sorted(whole_times)
    medians["whole_moe_wall"] = sorted_whole[len(sorted_whole) // 2]
    return medians


# ---------------------------------------------------------------------------
# Blob lane: production onednn_fused_moe_w4a8 single-dispatch path
# ---------------------------------------------------------------------------

def _median_stddev(samples):
    sorted_s = sorted(samples)
    median = sorted_s[len(sorted_s) // 2]
    stddev = statistics.pstdev(samples) if len(samples) > 1 else 0.0
    return median, stddev


def _time_whole_wall(run_fn, warmup, iters):
    for _ in range(warmup):
        run_fn()
    samples = []
    for _ in range(iters):
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        run_fn()
        torch.xpu.synchronize()
        samples.append((time.perf_counter() - t0) * 1e6)
    return samples


def blob_gate_blockers(activation):
    """Return reasons the production blob gate cannot be satisfied; empty means
    dispatchable. Mirrors the VLLM_XPU_W4A8_FUSED_BLOB gate in
    fused_moe_interface.xpu_fused_moe so the lane errors instead of silently
    running the unfused path. Env conditions are forced by the caller.
    """
    reasons = []
    if getattr(torch.ops._xpu_C, "onednn_fused_moe_w4a8", None) is None:
        reasons.append(
            "torch.ops._xpu_C.onednn_fused_moe_w4a8 op is not registered "
            "(rebuild csrc with the fused-blob op)")
    if getattr(torch.ops._moe_C,
               "remap_and_quant_hidden_states_int8", None) is None:
        reasons.append(
            "torch.ops._moe_C.remap_and_quant_hidden_states_int8 op is not "
            "registered (fused remap+quant required by blob gate)")
    if getattr(torch.ops._C,
               "swigluoai_and_mul_quant_int8_asym", None) is None:
        reasons.append(
            "torch.ops._C.swigluoai_and_mul_quant_int8_asym op is not "
            "registered (fused act+quant required by blob gate)")
    if activation != "swigluoai":
        reasons.append(
            f"activation={activation!r}; blob gate requires 'swigluoai' "
            "(inter_scale must be 1)")
    return reasons


def make_blob_runner(kwargs):
    from vllm_xpu_kernels.fused_moe_interface import xpu_fused_moe

    def run():
        xpu_fused_moe(
            kwargs["hidden_states"],
            kwargs["w13"], kwargs["w13_scales"], kwargs["w13_bias"],
            kwargs["w2"], kwargs["w2_scales"], kwargs["w2_bias"],
            kwargs["topk_weights"], kwargs["topk_ids"],
            kwargs["n_experts_per_token"], kwargs["activation"],
            kwargs["num_experts"],
            output=kwargs["output"],
            is_int4=kwargs["is_int4"], is_mxfp4=kwargs["is_mxfp4"])

    return run


def print_table(rows, header, footer_notes=()):
    widths = [max(len(str(r[i])) for r in [header] + rows)
              for i in range(len(header))]
    fmt = " | ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*header))
    print("-+-".join("-" * w for w in widths))
    for r in rows:
        print(fmt.format(*r))
    for note in footer_notes:
        print(note)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["decode", "prefill"], default="decode")
    parser.add_argument(
        "--num-tokens", type=int, default=None,
        help="Override num_rows. Decode default=17, prefill default=3072.")
    parser.add_argument("--top-k", type=int, default=TOPK_DEFAULT)
    parser.add_argument("--num-experts", type=int, default=E_DEFAULT)
    parser.add_argument("--hidden", type=int, default=2944,
                        help="hidden_size. Default=2944 matches TP1_GEMM1 K from roofline bench. "
                             "(gpt-oss-120b is 2880; we use 2944 to keep group_size=128 divisible)")
    parser.add_argument("--inter", type=int, default=3072,
                        help="single-GPU inter_size for TP=1. Default=3072 to match TP1_GEMM1 N=6144, TP1_GEMM2 K=3072")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--activation", default="swigluoai",
                        choices=["silu", "swigluoai"])
    parser.add_argument("--no-bias", action="store_true")
    parser.add_argument(
        "--no-fuse-act-quant", action="store_true",
        help="Disable fused swigluoai_and_mul_quant_int8_asym and force separate act + quant in the benchmark")
    parser.add_argument(
        "--backends", nargs="+",
        default=BACKENDS,
        choices=BACKENDS)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument(
        "--blob", action="store_true",
        help="Add a production fused-blob lane that drives onednn_fused_moe_w4a8 "
             "via xpu_fused_moe with VLLM_XPU_W4A8_FUSED_BLOB=1. Single dispatch, "
             "so only whole_moe_wall (median+stddev) is reported. Errors out "
             "rather than silently falling back to the unfused path.")
    args = parser.parse_args()

    if args.num_tokens is None:
        args.num_tokens = 17 if args.mode == "decode" else 3072

    has_bias = not args.no_bias
    print(f"\n{'=' * 100}")
    print(f"Fused MoE thunk benchmark — mode={args.mode}")
    print(f"hidden={args.hidden} inter={args.inter} (single-GPU, TP1) E={args.num_experts} "
          f"top_k={args.top_k} activation={args.activation} bias={'on' if has_bias else 'off'}")
    print(f"num_tokens={args.num_tokens} (M_total = num_tokens * top_k = {args.num_tokens * args.top_k})")
    print(f"{'=' * 100}\n")

    region_order = ["remap", "remap_quant_g1", "pre_quant_g1", "gemm_g1",
                    "act", "act_quant_fused",
                    "pre_quant_g2", "gemm_g2", "gather",
                    "whole_moe_wall"]

    results = {}
    for name in args.backends:
        print(f"--> {name}")
        env, kwargs = setup_backend(
            name, args.num_experts, args.hidden, args.inter,
            args.group_size, args.num_tokens, args.top_k,
            args.activation, has_bias)
        saved = apply_env(env)
        try:
            results[name] = bench_backend(
                name, kwargs, args.warmup, args.iters,
                fuse_act_quant=not args.no_fuse_act_quant)
        finally:
            restore_env(saved)

    # Per-thunk table
    print()
    print("=== Per-thunk wall (µs, median across iters) ===\n")
    header = ["thunk"] + [f"{b}" for b in args.backends]
    rows = []
    synced_regions = [r for r in region_order if r != "whole_moe_wall"]
    totals = {b: 0.0 for b in args.backends}
    has_synced_total = {b: False for b in args.backends}
    for region in region_order:
        row = [region]
        for b in args.backends:
            t = results[b].get(region, None)
            row.append(f"{t:.1f}" if t is not None else "-")
            if t is not None and region in synced_regions:
                totals[b] += t
                has_synced_total[b] = True
        rows.append(row)
    rows.append(["TOTAL_SYNCED_THUNKS"] + [
        f"{totals[b]:.1f}" if has_synced_total[b] else "-"
        for b in args.backends
    ])
    print_table(rows, header)

    print()
    print("=== Optimization budget analysis ===\n")
    print("Two distinct optimization vectors operate on different thunks:")
    print("  (A) PR #5134 post-ops (oneDNN-only, ipex cannot use)")
    print("        -> 'act' fusable into G1 epilogue (eltwise post-op)")
    print("  (B) Tier-1 wrapper fixes (oneDNN w4a8-only, kills Python launch overhead)")
    print("        -> 'pre_quant_g1' / 'pre_quant_g2' from 5 torch ops to 1 SYCL kernel\n")
    for b in args.backends:
        if not b.startswith("onednn"):
            continue
        r = results[b]
        budget_postop = r.get("act", 0.0)
        budget_wrapper = r.get("pre_quant_g1", 0.0) + r.get("pre_quant_g2", 0.0)
        before = totals[b]
        if before <= 0.0:
            continue
        after_postop = before - budget_postop
        after_both = before - budget_postop - budget_wrapper
        print(f"[{b}]")
        print(f"  Current synced-thunk total                  : {before:7.1f} µs")
        print(f"  (A) Fusable via PR #5134 (act post-op)       : {budget_postop:7.1f} µs "
              f"({100 * budget_postop / before:.1f}%)")
        if budget_wrapper > 0:
            print(f"  (B) Killable via Tier-1 fused-quant op       : {budget_wrapper:7.1f} µs "
                  f"({100 * budget_wrapper / before:.1f}%)")
        print(f"  Projected synced total after PR #5134       : {after_postop:7.1f} µs "
              f"(-{before - after_postop:.1f})")
        if budget_wrapper > 0:
            print(f"  Projected synced total after PR#5134+Tier1  : {after_both:7.1f} µs "
                  f"(-{before - after_both:.1f})")
        print()

    ipex_keys = [b for b in args.backends if b.startswith("ipex")]
    onednn_keys = [b for b in args.backends if b.startswith("onednn")]
    if ipex_keys and onednn_keys:
        print("=== oneDNN whole-MoE-block vs ipex monolithic (with optimization projections) ===\n")
        for ik in ipex_keys:
            it = results[ik]["whole_moe_wall"]
            print(f"  Baseline {ik} monolithic whole_moe_wall: {it:.1f} µs")
            for ok in onednn_keys:
                ot = totals[ok]
                ow = results[ok]["whole_moe_wall"]
                budget_postop = results[ok].get("act", 0.0)
                budget_wrapper = (results[ok].get("pre_quant_g1", 0.0) +
                                  results[ok].get("pre_quant_g2", 0.0))
                op = ot - budget_postop
                opb = op - budget_wrapper
                print(f"    {ok:>14} whole wall:   {ow:7.1f} µs ({ow - it:+7.1f} vs ipex)")
                print(f"    {ok:>14} synced now:   {ot:7.1f} µs ({ot - it:+7.1f} vs ipex)")
                print(f"    {ok:>14} +PR#5134:     {op:7.1f} µs ({op - it:+7.1f} vs ipex)")
                if budget_wrapper > 0:
                    print(f"    {ok:>14} +PR#5134+T1: {opb:7.1f} µs ({opb - it:+7.1f} vs ipex)")
            print()

    if args.blob:
        run_blob_lane(args, has_bias)


def run_blob_lane(args, has_bias):
    print("=== Fused-blob lane (production onednn_fused_moe_w4a8, single dispatch) ===\n")
    blockers = blob_gate_blockers(args.activation)
    if blockers:
        raise RuntimeError(
            "Blob lane cannot dispatch the production fused path (refusing to "
            "silently fall back to unfused):\n  - " + "\n  - ".join(blockers))

    blob_env = {
        "VLLM_XPU_GROUPED_GEMM_BACKEND": "onednn",
        "VLLM_XPU_USE_W4A8": "1",
        "VLLM_XPU_W4A8_FUSED_BLOB": "1",
    }

    lane_samples = {}

    _, blob_kwargs = setup_backend(
        "onednn_w4a8", args.num_experts, args.hidden, args.inter,
        args.group_size, args.num_tokens, args.top_k,
        args.activation, has_bias)
    saved = apply_env(blob_env)
    try:
        lane_samples["onednn_w4a8_blob"] = _time_whole_wall(
            make_blob_runner(blob_kwargs), args.warmup, args.iters)
    finally:
        restore_env(saved)

    for ik in [b for b in args.backends if b.startswith("ipex")]:
        env, kwargs = setup_backend(
            ik, args.num_experts, args.hidden, args.inter,
            args.group_size, args.num_tokens, args.top_k,
            args.activation, has_bias)
        saved = apply_env(env)
        try:
            lane_samples[ik] = _time_whole_wall(
                lambda kw=kwargs: run_ipex_monolithic(kw),
                args.warmup, args.iters)
        finally:
            restore_env(saved)

    header = ["lane", "whole_moe_wall median (µs)", "stddev (µs)"]
    rows = []
    for lane, samples in lane_samples.items():
        median, stddev = _median_stddev(samples)
        rows.append([lane, f"{median:.1f}", f"{stddev:.1f}"])
    print_table(rows, header)

    blob_median, _ = _median_stddev(lane_samples["onednn_w4a8_blob"])
    for ik in [b for b in args.backends if b.startswith("ipex")]:
        ik_median, _ = _median_stddev(lane_samples[ik])
        print(f"\n  onednn_w4a8_blob vs {ik}: "
              f"{blob_median - ik_median:+.1f} µs "
              f"({blob_median / ik_median:.2f}x)")


if __name__ == "__main__":
    main()
