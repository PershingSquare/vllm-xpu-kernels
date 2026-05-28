# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark grouped GEMM with roofline analysis: offline and server modes.

Usage:
    ZE_AFFINITY_MASK=2 python benchmark/benchmark_moe_roofline.py --mode offline
    ZE_AFFINITY_MASK=2 python benchmark/benchmark_moe_roofline.py --mode server --dist-file gpt-oss-120b-token-counts.md
    ZE_AFFINITY_MASK=2 python benchmark/benchmark_moe_roofline.py --mode offline --tp 4 --backend onednn_w4a8
"""

import torch
import numpy as np
torch.manual_seed(0)
np.random.seed(0)

import argparse
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
import gc
import os
import time

import vllm_xpu_kernels._xpu_C  # noqa: F401

DEVICE = "xpu"
ALL_BACKENDS = ["onednn_w4a8", "onednn_w4a16", "cutlass_w4a16", "cutlass_mxfp4", "ipex_int4", "ipex_mxfp4"]

# TP4 shapes
TP4_GEMM1 = {"K": 2944, "N": 1536, "group_size": 128}
TP4_GEMM2 = {"K": 768, "N": 2880, "group_size": 128}

# TP8 shapes
TP8_GEMM1 = {"K": 2944, "N": 768, "group_size": 128}
TP8_GEMM2 = {"K": 384, "N": 2880, "group_size": 128}

E = 128  # number of experts

SPEC_VERSION = "t6.1"
DEVICE_NAME = "B60"
HOST_ID = "10.98.74.64"
CONTAINER_NAME = "hans-gpt-oss-ww17"
OFFICIAL_BACKEND_ALIASES = {"ipex_w4a16": "ipex_int4"}
OFFICIAL_BACKENDS = ("onednn_w4a8", "ipex_int4")
OFFICIAL_DISTRIBUTIONS = ("balanced", "sparse-active", "routed/skew")
DECODE_M_VALUES = list(range(4, 129, 4))
PREFILL_M_VALUES = [8192, 12288]
CACHE_BANDWIDTH_GBS = 1800.0
DEFAULT_LAUNCH_US = 30.0

# Hardware constants (BMG single tile)
PEAK_INT8_TOPS = 198.0       # BMG single tile INT8 XMX
PEAK_BF16_TFLOPS = 99.0     # BMG single tile BF16 XMX
PEAK_BW_GBS = 456.0         # BMG HBM bandwidth GB/s

# Real expert token distribution from gpt-oss-120b inference (128 experts)
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


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def scale_tokens_to_M(tokens, M):
    """Scale token distribution to target M with deterministic largest remainder."""
    tokens = np.array(tokens, dtype=np.float64)
    if M < 0:
        raise ValueError(f"M must be nonnegative, got {M}")
    total = tokens.sum()
    if total <= 0:
        raise ValueError("token distribution must have positive total")

    scaled = tokens / total * M
    counts = np.floor(scaled).astype(int)
    remainder = int(M - counts.sum())
    if remainder:
        fractions = scaled - counts
        order = sorted(range(len(counts)), key=lambda idx: (-fractions[idx], idx))
        for idx in order[:remainder]:
            counts[idx] += 1

    return counts.tolist()


def scale_tokens_uniform(M, E):
    """Distribute M tokens evenly across E experts."""
    base = M // E
    rem = M % E
    counts = [base + (1 if i < rem else 0) for i in range(E)]
    return counts


def clear_xpu_cache():
    torch.xpu.synchronize()
    gc.collect()


# ---------------------------------------------------------------------------
# Per-token asymmetric uint8 quantization
# ---------------------------------------------------------------------------

def quantize_per_token_u8(x: torch.Tensor):
    """Per-token asymmetric u8 quant: scale=(max-min)/255, zp=round(-min/scale)."""
    flat = x.reshape(-1, x.shape[-1]).to(torch.float32)
    min_val = flat.min(dim=-1)[0]
    max_val = flat.max(dim=-1)[0]
    scale = ((max_val - min_val) / 255.0).clamp(min=1e-10)
    zero_point = torch.clamp(torch.round(-min_val / scale), 0, 255).to(torch.uint8)
    quantized = torch.clamp(
        torch.round(flat / scale.unsqueeze(-1) + zero_point.float().unsqueeze(-1)),
        0, 255,
    ).to(torch.uint8)
    return quantized.reshape(x.shape), scale.to(torch.bfloat16), zero_point


# ---------------------------------------------------------------------------
# Build expert offsets from token distribution
# ---------------------------------------------------------------------------

def build_offsets_from_distribution(token_counts, device):
    """Build cumulative [E+1] offsets from per-expert token counts."""
    offsets = [0]
    for c in token_counts:
        offsets.append(offsets[-1] + c)
    return torch.tensor(offsets, dtype=torch.int64, device=device)


def count_active_experts(offsets) -> int:
    """Count experts that have at least one token (count > 0) from cumulative offsets [E+1]."""
    counts = offsets[1:] - offsets[:-1]
    return int((counts > 0).sum().item())


# ---------------------------------------------------------------------------
# Cold-cache timing with rotating pool of pre-allocated buffers
# ---------------------------------------------------------------------------

def bench_cold(fns_list, warmup, iters, repetitions=3, return_samples=False):
    """Time with rotating pool of pre-allocated buffers for cold-cache simulation.

    fns_list[i] is the function for pool slot i (each has its own buffers).

    Uses batch-async timing: iters kernels are queued without syncing between
    them, mirroring how vLLM queues MoE ops to the SYCL command queue in
    production (CPU stays ahead of GPU; host dispatch is hidden by the async
    queue). A single torch.xpu.synchronize() at the end gates the wall time.

    To reduce run-to-run variance, `repetitions` independent batch runs are
    taken and the median is reported.
    """
    POOL = len(fns_list)
    for i in range(warmup):
        fns_list[i % POOL]()
    torch.xpu.synchronize()

    rep_results = []
    for _ in range(repetitions):
        t0 = time.perf_counter()
        for i in range(iters):
            fns_list[i % POOL]()
        torch.xpu.synchronize()
        rep_results.append((time.perf_counter() - t0) / iters * 1000.0)
    rep_results.sort()
    if return_samples:
        return rep_results
    return rep_results[len(rep_results) // 2]


# ---------------------------------------------------------------------------
# Backend runners with pool-based cold-cache simulation
# ---------------------------------------------------------------------------

def run_onednn_w4a8(M, N, K, E, group_size, offsets, max_expert_size, warmup, iters, POOL, use_bias=True, repetitions=3, return_samples=False):
    # Pool weights: in production weights >> L2 cache, always read from HBM.
    pool_Aq     = [torch.empty(M, K, dtype=torch.uint8,    device=DEVICE) for _ in range(POOL)]
    pool_Ascale = [torch.empty(M,    dtype=torch.bfloat16, device=DEVICE) for _ in range(POOL)]
    pool_Azp    = [torch.empty(M,    dtype=torch.uint8,    device=DEVICE) for _ in range(POOL)]
    pool_D      = [torch.empty(M, N, dtype=torch.bfloat16, device=DEVICE) for _ in range(POOL)]
    pool_B    = [(torch.randint(0,256,(E,N,K//2),dtype=torch.uint8,device=DEVICE)^0x88).contiguous() for _ in range(POOL)]
    pool_Bsc  = [torch.rand(E,K//group_size,N,dtype=torch.bfloat16,device=DEVICE)*0.5+0.01 for _ in range(POOL)]
    pool_bias = [(torch.randn(E,N,dtype=torch.bfloat16,device=DEVICE)*0.01).contiguous() if use_bias else None for _ in range(POOL)]

    def make_fn(slot):
        A_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device=DEVICE) * 0.1
        A_q, A_scale, A_zp = quantize_per_token_u8(A_bf16)
        pool_Aq[slot].copy_(A_q)
        pool_Ascale[slot].copy_(A_scale.flatten())
        pool_Azp[slot].copy_(A_zp.flatten())

        def fn():
            torch.ops._xpu_C.onednn_grouped_gemm_w4a8(
                pool_Aq[slot], pool_Ascale[slot], pool_Azp[slot],
                pool_B[slot], pool_Bsc[slot], pool_bias[slot],
                pool_D[slot], offsets, N, K, E, max_expert_size)
        return fn

    fns_list = [make_fn(i) for i in range(POOL)]
    return bench_cold(fns_list, warmup, iters, repetitions, return_samples)


def run_onednn_w4a16(M, N, K, E, group_size, offsets, max_expert_size, warmup, iters, POOL, use_bias=True, repetitions=3, return_samples=False):
    # Pool weights: force HBM reads, no L2 cache reuse.
    pool_A    = [torch.empty(M, K, dtype=torch.bfloat16, device=DEVICE) for _ in range(POOL)]
    pool_D    = [torch.empty(M, N, dtype=torch.bfloat16, device=DEVICE) for _ in range(POOL)]
    pool_B    = [(torch.randint(0,256,(E,N,K//2),dtype=torch.uint8,device=DEVICE)^0x88).contiguous() for _ in range(POOL)]
    pool_Bsc  = [torch.rand(E,K//group_size,N,dtype=torch.bfloat16,device=DEVICE)*0.5+0.01 for _ in range(POOL)]
    pool_bias = [(torch.randn(E,N,dtype=torch.bfloat16,device=DEVICE)*0.01).contiguous() if use_bias else None for _ in range(POOL)]

    def make_fn(slot):
        pool_A[slot].copy_(torch.randn(M, K, dtype=torch.bfloat16, device=DEVICE) * 0.1)

        def fn():
            torch.ops._xpu_C.onednn_grouped_gemm_w4a16(
                pool_A[slot], pool_B[slot], pool_Bsc[slot], pool_bias[slot],
                pool_D[slot], offsets, N, K, E,
                True, False, max_expert_size)
        return fn

    fns_list = [make_fn(i) for i in range(POOL)]
    return bench_cold(fns_list, warmup, iters, repetitions, return_samples)


def run_cutlass_w4a16(M, N, K, E, group_size, offsets, warmup, iters, POOL, use_bias=True, repetitions=3, return_samples=False):
    """Cutlass w4a16: bf16 activations, int4 weights."""
    old = os.environ.get("VLLM_XPU_GROUPED_GEMM_BACKEND", "")
    os.environ["VLLM_XPU_GROUPED_GEMM_BACKEND"] = ""
    try:
        # B and scales are fixed (shared across pool)
        B_packed = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8, device=DEVICE)
        B_scales = torch.rand(E, N, K // group_size, dtype=torch.bfloat16, device=DEVICE) * 0.5 + 0.01
        bias = (torch.randn(E, N, dtype=torch.bfloat16, device=DEVICE) * 0.01).contiguous() if use_bias else None

        # Pool of buffers: (Abf, D)
        pool_A = [torch.empty(M, K, dtype=torch.bfloat16, device=DEVICE) for _ in range(POOL)]
        pool_D = [torch.empty(M, N, dtype=torch.bfloat16, device=DEVICE) for _ in range(POOL)]

        def make_fn(slot):
            A = torch.randn(M, K, dtype=torch.bfloat16, device=DEVICE) * 0.1
            pool_A[slot].copy_(A)

            def fn():
                torch.ops._xpu_C.grouped_gemm_interface(
                    ptr_A=pool_A[slot], ptr_B=B_packed, ptr_scales=B_scales, ptr_bias=bias,
                    ptr_D=pool_D[slot], expert_first_token_offset=offsets,
                    N=N, K=K, num_experts=E, is_B_int4=True, is_B_mxfp4=False)
            return fn

        fns_list = [make_fn(i) for i in range(POOL)]
        return bench_cold(fns_list, warmup, iters, repetitions, return_samples)
    finally:
        os.environ["VLLM_XPU_GROUPED_GEMM_BACKEND"] = old


def run_cutlass_mxfp4(M, N, K, E, group_size, offsets, warmup, iters, POOL, use_bias=True, repetitions=3, return_samples=False):
    """Cutlass mxfp4: bf16 activations, mxfp4 weights, uint8 MX scales."""
    old = os.environ.get("VLLM_XPU_GROUPED_GEMM_BACKEND", "")
    os.environ["VLLM_XPU_GROUPED_GEMM_BACKEND"] = ""
    try:
        # B packed u8, scales uint8 [E,N,K//gs]
        B_packed = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8, device=DEVICE)
        B_scales = torch.randint(0, 256, (E, N, K // group_size), dtype=torch.uint8, device=DEVICE)
        bias = (torch.randn(E, N, dtype=torch.bfloat16, device=DEVICE) * 0.01).contiguous() if use_bias else None

        # Pool of buffers: (Abf, D)
        pool_A = [torch.empty(M, K, dtype=torch.bfloat16, device=DEVICE) for _ in range(POOL)]
        pool_D = [torch.empty(M, N, dtype=torch.bfloat16, device=DEVICE) for _ in range(POOL)]

        def make_fn(slot):
            A = torch.randn(M, K, dtype=torch.bfloat16, device=DEVICE) * 0.1
            pool_A[slot].copy_(A)

            def fn():
                torch.ops._xpu_C.grouped_gemm_interface(
                    ptr_A=pool_A[slot], ptr_B=B_packed, ptr_scales=B_scales, ptr_bias=bias,
                    ptr_D=pool_D[slot], expert_first_token_offset=offsets,
                    N=N, K=K, num_experts=E, is_B_int4=False, is_B_mxfp4=True)
            return fn

        fns_list = [make_fn(i) for i in range(POOL)]
        return bench_cold(fns_list, warmup, iters, repetitions, return_samples)
    finally:
        os.environ["VLLM_XPU_GROUPED_GEMM_BACKEND"] = old


def run_ipex_int4(M, N, K, E, group_size, offsets, warmup, iters, POOL, use_bias=True, repetitions=3, return_samples=False):
    """IPEX moe_gemm int4: bf16 activations, int4 weights, bf16 scales."""
    import torch.xpu as xpu
    import intel_extension_for_pytorch  # noqa

    ipex_group_num = K // group_size
    # IPEX weight layout: [E, K//2, N]
    B_packed = torch.randint(0, 256, (E, K // 2, N), dtype=torch.uint8, device=DEVICE)
    B_scales = torch.rand(E, ipex_group_num, N, dtype=torch.bfloat16, device=DEVICE) * 0.5 + 0.01
    # gpt-oss bias [E, N] in bf16
    bias = (torch.randn(E, N, dtype=torch.bfloat16, device=DEVICE) * 0.01).contiguous() if use_bias else None
    counts = offsets[1:] - offsets[:-1]
    rows_for_experts = counts.to(torch.int32)

    # Pool of buffers: Abf
    pool_A = [torch.empty(M, K, dtype=torch.bfloat16, device=DEVICE) for _ in range(POOL)]

    def make_fn(slot):
        A = torch.randn(M, K, dtype=torch.bfloat16, device=DEVICE) * 0.1
        pool_A[slot].copy_(A)

        def fn():
            xpu.moe_gemm(pool_A[slot], B_packed, rows_for_experts, E,
                         matrix_b_scale_inv=B_scales, bias=bias, is_int4=True)
        return fn

    fns_list = [make_fn(i) for i in range(POOL)]
    return bench_cold(fns_list, warmup, iters, repetitions, return_samples)


def run_ipex_mxfp4(M, N, K, E, group_size, offsets, warmup, iters, POOL, use_bias=True, repetitions=3, return_samples=False):
    """IPEX moe_gemm mxfp4: bf16 activations, mxfp4 weights, uint8 MX scales (group_size=32 only)."""
    import torch.xpu as xpu
    import intel_extension_for_pytorch  # noqa

    # IPEX mxfp4 requires group_size=32
    ipex_group_size = 32
    ipex_group_num = K // ipex_group_size
    # IPEX weight layout: [E, K//2, N] (transposed vs cutlass [E, N, K//2])
    counts = offsets[1:] - offsets[:-1]
    rows_for_experts = counts.to(torch.int32)
    # Pool weights: force HBM reads.
    pool_A    = [torch.empty(M, K, dtype=torch.bfloat16, device=DEVICE) for _ in range(POOL)]
    pool_B    = [torch.randint(0,256,(E,K//2,N),dtype=torch.uint8,device=DEVICE) for _ in range(POOL)]
    pool_Bsc  = [torch.randint(0,256,(E,ipex_group_num,N),dtype=torch.uint8,device=DEVICE) for _ in range(POOL)]
    pool_bias = [(torch.randn(E,N,dtype=torch.bfloat16,device=DEVICE)*0.01).contiguous() if use_bias else None for _ in range(POOL)]

    def make_fn(slot):
        A = torch.randn(M, K, dtype=torch.bfloat16, device=DEVICE) * 0.1
        pool_A[slot].copy_(A)

        def fn():
            xpu.moe_gemm(pool_A[slot], pool_B[slot], rows_for_experts, E,
                         matrix_b_scale_inv=pool_Bsc[slot], bias=pool_bias[slot], is_mxfp4=True)
        return fn

    fns_list = [make_fn(i) for i in range(POOL)]
    return bench_cold(fns_list, warmup, iters, repetitions, return_samples)


RUNNERS = {
    "onednn_w4a8": run_onednn_w4a8,
    "onednn_w4a16": run_onednn_w4a16,
    "cutlass_w4a16": run_cutlass_w4a16,
    "cutlass_mxfp4": run_cutlass_mxfp4,
    "ipex_int4": run_ipex_int4,
    "ipex_mxfp4": run_ipex_mxfp4,
}

ONEDNN_BACKENDS = {"onednn_w4a8", "onednn_w4a16"}


# ---------------------------------------------------------------------------
# Roofline analysis
# ---------------------------------------------------------------------------

def compute_bytes(total_M, N, K, e_active, group_size, backend, use_bias=True):
    # bias is [E, N] bf16; only active experts read it.
    bias_bytes = e_active * N * 2 if use_bias else 0
    if backend == "onednn_w4a8":
        bytes_a = total_M * K * 1
        bytes_b = e_active * N * K // 2
        bytes_d = total_M * N * 2
        bytes_scales_w = e_active * (K // group_size) * N * 2
        bytes_scales_a = total_M * 2
        return bytes_a + bytes_b + bytes_d + bytes_scales_w + bytes_scales_a + bias_bytes
    elif backend in ("onednn_w4a16", "cutlass_w4a16", "ipex_int4"):
        bytes_a = total_M * K * 2
        bytes_b = e_active * N * K // 2
        bytes_d = total_M * N * 2
        bytes_scales_w = e_active * (K // group_size) * N * 2
        return bytes_a + bytes_b + bytes_d + bytes_scales_w + bias_bytes
    elif backend in ("cutlass_mxfp4", "ipex_mxfp4"):
        bytes_a = total_M * K * 2
        bytes_b = e_active * N * K // 2
        bytes_d = total_M * N * 2
        bytes_scales_w = e_active * (K // group_size) * N * 1
        return bytes_a + bytes_b + bytes_d + bytes_scales_w + bias_bytes
    else:
        raise ValueError(f"Unknown backend: {backend}")


def roofline_stats(ms, total_M, N, K, e_active, group_size, backend, use_bias=True):
    flops = 2 * total_M * N * K
    bytes_total = compute_bytes(total_M, N, K, e_active, group_size, backend, use_bias)

    achieved_ops_per_sec = flops / (ms * 1e-3)   # ops/sec
    achieved_gbs = bytes_total / (ms * 1e-3) / 1e9
    ai = flops / bytes_total   # ops/byte (arithmetic intensity)

    # Choose correct peak for this backend
    peak_compute = PEAK_INT8_TOPS * 1e12 if backend == "onednn_w4a8" else PEAK_BF16_TFLOPS * 1e12
    ridge_point = peak_compute / (PEAK_BW_GBS * 1e9)  # ops/byte

    if ai >= ridge_point:
        bound = "compute"
        roofline_pct = achieved_ops_per_sec / peak_compute * 100
    else:
        bound = "memory"
        roofline_pct = achieved_gbs / PEAK_BW_GBS * 100

    achieved_tops_or_tflops = achieved_ops_per_sec / 1e12
    return achieved_tops_or_tflops, achieved_gbs, ai, bound, roofline_pct


# ---------------------------------------------------------------------------
# Distribution file parsing (server mode)
# ---------------------------------------------------------------------------

def parse_token_distribution(md_path):
    """Parse markdown table of token distribution."""
    dist = {}
    with open(md_path) as f:
        for line in f:
            line = line.strip()
            if not line.startswith('|') or 'tokens' in line.lower() or '---' in line:
                continue
            parts = [p.strip().replace(',', '') for p in line.split('|') if p.strip()]
            if len(parts) >= 2:
                try:
                    tokens = int(parts[0])
                    occurrences = int(parts[1])
                    dist[tokens] = occurrences
                except ValueError:
                    continue
    return dist  # {token_count: occurrences}


def filter_distribution(dist, coverage=0.95):
    """Keep the highest-frequency M values covering `coverage` fraction of total mass.

    Sorting by frequency descending ensures we keep the most representative
    M values first — e.g. at 95% coverage this yields ~60 values rather than
    all 2000+ entries that ascending-token-count ordering would produce.
    """
    total = sum(dist.values())
    threshold = total * coverage
    sorted_items = sorted(dist.items(), key=lambda x: -x[1])
    cumsum = 0
    kept = {}
    for tokens, occ in sorted_items:
        cumsum += occ
        kept[tokens] = occ
        if cumsum >= threshold:
            break
    return kept


# ---------------------------------------------------------------------------
# Run benchmark for a single (config, backend, M) combination
# ---------------------------------------------------------------------------

def run_benchmark(total_M, N, K, E, group_size, top_k, backend, warmup, iters, POOL, token_dist="real", use_bias=True, token_counts=None, repetitions=3, return_samples=False, raise_errors=False):
    if token_counts is None:
        if token_dist == "real":
            token_counts = scale_tokens_to_M(REAL_EXPERT_TOKENS, total_M)
        else:
            token_counts = scale_tokens_uniform(total_M, E)
    offsets = build_offsets_from_distribution(token_counts, DEVICE)
    max_expert_size = (total_M + top_k - 1) // top_k
    e_active = count_active_experts(offsets)

    try:
        if backend in ONEDNN_BACKENDS:
            result = RUNNERS[backend](total_M, N, K, E, group_size, offsets, max_expert_size, warmup, iters, POOL, use_bias, repetitions, return_samples)
        else:
            result = RUNNERS[backend](total_M, N, K, E, group_size, offsets, warmup, iters, POOL, use_bias, repetitions, return_samples)
        return result, e_active
    except Exception as e:
        if raise_errors:
            raise
        print(f"  {backend}: FAILED - {str(e)[:80]}")
        return None, e_active
    finally:
        gc.collect()



# ---------------------------------------------------------------------------
# Official paired JSON matrix output
# ---------------------------------------------------------------------------

def normalize_backend_name(backend):
    return OFFICIAL_BACKEND_ALIASES.get(backend, backend)


def normalize_backend_list(backends):
    return [normalize_backend_name(backend) for backend in backends]


def shape_for(tp, gemm):
    if tp == 4 and gemm == "GEMM1":
        return TP4_GEMM1
    if tp == 4 and gemm == "GEMM2":
        return TP4_GEMM2
    if tp == 8 and gemm == "GEMM1":
        return TP8_GEMM1
    if tp == 8 and gemm == "GEMM2":
        return TP8_GEMM2
    return None


def distribution_slug(distribution):
    return distribution.replace("/", "_").replace("-", "_")


def mode_for_m(total_M):
    return "prefill" if total_M in PREFILL_M_VALUES else "decode"


def row_id_for(tp, gemm, total_M, distribution):
    return f"B60_tp{tp}_{gemm}_{mode_for_m(total_M)}_M{total_M}_{distribution_slug(distribution)}"


def distribution_counts(distribution, total_M):
    if distribution == "balanced":
        return {
            "token_counts": scale_tokens_uniform(total_M, E),
            "runner_token_dist": "uniform",
            "distribution_source": "scale_tokens_uniform",
        }
    if distribution == "routed/skew":
        return {
            "token_counts": scale_tokens_to_M(REAL_EXPERT_TOKENS, total_M),
            "runner_token_dist": "real",
            "distribution_source": "REAL_EXPERT_TOKENS",
        }
    if distribution == "sparse-active":
        return {
            "token_counts": scale_tokens_to_M(REAL_EXPERT_TOKENS, total_M),
            "runner_token_dist": None,
            "distribution_source": "sparse_active_vector",
        }
    raise KeyError(f"missing distribution mapping: {distribution}")


def nearest_rank(values, percentile):
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(len(ordered) * percentile / 100.0))
    return ordered[min(rank - 1, len(ordered) - 1)]


def timing_summary(samples_ms, runner_backend):
    samples_us = [float(sample) * 1000.0 for sample in samples_ms]
    if not samples_us:
        raise ValueError("missing latency samples")
    if any((not math.isfinite(sample)) or sample <= 0 for sample in samples_us):
        raise ValueError(f"non-finite or non-positive latency sample for {runner_backend}: {samples_us}")

    median_us = float(statistics.median(samples_us))
    p10_us = float(nearest_rank(samples_us, 10))
    p90_us = float(nearest_rank(samples_us, 90))
    q1_us = float(nearest_rank(samples_us, 25))
    q3_us = float(nearest_rank(samples_us, 75))
    iqr_us = q3_us - q1_us
    mean_us = statistics.fmean(samples_us)
    cv_pct = (statistics.pstdev(samples_us) / mean_us * 100.0) if len(samples_us) > 1 and mean_us > 0 else 0.0
    iqr_pct = (iqr_us / median_us * 100.0) if median_us > 0 else 0.0
    outlier_count = sum(sample < 0.5 * median_us or sample > 1.5 * median_us for sample in samples_us)

    return {
        "runner_backend": runner_backend,
        "median_us": median_us,
        "p10_us": p10_us,
        "p90_us": p90_us,
        "samples_us": samples_us,
        "measured_iqr_us": iqr_us,
        "measured_cv_pct": cv_pct,
        "measured_iqr_pct": iqr_pct,
        "variance_pass": cv_pct <= 5.0 and iqr_pct <= 5.0,
        "outlier_count": outlier_count,
    }


def byte_breakdown(total_M, N, K, e_active, group_size, use_bias=True):
    activation_bytes = total_M * K
    activation_scale_bytes = total_M * 2
    activation_zero_point_bytes = total_M
    weight_bytes_ideal = e_active * N * (K // 2)
    weight_bytes_actual = weight_bytes_ideal
    scale_bytes = e_active * (K // group_size) * N * 2
    output_bytes = total_M * N * 2
    bias_bytes = e_active * N * 2 if use_bias else 0
    offsets_bytes = (E + 1) * 8
    scratch_bytes = 0
    memory_bytes = (
        activation_bytes
        + activation_scale_bytes
        + activation_zero_point_bytes
        + weight_bytes_actual
        + scale_bytes
        + output_bytes
        + bias_bytes
        + offsets_bytes
        + scratch_bytes
    )
    cache_bytes = output_bytes + activation_bytes
    return {
        "activation_bytes": activation_bytes,
        "activation_scale_bytes": activation_scale_bytes,
        "activation_zero_point_bytes": activation_zero_point_bytes,
        "weight_bytes_ideal": weight_bytes_ideal,
        "weight_bytes_actual": weight_bytes_actual,
        "scale_bytes": scale_bytes,
        "output_bytes": output_bytes,
        "bias_bytes": bias_bytes,
        "offsets_bytes": offsets_bytes,
        "scratch_bytes": scratch_bytes,
        "memory_bytes": memory_bytes,
        "cache_bytes": cache_bytes,
    }


def dominant_component(launch_us, compute_us, memory_us, cache_us):
    ordered = [
        ("launch", launch_us),
        ("compute", compute_us),
        ("memory", memory_us),
        ("cache", cache_us),
    ]
    best_name, best_value = ordered[0]
    for name, value in ordered[1:]:
        if value > best_value:
            best_name, best_value = name, value
    return best_name


def roofline_fields(total_M, N, K, e_active, group_size, measured_median_us, args):
    bytes_info = byte_breakdown(total_M, N, K, e_active, group_size, args.bias)
    total_ops = 2 * total_M * K * N
    peak_compute_ops_per_us = PEAK_INT8_TOPS * 1e6
    compute_us = total_ops / peak_compute_ops_per_us
    memory_us = bytes_info["memory_bytes"] / (PEAK_BW_GBS * 1000.0)
    cache_us = bytes_info["cache_bytes"] / (CACHE_BANDWIDTH_GBS * 1000.0)
    launch_us = float(args.launch_us)
    roof_us = max(launch_us, compute_us, memory_us, cache_us)
    efficiency = roof_us / measured_median_us

    return {
        **bytes_info,
        "total_ops": total_ops,
        "peak_compute_ops_per_us": peak_compute_ops_per_us,
        "memory_bandwidth_gbs": PEAK_BW_GBS,
        "cache_bandwidth_gbs": CACHE_BANDWIDTH_GBS,
        "memory_bytes_source": "benchmark_compute_bytes_ideal",
        "cache_bandwidth_source": "explicit_assumption",
        "weight_bytes_source": "benchmark_compute_bytes_ideal",
        "launch_us": launch_us,
        "launch_p10_us": launch_us,
        "launch_median_us": launch_us,
        "launch_p90_us": launch_us,
        "launch_source": args.launch_policy,
        "launch_scope": "tp_gemm_backend_family" if args.launch_policy == "calibrated_family" else "row",
        "compute_us": compute_us,
        "memory_us": memory_us,
        "cache_us": cache_us,
        "roof_us": roof_us,
        "efficiency": efficiency,
        "dominant_roof_component": dominant_component(launch_us, compute_us, memory_us, cache_us),
    }


def common_row_fields(row_spec, args, dist_info):
    shape = row_spec.get("shape") or {}
    return {
        "schema_version": 1,
        "spec_version": SPEC_VERSION,
        "row_id": row_id_for(row_spec["tp"], row_spec["gemm"], row_spec["M"], row_spec["distribution"]),
        "device": DEVICE_NAME,
        "host": HOST_ID,
        "container": CONTAINER_NAME,
        "ze_affinity_mask": os.environ.get("ZE_AFFINITY_MASK", ""),
        "mode": mode_for_m(row_spec["M"]),
        "tp": row_spec["tp"],
        "gemm": row_spec["gemm"],
        "M": row_spec["M"],
        "K": shape.get("K"),
        "N": shape.get("N"),
        "E": E,
        "group_size": shape.get("group_size"),
        "top_k": args.top_k,
        "use_bias": bool(args.bias),
        "distribution": row_spec["distribution"],
        "runner_token_dist": dist_info.get("runner_token_dist"),
        "distribution_source": dist_info.get("distribution_source"),
        "backend_aliases": dict(OFFICIAL_BACKEND_ALIASES),
        "warmup": args.warmup,
        "iters": args.iters,
        "repetitions": args.repetitions,
        "pool": args.pool,
        "cache_policy": args.cache_policy,
        "launch_policy": args.launch_policy,
        "seed": args.seed,
    }


def null_measurement_fields():
    return {
        "measured_median_us": None,
        "measured_p10_us": None,
        "measured_p90_us": None,
        "measured_iqr_us": None,
        "measured_cv_pct": None,
        "measured_iqr_pct": None,
        "onednn_w4a8_median_us": None,
        "ipex_w4a16_median_us": None,
        "competitor_margin": None,
        "competitor_pass": False,
        "launch_us": None,
        "compute_us": None,
        "memory_us": None,
        "cache_us": None,
        "roof_us": None,
        "efficiency": None,
        "dominant_roof_component": None,
        "total_ops": None,
        "memory_bytes": None,
        "cache_bytes": None,
        "activation_bytes": None,
        "weight_bytes_ideal": None,
        "weight_bytes_actual": None,
        "scale_bytes": None,
        "output_bytes": None,
        "peak_compute_ops_per_us": None,
        "memory_bandwidth_gbs": None,
        "cache_bandwidth_gbs": None,
        "memory_bytes_source": None,
        "cache_bandwidth_source": None,
        "weight_bytes_source": None,
        "variance_pass": False,
        "outlier_count": 0,
    }


def failure_row(row_spec, args, dist_info, status, message, backends=None):
    row = common_row_fields(row_spec, args, dist_info)
    row.update(null_measurement_fields())
    row.update({
        "status": status,
        "error_type": status,
        "error_message": message,
        "failure_reason": message,
        "backends": backends or {
            "onednn_w4a8": {"runner_backend": "onednn_w4a8", "status": status, "error_message": message},
            "ipex_w4a16": {"runner_backend": "ipex_int4", "status": status, "error_message": message},
        },
    })
    return row


def backend_official_name(runner_backend):
    if runner_backend == "ipex_int4":
        return "ipex_w4a16"
    return runner_backend


def run_backend_measurement(row_spec, args, runner_backend, token_counts):
    shape = row_spec["shape"]
    samples_ms, e_active = run_benchmark(
        row_spec["M"],
        shape["N"],
        shape["K"],
        E,
        shape["group_size"],
        args.top_k,
        runner_backend,
        args.warmup,
        args.iters,
        args.pool,
        token_dist="real",
        use_bias=args.bias,
        token_counts=token_counts,
        repetitions=args.repetitions,
        return_samples=True,
        raise_errors=True,
    )
    return timing_summary(samples_ms, runner_backend), e_active


def build_paired_row(row_spec, args):
    try:
        dist_info = distribution_counts(row_spec["distribution"], row_spec["M"])
    except KeyError as exc:
        return failure_row(row_spec, args, {"runner_token_dist": None, "distribution_source": None}, "missing_distribution", str(exc))

    shape = row_spec.get("shape")
    if shape is None:
        return failure_row(row_spec, args, dist_info, "unsupported_shape", "unsupported TP/GEMM shape")

    token_counts = dist_info["token_counts"]
    e_active = sum(1 for count in token_counts if count > 0)
    backend_results = {}
    backend_errors = {}

    for runner_backend in OFFICIAL_BACKENDS:
        official_name = backend_official_name(runner_backend)
        try:
            summary, measured_e_active = run_backend_measurement(row_spec, args, runner_backend, token_counts)
            e_active = measured_e_active
            backend_results[official_name] = summary
        except ValueError as exc:
            backend_errors[official_name] = ("nan", str(exc))
        except Exception as exc:
            backend_errors[official_name] = ("crash", str(exc))
        clear_xpu_cache()

    if backend_errors:
        backends = {}
        for official_name in ("onednn_w4a8", "ipex_w4a16"):
            if official_name in backend_results:
                backends[official_name] = backend_results[official_name]
            else:
                runner_backend = "ipex_int4" if official_name == "ipex_w4a16" else official_name
                error_type, error_message = backend_errors.get(official_name, ("crash", "missing backend measurement"))
                backends[official_name] = {
                    "runner_backend": runner_backend,
                    "status": error_type,
                    "error_message": error_message,
                }
        first_error = next(iter(backend_errors.values()))
        return failure_row(row_spec, args, dist_info, first_error[0], first_error[1], backends)

    target = backend_results["onednn_w4a8"]
    competitor = backend_results["ipex_w4a16"]
    competitor_margin = competitor["median_us"] / target["median_us"]
    competitor_pass = competitor_margin > 1.0
    roof_fields = roofline_fields(row_spec["M"], shape["N"], shape["K"], e_active, shape["group_size"], target["median_us"], args)

    row = common_row_fields(row_spec, args, dist_info)
    row.update({
        "measured_median_us": target["median_us"],
        "measured_p10_us": target["p10_us"],
        "measured_p90_us": target["p90_us"],
        "measured_iqr_us": target["measured_iqr_us"],
        "measured_cv_pct": target["measured_cv_pct"],
        "measured_iqr_pct": target["measured_iqr_pct"],
        "onednn_w4a8_median_us": target["median_us"],
        "onednn_w4a8_p10_us": target["p10_us"],
        "onednn_w4a8_p90_us": target["p90_us"],
        "onednn_w4a8_iters": args.iters,
        "ipex_w4a16_median_us": competitor["median_us"],
        "ipex_w4a16_p10_us": competitor["p10_us"],
        "ipex_w4a16_p90_us": competitor["p90_us"],
        "ipex_w4a16_iters": args.iters,
        "competitor_margin": competitor_margin,
        "competitor_pass": competitor_pass,
        "failure_reason": None if competitor_pass else f"{row_id_for(row_spec['tp'], row_spec['gemm'], row_spec['M'], row_spec['distribution'])}: competitor_margin={competitor_margin:.6f} <= 1.0",
        "variance_pass": target["variance_pass"],
        "outlier_count": target["outlier_count"],
        "status": "pass",
        "backends": backend_results,
    })
    row.update(roof_fields)
    return row


def full_matrix_specs():
    specs = []
    for total_M in [*DECODE_M_VALUES, *PREFILL_M_VALUES]:
        for tp in (4, 8):
            for gemm in ("GEMM1", "GEMM2"):
                for distribution in OFFICIAL_DISTRIBUTIONS:
                    specs.append({"tp": tp, "gemm": gemm, "M": total_M, "distribution": distribution, "shape": shape_for(tp, gemm)})
    return specs


def subset_matrix_specs():
    return [
        {"tp": 4, "gemm": "GEMM1", "M": 4, "distribution": "balanced", "shape": shape_for(4, "GEMM1")},
        {"tp": 4, "gemm": "GEMM1", "M": 8, "distribution": "balanced", "shape": shape_for(4, "GEMM1")},
        {"tp": 8, "gemm": "GEMM2", "M": 128, "distribution": "sparse-active", "shape": shape_for(8, "GEMM2")},
    ]


def single_matrix_specs(args, configs):
    if args.M is None:
        total_M = 3072 * args.top_k
    else:
        total_M = args.M
    distribution = args.distribution
    if distribution is None:
        distribution = "balanced" if args.token_dist == "uniform" else "routed/skew"
    specs = []
    for label, _ in configs:
        label_parts = label.split()
        tp = int(label_parts[0].replace("TP", ""))
        gemm = label_parts[1]
        specs.append({"tp": tp, "gemm": gemm, "M": total_M, "distribution": distribution, "shape": shape_for(tp, gemm)})
    return specs


def selected_matrix_specs(args, configs):
    if args.matrix == "full":
        return full_matrix_specs()
    if args.matrix == "subset":
        return subset_matrix_specs()
    return single_matrix_specs(args, configs)


def write_json_matrix(args, configs):
    rows = []
    specs = selected_matrix_specs(args, configs)
    for index, row_spec in enumerate(specs, 1):
        row_id = row_id_for(row_spec["tp"], row_spec["gemm"], row_spec["M"], row_spec["distribution"])
        print(f"[{index}/{len(specs)}] {row_id}")
        rows.append(build_paired_row(row_spec, args))

    payload = {
        "schema_version": 1,
        "spec_version": SPEC_VERSION,
        "task": "T6 shape matrix runner subset" if args.matrix == "subset" else "T6 shape matrix runner",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": DEVICE_NAME,
        "host": HOST_ID,
        "container": CONTAINER_NAME,
        "ze_affinity_mask": os.environ.get("ZE_AFFINITY_MASK", ""),
        "matrix": args.matrix,
        "row_count": len(rows),
        "status_counts": {status: sum(row.get("status") == status for row in rows) for status in sorted({row.get("status") for row in rows})},
        "rows": rows,
    }
    output_path = Path(args.json_out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {len(rows)} rows to {output_path}")

# ---------------------------------------------------------------------------
# Offline mode
# ---------------------------------------------------------------------------

def run_offline(args, configs, backends):
    """Fixed total_M offline benchmark with roofline analysis."""
    if args.M is not None:
        total_M = args.M
        tokens_label = "decode override"
    else:
        total_M = 3072 * args.top_k  # 12288 by default
        tokens_label = f"tokens=3072, top_k={args.top_k}"

    print("=" * 110)
    print("Roofline Benchmark — OFFLINE MODE")
    print(f"E={E}, Total M={total_M} ({tokens_label}), bias={'on' if args.bias else 'off'}")
    print(f"Peak INT8: {PEAK_INT8_TOPS} TOPS | Peak BF16: {PEAK_BF16_TFLOPS} TFLOPS | Peak BW: {PEAK_BW_GBS} GB/s")
    print(f"Backends: {backends}")
    print("=" * 110)

    header = (f"{'Config':<12} | {'Backend':<16} | {'M':>6} | {'ms':>8} | "
              f"{'TOPS/TFLOPS':>11} | {'BW(GB/s)':>9} | {'AI':>7} | {'Bound':<8} | {'Roofline%':>9}")
    print(header)
    print("-" * len(header))

    for label, cfg in configs:
        N, K, group_size = cfg["N"], cfg["K"], cfg["group_size"]
        for backend in backends:
            ms, e_active = run_benchmark(total_M, N, K, E, group_size, args.top_k, backend, args.warmup, args.iters, args.pool, use_bias=args.bias)
            if ms is not None:
                tops, bw, ai, bound, roofline_pct = roofline_stats(ms, total_M, N, K, e_active, group_size, backend, args.bias)
                print(f"{label:<12} | {backend:<16} | {total_M:>6} | {ms:>8.3f} | "
                      f"{tops:>11.2f} | {bw:>9.1f} | {ai:>7.2f} | {bound:<8} | {roofline_pct:>8.1f}%")
            else:
                print(f"{label:<12} | {backend:<16} | {total_M:>6} | {'FAILED':>8} | "
                      f"{'—':>11} | {'—':>9} | {'—':>7} | {'—':<8} | {'—':>9}")
        clear_xpu_cache()

    print()
    print("Done.")


# ---------------------------------------------------------------------------
# Server mode
# ---------------------------------------------------------------------------

def run_server(args, configs, backends):
    """Server mode: benchmark representative M values from distribution."""
    # Find distribution file
    dist_file = args.dist_file
    if dist_file is None:
        # Auto-find relative to script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        dist_file = os.path.join(os.path.dirname(script_dir), "gpt-oss-120b-token-counts.md")
    if not os.path.exists(dist_file):
        raise FileNotFoundError(f"Distribution file not found: {dist_file}")

    dist = parse_token_distribution(dist_file)
    # 3072 tokens is the offline measurement point; exclude from server stats.
    OFFLINE_TOKENS = 3072
    dropped_offline = dist.pop(OFFLINE_TOKENS, 0)
    filtered = filter_distribution(dist, coverage=args.coverage)

    total_occurrences = sum(dist.values())
    filtered_occurrences = sum(filtered.values())
    num_m_values = len(filtered)

    print("=" * 110)
    print("Roofline Benchmark — SERVER MODE")
    print(f"E={E}, top_k={args.top_k}, bias={'on' if args.bias else 'off'}")
    print(f"Distribution: {len(dist)} unique token counts, {total_occurrences} total occurrences "
          f"(dropped {dropped_offline} occurrences at tokens={OFFLINE_TOKENS} — offline-only)")
    print(f"After {args.coverage*100:.0f}% coverage filter: {num_m_values} M values, "
          f"{filtered_occurrences} occurrences ({filtered_occurrences/total_occurrences*100:.1f}%)")
    print(f"Peak INT8: {PEAK_INT8_TOPS} TOPS | Peak BF16: {PEAK_BF16_TFLOPS} TFLOPS | Peak BW: {PEAK_BW_GBS} GB/s")
    print(f"Backends: {backends}")
    print("=" * 110)

    # Convert token counts to total_M (tokens * top_k)
    m_values = sorted(filtered.keys())
    m_weights = {tokens: filtered[tokens] for tokens in m_values}

    header = (f"{'Config':<12} | {'Backend':<16} | {'M':>6} | {'ms':>8} | "
              f"{'TOPS/TFLOPS':>11} | {'BW(GB/s)':>9} | {'AI':>7} | {'Bound':<8} | {'Roofline%':>9}")
    print(header)
    print("-" * len(header))

    # Store results for weighted average: results[(label, backend)] = [(ms, total_M, weight), ...]
    all_results = {}

    for label, cfg in configs:
        N, K, group_size = cfg["N"], cfg["K"], cfg["group_size"]
        for backend in backends:
            key = (label, backend)
            all_results[key] = []
            for tokens in m_values:
                total_M = tokens * args.top_k
                if total_M < 1:
                    continue
                ms, e_active = run_benchmark(total_M, N, K, E, group_size, args.top_k, backend, args.warmup, args.iters, args.pool, args.token_dist, use_bias=args.bias)
                weight = m_weights[tokens]
                if ms is not None:
                    tops, bw, ai, bound, roofline_pct = roofline_stats(ms, total_M, N, K, e_active, group_size, backend, args.bias)
                    print(f"{label:<12} | {backend:<16} | {total_M:>6} | {ms:>8.3f} | "
                          f"{tops:>11.2f} | {bw:>9.1f} | {ai:>7.2f} | {bound:<8} | {roofline_pct:>8.1f}%")
                    all_results[key].append((ms, total_M, weight, e_active))
                else:
                    print(f"{label:<12} | {backend:<16} | {total_M:>6} | {'FAILED':>8} | "
                          f"{'—':>11} | {'—':>9} | {'—':>7} | {'—':<8} | {'—':>9}")
            clear_xpu_cache()

    # Weighted average summary
    print()
    print("=" * 110)
    print("=== Weighted Average ===")
    print("=" * 110)
    print(header)
    print("-" * len(header))

    for label, cfg in configs:
        N, K, group_size = cfg["N"], cfg["K"], cfg["group_size"]
        for backend in backends:
            key = (label, backend)
            entries = all_results.get(key, [])
            if not entries:
                print(f"{label:<12} | {backend:<16} | {'—':>6} | {'N/A':>8} | "
                      f"{'—':>11} | {'—':>9} | {'—':>7} | {'—':<8} | {'—':>9}")
                continue

            total_weight = sum(w for _, _, w, _ in entries)
            weighted_ms = sum(ms * w for ms, _, w, _ in entries) / total_weight
            weighted_M = sum(m * w for _, m, w, _ in entries) / total_weight
            avg_M = int(round(weighted_M))
            avg_e_active = int(round(sum(ea * w for _, _, w, ea in entries) / total_weight))

            tops, bw, ai, bound, roofline_pct = roofline_stats(weighted_ms, avg_M, N, K, avg_e_active, group_size, backend, args.bias)
            print(f"{label:<12} | {backend:<16} | {avg_M:>6} | {weighted_ms:>8.3f} | "
                  f"{tops:>11.2f} | {bw:>9.1f} | {ai:>7.2f} | {bound:<8} | {roofline_pct:>8.1f}%")

    # Tokens-bucket summary
    BUCKETS = [
        ("tokens<32",    lambda tm: tm < 128),
        ("tokens 32-128", lambda tm: 128 <= tm < 512),
        ("tokens 128-512", lambda tm: 512 <= tm < 2048),
        ("tokens>=512",  lambda tm: tm >= 2048),
    ]
    print()
    print("=" * 110)
    print("=== Tokens-Bucket Summary ===")
    print("=" * 110)
    print(header)
    print("-" * len(header))
    for bucket_label, bucket_fn in BUCKETS:
        for label, cfg in configs:
            N, K, group_size = cfg["N"], cfg["K"], cfg["group_size"]
            for backend in backends:
                key = (label, backend)
                entries = [e for e in all_results.get(key, []) if bucket_fn(e[1])]
                if not entries:
                    continue
                total_weight = sum(w for _, _, w, _ in entries)
                weighted_ms = sum(ms * w for ms, _, w, _ in entries) / total_weight
                weighted_M = sum(m * w for _, m, w, _ in entries) / total_weight
                avg_M = int(round(weighted_M))
                avg_e_active = int(round(sum(ea * w for _, _, w, ea in entries) / total_weight))
                tops, bw, ai, bound, roofline_pct = roofline_stats(weighted_ms, avg_M, N, K, avg_e_active, group_size, backend, args.bias)
                print(f"{label:<12} | {backend:<16} | {avg_M:>6} | {weighted_ms:>8.3f} | "
                      f"{tops:>11.2f} | {bw:>9.1f} | {ai:>7.2f} | {bound:<8} | {roofline_pct:>8.1f}% | {bucket_label}")

    print()
    print("Done.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark grouped GEMM with roofline analysis (offline/server modes)")
    parser.add_argument("--mode", type=str, choices=["offline", "server"], required=True,
                        help="Benchmark mode: offline (fixed M) or server (distribution-based)")
    parser.add_argument("--tp", type=str, choices=["4", "8", "both"], default="both",
                        help="Tensor parallelism: 4, 8, or both (default: both)")
    parser.add_argument("--backend", type=str, nargs="+",
                        default=["onednn_w4a8", "ipex_mxfp4"],
                        choices=ALL_BACKENDS + ["ipex_w4a16", "all"],
                        help="Backend(s) to benchmark (default: onednn_w4a8 ipex_mxfp4; pass 'all' for full sweep)")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100,
                        help="Kernel calls per batch-async timing rep. Default: 100.")
    parser.add_argument("--repetitions", type=int, default=3,
                        help="Timing batch repetitions for JSON output and runner samples. Default: 3.")
    parser.add_argument("--pool", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=4, dest="top_k")
    parser.add_argument("--dist-file", type=str, default=None,
                        help="Path to token distribution markdown file (default: auto-find)")
    parser.add_argument("--coverage", type=float, default=0.95,
                        help="Fraction of mass to keep for server mode (default: 0.95)")
    parser.add_argument("--token-dist", type=str, choices=["real", "uniform"], default="real",
                        help="Token distribution across experts: real (REAL_EXPERT_TOKENS) or uniform (default: real)")
    parser.add_argument("--bias", type=str, choices=["on", "off"], default="on",
                        help="Include per-expert bias [E,N] bf16 to match gpt-oss production (default: on)")
    parser.add_argument("--M", type=int, default=None,
                        help="Override total_M for offline mode (e.g. 68 for decode). Default: 3072*top_k.")
    parser.add_argument("--distribution", type=str, choices=OFFICIAL_DISTRIBUTIONS, default=None,
                        help="Official JSON distribution label. Defaults from --token-dist for single-row JSON.")
    parser.add_argument("--matrix", type=str, choices=["single", "subset", "full"], default="single",
                        help="JSON matrix shape set: current CLI selection, T6 subset, or full B60 matrix.")
    parser.add_argument("--subset", action="store_true",
                        help="Alias for --matrix subset.")
    parser.add_argument("--full-matrix", action="store_true",
                        help="Alias for --matrix full.")
    parser.add_argument("--json-out", type=str, default=None,
                        help="Write paired official JSON rows to this path.")
    parser.add_argument("--cache-policy", type=str, choices=["cold_pool", "warm_repeat"], default="cold_pool")
    parser.add_argument("--launch-policy", type=str,
                        choices=["measured_empty_submit", "measured_noop_kernel", "measured_backend_enqueue", "calibrated_family", "smoke_fixture"],
                        default="calibrated_family")
    parser.add_argument("--launch-us", type=float, default=DEFAULT_LAUNCH_US,
                        help="Calibrated launch lower bound in microseconds for JSON roofline rows.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.bias = (args.bias == "on")
    if args.subset:
        args.matrix = "subset"
    if args.full_matrix:
        args.matrix = "full"
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.json_out and args.mode != "offline":
        parser.error("--json-out currently supports --mode offline")
    if args.matrix != "single" and not args.json_out:
        parser.error("--matrix subset/full requires --json-out")

    backends = normalize_backend_list(ALL_BACKENDS if "all" in args.backend else args.backend)

    configs = []
    if args.tp in ("4", "both"):
        configs.append(("TP4 GEMM1", TP4_GEMM1))
        configs.append(("TP4 GEMM2", TP4_GEMM2))
    if args.tp in ("8", "both"):
        configs.append(("TP8 GEMM1", TP8_GEMM1))
        configs.append(("TP8 GEMM2", TP8_GEMM2))

    if args.json_out:
        write_json_matrix(args, configs)
    elif args.mode == "offline":
        run_offline(args, configs, backends)
    elif args.mode == "server":
        run_server(args, configs, backends)


if __name__ == "__main__":
    main()
