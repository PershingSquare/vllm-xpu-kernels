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
    """Scale token distribution to target M with proper rounding."""
    tokens = np.array(tokens, dtype=np.float64)
    scaled = tokens / tokens.sum() * M
    scaled = np.round(scaled).astype(int)

    # Fix rounding errors by adjusting first elements
    diff = M - scaled.sum()
    if diff != 0:
        scaled[0] += diff

    return scaled.tolist()


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

def bench_cold(fns_list, warmup, iters, repetitions=3):
    """Time with rotating pool of pre-allocated buffers for cold-cache simulation.

    fns_list[i] is the function for pool slot i (each has its own buffers).

    Each iter is timed individually with a per-iter sync. This is slower than
    timing a batch of iters with one sync at the end, but it gives accurate
    per-call latency (matches what a server actually pays per request) and
    eliminates launch-queue variance that would otherwise inflate run-to-run
    noise to 10-30%. We then take the median of `iters` per-iter latencies
    within each repetition, and report the median across `repetitions`.
    """
    POOL = len(fns_list)
    for i in range(warmup):
        fns_list[i % POOL]()
    torch.xpu.synchronize()

    rep_medians = []
    for r in range(repetitions):
        per_iter = []
        for i in range(iters):
            t0 = time.perf_counter()
            fns_list[i % POOL]()
            torch.xpu.synchronize()
            per_iter.append((time.perf_counter() - t0) * 1000.0)
        per_iter.sort()
        rep_medians.append(per_iter[len(per_iter) // 2])
    rep_medians.sort()
    return rep_medians[len(rep_medians) // 2]


# ---------------------------------------------------------------------------
# Backend runners with pool-based cold-cache simulation
# ---------------------------------------------------------------------------

def run_onednn_w4a8(M, N, K, E, group_size, offsets, max_expert_size, warmup, iters, POOL):
    B_s4 = (torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8, device=DEVICE) ^ 0x88).contiguous()
    B_scales = torch.rand(E, K // group_size, N, dtype=torch.bfloat16, device=DEVICE) * 0.5 + 0.01

    pool_Aq = [torch.empty(M, K, dtype=torch.uint8, device=DEVICE) for _ in range(POOL)]
    pool_Ascale = [torch.empty(M, dtype=torch.bfloat16, device=DEVICE) for _ in range(POOL)]
    pool_Azp = [torch.empty(M, dtype=torch.uint8, device=DEVICE) for _ in range(POOL)]
    pool_D = [torch.empty(M, N, dtype=torch.bfloat16, device=DEVICE) for _ in range(POOL)]

    def make_fn(slot):
        A_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device=DEVICE) * 0.1
        A_q, A_scale, A_zp = quantize_per_token_u8(A_bf16)
        pool_Aq[slot].copy_(A_q)
        pool_Ascale[slot].copy_(A_scale.flatten())
        pool_Azp[slot].copy_(A_zp.flatten())

        def fn():
            torch.ops._xpu_C.onednn_grouped_gemm_w4a8(
                pool_Aq[slot], pool_Ascale[slot], pool_Azp[slot],
                B_s4, B_scales, None,
                pool_D[slot], offsets, N, K, E, max_expert_size)
        return fn

    fns_list = [make_fn(i) for i in range(POOL)]
    return bench_cold(fns_list, warmup, iters)


def run_onednn_w4a16(M, N, K, E, group_size, offsets, max_expert_size, warmup, iters, POOL):
    B_s4 = (torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8, device=DEVICE) ^ 0x88).contiguous()
    B_scales = torch.rand(E, K // group_size, N, dtype=torch.bfloat16, device=DEVICE) * 0.5 + 0.01

    pool_A = [torch.empty(M, K, dtype=torch.bfloat16, device=DEVICE) for _ in range(POOL)]
    pool_D = [torch.empty(M, N, dtype=torch.bfloat16, device=DEVICE) for _ in range(POOL)]

    def make_fn(slot):
        pool_A[slot].copy_(torch.randn(M, K, dtype=torch.bfloat16, device=DEVICE) * 0.1)

        def fn():
            torch.ops._xpu_C.onednn_grouped_gemm_w4a16(
                pool_A[slot], B_s4, B_scales, None,
                pool_D[slot], offsets, N, K, E,
                True, False, max_expert_size)
        return fn

    fns_list = [make_fn(i) for i in range(POOL)]
    return bench_cold(fns_list, warmup, iters)


def run_cutlass_w4a16(M, N, K, E, group_size, offsets, warmup, iters, POOL):
    """Cutlass w4a16: bf16 activations, int4 weights."""
    old = os.environ.get("VLLM_XPU_GROUPED_GEMM_BACKEND", "")
    os.environ["VLLM_XPU_GROUPED_GEMM_BACKEND"] = ""
    try:
        # B and scales are fixed (shared across pool)
        B_packed = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8, device=DEVICE)
        B_scales = torch.rand(E, N, K // group_size, dtype=torch.bfloat16, device=DEVICE) * 0.5 + 0.01

        # Pool of buffers: (Abf, D)
        pool_A = [torch.empty(M, K, dtype=torch.bfloat16, device=DEVICE) for _ in range(POOL)]
        pool_D = [torch.empty(M, N, dtype=torch.bfloat16, device=DEVICE) for _ in range(POOL)]

        def make_fn(slot):
            A = torch.randn(M, K, dtype=torch.bfloat16, device=DEVICE) * 0.1
            pool_A[slot].copy_(A)

            def fn():
                torch.ops._xpu_C.grouped_gemm_interface(
                    ptr_A=pool_A[slot], ptr_B=B_packed, ptr_scales=B_scales, ptr_bias=None,
                    ptr_D=pool_D[slot], expert_first_token_offset=offsets,
                    N=N, K=K, num_experts=E, is_B_int4=True, is_B_mxfp4=False)
            return fn

        fns_list = [make_fn(i) for i in range(POOL)]
        return bench_cold(fns_list, warmup, iters)
    finally:
        os.environ["VLLM_XPU_GROUPED_GEMM_BACKEND"] = old


def run_cutlass_mxfp4(M, N, K, E, group_size, offsets, warmup, iters, POOL):
    """Cutlass mxfp4: bf16 activations, mxfp4 weights, uint8 MX scales."""
    old = os.environ.get("VLLM_XPU_GROUPED_GEMM_BACKEND", "")
    os.environ["VLLM_XPU_GROUPED_GEMM_BACKEND"] = ""
    try:
        # B packed u8, scales uint8 [E,N,K//gs]
        B_packed = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8, device=DEVICE)
        B_scales = torch.randint(0, 256, (E, N, K // group_size), dtype=torch.uint8, device=DEVICE)

        # Pool of buffers: (Abf, D)
        pool_A = [torch.empty(M, K, dtype=torch.bfloat16, device=DEVICE) for _ in range(POOL)]
        pool_D = [torch.empty(M, N, dtype=torch.bfloat16, device=DEVICE) for _ in range(POOL)]

        def make_fn(slot):
            A = torch.randn(M, K, dtype=torch.bfloat16, device=DEVICE) * 0.1
            pool_A[slot].copy_(A)

            def fn():
                torch.ops._xpu_C.grouped_gemm_interface(
                    ptr_A=pool_A[slot], ptr_B=B_packed, ptr_scales=B_scales, ptr_bias=None,
                    ptr_D=pool_D[slot], expert_first_token_offset=offsets,
                    N=N, K=K, num_experts=E, is_B_int4=False, is_B_mxfp4=True)
            return fn

        fns_list = [make_fn(i) for i in range(POOL)]
        return bench_cold(fns_list, warmup, iters)
    finally:
        os.environ["VLLM_XPU_GROUPED_GEMM_BACKEND"] = old


def run_ipex_int4(M, N, K, E, group_size, offsets, warmup, iters, POOL):
    """IPEX moe_gemm int4: bf16 activations, int4 weights, bf16 scales."""
    import torch.xpu as xpu
    import intel_extension_for_pytorch  # noqa

    ipex_group_num = K // group_size
    # IPEX weight layout: [E, K//2, N]
    B_packed = torch.randint(0, 256, (E, K // 2, N), dtype=torch.uint8, device=DEVICE)
    B_scales = torch.rand(E, ipex_group_num, N, dtype=torch.bfloat16, device=DEVICE) * 0.5 + 0.01
    counts = offsets[1:] - offsets[:-1]
    rows_for_experts = counts.to(torch.int32)

    # Pool of buffers: Abf
    pool_A = [torch.empty(M, K, dtype=torch.bfloat16, device=DEVICE) for _ in range(POOL)]

    def make_fn(slot):
        A = torch.randn(M, K, dtype=torch.bfloat16, device=DEVICE) * 0.1
        pool_A[slot].copy_(A)

        def fn():
            xpu.moe_gemm(pool_A[slot], B_packed, rows_for_experts, E,
                         matrix_b_scale_inv=B_scales, is_int4=True)
        return fn

    fns_list = [make_fn(i) for i in range(POOL)]
    return bench_cold(fns_list, warmup, iters)


def run_ipex_mxfp4(M, N, K, E, group_size, offsets, warmup, iters, POOL):
    """IPEX moe_gemm mxfp4: bf16 activations, mxfp4 weights, uint8 MX scales (group_size=32 only)."""
    import torch.xpu as xpu
    import intel_extension_for_pytorch  # noqa

    # IPEX mxfp4 requires group_size=32
    ipex_group_size = 32
    ipex_group_num = K // ipex_group_size
    # IPEX weight layout: [E, K//2, N] (transposed vs cutlass [E, N, K//2])
    B_packed = torch.randint(0, 256, (E, K // 2, N), dtype=torch.uint8, device=DEVICE)
    B_scales = torch.randint(0, 256, (E, ipex_group_num, N), dtype=torch.uint8, device=DEVICE)
    counts = offsets[1:] - offsets[:-1]
    rows_for_experts = counts.to(torch.int32)

    # Pool of buffers: Abf
    pool_A = [torch.empty(M, K, dtype=torch.bfloat16, device=DEVICE) for _ in range(POOL)]

    def make_fn(slot):
        A = torch.randn(M, K, dtype=torch.bfloat16, device=DEVICE) * 0.1
        pool_A[slot].copy_(A)

        def fn():
            xpu.moe_gemm(pool_A[slot], B_packed, rows_for_experts, E,
                         matrix_b_scale_inv=B_scales, is_mxfp4=True)
        return fn

    fns_list = [make_fn(i) for i in range(POOL)]
    return bench_cold(fns_list, warmup, iters)


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

def compute_bytes(total_M, N, K, e_active, group_size, backend):
    if backend == "onednn_w4a8":
        bytes_a = total_M * K * 1
        bytes_b = e_active * N * K // 2
        bytes_d = total_M * N * 2
        bytes_scales_w = e_active * (K // group_size) * N * 2
        bytes_scales_a = total_M * 2
        return bytes_a + bytes_b + bytes_d + bytes_scales_w + bytes_scales_a
    elif backend in ("onednn_w4a16", "cutlass_w4a16", "ipex_int4"):
        bytes_a = total_M * K * 2
        bytes_b = e_active * N * K // 2
        bytes_d = total_M * N * 2
        bytes_scales_w = e_active * (K // group_size) * N * 2
        return bytes_a + bytes_b + bytes_d + bytes_scales_w
    elif backend in ("cutlass_mxfp4", "ipex_mxfp4"):
        bytes_a = total_M * K * 2
        bytes_b = e_active * N * K // 2
        bytes_d = total_M * N * 2
        bytes_scales_w = e_active * (K // group_size) * N * 1
        return bytes_a + bytes_b + bytes_d + bytes_scales_w
    else:
        raise ValueError(f"Unknown backend: {backend}")


def roofline_stats(ms, total_M, N, K, e_active, group_size, backend):
    flops = 2 * total_M * N * K
    bytes_total = compute_bytes(total_M, N, K, e_active, group_size, backend)

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

def run_benchmark(total_M, N, K, E, group_size, top_k, backend, warmup, iters, POOL, token_dist="real"):
    if token_dist == "real":
        token_counts = scale_tokens_to_M(REAL_EXPERT_TOKENS, total_M)
    else:
        token_counts = scale_tokens_uniform(total_M, E)
    offsets = build_offsets_from_distribution(token_counts, DEVICE)
    max_expert_size = (total_M + top_k - 1) // top_k
    e_active = count_active_experts(offsets)

    try:
        if backend in ONEDNN_BACKENDS:
            ms = RUNNERS[backend](total_M, N, K, E, group_size, offsets, max_expert_size, warmup, iters, POOL)
        else:
            ms = RUNNERS[backend](total_M, N, K, E, group_size, offsets, warmup, iters, POOL)
        return ms, e_active
    except Exception as e:
        print(f"  {backend}: FAILED - {str(e)[:80]}")
        return None, e_active
    finally:
        gc.collect()


# ---------------------------------------------------------------------------
# Offline mode
# ---------------------------------------------------------------------------

def run_offline(args, configs, backends):
    """Fixed total_M offline benchmark with roofline analysis."""
    total_M = 3072 * args.top_k  # 12288 by default

    print("=" * 110)
    print("Roofline Benchmark — OFFLINE MODE")
    print(f"E={E}, Total M={total_M} (tokens=3072, top_k={args.top_k})")
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
            ms, e_active = run_benchmark(total_M, N, K, E, group_size, args.top_k, backend, args.warmup, args.iters, args.pool)
            if ms is not None:
                tops, bw, ai, bound, roofline_pct = roofline_stats(ms, total_M, N, K, e_active, group_size, backend)
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
    print(f"E={E}, top_k={args.top_k}")
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
                ms, e_active = run_benchmark(total_M, N, K, E, group_size, args.top_k, backend, args.warmup, args.iters, args.pool, args.token_dist)
                weight = m_weights[tokens]
                if ms is not None:
                    tops, bw, ai, bound, roofline_pct = roofline_stats(ms, total_M, N, K, e_active, group_size, backend)
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

            tops, bw, ai, bound, roofline_pct = roofline_stats(weighted_ms, avg_M, N, K, avg_e_active, group_size, backend)
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
                tops, bw, ai, bound, roofline_pct = roofline_stats(weighted_ms, avg_M, N, K, avg_e_active, group_size, backend)
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
                        choices=ALL_BACKENDS + ["all"],
                        help="Backend(s) to benchmark (default: onednn_w4a8 ipex_mxfp4; pass 'all' for full sweep)")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=30,
                        help="Per-iter timed calls (synced per iter). Default: 30.")
    parser.add_argument("--pool", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=4, dest="top_k")
    parser.add_argument("--dist-file", type=str, default=None,
                        help="Path to token distribution markdown file (default: auto-find)")
    parser.add_argument("--coverage", type=float, default=0.95,
                        help="Fraction of mass to keep for server mode (default: 0.95)")
    parser.add_argument("--token-dist", type=str, choices=["real", "uniform"], default="real",
                        help="Token distribution across experts: real (REAL_EXPERT_TOKENS) or uniform (default: real)")
    args = parser.parse_args()

    backends = ALL_BACKENDS if "all" in args.backend else args.backend

    configs = []
    if args.tp in ("4", "both"):
        configs.append(("TP4 GEMM1", TP4_GEMM1))
        configs.append(("TP4 GEMM2", TP4_GEMM2))
    if args.tp in ("8", "both"):
        configs.append(("TP8 GEMM1", TP8_GEMM1))
        configs.append(("TP8 GEMM2", TP8_GEMM2))

    if args.mode == "offline":
        run_offline(args, configs, backends)
    elif args.mode == "server":
        run_server(args, configs, backends)


if __name__ == "__main__":
    main()
