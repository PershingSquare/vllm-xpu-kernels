# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark grouped GEMM: oneDNN w4a8 vs oneDNN w4a16 vs Cutlass w4a16.

Uses realistic gpt-oss-120b MoE shapes and real expert token distributions.

Usage:
    ZE_AFFINITY_MASK=5 python benchmark/benchmark_grouped_gemm_w4a8.py
    ZE_AFFINITY_MASK=5 python benchmark/benchmark_grouped_gemm_w4a8.py --backend onednn_w4a8 onednn_w4a16
    ZE_AFFINITY_MASK=5 python benchmark/benchmark_grouped_gemm_w4a8.py --gemm both
"""

import argparse
import gc
import os
import time

import torch

import vllm_xpu_kernels._xpu_C  # noqa: F401

DEVICE = "xpu"
ALL_BACKENDS = ["onednn_w4a8", "onednn_w4a16", "cutlass_w4a16"]

# gpt-oss-120b: hidden_size=2880, intermediate_size=2880, num_experts=128
# GEMM1 (gate+up fused): K=2944, N=1536 (padded K=ceil(2880/128)*128, N=2*768)
# GEMM2 (down):          K=768, N=2880
GPT_OSS_120B_GEMM1 = {"K": 2944, "N": 1536, "E": 128, "group_size": 128}
GPT_OSS_120B_GEMM2 = {"K": 768, "N": 2880, "E": 128, "group_size": 128}

# Real expert token distribution from gpt-oss-120b inference (sums to 12230)
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
assert sum(REAL_EXPERT_TOKENS) == 12330


def clear_xpu_cache():
    torch.xpu.synchronize()
    torch.xpu.empty_cache()
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


# ---------------------------------------------------------------------------
# Timing utility — ensures data comes from memory not cache
# ---------------------------------------------------------------------------

def bench_fn_no_cache(fn_factory, warmup, iters):
    """Time fn with fresh data each iteration to avoid L2 cache hits.

    fn_factory() returns a new callable with fresh tensors each time.
    We pre-generate all iterations' data, then time execution.
    """
    # Pre-generate all callables with fresh data
    fns = [fn_factory() for _ in range(warmup + iters)]

    # Warmup with fresh data
    for i in range(warmup):
        fns[i]()
    torch.xpu.synchronize()

    # Timed with fresh data
    start = time.perf_counter()
    for i in range(warmup, warmup + iters):
        fns[i]()
    torch.xpu.synchronize()
    elapsed = time.perf_counter() - start
    return (elapsed / iters) * 1000.0  # ms


# ---------------------------------------------------------------------------
# Backend runners
# ---------------------------------------------------------------------------

def run_onednn_w4a8(M, N, K, E, group_size, offsets, warmup, iters):
    """oneDNN w4a8: u8 activations, s4 weights, bf16 scales."""
    # Pre-allocate fixed weights (these would be in memory anyway)
    B_packed = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8, device=DEVICE)
    B_scales = torch.rand(E, N, K // group_size, dtype=torch.bfloat16, device=DEVICE) * 0.5 + 0.01
    D = torch.empty(M, N, dtype=torch.bfloat16, device=DEVICE)

    def make_fn():
        # Fresh activations each call to avoid cache
        A_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device=DEVICE) * 0.1
        A_q, A_scale, A_zp = quantize_per_token_u8(A_bf16)
        A_scale_flat = A_scale.flatten()
        A_zp_flat = A_zp.flatten()
        def fn():
            torch.ops._xpu_C.onednn_grouped_gemm_w4a8(
                A_q, A_scale_flat, A_zp_flat, B_packed, B_scales, None,
                D, offsets, N, K, E)
        return fn

    return bench_fn_no_cache(make_fn, warmup, iters)


def run_onednn_w4a16(M, N, K, E, group_size, offsets, warmup, iters):
    """oneDNN w4a16: bf16 activations, s4 weights, bf16 scales."""
    old = os.environ.get("VLLM_XPU_GROUPED_GEMM_BACKEND", "")
    os.environ["VLLM_XPU_GROUPED_GEMM_BACKEND"] = "onednn"
    try:
        B_packed = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8, device=DEVICE)
        B_scales = torch.rand(E, N, K // group_size, dtype=torch.bfloat16, device=DEVICE) * 0.5 + 0.01
        D = torch.empty(M, N, dtype=torch.bfloat16, device=DEVICE)

        def make_fn():
            A = torch.randn(M, K, dtype=torch.bfloat16, device=DEVICE) * 0.1
            def fn():
                torch.ops._xpu_C.grouped_gemm_interface(
                    ptr_A=A, ptr_B=B_packed, ptr_scales=B_scales, ptr_bias=None,
                    ptr_D=D, expert_first_token_offset=offsets,
                    N=N, K=K, num_experts=E, is_B_int4=True, is_B_mxfp4=False)
            return fn

        return bench_fn_no_cache(make_fn, warmup, iters)
    finally:
        os.environ["VLLM_XPU_GROUPED_GEMM_BACKEND"] = old


def run_cutlass_w4a16(M, N, K, E, group_size, offsets, warmup, iters):
    """Cutlass w4a16: bf16 activations, int4 weights."""
    old = os.environ.get("VLLM_XPU_GROUPED_GEMM_BACKEND", "")
    os.environ["VLLM_XPU_GROUPED_GEMM_BACKEND"] = ""
    try:
        B_packed = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8, device=DEVICE)
        B_scales = torch.rand(E, N, K // group_size, dtype=torch.bfloat16, device=DEVICE) * 0.5 + 0.01
        D = torch.empty(M, N, dtype=torch.bfloat16, device=DEVICE)

        def make_fn():
            A = torch.randn(M, K, dtype=torch.bfloat16, device=DEVICE) * 0.1
            def fn():
                torch.ops._xpu_C.grouped_gemm_interface(
                    ptr_A=A, ptr_B=B_packed, ptr_scales=B_scales, ptr_bias=None,
                    ptr_D=D, expert_first_token_offset=offsets,
                    N=N, K=K, num_experts=E, is_B_int4=True, is_B_mxfp4=False)
            return fn

        return bench_fn_no_cache(make_fn, warmup, iters)
    finally:
        os.environ["VLLM_XPU_GROUPED_GEMM_BACKEND"] = old


RUNNERS = {
    "onednn_w4a8": run_onednn_w4a8,
    "onednn_w4a16": run_onednn_w4a16,
    "cutlass_w4a16": run_cutlass_w4a16,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_single_config(label, N, K, E, group_size, token_counts, backends, warmup, iters):
    """Run benchmark for a single GEMM config."""
    M = sum(token_counts)
    offsets = build_offsets_from_distribution(token_counts, DEVICE)

    print(f"\n--- {label}: M={M}, N={N}, K={K}, E={E}, group_size={group_size} ---")

    # Header
    hdr = f"{'Backend':<16} | {'Time (ms)':>12} | {'TFLOPS':>10} | {'vs cutlass':>12}"
    print(hdr)
    print("-" * len(hdr))

    results = {}
    for b in backends:
        try:
            ms = RUNNERS[b](M, N, K, E, group_size, offsets, warmup, iters)
            # Effective FLOPS: sum over experts of 2*tokens_e*N*K
            flops = sum(2 * t * N * K for t in token_counts)
            tflops = flops / (ms / 1000.0) / 1e12
            results[b] = ms
        except Exception as e:
            results[b] = None
            print(f"  {'[FAIL] ' + b:<16} | {str(e)[:50]}")
            continue

        # Speedup vs cutlass
        cutlass_t = results.get("cutlass_w4a16")
        if cutlass_t and b != "cutlass_w4a16":
            speedup = f"{cutlass_t / ms:.2f}x"
        elif b == "cutlass_w4a16":
            speedup = "(baseline)"
        else:
            speedup = "N/A"

        print(f"  {b:<16} | {ms:>9.3f} ms | {tflops:>8.2f} T | {speedup:>12}")

    clear_xpu_cache()
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark grouped GEMM with gpt-oss-120b MoE shapes")
    parser.add_argument("--gemm", type=str, choices=["gemm1", "gemm2", "both"], default="both",
                        help="Which GEMM to benchmark: gemm1 (gate+up), gemm2 (down), or both")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--backend", type=str, nargs="+", default=["all"],
                        choices=ALL_BACKENDS + ["all"])
    # Allow overriding the token distribution with a simple uniform one
    parser.add_argument("--uniform-tokens", type=int, default=None,
                        help="Use uniform token distribution with this total M instead of real distribution")
    args = parser.parse_args()

    backends = ALL_BACKENDS if "all" in args.backend else args.backend

    if args.uniform_tokens:
        E = 128
        tpe = args.uniform_tokens // E
        rem = args.uniform_tokens % E
        token_counts = [tpe + (1 if i < rem else 0) for i in range(E)]
    else:
        token_counts = REAL_EXPERT_TOKENS

    total_M = sum(token_counts)
    E = len(token_counts)

    print("=" * 70)
    print("Grouped GEMM Benchmark — gpt-oss-120b MoE (TP=4, EP=1)")
    print(f"E={E}, Total M={total_M}, warmup={args.warmup}, iters={args.iters}")
    print(f"Backends: {backends}")
    print(f"Token distribution: real (skewed)" if not args.uniform_tokens else f"Token distribution: uniform")
    # Show weight memory footprint
    for label, cfg in [("GEMM1 (gate+up)", GPT_OSS_120B_GEMM1), ("GEMM2 (down)", GPT_OSS_120B_GEMM2)]:
        weight_bytes = E * cfg["N"] * cfg["K"] // 2  # int4 packed
        scale_bytes = E * cfg["N"] * (cfg["K"] // cfg["group_size"]) * 2  # bf16
        print(f"  {label}: weights={weight_bytes/1024/1024:.1f} MB, scales={scale_bytes/1024/1024:.1f} MB")
    print("=" * 70)

    if args.gemm in ("gemm1", "both"):
        cfg = GPT_OSS_120B_GEMM1
        run_single_config("GEMM1 (gate+up)", cfg["N"], cfg["K"], E, cfg["group_size"],
                          token_counts, backends, args.warmup, args.iters)

    if args.gemm in ("gemm2", "both"):
        cfg = GPT_OSS_120B_GEMM2
        run_single_config("GEMM2 (down)", cfg["N"], cfg["K"], E, cfg["group_size"],
                          token_counts, backends, args.warmup, args.iters)

    print("\nDone.")


if __name__ == "__main__":
    main()