# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark grouped GEMM: oneDNN w4a8 vs oneDNN w4a16 vs Cutlass vs IPEX.

Uses realistic MoE shapes with TP4/TP8 support and real expert token distributions.

Usage:
    ZE_AFFINITY_MASK=2 python benchmark/benchmark_grouped_gemm_w4a8.py
    ZE_AFFINITY_MASK=2 python benchmark/benchmark_grouped_gemm_w4a8.py --tp 4 --backend onednn_w4a8 ipex_int4
    ZE_AFFINITY_MASK=2 python benchmark/benchmark_grouped_gemm_w4a8.py --warmup 20 --iters 100
    ZE_AFFINITY_MASK=2 python benchmark/benchmark_grouped_gemm_w4a8.py --hint total_M_over_topk actual_max
"""

import argparse
import gc
import os
import time

import torch
import numpy as np

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


# ---------------------------------------------------------------------------
# Cold-cache timing with rotating pool of pre-allocated buffers
# ---------------------------------------------------------------------------

def bench_cold(fns_list, warmup, iters):
    """Time with rotating pool of pre-allocated buffers for cold-cache simulation.

    fns_list[i] is the function for pool slot i (each has its own buffers).
    """
    POOL = len(fns_list)
    for i in range(warmup):
        fns_list[i % POOL]()
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for i in range(iters):
        fns_list[i % POOL]()
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


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


def compute_hint(hint_mode: str, total_M: int, top_k: int, token_counts: list) -> int:
    if hint_mode == "total_M_over_topk":
        return (total_M + top_k - 1) // top_k
    elif hint_mode == "actual_max":
        return max(token_counts)
    else:
        raise ValueError(f"Unknown hint mode: {hint_mode}")


RUNNERS = {
    "onednn_w4a8": run_onednn_w4a8,
    "onednn_w4a16": run_onednn_w4a16,
    "cutlass_w4a16": run_cutlass_w4a16,
    "cutlass_mxfp4": run_cutlass_mxfp4,
    "ipex_int4": run_ipex_int4,
    "ipex_mxfp4": run_ipex_mxfp4,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_single_config(label, N, K, E, group_size, token_counts, top_k, hint_mode, backends, warmup, iters, POOL):
    M = sum(token_counts)
    offsets = build_offsets_from_distribution(token_counts, DEVICE)
    max_expert_size = compute_hint(hint_mode, M, top_k, token_counts)

    onednn_backends = {"onednn_w4a8", "onednn_w4a16"}

    results = {}
    for b in backends:
        try:
            if b in onednn_backends:
                ms = RUNNERS[b](M, N, K, E, group_size, offsets, max_expert_size, warmup, iters, POOL)
            else:
                ms = RUNNERS[b](M, N, K, E, group_size, offsets, warmup, iters, POOL)
            results[b] = ms
        except Exception as e:
            results[b] = None
            print(f"  {b}: FAILED - {str(e)[:60]}")
        import gc; gc.collect()

    clear_xpu_cache()
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark grouped GEMM with TP4/TP8 MoE shapes")
    parser.add_argument("--tp", type=str, choices=["4", "8", "both"], default="both",
                        help="Tensor parallelism: 4, 8, or both (default: both)")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--pool", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=4,
                        dest="top_k")
    parser.add_argument("--M", type=int, default=12288,
                        help="Total tokens M (default: 12288)")
    parser.add_argument("--backend", type=str, nargs="+", default=["all"],
                        choices=ALL_BACKENDS + ["all"])
    parser.add_argument("--hint", type=str, nargs="+",
                        choices=["total_M_over_topk", "actual_max"],
                        default=["total_M_over_topk"],
                        help="oneDNN DNNL_ARG_HINT_MAX_GROUP_SIZE strategy (default: total_M_over_topk)")
    args = parser.parse_args()

    backends = ALL_BACKENDS if "all" in args.backend else args.backend
    hint_modes = args.hint

    token_counts = scale_tokens_to_M(REAL_EXPERT_TOKENS, args.M)
    total_M = sum(token_counts)
    actual_max = max(token_counts)

    configs = []
    if args.tp in ("4", "both"):
        configs.append(("TP4 GEMM1", TP4_GEMM1))
        configs.append(("TP4 GEMM2", TP4_GEMM2))
    if args.tp in ("8", "both"):
        configs.append(("TP8 GEMM1", TP8_GEMM1))
        configs.append(("TP8 GEMM2", TP8_GEMM2))

    backend_cols = {
        "onednn_w4a8": "oneDNN w4a8",
        "onednn_w4a16": "oneDNN w4a16",
        "cutlass_w4a16": "Cutlass w4a16",
        "cutlass_mxfp4": "Cutlass mxfp4",
        "ipex_int4": "IPEX int4",
        "ipex_mxfp4": "IPEX mxfp4",
    }

    for hint_mode in hint_modes:
        hint_val = compute_hint(hint_mode, total_M, args.top_k, token_counts)

        print("=" * 90)
        print("Grouped GEMM Benchmark — MoE (TP4/TP8)")
        print(f"E={E}, Total M={total_M}, top_k={args.top_k}, actual_max={actual_max}")
        print(f"Hint mode: {hint_mode}  => DNNL_ARG_HINT_MAX_GROUP_SIZE={hint_val}")
        print(f"Backends: {backends}")
        print("=" * 90)

        header = f"{'Config':<14}"
        for b in backends:
            header += f" | {backend_cols[b]:>22}"
        print(header)
        print("-" * len(header))

        for label, cfg in configs:
            results = run_single_config(
                label, cfg["N"], cfg["K"], E, cfg["group_size"],
                token_counts, args.top_k, hint_mode, backends,
                args.warmup, args.iters, args.pool)

            N, K = cfg["N"], cfg["K"]
            flops = 2 * total_M * N * K
            bytes_a = total_M * K
            bytes_b = E * N * K // 2
            bytes_c = total_M * N * 2
            bytes_sc = E * N * (K // cfg["group_size"]) * 2 + total_M * 2
            bytes_total = bytes_a + bytes_b + bytes_c + bytes_sc
            BMG_BW_GBS = 456.0

            row = f"{label:<14}"
            mxfp4_ms = results.get("ipex_mxfp4") if "ipex_mxfp4" in backends else None
            for b in backends:
                ms = results.get(b)
                if ms is not None:
                    tf = flops / (ms * 1e-3) / 1e12
                    bw_pct = bytes_total / (ms * 1e-3) / (BMG_BW_GBS * 1e9) * 100
                    if mxfp4_ms is not None and b != "ipex_mxfp4":
                        speedup = mxfp4_ms / ms
                        cell = f"{ms:>5.3f}ms {tf:>4.1f}TF {bw_pct:>3.0f}% {speedup:>4.2f}x"
                    else:
                        cell = f"{ms:>5.3f}ms {tf:>4.1f}TF {bw_pct:>3.0f}%"
                    row += f" | {cell:>22}"
                else:
                    row += f" | {'N/A':>22}"
            print(row)

        print()

    print("Done.")


if __name__ == "__main__":
    main()