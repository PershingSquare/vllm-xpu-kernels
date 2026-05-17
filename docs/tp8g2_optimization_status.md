# TP8 GEMM2 Optimization Status

## Current State (Phase 1.6, shipped)

**Stable warm-cache benchmark numbers** (BMG, gpt-oss-120b distribution, hint=actual_max,
30 warmup + 100 iters):

| Shape | oneDNN w4a8 | TFLOPS | %BW | vs IPEX int4 | vs IPEX mxfp4 |
|---|---|---|---|---|---|
| TP4 GEMM1 | 1.264ms | 87.9 | 65% | 1.69x | 2.96x |
| TP4 GEMM2 | 0.782ms | 69.5 | 63% | 1.42x | 2.38x |
| TP8 GEMM1 | 0.648ms | 85.7 | 69% | 1.67x | 2.93x |
| **TP8 GEMM2** | **0.523ms** | **52.0** | **62%** | **1.28x** | **1.90x** |
| **Total** | **3.217ms** | | | | |

Roofline at BMG 456 GB/s peak: TP8G2 = 0.326ms (80% BW would be 0.408ms).
Gap to roofline: **~197us** = 38% of current latency.

## Quantitative Gap Breakdown (Oracle analysis)

The 38% gap from roofline decomposes into four mechanisms:

| Component | Lost time | % of gap | Mechanism |
|---|---|---|---|
| **C write-allocate** | ~155us | 33% | tile_store at SUBGROUP_SIZE=16 lanes x bf16=2B = 32B/iter (HALF cache line). First half-line write triggers 64B DRAM read-allocate. Eliminating this saves 70.8MB of needless DRAM traffic. |
| **B-weight L3 re-fetch latency** | ~90us | 28% | Token-centric dispatch puts adjacent tiles for same expert on different Xe-cores. B reuse goes via L3 (~150 cycles) instead of L1 (~20 cycles). 3 avg M-tiles/expert each pay full L3 latency. |
| **K-loop pipeline fill/drain** | ~40us | 18% | K=384 / kb_load=64 = 6 K-iterations only. Prefetch fill (2) + drain (2) leaves 2-3 steady-state iterations. Pipeline never reaches full throughput. |
| **Per-tile dispatch overhead** | ~60us | 21% | Avg 3 tiles/expert with binary search + accum init + dequant + store = ~50% overhead per tile. |

## Levers Investigated This Session

| Lever | Result | Detail |
|---|---|---|
| tile_store_block2d (block 2D writes) | Compile error | Requires DECLARE_2D_TILE_BLOCK2D_OPS for c_tile_type_dst. Gemmstone-defined block decomposition doesn't match supported (br,bc,sg) tuples in tile_ops.h:265-271. |
| Prefetch distance @3 (was @128) | REGRESSION (+11% TP8G2) | Oracle hypothesized @128 overshoots K=384 by 21x. Empirically wrong: @128 is well-tuned for the gemmstone microkernel. Different units / semantics than Cutlass's explicit prefetch_dist. |
| unroll[1]=64 (larger N-tile) | Codegen failure | Gemmstone xe2 path rejects this combination with wg 4x1. Primitive_desc creation throws. |

## Remaining ROI-Ranked Levers

### 1. Block 2D / streaming store infrastructure (HIGHEST ROI, 3-5 days)

**Expected gain on TP8G2**: ~155us -> 0.37ms (84% BW)
**Risk**: Low (additive, optional fast-path)
**Why it works**: Block 2D writes via __builtin_IB_subgroup_block_write_flat_*
emit single LSC stores writing 256B+ at once. No half-line allocate penalty.

**Implementation outline:**
1. Add DECLARE_2D_TILE_BLOCK2D_OPS instantiation for c_tile_type_dst tile in
   grouped_micro_gemm.cl. Need to figure out what (br, bc, nbr, nbc) values
   match the gemmstone-emitted ugemm_grouped_c_type_dst layout.
2. Add fast-path in store_results():
   if (full_tile_in_bounds && tile_dimensions_supported)
       tile_store_block2d(tile_dst, ptr, n, m, lddst, sg_j0, sg_i0);
   else
       tile_store(tile_dst, ptr, n, m, lddst, sg_i0, sg_j0);
3. Verify accuracy + 318 CI + production benchmarks.

**Where to start**: Look at how sdpa/micro.cpp or other oneDNN kernels invoke
DECLARE_2D_TILE_BLOCK2D_OPS for their output tiles. Match the pattern.

### 2. Persistent-WG K-reuse (Cutlass-style, 7+ days)

**Expected gain on TP8G2**: ~90us -> 0.43ms (75% BW)
**Risk**: HIGH - blocked on gemmstone reentrancy (vISA-level shim work, 3-4 weeks)
**Why it works**: Same Xe-core processes all M-tiles for an expert sequentially.
B weights stay in L1 across tiles instead of being re-fetched from L3.

**Blocker**: gemmstone microkernel's inline-asm declarations can't be re-executed
in a loop within one WG (vISA decl collision). Must fix the inline-asm shim layer
first - this is the same blocker we hit in Layer 3 reentrancy work earlier.

**Reference pattern**: vllm-xpu-kernels/csrc/xpu/grouped_gemm/xe_2/grouped_gemm_xe2.hpp:172-214
shows the while-loop pattern with atomic work-stealing.

### 3. Cutlass w4a8 parallel track (2-3 weeks)

**Expected**: Unknown ceiling. Cutlass mxfp4 hits 0.822ms on TP8G2 (with 2x weight
bytes). w4a8 has lower BW pressure, so headroom should be high.
**Risk**: Medium - Cutlass infrastructure for w4a16/mxfp4 is mature; w4a8 is a
delta. CUTE compile times are slow (template-heavy).

**Files to extend:**
- vllm-xpu-kernels/csrc/xpu/grouped_gemm/xe_2/gemm_xe2_policy.hpp:76-98
  (add w4a8_policy, w4a8_policy_m_8/16/32 mirroring w4a16_policy)
- vllm-xpu-kernels/csrc/xpu/grouped_gemm/xe_2/gemm_xe2.hpp:320 onwards
  (w4a8 dequant pipeline; reuse 4bits prefetch_dist=6)
- vllm-xpu-kernels/csrc/xpu/onednn/grouped_gemm_w4a8.cpp would need branching to
  pick Cutlass vs oneDNN backend.

### 4. Per-tile overhead reduction (~30us saved, 2 days)

- Skip binary search when ngroups <= 16 (linear scan is faster for small E)
- Cache last-found expert across consecutive WGs (only invalidate on tile_idx
  crossing tile_starts boundary)
- These are micro-optimizations; small ROI.

## Recommendation

**Don't pursue further optimization without explicit mlperf signal.** Phase 1.6
already delivers:
- 1.28x faster than IPEX int4 (Intel reference) on TP8G2
- 1.90x faster than IPEX mxfp4 on TP8G2
- 62% BW utilization (well above typical mixed-precision MoE GEMM)

If mlperf demands sub-0.5ms TP8G2: **block 2D store infrastructure** (#1) is the
right next step. 3-5 days, low risk, ~84% BW projected. Beneficial to all
bf16-output grouped GEMM shapes, not just TP8G2.

If mlperf demands sub-0.4ms: combine block 2D stores + persistent-WG (#2). The
gemmstone reentrancy work becomes worth the multi-week investment.

If oneDNN path stalls: pivot to Cutlass w4a8 (#3) as a parallel track.

## Files Modified (Phase 1.6 ship)

- /data/josephku/oneDNN (branch joseph/token-centric-dispatch, HEAD fd39414adc)
  - src/gpu/intel/matmul/grouped_micro_gemm.cpp: token-centric heuristic + dispatch + scratchpad book + dual-kernel init
  - src/gpu/intel/matmul/grouped_micro_gemm.hpp: precompute_kernel_ member, use_token_centric_ flag
  - src/gpu/intel/matmul/grouped_micro_gemm.cl: WITH_TILE_INDEX path, grouped_compute_tile_starts kernel
- vllm-xpu-kernels wheel: dist/vllm_xpu_kernels-0.1.7.dev13+ga971efd1.xe2v4-cp312-cp312-linux_x86_64.whl

## Investigation Receipts

- Oracle session ses_1cca7597affe9ZLJdv1GCsW55t (architectural review, quantitative breakdown)
- Cutlass explore session ses_1cca69dabffeGrvNM7QezrAgTX (Cutlass xe2 patterns)
- Phase 1 commit: 20e79f4724 (env-gated infrastructure, naive m_all - regression)
- Phase 1.5 commit: 14016bc003 (per-WG SLM scan - regression)
- Phase 1.6 commit: fd39414adc (precompute kernel + scratchpad - SHIPPED, 8.9% total speedup, 20% on TP8G2)
