# Learnings

## Task 4 — Blob lane in benchmark_fused_moe_thunks.py (2026-06-12)

- Production blob path: `vllm_xpu_kernels/fused_moe_interface.py:335-355` (`xpu_fused_moe`).
  Gate (`_use_blob`) requires ALL of:
  - `using_w4a8` = onednn backend (`VLLM_XPU_GROUPED_GEMM_BACKEND=onednn`) AND `VLLM_XPU_USE_W4A8=1`
  - `_pool is not None` (auto-true whenever `using_w4a8`)
  - `remap_and_quant_hidden_states_int8` op registered (`torch.ops._moe_C`)
  - `swigluoai_and_mul_quant_int8_asym` op registered (`torch.ops._C`)
  - `onednn_fused_moe_w4a8` op registered (`torch.ops._xpu_C`)
  - activation == `swigluoai` and `_inter_scale == 1` (not `relu2_no_mul`)
  - `VLLM_XPU_W4A8_FUSED_BLOB != "0"` (production DEFAULTS to "1")
- Blob is a single dispatch — NO per-thunk breakdown. Only `whole_moe_wall` is
  meaningful; the lane reports median + population stddev.
- No-silent-fallback: `blob_gate_blockers(activation)` mirrors the op/activation
  half of the gate and raises RuntimeError listing missing ops instead of letting
  `xpu_fused_moe` quietly run the unfused path. Env conditions forced via `apply_env`.
- Blob lane is SEPARATE from `args.backends`/`region_order`, so the unfused
  (`onednn_w4a8`) attribution lane and IPEX monolithic lane keep their per-thunk
  tables untouched. `--blob` is additive only.
- Lane drives the REAL `xpu_fused_moe` (not a reimplementation) -> production truth.
  `setup_backend("onednn_w4a8", ...)` reuses w4a8 weight setup; `make_blob_runner`
  calls `xpu_fused_moe(...)` with prebuilt kwargs.

### Smoke result (hans-gpt-oss-ww17, ZE_AFFINITY_MASK=0, decode, num-tokens 8)
- `onednn_w4a8_blob` whole_moe_wall: 482.4 us (stddev 39.1 us)
- `ipex_mxfp4`        whole_moe_wall: 498.6 us (stddev 5.8 us)
- Blob beats IPEX monolithic by ~16 us (0.97x) at M=32, BUT its stddev is ~7x
  larger than IPEX's — blob single-dispatch wall is noticeably noisier at tiny-M
  decode. Future sweeps MUST report stddev (not just median). Confirms the plan's
  "noise floor visible" rationale.
- The unfused `onednn_w4a8` whole wall (536.1 us) is SLOWER than the blob
  (482.4 us); use the blob lane, not the unfused synced-thunk total, for
  apples-to-apples vs IPEX.

### Gotchas
- Host has no `python`; runs go through
  `docker exec -w /data/josephku/vllm-xpu-kernels hans-gpt-oss-ww17 ... /opt/venv/bin/python`.
- `--blob` with non-swigluoai activation correctly errors out (gate requires
  swigluoai / inter_scale==1).

## Task 7 — Baseline capture + noise-floor harness (2026-06-12)

- New reusable harness: `benchmark/baseline_capture_noise_floor.py`. Drives the
  REAL production blob (`xpu_fused_moe` -> `onednn_fused_moe_w4a8`) vs IPEX
  monolithic `GatedMLPMOE`, for TP4/TP8 x decode{1..32} x prefill{2048,3072},
  each config measured `--runs` times (default 2). Reuses token distribution +
  routing helpers from `benchmark_fused_moe_thunks.py` so both lanes see an
  identical expert-load profile. Weights are built ONCE per TP (independent of
  num_rows); only activations/routing rebuild per config -> fast full sweep.
- Shapes (match tp4_tp8_moe_wall_results.md): hidden=2944, E=128, top_k=4,
  group=128, swigluoai, bias on. TP4 inter=768, TP8 inter=384.
- WIN-THRESHOLD RULE (single source of truth, embedded in script + every report):
  lane A beats baseline B iff (B_med - A_med) > max(2*B_stddev, 1%*B_med).
  Per-config noise_floor = max(2*stddev, run_to_run_delta). Matches plan
  DECODE PARITY def (max(2*stddev, 1%)).

### Headline result (full sweep, warmup=30 iters=50 runs=2; evidence/task-7-baselines.md)
- 68 configs, ZERO missing cells. Tally: 54 BLOB_WIN, 0 BLOB_LOSS, 14 TIE.
- The fused BLOB BEATS or TIES IPEX at EVERY decode config (1..32) at both TP4
  and TP8 — the OPPOSITE of the old unfused `tp4_tp8_moe_wall_results.md` (which
  showed oneDNN losing badly). CONFIRMS Finding F1: the unfused benchmark lane
  massively overstated the decode gap; on the production blob path the gap is
  already closed/inverted. **Goal B (decode parity) may already be largely met
  on the blob — later kernel work should re-baseline against THIS file, not the
  unfused numbers.**
- Prefill: blob wins hugely (TP4 3072: 2849us vs IPEX 6463us; TP8 3072: 1914us
  vs 3603us). Goal A (prefill non-regression) has large headroom.

### Noise-floor / reproducibility findings (evidence/task-7-repro.txt)
- WARM single-process reproducibility (--runs 5, GPU warm) is clean: blob
  run-medians 117-121us (run_delta 4.1 <= nf 9.4), ipex 128.7-133.9us
  (run_delta 5.2 <= nf 7.2). Both reproducible=YES.
- CROSS-INVOCATION absolute medians drift ~10-13us (~8-10%) at tiny-M, dominated
  by process cold-start (first measured config in a fresh process pays GPU
  freq/thermal spin-up that the full sweep amortizes on its first config).
  => Do NOT compare absolute medians captured in DIFFERENT processes below ~13us.
- DECISION RULE for A/B tasks: measure flag-OFF vs flag-ON in the SAME process
  invocation (blob and ipex are already measured back-to-back here) and compare
  the DELTA against max(2*stddev,1%). The in-invocation delta is robust; absolute
  cross-process medians are not. Near-boundary configs (small win_thr, delta only
  marginally above it, e.g. TP8/decode/8, 27-31, 30) flip BLOB_WIN<->TIE across
  invocations — treat single-invocation BLOB_WIN with delta < ~2x win_thr as
  provisional; confirm with --runs>=5.
- BLOB lane needs warmup>=30 to stabilize: at warmup=5 blob stddev was ~52us
  (nf 105us!); at warmup=30 it dropped to ~4us (nf ~8us). IPEX stabilizes much
  faster. ALWAYS use warmup>=30 for blob decode measurements.

### Gotchas
- `/tmp/opencode` is a HOST path, not present inside the container — use `/tmp`
  for in-container scratch output.
- Validation: `python -m py_compile benchmark/baseline_capture_noise_floor.py`
  and `git diff --check` both clean in-container.

## 2026-06-12 20:20 Task: 15 — Large-M W4A8 microkernel tiling fix

- Root cause confirmed: Task 15 had added in-kernel source zero-point correction to `grouped_gemm_w4a8_xe2.cl` (`sum_s8x4`, `weight_sum`, and `WITH_SRC_ZP` subtract). W4A8 source ZP is already handled before the oneDNN kernel by the host-side/remap quantization pipeline, so this double-applied the correction and caused the large-M flag path to fail parity.
- Fix was surgical: removed only the kernel-local ZP correction machinery and kept the intended large-M changes (`W4A8_M_TILE` 4→8 under `W4A8_LARGE_M_TILE`, `acc1/iacc1`, rows 4-7 loads, and acc0/acc1 output selection). Also kept the dtype-aware `BIA_TO_REF` / `WEI_SCALES_TO_REF` usage and `MATH_UTILS_DECLARE_BF16` define.
- Verification after rebuild in `hans-gpt-oss-ww17`: flag-OFF parity `48 passed in 3.10s`; flag-ON parity `48 passed in 2.82s`.
- Blob prefill A/B was run as separate OFF/ON processes because this checkout's `benchmark_fused_moe_thunks.py` has no `--tp` or multi-token CLI. TP was mapped through `--inter` (TP4=768, TP8=384), warmup=30, iters=50. Results were mostly ties; TP8/tokens=2048 showed a cross-process threshold regression (OFF 1462.4us std 2.9 vs ON 1480.4us std 23.2, threshold 14.6us). Treat that as performance follow-up, not correctness blocker.

## Task 3 — Fused W4A8 MoE blob parity test (2026-06-11)

- New test: `tests/test_fused_moe_w4a8_blob_onednn.py`. Drives the production
  blob op `torch.ops._xpu_C.onednn_fused_moe_w4a8` directly with caller-provided
  scratch and asserts parity vs a pure-torch CPU reference. Reuses the EXACT
  dequant math + tolerances from `tests/test_grouped_gemm_w4a8_onednn.py`
  (acts `(a_q-a_zp)*a_scale`, weights `(u4-8)*scale`; fp16 atol/rtol=5e-2,
  bf16=3e-1). 16 params: bias{off,on} x dtype{fp16,bf16} x gs{128,256} x
  2 expert configs (E4/topk2 empty-middle-expert, E6/topk1 two empty experts).
- Reference reconstructs the WHOLE blob chain per (token,expert): remap+quant
  (uint8 asym, scale stored at out_dtype) -> GEMM1 -> swigluoai(alpha=1.702,
  limit=7) -> requant -> GEMM2 -> weighted gather. Permutation/offset details
  are NOT replicated — iterating rows x topk and summing weighted g2 is
  permutation-invariant and matches `moe_gather`'s reduction.
- Quant formulas verified against the kernels: `remap_and_quant_hidden_states_int8`
  and `swigluoai_and_mul_quant_int8_asym` both do `scale=max((max-min)/255,1e-10)`,
  `zp=clamp(round(-min/scale),0,255)`, `q=clamp(round(x/scale+zp),0,255)`. swigluoai
  gate=EVEN cols, up=ODD cols; gate upper-clamped to limit, up clamped [-limit,limit].
  `remap` requires topk in {1,2,4,6,8,10} and hidden%4==0.

### KEY FINDING — bf16 w4a8 grouped GEMM emits ZEROS when group_size == K (gn=1)
- Isolated on the direct op `onednn_grouped_gemm_w4a8` (decoupled from the blob):
  fp16 works for ALL group counts incl. gn=1; **bf16 produces all-zero output
  ONLY when group_size == K (single group, gn=1)**. bf16 with gn>=2 (gs128, or
  K=512+gs256) is correct. This is why an initial hidden=inter=256 + gs256 gave
  99% mismatch — the GEMM silently returned zeros.
- NON-PRODUCTION edge case: gpt-oss has hidden/inter ~2944/3072, group_size 256
  -> gn ~11-12 (always >>1), and production runs fp16 (`--dtype=float16`), where
  even gn=1 works. Still a silent-zero footgun worth a dedicated kernel-side guard.
- Test sidesteps it by sizing hidden=inter=512 so gs256->gn2, gs128->gn4 (gn>=2
  asserted in-test). Comment in the test documents the constraint.

### bf16 chained-requant boundary noise (tolerance, NOT a bug)
- bf16 GEMM1 output matches reference to ~1 ULP, but the swigluoai activation
  range can be large; with `scale_act = range/255` a single +-1 int8 quant-bucket
  flip (triggered by bf16 rounding straddling a bucket boundary) -> error
  `scale_act * |w2_deq|` that can EXCEED atol 0.3 by itself, then gemm2 fans it
  across the row. fp16's finer mantissa avoids the straddles -> passes easily.
- FIX (no tolerance loosening): condition inputs so g1 stays well below the
  swigluoai clamp limit -> small activation range -> small scale_act. Chosen:
  hidden*0.05, w13 scale [0.01,0.2], w2 scale [0.01,0.1]. Empirical sweep: this
  gives max-abs-diff 0.09-0.13 (0 elements over tol, refmax ~13-16). Aggressive
  (w13 0.1 / w2 0.05) gives ~0.03 with 10x margin. Lesson: when testing a
  CHAINED int8-requant path at element-level atol, keep intermediate ranges
  modest or boundary flips will spuriously fail bf16.

### Result + negative control
- `ZE_AFFINITY_MASK=0 /opt/venv/bin/python -m pytest -q tests/test_fused_moe_w4a8_blob_onednn.py`
  -> 16 passed (evidence/task-3-blob-test-pass.txt).
- Negative sensitivity: scratch copy with reference zp `z1=z1+1` -> ALL 16 FAIL
  (evidence/task-3-negative-sensitivity.txt); scratch deleted, real test clean.

### Gotchas
- `/opt/venv` (the env with the kernels) had NO pytest, and EVERY pre-existing
  venv (`.venv`, `.pytest_venv*`) ships a TAMPERED `_pytest/compat.py` with an
  injected `import py` + `LEGACY_PATH = py.path.local` (the `py` pkg is absent)
  and a comment "intentional space to create a fake difference for the
  verification" -> pytest import crashes. Network pip stalls at "Looking in
  indexes" (effectively blocked). FIX: copied pytest+deps from `.venv` into
  `/opt/venv` and repointed compat.py at the vendored
  `from _pytest._py.path import LocalPath as LEGACY_PATH` (what real pytest 9.0.2
  uses). pytest 9.0.2 now runs cleanly under `/opt/venv/bin/python -m pytest`.
- Blob op needs all three extensions imported (`_xpu_C`, `_C`, `_moe_C`) because
  it dispatches remap/gather via `_moe_C` and act-quant via `_C`; the test
  importorskips all three.

## Task 14 — G1 epilogue SwiGLU-OAI via oneDNN post-ops is not ABI-feasible (2026-06-12)

- `grouped_micro_gemm.cpp::generate_post_ops_header` currently emits only
  shape-preserving unary eltwise over each accumulator element plus binary-mul
  scale helpers. The eltwise path is SiLU-like `v * sigmoid(alpha*v)` and has no
  access to a paired `up` column.
- SwiGLU-OAI consumes paired G1 columns and contracts `[rows, 2*inter] ->
  [rows, inter]`: gate is even columns, up is odd columns, with OAI clamps and
  `(up + 1) * gate * sigmoid(alpha*gate)`. A normal oneDNN matmul post-op must
  preserve the destination descriptor shape, so simply appending an eltwise
  post-op in `grouped_gemm_w4a8.cpp` cannot feed G2.
- Task 12 KEEP evidence supports the separate Tier-1 act+quant SYCL kernel, not
  dynamic quant fusion into oneDNN. For Task 14, dynamic-quant fusion remains out
  of scope and was not attempted.
- Future work would need a custom grouped_micro_gemm post-op ABI or dedicated G1
  epilogue kernel path that explicitly supports paired-column contraction; the
  existing gemmstone post-op surface is insufficient.


## Task 13 — Token-centric prefill threshold flag (2026-06-12)

- Added default-off production flag `VLLM_XPU_ONEDNN_TOKEN_CENTRIC_PREFILL_TUNE`.
  Kernels-side cache key includes the flag, so OFF/ON primitive descriptors do not
  alias during same-process A/B. oneDNN reads the same flag directly.
- Tuned behavior raises token-centric auto-entry from `M >= ngroups*8` to
  `M >= ngroups*64`. For the blob path, prefill 2048 corresponds to `M=8192`
  routed rows with `ngroups=128`, so it remains token-centric and uses the
  existing large-M tiling branch; decode and medium-M guards stay out of the
  tuned token-centric path.
- Sweep finding: forcing alternate gemmstone strategies (`wg 1x1`, `2x1`, `4x1`,
  `8x1`, `4x2`, and smaller row tile) did not beat the current shape heuristic
  on blob wall time. The safest landed change is therefore threshold gating, not
  replacing the existing large-M tile heuristic.
- Same-process A/B (`evidence/task-13-prefill-ab.md`, runs=2/warmup=30/iters=50):
  prefill deltas were positive but below the task-7 threshold (TP4 2048 +3.3us
  vs thr 22.2; TP4 3072 +7.6us vs thr 39.3; TP8 2048 -0.2us vs thr 39.8;
  TP8 3072 +1.3us vs thr 54.8). Verdict: 4/4 prefill TIE, no prefill win above
  noise. Decode guard: 3 wins, 3 ties, 0 regressions. Medium guard: 4 ties,
  0 regressions.
- Parity: `pytest -q tests/test_fused_moe_w4a8_blob_onednn.py tests/test_grouped_gemm_w4a8_onednn.py`
  passed 48 tests (`evidence/task-13-parity.txt`).

## Task 15 — Large-M W4A8 tile/unroll attempt did not land (2026-06-12)

- Production default-off flag plumbing was added for `VLLM_XPU_ONEDNN_W4A8_LARGE_M_TILE`, including cache-key separation for same-process OFF/ON A/B. The cache bit was later narrowed to `total_M >= 1024` so decode/small-M descriptors do not split when the flag cannot affect strategy.
- oneDNN gemmstone large-M Xe2 strategy was tested with `strat.unroll[1] = 64` behind the flag, then narrowed to `M() >= 1024`. Correctness passed in production mode: default-off `48 passed` and flag-on `48 passed` for `tests/test_fused_moe_w4a8_blob_onednn.py tests/test_grouped_gemm_w4a8_onednn.py`.
- Same-process blob A/B did not produce a reliable prefill win. Clean rerun after cache-key narrowing showed all prefill configs were TIE and no decode regressions: TP4 2048 `+16.5us` vs threshold `1426.8us`, TP4 3072 `+2.6us` vs `3982.4us`, TP8 2048 `-2.7us` vs `61.6us`, TP8 3072 `-26.8us` vs `44.0us`.
- Forced hand-written Xe2 W4A8 path (`GRPGEMM_W4A8_XE2_KERNEL=1 VLLM_XPU_ONEDNN_W4A8_LARGE_M_TILE=1`) still failed parity badly: 48/48 W4A8 tests failed with large tensor mismatches, so it must remain out of production/default flag behavior.
- Do not commit the Task 15 tuning as a performance win. The safe pieces are correctness-preserving/default-off, but the required A/B win rule was not met after three approaches: gemmstone unroll64, large-M/cache gating, and forced hand Xe2 kernel.

## [2026-06-12 20:30] Task: 16 — TP4 acceptance sweep
- Ran full A/B at TP4 with all validated flags ON
  (VLLM_XPU_ONEDNN_TOKEN_CENTRIC_PREFILL_TUNE=1, VLLM_XPU_ONEDNN_W4A8_LARGE_M_TILE=1)
  via baseline_capture_noise_floor.py --tp 4 --runs 2 --warmup 30 --iters 50.
  Same-process blob-vs-ipex delta (avoids ~10-13us tiny-M cross-invocation drift).
- Result: 34 configs (decode 1..32 + prefill 2048,3072): 27 BLOB_WIN, 7 TIE,
  **0 BLOB_LOSS**. DECODE PARITY HOLDS at every integer token count 1..32.
- Prefill non-regression: blob ~2.1x (2048: 2217.7 vs 4650.2us) and ~2.3x
  (3072: 2845.3 vs 6467.7us) faster than IPEX; blob medians within Task-7 nf.
- Worst decode config = tokens=9 (only point where blob_med > ipex_med, delta
  -1.8us << win_thr 7.5us = TIE). High-runs confirm (--runs 6): delta flips to
  +1.1us, still TIE. M=9 blob/ipex are statistically indistinguishable; never
  near a BLOB_LOSS. tokens=13 firmed to BLOB_WIN at 6 runs.
- blob lane non-regression vs Task-7 (flags OFF): all blob-ON medians within
  baseline noise floor / known tiny-M drift; tokens=7,10 marginally above own nf
  (+11.4/+12.0) but inside documented ~10-13us drift and decisive A/B wins.
- Evidence: .omo/evidence/task-16-tp4-acceptance.md (full table + verdicts),
  .omo/evidence/task-16-tp4-worst-case.md (worst-case analysis).

## [2026-06-12] Task: 18 — No-regression validation (all validated flags ON)
- All validated flags ON: VLLM_XPU_ONEDNN_TOKEN_CENTRIC_PREFILL_TUNE=1 +
  VLLM_XPU_ONEDNN_W4A8_LARGE_M_TILE=1 (both stay default-OFF in build; only
  measured, not flipped). Wave 2 opts baked into build.
- CORRECTNESS (flags ON): the 3 files pass — `pytest -q
  test_grouped_gemm_w4a8_onednn.py test_grouped_gemm_w4a16_onednn.py
  test_fused_moe_w4a8_blob_onednn.py` => 54 passed in 11.81s. This covers
  bias{off,on} x dtype{fp16,bf16} x gs{128,256}/{64,128} in one shot — axes 3/4/5
  are correctness-only and need no separate benchmark. bf16+gs256 isolated
  evidence: evidence/task-18-bf16-g256.txt (4 bf16+gs256 + 24 bf16 total pass).
- PERF no-regression (blob whole_moe_wall, OFF vs ON, same task-7 rule with
  OFF_run_to_run_delta folded into threshold): 8/8 TIE. prefill TP4/TP8 2048+3072
  + TP1 decode{1,4,8} + TP1 prefill 2048. Evidence: evidence/task-18-no-regression.md.
  Driver: .omo/scripts/task18_noregression.py (TP mapped via --inter 768/384/3072
  since this checkout's benchmark_fused_moe_thunks.py has single --num-tokens, no --tp).
- KEY NOISE LESSON: REPS=2 is NOT enough for blob whole-wall on this box. The
  first pass flagged prefill TP8 2048+3072 as REGRESSION, but the ON run pairs
  were `[1540,2751]` and `[4356,1909]` — one clean run matching OFF (1909.8 vs
  OFF 1911.8) + one wild outlier. The SAME transient GPU contention also blew up
  OFF runs of TP1 decode 4/8 (OFF `[446,1605]`, `[516,1628]`), proving it's
  process/device contention, not a flag effect. A 2-run median averages the
  outlier in. REPS=5 (median rejects 1-2 outliers) -> all 4 confirm TIE
  (TP8 3072 ON median 1908.7 actually < OFF 1911.8). RULE: for blob A/B always
  REPS>=5 and use median-of-medians; do not trust a 2-rep REGRESSION before
  re-running with more reps. Within-run iter stddev can spike from unrelated
  contention; use run-to-run dispersion of medians for the noise floor instead.

## [2026-06-12 20:35] Task: 17 — TP8 acceptance sweep
- Rebuild required first: build/temp .so was mid-write ("file too short") + uncommitted
  csrc/xpu/onednn/grouped_gemm_w4a8.cpp change (gate w4a8_large_m_tile to total_M>=1024 so
  decode small-M is untouched). Rebuilt `pip install --no-build-isolation -e .` (exit 0)
  so the .so matches source before benchmarking.
- Full TP8 A/B (decode 1..32 + prefill {2048,3072} = 34 configs), warmup=30 iters=50 runs=2.
  FLAGS ON (TOKEN_CENTRIC_PREFILL_TUNE=1, W4A8_LARGE_M_TILE=1):
    BLOB_WIN 15, BLOB_LOSS 0, TIE 19. delta(ipex-blob) POSITIVE at ALL 34 configs.
  => DECODE PARITY PASS (32/32), PREFILL NON-REGRESSION PASS (2/2). All 68 lanes reproducible=YES.
- Worst decode config = tokens=28: blob 241.2us vs ipex 242.0us, delta +0.8us, TIE -> PASS.
  No config where blob is slower (no BLOB_LOSS). Thin-margin band is M~26-30.
- FLAGS OFF reference: BLOB_WIN 11, BLOB_LOSS 1, TIE 22. The single loss (tokens=27, blob
  270.1us blob_std 113.1us vs ipex 233.5us, -36.6us) is a COLD-ITER NOISE ARTIFACT
  (blob stddev 113 >> 36.6 gap). With flags ON same config is clean TIE (234.1us, std 11.1,
  +4.8us, reproducible). => validated flags eliminate the only flags-OFF anomaly; no regression.
- Confirms Task-7 inherited wisdom: IPEX cold-iter stddev can blow noise floors to 400-900us at
  random configs, turning real blob wins into TIE. Use same-process delta + the median; the
  blob median is faster in every decode config at TP8.
- Evidence: .omo/evidence/task-17-tp8-acceptance.md, task-17-tp8-worst-case.md.
  Raw: container /tmp/task17_flags_on.md, task17_flags_off.md, *_repro.txt, *_run.log.
