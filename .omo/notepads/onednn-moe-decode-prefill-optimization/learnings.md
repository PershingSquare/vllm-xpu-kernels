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
