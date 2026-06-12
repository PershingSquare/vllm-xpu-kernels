# Task 13: token-centric prefill threshold/tiling A/B

Generated: 2026-06-12 19:58:29

Container: `hans-gpt-oss-ww17`; device: `ZE_AFFINITY_MASK=0`
Runs/config=2, warmup=30, iters/run=50
Flag: `VLLM_XPU_ONEDNN_TOKEN_CENTRIC_PREFILL_TUNE` (default OFF)

Win/regression rule: ON beats OFF iff `(OFF_med - ON_med) > max(2*OFF_stddev, 1%*OFF_med)`. Symmetric for regression.

## Results

| Mode | TP | Tokens | OFF med | OFF std | ON med | ON std | delta OFF-ON | threshold | verdict |
|:--|---:|---:|---:|---:|---:|---:|---:|---:|:--|
| prefill | 4 | 2048 | 2219.2 | 9.2 | 2215.9 | 16.9 | +3.3 | 22.2 | TIE |
| prefill | 4 | 3072 | 2857.1 | 19.7 | 2849.5 | 6.3 | +7.6 | 39.3 | TIE |
| decode | 4 | 1 | 114.1 | 3.3 | 112.0 | 3.0 | +2.2 | 6.5 | TIE |
| decode | 4 | 8 | 219.9 | 8.0 | 190.5 | 3.7 | +29.3 | 16.1 | ON_WIN |
| decode | 4 | 32 | 420.3 | 37.5 | 415.8 | 3.8 | +4.5 | 75.1 | TIE |
| medium | 4 | 512 | 1296.0 | 22.8 | 1292.4 | 8.8 | +3.6 | 45.7 | TIE |
| medium | 4 | 1024 | 1628.5 | 9.6 | 1627.2 | 19.6 | +1.3 | 19.3 | TIE |
| prefill | 8 | 2048 | 1471.3 | 19.9 | 1471.5 | 25.0 | -0.2 | 39.8 | TIE |
| prefill | 8 | 3072 | 1917.6 | 27.4 | 1916.3 | 4.0 | +1.3 | 54.8 | TIE |
| decode | 8 | 1 | 132.6 | 5.1 | 111.9 | 14.7 | +20.7 | 10.3 | ON_WIN |
| decode | 8 | 8 | 154.9 | 6.6 | 135.6 | 8.5 | +19.3 | 13.2 | ON_WIN |
| decode | 8 | 32 | 314.6 | 42.0 | 266.6 | 3.8 | +48.0 | 84.0 | TIE |
| medium | 8 | 512 | 842.6 | 55.1 | 843.0 | 17.5 | -0.5 | 110.3 | TIE |
| medium | 8 | 1024 | 1064.6 | 18.4 | 1063.8 | 24.2 | +0.9 | 36.9 | TIE |

- prefill: {'ON_WIN': 0, 'ON_REGRESSION': 0, 'TIE': 4}
- decode: {'ON_WIN': 3, 'ON_REGRESSION': 0, 'TIE': 3}
- medium: {'ON_WIN': 0, 'ON_REGRESSION': 0, 'TIE': 4}

## Notes

- Same Python process measured flag-OFF and flag-ON for each config; the primitive cache key includes the flag to avoid descriptor reuse across lanes.
- The tuned flag raises token-centric auto-entry from `M >= ngroups*8` to `M >= ngroups*64`; target prefill remains token-centric, while decode and medium-M guard shapes avoid the tuned token-centric path.
