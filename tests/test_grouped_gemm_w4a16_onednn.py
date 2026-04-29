# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from itertools import accumulate

import pytest
import torch

# Ensure the vLLM XPU extension is loaded so torch.ops._xpu_C is registered.
pytest.importorskip(
    "vllm_xpu_kernels._xpu_C",
    reason="vllm_xpu_kernels XPU extension is not available",
)


def _skip_if_xpu_unavailable() -> None:
    if not hasattr(torch, "xpu"):
        pytest.skip("PyTorch was built without XPU support")
    if not torch.xpu.is_available():
        pytest.skip("XPU device is not available")


def _pack_s4_from_u4_zp8(packed_u4: torch.Tensor) -> torch.Tensor:
    """Convert packed uint4 (zp=8) to packed signed int4 (two's complement).

    vLLM stores int4 weights as packed uint4 with an implicit zero-point 8.
    oneDNN's grouped GEMM micro-kernel interprets `u4` as *unsigned* [0..15]
    (no implicit zero-point), and grouped GEMM does not support explicit
    zero-points.

    Therefore, to represent signed weights in [-8..7], the wrapper converts the
    packed u4(zp=8) buffer into packed s4 (two's complement) by flipping bit3 of
    each nibble (XOR 0x88).

    For 4-bit values, converting (u4 - 8) into two's complement s4 is
    equivalent to flipping bit3 of each nibble => XOR with 0x88 per byte.
    """
    assert packed_u4.dtype == torch.uint8
    return packed_u4 ^ 0x88


def _dequantize_u4_zp8(
    packed_u4: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Dequantize packed uint4 (zp=8) weights into fp32.

    Args:
        packed_u4: [N, K//2] uint8, two u4 values per byte.
        scales:    [N, K//group_size] fp16 per-channel per-group scales.
        group_size: int, number of K elements per group.

    Returns:
        weight_fp32: [N, K] float32 (transposed weight layout), suitable for
        reference matmul via: A @ weight_fp32.T
    """
    assert packed_u4.dtype == torch.uint8
    assert scales.dtype == torch.float16
    assert scales.dim() == 2
    n, k_half = packed_u4.shape
    k = k_half * 2
    group_num = scales.shape[1]
    assert k == group_num * group_size
    assert scales.shape[0] == n

    low_u4 = (packed_u4 & 0x0F).to(torch.int16) - 8
    high_u4 = ((packed_u4 >> 4) & 0x0F).to(torch.int16) - 8

    w_int = torch.empty((n, k), dtype=torch.int16, device="cpu")
    w_int[:, 0::2] = low_u4
    w_int[:, 1::2] = high_u4

    group_idx = torch.arange(k, device="cpu", dtype=torch.int64)
    group_idx = group_idx // group_size
    scales_expanded = scales[:, group_idx]

    return w_int.to(torch.float32) * scales_expanded.to(torch.float32)


@pytest.mark.parametrize(
    "num_experts,token_counts",
    [
        (2, [5, 3]),
        (4, [1, 4, 2, 3]),
        # Include empty experts to validate `expert_first_token_offset` semantics.
        (4, [0, 4, 0, 3]),
    ],
)
@pytest.mark.parametrize("group_size", [64, 128])
def test_grouped_gemm_onednn_w4a16_int4(num_experts: int,
                                         token_counts: list[int],
                                         group_size: int,
                                         monkeypatch: pytest.MonkeyPatch):
    """Validate oneDNN w4a16 grouped GEMM with in-kernel bias application."""
    _skip_if_xpu_unavailable()

    # Select oneDNN backend for grouped GEMM.
    monkeypatch.setenv("VLLM_XPU_GROUPED_GEMM_BACKEND", "onednn")
    # Do not override externally-forced ZE_AFFINITY_MASK (CI sets this to select
    # a known-good device for the int4 micro-kernel).
    assert hasattr(torch.ops._xpu_C, "grouped_gemm_interface")
    assert hasattr(torch.ops._xpu_C, "cutlass_grouped_gemm_interface")

    device = torch.device("xpu")
    dtype = torch.float16

    # Small, deterministic shapes.
    torch.manual_seed(0)
    n = 64
    k = 256
    assert k % group_size == 0
    group_num = k // group_size

    assert len(token_counts) == num_experts
    total_m = sum(token_counts)
    assert total_m > 0

    # A: [Total_M, K] fp16
    A = (torch.randn((total_m, k), device=device, dtype=dtype) * 0.1)
    A = A.contiguous()

    # B (input for packing): [E, N, K/2] uint8, packed u4 with zp=8.
    packed_u4 = torch.randint(0,
                              256,
                              (num_experts, n, k // 2),
                              device=device,
                              dtype=torch.int32).to(torch.uint8)
    packed_u4 = packed_u4.contiguous()

    # Scales: [E, N, K/group_size] fp16 (required by oneDNN path).
    scales = (torch.rand((num_experts, n, group_num), device=device, dtype=dtype)
              * 0.5 + 0.01)
    scales = scales.contiguous()

    bias = (torch.randn((num_experts, n), device=device, dtype=dtype) * 0.1)
    bias = bias.contiguous()

    # Pass packed u4(zp=8) into the op. The oneDNN wrapper will convert it to
    # packed s4 (two's complement) internally.

    # Expert offsets: [E+1] int64 (non-uniform token distribution).
    offsets = torch.tensor([0] + list(accumulate(token_counts)),
                           device=device,
                           dtype=torch.int64)
    offsets = offsets.contiguous()

    # Output: [Total_M, N] fp16
    out = torch.empty((total_m, n), device=device, dtype=dtype).contiguous()
    out_repeat = torch.empty((total_m, n), device=device, dtype=dtype).contiguous()
    out_no_bias = torch.empty((total_m, n), device=device, dtype=dtype).contiguous()

    op_kwargs = dict(
        ptr_A=A,
        ptr_B=packed_u4,
        ptr_scales=scales,
        expert_first_token_offset=offsets,
        N=n,
        K=k,
        num_experts=num_experts,
        is_B_int4=True,
        is_B_mxfp4=False,
    )

    # Run the op (routes to oneDNN when VLLM_XPU_GROUPED_GEMM_BACKEND=onednn).
    try:
        out_ret = torch.ops._xpu_C.grouped_gemm_interface(
            ptr_D=out,
            ptr_bias=bias,
            **op_kwargs,
        )

        out_ret_repeat = torch.ops._xpu_C.cutlass_grouped_gemm_interface(
            ptr_D=out_repeat,
            ptr_bias=bias,
            **op_kwargs,
        )

        out_no_bias_ret = torch.ops._xpu_C.grouped_gemm_interface(
            ptr_D=out_no_bias,
            ptr_bias=None,
            **op_kwargs,
        )
    except RuntimeError as e:
        msg = str(e)
        if ("oneDNN grouped GEMM is not enabled in this build" in msg
                or "DNNL_EXPERIMENTAL_GROUPED_MEMORY=0" in msg):
            pytest.skip(msg)
        raise

    torch.xpu.synchronize()

    # The kernel returns ptr_D.
    assert out_ret.data_ptr() == out.data_ptr()
    assert out_ret_repeat.data_ptr() == out_repeat.data_ptr()
    assert out_no_bias_ret.data_ptr() == out_no_bias.data_ptr()

    torch.testing.assert_close(
        out_repeat.cpu().to(torch.float32),
        out.cpu().to(torch.float32),
        atol=0,
        rtol=0,
    )

    # -----------------------------
    # Reference (fp32): per-expert matmuls using offsets
    # -----------------------------
    A_cpu = A.cpu().to(torch.float32)
    packed_u4_cpu = packed_u4.cpu()
    scales_cpu = scales.cpu()
    offsets_cpu = offsets.cpu().tolist()
    bias_cpu = bias.cpu().to(torch.float32)

    expected_bias = torch.empty((total_m, n), dtype=torch.float32, device="cpu")
    for e in range(num_experts):
        start = offsets_cpu[e]
        end = offsets_cpu[e + 1]
        if end <= start:
            continue
        expected_bias[start:end] = bias_cpu[e]

    torch.testing.assert_close(out.cpu().to(torch.float32) -
                               out_no_bias.cpu().to(torch.float32),
                               expected_bias,
                               atol=2e-2,
                               rtol=2e-2)

    ref = torch.empty((total_m, n), dtype=torch.float32, device="cpu")
    for e in range(num_experts):
        start = offsets_cpu[e]
        end = offsets_cpu[e + 1]
        if end <= start:
            continue
        W_fp32 = _dequantize_u4_zp8(packed_u4_cpu[e], scales_cpu[e], group_size)
        ref[start:end] = A_cpu[start:end] @ W_fp32.T
        ref[start:end] += bias_cpu[e]

    torch.testing.assert_close(out.cpu().to(torch.float32),
                               ref,
                               atol=2e-2,
                               rtol=2e-2)
