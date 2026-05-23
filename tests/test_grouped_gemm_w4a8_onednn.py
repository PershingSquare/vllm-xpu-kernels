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


def _quantize_asymmetric_int8_per_row(
    a_fp32: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize fp32 activations to signed int8 with row-wise asymmetric scale/zp."""
    assert a_fp32.dtype == torch.float32
    row_min = a_fp32.amin(dim=1)
    row_max = a_fp32.amax(dim=1)
    qmin, qmax = -128, 127
    scale = (row_max - row_min) / float(qmax - qmin)
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    zp = torch.round(qmin - row_min / scale)
    zp = zp.clamp(qmin, qmax).to(torch.int32)
    a_q = torch.round(a_fp32 / scale.unsqueeze(1) + zp.to(torch.float32).unsqueeze(1))
    a_q = a_q.clamp(qmin, qmax).to(torch.int8)
    return a_q, scale.to(torch.float32), zp


def _quantize_asymmetric_uint8_per_row(
    a_fp32: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Row-wise asymmetric uint8 quantization matching the oneDNN w4a8 kernel contract."""
    assert a_fp32.dtype == torch.float32
    row_min = a_fp32.amin(dim=1)
    row_max = a_fp32.amax(dim=1)
    qmin, qmax = 0, 255
    scale = (row_max - row_min) / float(qmax - qmin)
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    zp = torch.round(qmin - row_min / scale).clamp(qmin, qmax)
    a_q = torch.round(a_fp32 / scale.unsqueeze(1) + zp.unsqueeze(1)).clamp(qmin, qmax)
    a_q = a_q.to(torch.uint8)
    zp = zp.to(torch.uint8)
    return a_q, scale.to(torch.float32), zp


def _dequantize_u4_zp8(
    packed_u4: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Dequantize packed uint4 (zp=8) weights into fp32.

    Args:
        packed_u4: [N, K//2] uint8, two u4 values per byte.
        scales:    [N, K//group_size] fp16/fp32 per-channel per-group scales.
        group_size: int, number of K elements per group.

    Returns:
        weight_fp32: [N, K] float32 (transposed weight layout), suitable for
        reference matmul via: A @ weight_fp32.T
    """
    assert packed_u4.dtype == torch.uint8
    assert scales.dtype in (torch.float16, torch.float32)
    assert packed_u4.dim() == 2
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

    group_idx = torch.arange(k, dtype=torch.int64, device="cpu") // group_size
    scales_expanded = scales[:, group_idx]
    return w_int.to(torch.float32) * scales_expanded.to(torch.float32)


@pytest.mark.parametrize(
    "num_experts,token_counts",
    [
        # Include empty experts to validate `expert_first_token_offset` semantics.
        (4, [2, 0, 3, 1]),
        (5, [0, 4, 0, 2, 1]),
    ],
)
@pytest.mark.parametrize("group_size", [64, 128])
@pytest.mark.parametrize("out_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("has_bias", [False, True])
def test_grouped_gemm_onednn_w4a8_int8(
    num_experts: int,
    token_counts: list[int],
    group_size: int,
    out_dtype: torch.dtype,
    has_bias: bool,
):
    _skip_if_xpu_unavailable()
    if not hasattr(torch.ops, "_xpu_C") or not hasattr(torch.ops._xpu_C, "onednn_grouped_gemm_w4a8"):
        pytest.skip("torch.ops._xpu_C.onednn_grouped_gemm_w4a8 is not available in this build")

    device = torch.device("xpu")

    # Small, deterministic shapes.
    torch.manual_seed(0)
    n = 64
    k = 256
    assert (k % 2) == 0
    assert (n % 2) == 0
    assert k % group_size == 0
    group_num = k // group_size

    assert len(token_counts) == num_experts
    total_m = sum(token_counts)
    assert total_m > 0

    # -----------------------------
    # Inputs
    # -----------------------------
    # A_fp32 (CPU): [Total_M, K]
    # Generate per-expert segments (different magnitudes) to validate offsets.
    offsets_cpu = torch.tensor(
        [0] + list(accumulate(token_counts)),
        dtype=torch.int64,
        device="cpu",
    )
    a_fp32_cpu = torch.empty((total_m, k), dtype=torch.float32, device="cpu")
    for e in range(num_experts):
        start = int(offsets_cpu[e].item())
        end = int(offsets_cpu[e + 1].item())
        if end <= start:
            continue
        # Scale by expert id to avoid identical segments.
        a_fp32_cpu[start:end] = torch.randn(
            (end - start, k),
            dtype=torch.float32,
            device="cpu",
        ) * (0.1 + 0.05 * e)

    # Quantize activations to int8 with row-wise asymmetric scale / zero-point.
    a_q_cpu, a_scale_cpu, a_zp_cpu = _quantize_asymmetric_uint8_per_row(a_fp32_cpu)

    a_q = a_q_cpu.to(device).contiguous()
    a_scale = a_scale_cpu.to(device=device, dtype=torch.float32).contiguous()
    a_zp = a_zp_cpu.to(device=device, dtype=torch.uint8).contiguous()

    packed_u4_cpu = torch.randint(
        0,
        256,
        (num_experts, n, k // 2),
        dtype=torch.int32,
        device="cpu",
    ).to(torch.uint8)
    b_packed_s4 = ((packed_u4_cpu.to(device) ^ 0x88)).contiguous()

    b_scales_cpu = (
        torch.rand(
            (num_experts, n, group_num),
            dtype=torch.float16,
            device="cpu",
        ) * 0.5 + 0.01
    )
    b_scales = b_scales_cpu.permute(0, 2, 1).contiguous().to(device)

    bias_cpu: torch.Tensor | None
    bias: torch.Tensor | None
    if has_bias:
        bias_cpu = (
            torch.randn((num_experts, n), dtype=torch.float16, device="cpu")
            * 0.1
        )
        bias = bias_cpu.to(device).contiguous()
    else:
        bias_cpu = None
        bias = None

    offsets = offsets_cpu.to(device).contiguous()
    out = torch.empty((total_m, n), device=device, dtype=out_dtype).contiguous()
    out_repeat = torch.empty((total_m, n), device=device, dtype=out_dtype).contiguous()

    # -----------------------------
    # Run oneDNN op
    # -----------------------------
    max_expert_size = max(token_counts) if token_counts else 0

    try:
        out_ret = torch.ops._xpu_C.onednn_grouped_gemm_w4a8(
            a_q,
            a_scale,
            a_zp,
            b_packed_s4,
            b_scales,
            bias,
            out,
            offsets,
            n,
            k,
            num_experts,
            max_expert_size,
        )

        out_ret_repeat = torch.ops._xpu_C.onednn_grouped_gemm_w4a8(
            a_q,
            a_scale,
            a_zp,
            b_packed_s4,
            b_scales,
            bias,
            out_repeat,
            offsets,
            n,
            k,
            num_experts,
            max_expert_size,
        )
    except RuntimeError as e:
        msg = str(e)
        if (
            "oneDNN grouped GEMM is not enabled in this build" in msg
            or "DNNL_EXPERIMENTAL_GROUPED_MEMORY=0" in msg
        ):
            pytest.skip(msg)
        raise

    torch.xpu.synchronize()

    # The kernel returns D.
    assert out_ret.data_ptr() == out.data_ptr()
    assert out_ret_repeat.data_ptr() == out_repeat.data_ptr()

    torch.testing.assert_close(
        out_repeat.cpu().to(torch.float32),
        out.cpu().to(torch.float32),
        atol=0,
        rtol=0,
    )

    # -----------------------------
    # Reference (fp32): A_deq @ W_deq^T (+ bias)
    # -----------------------------
    a_deq_cpu = (
        a_q_cpu.to(torch.float32) - a_zp_cpu.to(torch.float32).unsqueeze(1)
    ) * a_scale_cpu.to(torch.float32).unsqueeze(1)

    ref = torch.empty((total_m, n), dtype=torch.float32, device="cpu")
    offsets_list = offsets_cpu.tolist()
    for e in range(num_experts):
        start = offsets_list[e]
        end = offsets_list[e + 1]
        if end <= start:
            continue
        w_fp32 = _dequantize_u4_zp8(packed_u4_cpu[e], b_scales_cpu[e], group_size)
        ref[start:end] = a_deq_cpu[start:end] @ w_fp32.T
        if bias_cpu is not None:
            ref[start:end] += bias_cpu[e].to(torch.float32)

    ref_out = ref.to(out_dtype).to(torch.float32)

    if out_dtype == torch.float16:
        atol = 5e-2
        rtol = 5e-2
    else:
        atol = 3e-1
        rtol = 3e-1

    torch.testing.assert_close(out.cpu().to(torch.float32), ref_out, atol=atol, rtol=rtol)
