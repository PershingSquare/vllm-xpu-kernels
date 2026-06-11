# SPDX-License-Identifier: Apache-2.0
import os
import torch

try:
    from . import _C  # noqa: F401
    from . import _xpu_C  # noqa: F401
    FUSEDMOE_UNAVAILABLE_REASON = None
    FUSEDMOE_AVAILABLE = True
except ImportError as e:
    FUSEDMOE_UNAVAILABLE_REASON = str(e)
    FUSEDMOE_AVAILABLE = False


class _W4A8ScratchPool:
    """Per-shape reusable device buffers for the W4A8 fused-MoE path.

    vLLM runs MoE layers sequentially in eager mode (--enforce-eager), so a
    single buffer set is reused across all layers and decode steps. Buffers are
    keyed by shape/dtype/device; a key change reallocates the whole set. This
    removes the per-layer torch.empty/zeros and the int64->int32 offset cast
    (~25us host work per layer) from the hot path.
    """

    __slots__ = ("_key", "_buffers")

    def __init__(self):
        self._key = None
        self._buffers = None

    def get(self, num_rows, top_k, hidden, inter, num_experts, dtype, device):
        num_moe = top_k * num_rows
        key = (num_rows, top_k, hidden, inter, num_experts, dtype, str(device))
        if key != self._key:
            self._buffers = {
                "gemm1_output": torch.empty((num_moe, 2 * inter),
                                            dtype=dtype, device=device),
                "gemm2_output": torch.empty((num_moe, hidden),
                                            dtype=dtype, device=device),
                "a_q1": torch.empty((num_moe, hidden),
                                    dtype=torch.uint8, device=device),
                "a_scale1": torch.empty(num_moe, dtype=dtype, device=device),
                "a_zp1": torch.empty(num_moe, dtype=torch.uint8, device=device),
                "a_q2": torch.empty((num_moe, inter),
                                    dtype=torch.uint8, device=device),
                "a_scale2": torch.empty(num_moe, dtype=dtype, device=device),
                "a_zp2": torch.empty(num_moe, dtype=torch.uint8, device=device),
                "offset_i64": torch.empty((num_experts + 1,),
                                          dtype=torch.int64, device=device),
                "offset_i32": torch.empty((num_experts + 1,),
                                          dtype=torch.int32, device=device),
                "row_map": torch.empty((num_rows, top_k),
                                       dtype=torch.int32, device=device),
            }
            self._key = key
        return self._buffers


_w4a8_scratch = _W4A8ScratchPool()


def _use_w4a8() -> bool:
    return os.environ.get("VLLM_XPU_USE_W4A8", "0") == "1"


def _dynamic_per_token_quant_int8(x: torch.Tensor):
    """Per-token asymmetric uint8 quant: scale=(max-min)/255, zp=round(-min/scale), q in [0,255].

    Uses the fused SYCL kernel `torch.ops._C.dynamic_per_token_quant_int8_asym`
    when available (single kernel launch on XPU); falls back to a 5-torch-op
    chain for portability.
    """
    fused = getattr(torch.ops._C, "dynamic_per_token_quant_int8_asym", None)
    if fused is not None and x.dtype in (torch.bfloat16, torch.float16):
        d = x.shape[-1]
        x_flat = x.reshape(-1, d).contiguous()
        n = x_flat.shape[0]
        q = torch.empty_like(x_flat, dtype=torch.uint8)
        s = torch.empty(n, dtype=x.dtype, device=x.device)
        z = torch.empty(n, dtype=torch.uint8, device=x.device)
        fused(q, s, z, x_flat)
        return (
            q.reshape(x.shape),
            s.reshape(x.shape[:-1] + (1,)),
            z.reshape(x.shape[:-1] + (1,)),
        )

    flat = x.reshape(-1, x.shape[-1])
    min_val = flat.to(torch.float32).min(dim=-1)[0].unsqueeze(-1)
    max_val = flat.to(torch.float32).max(dim=-1)[0].unsqueeze(-1)
    scale = ((max_val - min_val) / 255.0).clamp(min=1e-10)
    zero_point = torch.clamp(torch.round(-min_val / scale), 0, 255).to(torch.uint8)
    quantized = torch.clamp(
        torch.round(flat.to(torch.float32) / scale + zero_point.to(torch.float32)),
        0, 255,
    ).to(torch.uint8)
    return (
        quantized.reshape(x.shape),
        scale.reshape(x.shape[:-1] + (1,)).to(x.dtype),
        zero_point.reshape(x.shape[:-1] + (1,)),
    )


def cutlass_grouped_gemm(input_A, input_B, bias, output, expert_token_count, n,
                         k, num_experts):
    # expert_token_count_ = torch.tensor(expert_token_count,
    #                                    dtype=torch.int64,
    #                                    device=input_A.device)
    # if bias is not None:
    #     bias = bias.repeat_interleave(expert_token_count_, dim=0).float()

    def exclusive_prefix_sum(arr):
        prefix = [0]
        for i, x in enumerate(arr):
            prefix.append(prefix[-1] + x)
        return prefix

    expert_offset = torch.tensor(exclusive_prefix_sum(expert_token_count),
                                 dtype=torch.int64,
                                 device="xpu")
    torch.ops._xpu_C.grouped_gemm_interface(
        ptr_A=input_A,
        ptr_B=input_B,
        ptr_scales=None,
        ptr_bias=bias,
        ptr_D=output,
        expert_first_token_offset=expert_offset,
        N=n,
        K=k,
        num_experts=num_experts,
        is_B_int4=False,
        is_B_mxfp4=False)


def cutlass_grouped_gemm_xe2(input_A, input_B, scales, bias, output,
                             num_rows_per_expert, n, k, num_experts, is_B_int4,
                             is_B_mxfp4):
    expert_first_token_offset = torch.cat([
        torch.tensor([0],
                     dtype=num_rows_per_expert.dtype,
                     device=num_rows_per_expert.device),
        torch.cumsum(num_rows_per_expert, dim=0)
    ]).to(torch.int64)
    torch.ops._xpu_C.grouped_gemm_interface(
        ptr_A=input_A,
        ptr_B=input_B,
        ptr_scales=scales,
        ptr_bias=bias,
        ptr_D=output,
        expert_first_token_offset=expert_first_token_offset,
        N=n,
        K=k,
        num_experts=num_experts,
        is_B_int4=is_B_int4,
        is_B_mxfp4=is_B_mxfp4)


def ceilDiv(a, b):
    return (a + b - 1) // b


def compute_num_tokens_per_block(num_tokens, num_experts_per_node):
    for num_tokens_per_block in [32, 64, 128, 256, 512, 1024]:
        num_blocks_per_seq = ceilDiv(num_tokens, num_tokens_per_block)
        if num_blocks_per_seq * num_experts_per_node <= num_tokens_per_block:
            return num_tokens_per_block
    return 1024


def implement_zp(qweight):
    # change u4 to s4 to avoid zero point in gemm kernel
    # only support default zero point now
    assert qweight.dtype == torch.uint8, "Input tensor must be uint8"

    high_u4 = (qweight >> 4) & 0x0F
    low_u4 = qweight & 0x0F

    high_s8 = high_u4.to(torch.int8)
    low_s8 = low_u4.to(torch.int8)

    high_s8 = high_s8 - 8
    low_s8 = low_s8 - 8

    def pack_compact(a, b):

        def process_number(x):
            sign = (x < 0).to(torch.uint8)
            abs_low3 = (x.view(torch.uint8) & 0x7).to(torch.uint8)
            return (sign << 3) | abs_low3

        packed_a = process_number(a)
        packed_b = process_number(b)

        return (packed_a << 4) | packed_b

    result = pack_compact(high_s8, low_s8)

    return result


def implement_zp_xor88(qweight):
    assert qweight.dtype == torch.uint8, "Input tensor must be uint8"
    return qweight ^ 0x88


def xpu_fused_moe(hidden_states,
                  w13,
                  w13_scales,
                  w13_bias,
                  w2,
                  w2_scales,
                  w2_bias,
                  topk_weights,
                  topk_ids,
                  n_experts_per_token,
                  activation,
                  num_experts,
                  ep_rank=0,
                  ep_size=1,
                  expert_map=None,
                  output=None,
                  is_fp8=False,
                  is_int4=False,
                  is_mxfp4=False):
    '''
    hidden_states: [num_rows, hidden_size]
    w13: [num_experts, 2*inter_size, hidden_size]
    w13_scales:
        None for bf16/fp16
        or [num_experts] for fp8
        or [num_experts, 2*inter_size, hidden_size // group_size] for 4bits
    w13_bias: [num_experts, 2*inter_size] or None
    w2: [num_experts, hidden_size, inter_size]
    w2_scales:
        None for bf16/fp16
        or [num_experts] for fp8
        or [num_experts, hidden_size, inter_size // group_size] for 4bits
    w2_bias: [num_experts, hidden_size] or None
    topk_weights: [num_rows, topk]
    topk_ids: [num_rows, topk]
    n_experts_per_token: int
    activation: str
    num_experts: int
    is_int4: bool
    is_mxfp4: bool
    '''
    if output is None:
        output = torch.empty_like(hidden_states)
    else:
        assert output.shape == hidden_states.shape, \
            "output shape must be the same as hidden_states shape"

    backend = os.environ.get("VLLM_XPU_GROUPED_GEMM_BACKEND", "").strip().lower()
    using_onednn_int4_backend = is_int4 and backend == "onednn"
    using_w4a8 = using_onednn_int4_backend and _use_w4a8()

    if not is_int4 and not is_mxfp4:
        inter_size = list(w13.shape)[-1] // 2
    else:
        inter_size = list(w13.shape)[-2] // 2

    assert w13.is_contiguous() and w2.is_contiguous()

    hidden_size = int(hidden_states.shape[1])
    if is_int4 and not hasattr(w13, 'xpu_fused_moe'):
        if backend != "onednn":
            w13_tmp = torch.empty_like(w13)
            w2_tmp = torch.empty_like(w2)
            for i in range(num_experts):
                w13_tmp[i] = implement_zp(w13[i])
                w2_tmp[i] = implement_zp(w2[i])
            w13.data = w13_tmp.contiguous()
            w2.data = w2_tmp.contiguous()
        w13.xpu_fused_moe = True

    num_rows, hidden_size = list(hidden_states.shape)
    num_moe_inputs = n_experts_per_token * num_rows
    if topk_ids.dtype == torch.int32:
        topk_ids = topk_ids.to(torch.int64)

    _pool = (_w4a8_scratch.get(num_rows, n_experts_per_token, hidden_size,
                               inter_size, num_experts, hidden_states.dtype,
                               hidden_states.device)
             if using_w4a8 else None)

    if _pool is not None:
        gemm1_output = _pool["gemm1_output"]
    else:
        gemm1_output = torch.empty((num_moe_inputs, 2 * inter_size),
                                   dtype=hidden_states.dtype,
                                   device=hidden_states.device)

    if not is_fp8 and not is_int4 and not is_mxfp4:
        gemm1_scales = None
        gemm2_scales = None
    else:
        gemm1_scales = w13_scales
        gemm2_scales = w2_scales

    if expert_map is None and ep_size > 1:
        expert_map = torch.empty((num_experts * ep_size),
                                 dtype=torch.int32,
                                 device=hidden_states.device)
        torch.ops._moe_C.init_expert_map(expert_map, num_experts, ep_rank,
                                         ep_size)

    if expert_map is not None:
        total_experts_num = expert_map.shape[0]
    else:
        total_experts_num = num_experts * ep_size
    local_experts_num = num_experts

    if _pool is not None:
        expert_first_token_offset = _pool["offset_i64"]
        expert_first_token_offset.zero_()
        unpermuted_row_to_permuted_row = _pool["row_map"]
    else:
        expert_first_token_offset = torch.zeros((num_experts + 1),
                                                dtype=torch.int64,
                                                device=hidden_states.device)
        unpermuted_row_to_permuted_row = torch.empty(
            (num_rows, n_experts_per_token),
            dtype=torch.int32,
            device=hidden_states.device)

    _remap_quant_op = getattr(torch.ops._moe_C,
                              "remap_and_quant_hidden_states_int8", None)
    _use_fused_remap_quant = (using_w4a8 and _remap_quant_op is not None)

    _is_swigluoai = (activation == "swigluoai" or
                     ("SWIGLUOAI" in str(activation)))
    _inter_scale = 2 if activation == "relu2_no_mul" else 1
    _act_quant_op = getattr(torch.ops._C,
                            "swigluoai_and_mul_quant_int8_asym", None)
    _blob_op = getattr(torch.ops._xpu_C, "onednn_fused_moe_w4a8", None)
    _blob_enabled = os.environ.get("VLLM_XPU_W4A8_FUSED_BLOB", "1") == "1"
    _use_blob = (_blob_enabled and using_w4a8 and _pool is not None
                 and _use_fused_remap_quant
                 and _is_swigluoai and _inter_scale == 1
                 and _act_quant_op is not None and _blob_op is not None)

    if _use_blob:
        _blob_op(
            hidden_states, w13, gemm1_scales, w13_bias,
            w2, gemm2_scales, w2_bias,
            topk_weights, topk_ids, expert_map, output,
            _pool["a_q1"], _pool["a_scale1"], _pool["a_zp1"],
            gemm1_output,
            _pool["a_q2"], _pool["a_scale2"], _pool["a_zp2"],
            _pool["gemm2_output"],
            expert_first_token_offset, _pool["offset_i32"],
            unpermuted_row_to_permuted_row,
            inter_size, hidden_size, num_experts, n_experts_per_token,
            total_experts_num, 1.702, 7.0)
        return output

    if _use_fused_remap_quant:
        if _pool is not None:
            A_q = _pool["a_q1"]
            A_scale = _pool["a_scale1"]
            A_zp = _pool["a_zp1"]
        else:
            A_q = torch.empty((num_moe_inputs, hidden_size),
                              dtype=torch.uint8, device=hidden_states.device)
            A_scale = torch.empty(num_moe_inputs, dtype=hidden_states.dtype,
                                  device=hidden_states.device)
            A_zp = torch.empty(num_moe_inputs, dtype=torch.uint8,
                               device=hidden_states.device)
        _remap_quant_op(
            hidden_states=hidden_states,
            remapped_q=A_q,
            remapped_scale=A_scale,
            remapped_zp=A_zp,
            expert_map=expert_map,
            expert_first_token_offset=expert_first_token_offset,
            unpermuted_row_to_permuted_row=unpermuted_row_to_permuted_row,
            topk_ids=topk_ids,
            total_experts_num=total_experts_num,
            local_experts_num=local_experts_num)
    else:
        remapped_hidden_states = torch.empty(
            (num_rows * n_experts_per_token, hidden_size),
            dtype=hidden_states.dtype,
            device=hidden_states.device)
        torch.ops._moe_C.remap_hidden_states(
            hidden_states=hidden_states,
            hidden_states_scales=None,
            remapped_hidden_states=remapped_hidden_states,
            remapped_hidden_states_scales=None,
            expert_map=expert_map,
            expert_first_token_offset=expert_first_token_offset,
            unpermuted_row_to_permuted_row=unpermuted_row_to_permuted_row,
            topk_ids=topk_ids,
            total_experts_num=total_experts_num,
            local_experts_num=local_experts_num)

    if using_w4a8:
        if _pool is not None:
            w4a8_expert_first_token_offset = _pool["offset_i32"]
            w4a8_expert_first_token_offset.copy_(expert_first_token_offset)
        else:
            w4a8_expert_first_token_offset = expert_first_token_offset.to(
                torch.int32)
    else:
        w4a8_expert_first_token_offset = None

    input_B = w13

    max_expert_size = (num_moe_inputs + n_experts_per_token - 1) // n_experts_per_token

    if using_w4a8:
        if not _use_fused_remap_quant:
            A_q, A_scale, A_zp = _dynamic_per_token_quant_int8(
                remapped_hidden_states)
        torch.ops._xpu_C.onednn_grouped_gemm_w4a8(
            A_q, A_scale.reshape(-1), A_zp.reshape(-1),
            input_B, gemm1_scales, w13_bias,
            gemm1_output, w4a8_expert_first_token_offset,
            2 * inter_size, hidden_size, num_experts, max_expert_size)
    else:
        torch.ops._xpu_C.grouped_gemm_interface(
            ptr_A=remapped_hidden_states,
            ptr_B=input_B,
            ptr_scales=gemm1_scales,
            ptr_bias=w13_bias,
            ptr_D=gemm1_output,
            expert_first_token_offset=expert_first_token_offset,
            N=2 * inter_size,
            K=hidden_size,
            num_experts=num_experts,
            is_B_int4=is_int4,
            is_B_mxfp4=is_mxfp4,
            max_expert_size=max_expert_size)

    inter_size_scale = 2 if activation == "relu2_no_mul" else 1
    is_swigluoai = (activation == "swigluoai" or
                    ("SWIGLUOAI" in str(activation)))
    fused_act_quant_op = getattr(torch.ops._C,
                                 "swigluoai_and_mul_quant_int8_asym", None)
    can_fuse_act_quant = (using_w4a8 and is_swigluoai
                          and inter_size_scale == 1
                          and fused_act_quant_op is not None)

    input_B = w2
    if _pool is not None:
        gemm2_output = _pool["gemm2_output"]
    else:
        gemm2_output = torch.empty((num_moe_inputs, hidden_size),
                                    dtype=hidden_states.dtype,
                                    device=hidden_states.device)

    if can_fuse_act_quant:
        if _pool is not None:
            A_q2 = _pool["a_q2"]
            A_scale2 = _pool["a_scale2"]
            A_zp2 = _pool["a_zp2"]
        else:
            A_q2 = torch.empty((num_moe_inputs, inter_size),
                               dtype=torch.uint8, device=gemm1_output.device)
            A_scale2 = torch.empty(num_moe_inputs, dtype=gemm1_output.dtype,
                                   device=gemm1_output.device)
            A_zp2 = torch.empty(num_moe_inputs, dtype=torch.uint8,
                                device=gemm1_output.device)
        fused_act_quant_op(A_q2, A_scale2, A_zp2, gemm1_output, 1.702, 7.0)
        torch.ops._xpu_C.onednn_grouped_gemm_w4a8(
            A_q2, A_scale2, A_zp2,
            input_B, gemm2_scales, w2_bias,
            gemm2_output, w4a8_expert_first_token_offset,
            hidden_size, inter_size, num_experts, max_expert_size)
    else:
        act_output = torch.empty(
            (num_moe_inputs, inter_size * inter_size_scale),
            dtype=gemm1_output.dtype, device=gemm1_output.device)
        if activation == "silu":
            torch.ops._C.silu_and_mul(act_output, gemm1_output)
        elif activation == "gelu":
            torch.ops._C.gelu_and_mul(act_output, gemm1_output)
        elif is_swigluoai:
            torch.ops._C.swigluoai_and_mul(act_output, gemm1_output, 1.702, 7.0)
        elif activation == "relu2_no_mul":
            torch.ops._C.relu2_no_mul(act_output, gemm1_output)
        elif activation == "swiglustep":
            torch.ops._C.swiglustep_and_mul(act_output, gemm1_output, 7.0)
        else:
            raise ValueError(f"Unsupported FusedMoe activation: {activation}.")
        input_A = act_output.contiguous()
        if using_w4a8:
            A_q2, A_scale2, A_zp2 = _dynamic_per_token_quant_int8(input_A)
            torch.ops._xpu_C.onednn_grouped_gemm_w4a8(
                A_q2, A_scale2.reshape(-1), A_zp2.reshape(-1),
                input_B, gemm2_scales, w2_bias,
                gemm2_output, w4a8_expert_first_token_offset,
                hidden_size, inter_size * inter_size_scale, num_experts, max_expert_size)
        else:
            torch.ops._xpu_C.grouped_gemm_interface(
                ptr_A=input_A,
                ptr_B=input_B,
                ptr_scales=gemm2_scales,
                ptr_bias=w2_bias,
                ptr_D=gemm2_output,
                expert_first_token_offset=expert_first_token_offset,
                N=hidden_size,
                K=inter_size * inter_size_scale,
                num_experts=num_experts,
                is_B_int4=is_int4,
                is_B_mxfp4=is_mxfp4,
                max_expert_size=max_expert_size)

    torch.ops._moe_C.moe_gather(output, gemm2_output, topk_weights,
                                unpermuted_row_to_permuted_row,
                                expert_first_token_offset, num_experts)

    return output
