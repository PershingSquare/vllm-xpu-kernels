#pragma once

#include <torch/all.h>

namespace oneDNN {

// oneDNN-backed MoE grouped GEMM for w4a16:
//   A: [Total_M, K] fp16/bf16
//   B: [E, N, K/2] uint8 packed signed int4 (s4) in two nibbles per byte
//   D: [Total_M, N] fp16/bf16
//   expert_first_token_offset: [E+1] int64
torch::Tensor grouped_gemm_w4a16(
    torch::Tensor ptr_A,
    torch::Tensor ptr_B,
    const c10::optional<at::Tensor>& ptr_scales,
    const c10::optional<at::Tensor>& ptr_bias,
    torch::Tensor ptr_D,
    torch::Tensor expert_first_token_offset,
    int64_t N,
    int64_t K,
    int64_t num_experts,
    bool is_B_int4,
    bool is_B_mxfp4);

}  // namespace oneDNN
