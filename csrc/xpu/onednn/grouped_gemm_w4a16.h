#pragma once

#include <torch/all.h>

namespace oneDNN {

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

std::tuple<torch::Tensor, torch::Tensor> grouped_gemm_w4a16_prepack(
    torch::Tensor ptr_B,
    torch::Tensor ptr_scales);

torch::Tensor grouped_gemm_w4a16_prepacked(
    torch::Tensor ptr_A,
    torch::Tensor ptr_B_s4,
    const c10::optional<at::Tensor>& ptr_scales_permuted,
    const c10::optional<at::Tensor>& ptr_bias,
    torch::Tensor ptr_D,
    torch::Tensor expert_first_token_offset,
    int64_t N,
    int64_t K,
    int64_t num_experts);

// v2: Zero-overhead hot-path. Caller pre-builds expert_ends_i32 and
// max_group_size outside the timing loop.
torch::Tensor grouped_gemm_w4a16_prepacked_v2(
    torch::Tensor ptr_A,
    torch::Tensor ptr_B_s4,
    const c10::optional<at::Tensor>& ptr_scales_permuted,
    const c10::optional<at::Tensor>& ptr_bias,
    torch::Tensor ptr_D,
    torch::Tensor expert_ends_i32,
    int64_t N,
    int64_t K,
    int64_t num_experts,
    int64_t max_group_size);

}  // namespace oneDNN
