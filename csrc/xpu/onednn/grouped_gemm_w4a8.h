#pragma once

#include <torch/all.h>

namespace oneDNN {

torch::Tensor grouped_gemm_w4a8(
    torch::Tensor A_q,
    torch::Tensor A_scale,
    torch::Tensor A_zp,
    torch::Tensor B_packed_s4,
    torch::Tensor B_scales,
    const c10::optional<at::Tensor>& bias,
    torch::Tensor D,
    torch::Tensor expert_first_token_offset,
    int64_t N,
    int64_t K,
    int64_t num_experts);

// Prepack weights once at model-load time: converts u4->s4 and permutes scales.
std::tuple<torch::Tensor, torch::Tensor> grouped_gemm_w4a8_prepack(
    torch::Tensor B_packed_u4,
    torch::Tensor B_scales);

// Fast-path GEMM using prepacked weights (skips per-call XOR and permute).
torch::Tensor grouped_gemm_w4a8_prepacked(
    torch::Tensor A_q,
    torch::Tensor A_scale,
    torch::Tensor A_zp,
    torch::Tensor B_s4,
    torch::Tensor B_scales_permuted,
    const c10::optional<at::Tensor>& bias,
    torch::Tensor D,
    torch::Tensor expert_first_token_offset,
    int64_t N,
    int64_t K,
    int64_t num_experts);

// v2: Zero-overhead hot-path. Caller pre-builds expert_ends_i32 and
// max_group_size outside the timing loop, eliminating per-call allocations
// and device→host syncs.
torch::Tensor grouped_gemm_w4a8_prepacked_v2(
    torch::Tensor A_q,
    torch::Tensor A_scale,
    torch::Tensor A_zp,
    torch::Tensor B_s4,
    torch::Tensor B_scales_permuted,
    const c10::optional<at::Tensor>& bias,
    torch::Tensor D,
    torch::Tensor expert_ends_i32,
    int64_t N,
    int64_t K,
    int64_t num_experts,
    int64_t max_group_size);

}  // namespace oneDNN
