#pragma once

#include <torch/all.h>

namespace oneDNN {

torch::Tensor grouped_gemm_w4a8(
    torch::Tensor A_q,
    torch::Tensor A_scale,
    torch::Tensor A_zp,
    torch::Tensor B_packed_u4,
    torch::Tensor B_scales,
    const c10::optional<at::Tensor>& bias,
    torch::Tensor D,
    torch::Tensor expert_first_token_offset,
    int64_t N,
    int64_t K,
    int64_t num_experts,
    int64_t max_expert_size);

void onednn_fused_moe_w4a8(
    torch::Tensor hidden_states,
    torch::Tensor w13,
    torch::Tensor w13_scales,
    const c10::optional<at::Tensor>& w13_bias,
    torch::Tensor w2,
    torch::Tensor w2_scales,
    const c10::optional<at::Tensor>& w2_bias,
    torch::Tensor topk_weights,
    torch::Tensor topk_ids,
    const c10::optional<at::Tensor>& expert_map,
    torch::Tensor output,
    torch::Tensor A_q1,
    torch::Tensor A_scale1,
    torch::Tensor A_zp1,
    torch::Tensor gemm1_output,
    torch::Tensor A_q2,
    torch::Tensor A_scale2,
    torch::Tensor A_zp2,
    torch::Tensor gemm2_output,
    torch::Tensor expert_first_token_offset,
    torch::Tensor expert_first_token_offset_i32,
    torch::Tensor unpermuted_row_to_permuted_row,
    int64_t inter_size,
    int64_t hidden_size,
    int64_t num_experts,
    int64_t n_experts_per_token,
    int64_t total_experts_num,
    double alpha,
    double limit);

}  // namespace oneDNN
