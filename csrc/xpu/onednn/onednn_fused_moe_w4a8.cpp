#include "xpu/onednn/grouped_gemm_w4a8.h"

#include "utils.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/all.h>

#include <cstdint>

namespace oneDNN {

namespace {

// Cached dispatcher handles for the cross-extension ops the blob sequences.
// _moe_C and _C live in separate shared objects, so they are reached through
// the torch dispatcher rather than direct linkage. Handles are resolved once.
const c10::OperatorHandle& remap_quant_handle() {
  static auto h = c10::Dispatcher::singleton()
      .findSchemaOrThrow("_moe_C::remap_and_quant_hidden_states_int8", "");
  return h;
}

const c10::OperatorHandle& act_quant_handle() {
  static auto h = c10::Dispatcher::singleton()
      .findSchemaOrThrow("_C::swigluoai_and_mul_quant_int8_asym", "");
  return h;
}

const c10::OperatorHandle& gather_handle() {
  static auto h = c10::Dispatcher::singleton()
      .findSchemaOrThrow("_moe_C::moe_gather", "");
  return h;
}

}  // namespace

// Single-entry fused W4A8 MoE block: remap+quant -> GEMM1 -> swiglu+quant ->
// GEMM2 -> gather, sequenced in C++ to collapse the Python per-thunk dispatch
// floor. The two GEMMs link directly; the three cross-.so kernels go through
// the dispatcher. All scratch is caller-provided so the op stays functional
// (no internal allocation state) and bit-identical to the Python path.
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
    double limit) {
  const int64_t num_moe_inputs = n_experts_per_token * hidden_states.size(0);
  const int64_t max_expert_size =
      (num_moe_inputs + n_experts_per_token - 1) / n_experts_per_token;

  expert_first_token_offset.zero_();

  remap_quant_handle().typed<void(
      const at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&,
      const c10::optional<at::Tensor>&, at::Tensor&, at::Tensor&, at::Tensor&,
      int64_t, int64_t, const c10::optional<at::Tensor>&)>()
      .call(hidden_states, A_q1, A_scale1, A_zp1, expert_map,
            expert_first_token_offset, unpermuted_row_to_permuted_row,
            topk_ids, total_experts_num, num_experts,
            expert_first_token_offset_i32);

  grouped_gemm_w4a8(A_q1, A_scale1, A_zp1, w13, w13_scales, w13_bias,
                    gemm1_output, expert_first_token_offset_i32,
                    2 * inter_size, hidden_size, num_experts, max_expert_size);

  act_quant_handle().typed<void(
      at::Tensor&, at::Tensor&, at::Tensor&, const at::Tensor&, double,
      double)>()
      .call(A_q2, A_scale2, A_zp2, gemm1_output, alpha, limit);

  grouped_gemm_w4a8(A_q2, A_scale2, A_zp2, w2, w2_scales, w2_bias,
                    gemm2_output, expert_first_token_offset_i32,
                    hidden_size, inter_size, num_experts, max_expert_size);

  gather_handle().typed<void(
      at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
      const at::Tensor&, int64_t)>()
      .call(output, gemm2_output, topk_weights,
            unpermuted_row_to_permuted_row, expert_first_token_offset,
            num_experts);
}

}  // namespace oneDNN
