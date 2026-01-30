#include "csrc/utils.h"
#include "grouped_gemm_interface.h"
#include <stdio.h>

#include <algorithm>
#include <cctype>
#include <string>

#include "csrc/xpu/onednn/grouped_gemm_w4a16.h"

#ifdef VLLM_XPU_ENABLE_XE2
  #include "xe_2/grouped_gemm_xe2.h"
#endif
#ifdef VLLM_XPU_ENABLE_XE_DEFAULT
  #include "xe_default/grouped_gemm_xe_default.h"
#endif

torch::Tensor cutlass_grouped_gemm_interface(
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
    bool is_B_mxfp4) {
  auto backend = vllm::xpu::getEnv("VLLM_XPU_GROUPED_GEMM_BACKEND");
  if (backend.has_value() && is_B_int4) {
    std::string backend_lc = backend.value();
    std::transform(
        backend_lc.begin(),
        backend_lc.end(),
        backend_lc.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (backend_lc == "onednn") {
      return oneDNN::grouped_gemm_w4a16(
          ptr_A,
          ptr_B,
          ptr_scales,
          ptr_bias,
          ptr_D,
          expert_first_token_offset,
          N,
          K,
          num_experts,
          is_B_int4,
          is_B_mxfp4);
    }
  }

  if (vllm::xpu::force_xe_default_kernel()) {
#ifdef VLLM_XPU_ENABLE_XE_DEFAULT
    int64_t groups = num_experts;
    return cutlass_grouped_gemm_xe_default(
        ptr_A, ptr_B, ptr_bias, ptr_D, expert_first_token_offset, N, K, groups);
#else
    TORCH_CHECK(
        false,
        "XE default cutlass kernel is not enabled in this build, force use XE "
        "default kernel failed.");
#endif
  } else if (vllm::xpu::is_xe2_arch()) {
#ifdef VLLM_XPU_ENABLE_XE2
    // Use XE2 cutlass kernel
    TORCH_CHECK(false, "cutlass kernel was called")
    return cutlass_grouped_gemm_xe2(
        ptr_A,
        ptr_B,
        ptr_scales,
        ptr_bias,
        ptr_D,
        expert_first_token_offset,
        N,
        K,
        num_experts,
        is_B_int4,
        is_B_mxfp4);
#else
    TORCH_CHECK(false, "XE2 cutlass kernel is not enabled in this build.");
#endif
  } else {
#ifdef VLLM_XPU_ENABLE_XE_DEFAULT
    int64_t groups = num_experts;
    return cutlass_grouped_gemm_xe_default(
        ptr_A, ptr_B, ptr_bias, ptr_D, expert_first_token_offset, N, K, groups);
#else
    TORCH_CHECK(
        false, "XE default cutlass kernel is not enabled in this build.");
#endif
  }
}
