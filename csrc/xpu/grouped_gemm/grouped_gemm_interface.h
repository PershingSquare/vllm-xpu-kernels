#include <torch/all.h>

torch::Tensor grouped_gemm_interface(
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
    bool is_B_mxfp4,
    // Used as DNNL_ARG_HINT_MAX_GROUP_SIZE when routing to oneDNN backend.
    // Pass the actual max tokens per expert for best primitive selection.
    int64_t max_expert_size = 0);
