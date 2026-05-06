#include "csrc/xpu/onednn/grouped_gemm_w4a16.h"

#include "csrc/utils.h"
#include "csrc/xpu/onednn/onednn_grouped_gemm_cache.h"
#include "csrc/xpu/onednn/onednn_runtime.h"

#include <cstdint>
#include <exception>
#include <limits>
#include <string_view>
#include <unordered_map>
#include <vector>

#ifndef DNNL_ARG_HINT_MAX_GROUP_SIZE
#define DNNL_ARG_HINT_MAX_GROUP_SIZE 384
#endif

namespace oneDNN {

namespace {

// Forward decl (used by slow fallback helper below).
static inline dnnl::memory::data_type to_onednn_type(at::ScalarType t);

static inline bool message_contains(std::string_view haystack, std::string_view needle) {
  return haystack.find(needle) != std::string_view::npos;
}

// Slow but robust fallback for environments where oneDNN's int4 GPU path
// fails due to missing runtime features (e.g. named barriers).
// Dequantizes packed int4 weights on CPU and runs grouped matmul with fp16/bf16 weights.
static torch::Tensor grouped_gemm_w4a16_fallback_dequant_cpu(
    torch::Tensor ptr_A,
    torch::Tensor ptr_B_packed_u4,
    torch::Tensor scales_enk,
    const c10::optional<at::Tensor>& ptr_bias,
    torch::Tensor ptr_D,
    torch::Tensor expert_first_token_offset_i32,
    int64_t N,
    int64_t K,
    int64_t num_experts,
    int64_t group_num,
    int64_t group_size,
    const dnnl::memory::desc& src_md,
    const dnnl::memory::desc& dst_md,
    dnnl::engine& engine,
    dnnl::stream& stream) {
  (void)src_md;
  (void)dst_md;
  // Move packed weights and scales to CPU.
  auto B_cpu = ptr_B_packed_u4.cpu().contiguous();
  auto scales_cpu_f32 = scales_enk.cpu().to(at::kFloat).contiguous();

  // Build fp32 weights in logical layout [E, K, N] (contiguous).
  // This is a slow compatibility path; prioritize clarity/correctness.
  auto W_cpu_f32 = torch::empty(
      {num_experts, K, N},
      torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat));

  auto B_acc = B_cpu.accessor<uint8_t, 3>();
  auto S_acc = scales_cpu_f32.accessor<float, 3>();
  auto W_acc = W_cpu_f32.accessor<float, 3>();

  // ptr_B layout is [E, N, K/2] bytes (two int4 values per byte).
  const int64_t K_half = K / 2;
  for (int64_t e = 0; e < num_experts; ++e) {
    for (int64_t n = 0; n < N; ++n) {
      for (int64_t j = 0; j < K_half; ++j) {
        const uint8_t byte = B_acc[e][n][j];
        // vLLM convention: packed u4 with implicit zero-point 8.
        // Convert to signed [-8..7] by subtracting 8 per nibble.
        int8_t lo = static_cast<int8_t>(static_cast<int>(byte & 0x0F) - 8);
        int8_t hi = static_cast<int8_t>(static_cast<int>((byte >> 4) & 0x0F) - 8);

        const int64_t k0 = 2 * j;
        const int64_t k1 = k0 + 1;

        const int64_t g0 = k0 / group_size;
        const int64_t g1 = k1 / group_size;
        // scales are [E, N, group_num]
        const float s0 = S_acc[e][n][g0];
        const float s1 = S_acc[e][n][g1];

        // W is stored as [E, K, N]
        W_acc[e][k0][n] = static_cast<float>(lo) * s0;
        W_acc[e][k1][n] = static_cast<float>(hi) * s1;
      }
    }
  }

  // Move weights to device with same dtype as A (fp16/bf16).
  auto W_dev = W_cpu_f32.to(ptr_A.scalar_type()).to(ptr_A.device()).contiguous();
  CHECK_DEVICE(W_dev);
  CHECK_CONTIGUOUS(W_dev);

  // Run per-expert matmuls with standard (non-grouped) memory descriptors.
  // This avoids relying on experimental grouped offset semantics.
  auto offsets_cpu = expert_first_token_offset_i32.cpu().contiguous();
  auto offsets_ptr = offsets_cpu.data_ptr<int>();

  const auto src_dt = to_onednn_type(ptr_A.scalar_type());
  const size_t a_elem = ptr_A.element_size();
  const size_t d_elem = ptr_D.element_size();
  const size_t w_elem = W_dev.element_size();

  char* A_base = static_cast<char*>(ptr_A.data_ptr());
  char* D_base = static_cast<char*>(ptr_D.data_ptr());
  char* W_base = static_cast<char*>(W_dev.data_ptr());

  for (int64_t e = 0; e < num_experts; ++e) {
    const int64_t start = offsets_ptr[e];
    const int64_t end = offsets_ptr[e + 1];
    if (end <= start) continue;
    const int64_t M_e = end - start;

    auto src_md_e = dnnl::memory::desc(
        {M_e, K}, src_dt, dnnl::memory::format_tag::ab);
    auto dst_md_e = dnnl::memory::desc(
        {M_e, N}, src_dt, dnnl::memory::format_tag::ab);
    auto wei_md_e = dnnl::memory::desc(
        {K, N}, src_dt, dnnl::memory::format_tag::ab);

    // Slice pointers.
    void* A_ptr_e = A_base + static_cast<size_t>(start) * static_cast<size_t>(K) * a_elem;
    void* D_ptr_e = D_base + static_cast<size_t>(start) * static_cast<size_t>(N) * d_elem;
    void* W_ptr_e = W_base + static_cast<size_t>(e) * static_cast<size_t>(K) *
        static_cast<size_t>(N) * w_elem;

    dnnl::memory::desc bias_md_e;
    bool has_bias = false;
    void* bias_ptr_e = nullptr;
    if (ptr_bias.has_value()) {
      const at::Tensor& bias = *ptr_bias;
      has_bias = true;
      bias_md_e = dnnl::memory::desc({N}, to_onednn_type(bias.scalar_type()), dnnl::memory::format_tag::a);
      bias_ptr_e = static_cast<char*>(bias.data_ptr()) + static_cast<size_t>(e) * static_cast<size_t>(N) * bias.element_size();
    }

    dnnl::matmul::primitive_desc pd;
    if (has_bias) {
      pd = dnnl::matmul::primitive_desc(engine, src_md_e, wei_md_e, bias_md_e, dst_md_e);
    } else {
      pd = dnnl::matmul::primitive_desc(engine, src_md_e, wei_md_e, dst_md_e);
    }
    dnnl::matmul prim(pd);

    auto src_mem = oneDNN::make_onednn_memory(src_md_e, engine, A_ptr_e);
    auto dst_mem = oneDNN::make_onednn_memory(dst_md_e, engine, D_ptr_e);
    auto wei_mem = oneDNN::make_onednn_memory(wei_md_e, engine, W_ptr_e);

    std::unordered_map<int, dnnl::memory> args;
    args.emplace(DNNL_ARG_SRC, std::move(src_mem));
    args.emplace(DNNL_ARG_WEIGHTS, std::move(wei_mem));
    args.emplace(DNNL_ARG_DST, std::move(dst_mem));
    if (has_bias) {
      auto bias_mem = oneDNN::make_onednn_memory(bias_md_e, engine, bias_ptr_e);
      args.emplace(DNNL_ARG_BIAS, std::move(bias_mem));
    }

    (void)dnnl::sycl_interop::execute(prim, stream, args);
  }

  return ptr_D;
}

static inline dnnl::memory::data_type to_onednn_type(at::ScalarType t) {
  switch (t) {
    case at::ScalarType::Half: return dnnl::memory::data_type::f16;
    case at::ScalarType::BFloat16: return dnnl::memory::data_type::bf16;
    case at::ScalarType::Float: return dnnl::memory::data_type::f32;
    case at::ScalarType::Int: return dnnl::memory::data_type::s32;
    default: break;
  }
  TORCH_CHECK(false, "Unsupported dtype in oneDNN grouped GEMM: ", t);
  return dnnl::memory::data_type::undef;
}

static inline bool is_fp16_or_bf16(at::ScalarType t) {
  return t == at::ScalarType::Half || t == at::ScalarType::BFloat16;
}

}  // namespace

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
    bool is_B_mxfp4) {
#if !(defined(DNNL_EXPERIMENTAL_GROUPED_MEMORY) && DNNL_EXPERIMENTAL_GROUPED_MEMORY) && \
    !(defined(DNNL_EXPERIMENTAL_GROUPED_GEMM) && DNNL_EXPERIMENTAL_GROUPED_GEMM)
  (void)ptr_A;
  (void)ptr_B;
  (void)ptr_scales;
  (void)ptr_bias;
  (void)ptr_D;
  (void)expert_first_token_offset;
  (void)N;
  (void)K;
  (void)num_experts;
  (void)is_B_int4;
  (void)is_B_mxfp4;
  TORCH_CHECK(
      false,
      "oneDNN grouped GEMM is not enabled in this build "
      "(DNNL_EXPERIMENTAL_GROUPED_MEMORY=0 / DNNL_EXPERIMENTAL_GROUPED_GEMM=0)");
#else
  TORCH_CHECK(is_B_int4, "oneDNN grouped GEMM path only supports is_B_int4");
  TORCH_CHECK(
      !is_B_mxfp4,
      "oneDNN grouped GEMM path does not support is_B_mxfp4 (w4a16 only)");
  TORCH_WARN_ONCE(
      "Consider using grouped_gemm_w4a16_prepacked for better performance.");

  // -----------------------------
  // Validate inputs
  // -----------------------------
  CHECK_DEVICE(ptr_A);
  CHECK_DEVICE(ptr_B);
  CHECK_DEVICE(ptr_D);
  CHECK_DEVICE(expert_first_token_offset);

  TORCH_CHECK(ptr_A.dim() == 2, "ptr_A must be 2D [Total_M, K]");
  TORCH_CHECK(ptr_B.dim() == 3, "ptr_B must be 3D [E, N, K/2]");
  TORCH_CHECK(ptr_D.dim() == 2, "ptr_D must be 2D [Total_M, N]");
  TORCH_CHECK(
      expert_first_token_offset.dim() == 1,
      "expert_first_token_offset must be 1D [E+1]");

  CHECK_CONTIGUOUS(ptr_A);
  CHECK_CONTIGUOUS(ptr_B);
  CHECK_CONTIGUOUS(ptr_D);
  CHECK_CONTIGUOUS(expert_first_token_offset);

  auto A_dtype = ptr_A.scalar_type();
  TORCH_CHECK(
      is_fp16_or_bf16(A_dtype),
      "ptr_A must be fp16 or bf16 for oneDNN w4a16 grouped GEMM");
  TORCH_CHECK(
      ptr_D.scalar_type() == A_dtype,
      "ptr_D dtype must match ptr_A dtype for oneDNN w4a16 grouped GEMM");

  TORCH_CHECK(
      ptr_B.scalar_type() == at::ScalarType::Byte,
      "ptr_B must be uint8 (packed int4) for oneDNN w4a16 grouped GEMM");

  TORCH_CHECK(
      expert_first_token_offset.scalar_type() == at::ScalarType::Long,
      "expert_first_token_offset must be int64");

  const int64_t total_M = ptr_A.size(0);
  TORCH_CHECK(
      total_M <= std::numeric_limits<int>::max(),
      "oneDNN grouped_gemm_w4a16: total_M exceeds int32 range (total_M=",
      total_M,
      ")");
  TORCH_CHECK(
      ptr_A.size(1) == K,
      "ptr_A.size(1) must match K (expected K=",
      K,
      ", got ",
      ptr_A.size(1),
      ")");

  TORCH_CHECK(
      ptr_B.size(0) == num_experts,
      "ptr_B.size(0) must match num_experts");
  TORCH_CHECK(ptr_B.size(1) == N, "ptr_B.size(1) must match N");
  TORCH_CHECK(
      ptr_B.size(2) * 2 == K,
      "ptr_B.size(2) must be K/2 bytes for packed int4 weights");

  TORCH_CHECK(
      ptr_D.size(0) == total_M,
      "ptr_D.size(0) must match ptr_A.size(0)");
  TORCH_CHECK(ptr_D.size(1) == N, "ptr_D.size(1) must match N");

  TORCH_CHECK(
      expert_first_token_offset.numel() == (num_experts + 1),
      "expert_first_token_offset must have length num_experts+1");

  // Scales are required for w4a16.
  TORCH_CHECK(ptr_scales.has_value(), "w4a16 grouped GEMM must have scales");
  const at::Tensor& scales = *ptr_scales;
  CHECK_DEVICE(scales);
  CHECK_CONTIGUOUS(scales);
  TORCH_CHECK(scales.dim() == 3, "ptr_scales must be 3D [E, N, K/group]");
  TORCH_CHECK(
      scales.size(0) == num_experts, "ptr_scales.size(0) must match num_experts");
  TORCH_CHECK(scales.size(1) == N, "ptr_scales.size(1) must match N");
  TORCH_CHECK(
      scales.size(2) > 0, "ptr_scales.size(2) (group_num) must be > 0");
  TORCH_CHECK(
      K % scales.size(2) == 0,
      "ptr_scales.size(2) (group_num) must divide K");
  const int64_t group_num = scales.size(2);
  const int64_t group_size = K / group_num;

  // Grouped GPU int4 micro-kernel requires K and N even for WEI=s4/u4.
  TORCH_CHECK(
      (K % 2) == 0,
      "oneDNN grouped_gemm_w4a16 requires K even for int4 weights (K=",
      K,
      ")");
  TORCH_CHECK(
      (N % 2) == 0,
      "oneDNN grouped_gemm_w4a16 requires N even for int4 weights (N=",
      N,
      ")");

  // oneDNN expects weight scales for mask=7 (per-expert, per-K-group, per-N)
  // as a physically contiguous tensor with logical layout [E, group_num, N]
  // (N is the innermost dim). vLLM provides scales as [E, N, group_num].
  auto scales_repacked = scales.permute({0, 2, 1}).contiguous();
  CHECK_DEVICE(scales_repacked);
  CHECK_CONTIGUOUS(scales_repacked);

  // Optional bias: [E, N].
  if (ptr_bias.has_value()) {
    const at::Tensor& bias = *ptr_bias;
    CHECK_DEVICE(bias);
    CHECK_CONTIGUOUS(bias);
    TORCH_CHECK(bias.dim() == 2, "ptr_bias must be 2D [E, N]");
    TORCH_CHECK(bias.size(0) == num_experts, "ptr_bias.size(0) must match E");
    TORCH_CHECK(bias.size(1) == N, "ptr_bias.size(1) must match N");
  }

  // -----------------------------
  // oneDNN descriptors
  // -----------------------------
  const auto src_dt = to_onednn_type(A_dtype);
  const auto dst_dt = src_dt;
  const auto scales_dt = to_onednn_type(scales.scalar_type());

  // expert_first_token_offset is int64 in vLLM; oneDNN grouped encoding expects s32.
  auto expert_first_token_offset_i32 =
      (expert_first_token_offset.scalar_type() == at::ScalarType::Int)
      ? expert_first_token_offset
      : expert_first_token_offset.to(at::ScalarType::Int);
  CHECK_DEVICE(expert_first_token_offset_i32);
  CHECK_CONTIGUOUS(expert_first_token_offset_i32);

  // oneDNN expects cumulative ends: ends[e] = offsets[e+1].
  torch::Tensor expert_ends_onednn_i32 = torch::empty(
      {num_experts + 1},
      expert_first_token_offset_i32.options().dtype(at::ScalarType::Int));
  CHECK_DEVICE(expert_ends_onednn_i32);
  CHECK_CONTIGUOUS(expert_ends_onednn_i32);
  expert_ends_onednn_i32.narrow(0, 0, num_experts)
      .copy_(expert_first_token_offset_i32.narrow(0, 1, num_experts));
  expert_ends_onednn_i32.narrow(0, num_experts, 1).fill_(static_cast<int>(total_M));

  // Grouped encoding on dim0 (M) for src/dst.
  const dnnl::memory::dim ngroups = static_cast<dnnl::memory::dim>(num_experts);
  auto src_md = dnnl::memory::desc::grouped(
      {total_M, K},
      src_dt,
      /*variable_dim_idx=*/0,
      /*group_count=*/ngroups,
      /*offsets_dt=*/dnnl::memory::data_type::s32);
  auto dst_md = dnnl::memory::desc::grouped(
      {total_M, N},
      dst_dt,
      /*variable_dim_idx=*/0,
      /*group_count=*/ngroups,
      /*offsets_dt=*/dnnl::memory::data_type::s32);

  // Weights are dense logical [E, K, N] with dtype s4.
  // vLLM physical buffer is [E, N, K/2] bytes (two int4 per byte).
  // Describe as tag `acb` so oneDNN indexes as (e-major, n-major, k-contig).
  auto wei_md = dnnl::memory::desc(
      {num_experts, K, N}, dnnl::memory::data_type::s4, dnnl::memory::format_tag::acb);

  // Scales are expected by oneDNN as [E, K/group, N] (contiguous).
  auto scales_md = dnnl::memory::desc(
      {num_experts, group_num, N}, scales_dt, dnnl::memory::format_tag::abc);

  // Optional bias: [E, N].
  dnnl::memory::desc bias_md;
  if (ptr_bias.has_value()) {
    const at::Tensor& bias = *ptr_bias;
    bias_md = dnnl::memory::desc(
        {num_experts, N},
        to_onednn_type(bias.scalar_type()),
        {/*stride_e=*/N, /*stride_n=*/1});
  }

  // Configure weight scales.
  dnnl::primitive_attr attr;
  // weights dims are [E, K, N]. mask 0/1/2 => per-(E, K-group, N).
  // groups correspond to (K, N) dims; K is grouped by group_size, N is not.
  attr.set_scales(
      DNNL_ARG_WEIGHTS,
      /*mask=*/(1 << 0) | (1 << 1) | (1 << 2),
      /*groups=*/{group_size, 1},
      scales_dt);

  // -----------------------------
  // Execute oneDNN matmul
  // -----------------------------
  const at::Device cur_device = ptr_A.device();
  const int device_id = cur_device.index();
  auto& engine = oneDNN::GpuEngineManager::Instance().get_engine(cur_device);
  auto& stream = oneDNN::GpuStreamManager::Instance().get_stream(device_id);

  grouped_gemm_primitive_key_t cache_key{};
  cache_key.device_id = device_id;
  cache_key.src_dtype = static_cast<int64_t>(src_dt);
  cache_key.wei_dtype = static_cast<int64_t>(dnnl::memory::data_type::s4);
  cache_key.dst_dtype = static_cast<int64_t>(dst_dt);
  cache_key.src_scales_dtype =
      static_cast<int64_t>(dnnl::memory::data_type::undef);
  cache_key.wei_scales_dtype = static_cast<int64_t>(scales_dt);
  cache_key.bias_dtype = ptr_bias.has_value()
      ? static_cast<int64_t>(to_onednn_type(ptr_bias.value().scalar_type()))
      : static_cast<int64_t>(dnnl::memory::data_type::undef);
  cache_key.requested_dst_dtype = static_cast<int64_t>(dst_dt);
  cache_key.total_m = total_M;
  cache_key.n = N;
  cache_key.k = K;
  cache_key.num_experts = num_experts;
  cache_key.group_num = group_num;
  cache_key.group_size = group_size;
  cache_key.has_bias = ptr_bias.has_value() ? 1 : 0;

  auto& primitive_cache = get_grouped_gemm_primitive_cache(device_id);
  auto iter = primitive_cache.find(cache_key);
  if (iter == primitive_cache.end()) {
    dnnl::matmul::primitive_desc pd;
    try {
      if (ptr_bias.has_value()) {
        pd = dnnl::matmul::primitive_desc(engine, src_md, wei_md, bias_md, dst_md, attr);
      } else {
        pd = dnnl::matmul::primitive_desc(engine, src_md, wei_md, dst_md, attr);
      }
    } catch (const dnnl::error& e) {
      TORCH_WARN(
          "oneDNN grouped_gemm_w4a16: matmul primitive_desc creation failed: ",
          e.what());
      auto ocl_icd_vendors = vllm::xpu::getEnv("OCL_ICD_VENDORS");
      if (ocl_icd_vendors.has_value()) {
        TORCH_CHECK(
            false,
            "oneDNN grouped_gemm_w4a16: matmul primitive_desc creation failed on GPU while "
            "OCL_ICD_VENDORS is set to '",
            ocl_icd_vendors.value(),
            "'. CPU fallback is disabled. Detail: ",
            e.what());
      }
      throw;
    }

    dnnl::matmul prim;
    try {
      prim = dnnl::matmul(pd);
    } catch (const sycl::exception& e) {
      TORCH_WARN(
          "oneDNN grouped_gemm_w4a16: SYCL exception during primitive construction: ",
          e.what());
      throw;
    } catch (const dnnl::error& e) {
      TORCH_WARN("oneDNN grouped_gemm_w4a16: matmul primitive creation failed: ", e.what());
      throw;
    } catch (const std::exception& e) {
      TORCH_WARN(
          "oneDNN grouped_gemm_w4a16: std::exception during primitive construction: ",
          e.what());
      if (message_contains(e.what(), "Named barriers not yet implemented")) {
        TORCH_CHECK(
            false,
            "oneDNN grouped_gemm_w4a16: GPU primitive construction failed because named barriers "
            "are not implemented in this environment. CPU fallback is disabled. Detail: ",
            e.what());
      }
      throw;
    }

    iter = primitive_cache
               .insert(
                   {cache_key,
                    grouped_gemm_cached_primitive_t{std::move(pd), std::move(prim)}})
               .first;
  }

  // Convert vLLM packed u4(zp=8) to packed s4 (two's complement) for oneDNN.
  torch::Tensor B_for_onednn = (ptr_B ^ 0x88).contiguous();

  // Grouped src/dst memories carry both (values, offsets) handles.
  auto src_mem = dnnl::sycl_interop::make_memory(
      src_md,
      engine,
      dnnl::sycl_interop::memory_kind::usm,
      std::vector<void*>{ptr_A.data_ptr(), expert_ends_onednn_i32.data_ptr()});
  auto dst_mem = dnnl::sycl_interop::make_memory(
      dst_md,
      engine,
      dnnl::sycl_interop::memory_kind::usm,
      std::vector<void*>{ptr_D.data_ptr(), expert_ends_onednn_i32.data_ptr()});

  // Dense weights/scales/bias.
  auto wei_mem = oneDNN::make_onednn_memory(wei_md, engine, B_for_onednn.data_ptr());
  auto scales_mem =
      oneDNN::make_onednn_memory(scales_md, engine, scales_repacked.data_ptr());

  std::unordered_map<int, dnnl::memory> args;
  args.emplace(DNNL_ARG_SRC, std::move(src_mem));
  args.emplace(DNNL_ARG_WEIGHTS, std::move(wei_mem));
  args.emplace(DNNL_ARG_DST, std::move(dst_mem));
  args.emplace(DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, std::move(scales_mem));
  if (ptr_bias.has_value()) {
    const at::Tensor& bias = *ptr_bias;
    auto bias_mem = oneDNN::make_onednn_memory(bias_md, engine, bias.data_ptr());
    args.emplace(DNNL_ARG_BIAS, std::move(bias_mem));
  }

  try {
    (void)dnnl::sycl_interop::execute(iter->second.prim, stream, args);
  } catch (const sycl::exception& e) {
    TORCH_WARN(
        "oneDNN grouped_gemm_w4a16: SYCL exception during execute: ",
        e.what());
    throw;
  } catch (const dnnl::error& e) {
    TORCH_WARN(
        "oneDNN grouped_gemm_w4a16: oneDNN exception during execute: ",
        e.what());
    throw;
  } catch (const std::exception& e) {
    TORCH_WARN(
        "oneDNN grouped_gemm_w4a16: std::exception during execute: ",
        e.what());
    throw;
  }
  return ptr_D;
#endif
}

std::tuple<torch::Tensor, torch::Tensor> grouped_gemm_w4a16_prepack(
    torch::Tensor ptr_B,
    torch::Tensor ptr_scales) {
  // Convert vLLM packed u4(zp=8) to packed s4 (two's complement) for oneDNN.
  auto B_s4 = (ptr_B ^ 0x88).contiguous();
  // Permute scales from vLLM [E, N, G] to oneDNN [E, G, N].
  auto scales_permuted = ptr_scales.permute({0, 2, 1}).contiguous();
  return std::make_tuple(std::move(B_s4), std::move(scales_permuted));
}

torch::Tensor grouped_gemm_w4a16_prepacked(
    torch::Tensor ptr_A,
    torch::Tensor ptr_B_s4,
    const c10::optional<at::Tensor>& ptr_scales_permuted,
    const c10::optional<at::Tensor>& ptr_bias,
    torch::Tensor ptr_D,
    torch::Tensor expert_first_token_offset,
    int64_t N,
    int64_t K,
    int64_t num_experts) {
#if !(defined(DNNL_EXPERIMENTAL_GROUPED_MEMORY) && DNNL_EXPERIMENTAL_GROUPED_MEMORY) && \
    !(defined(DNNL_EXPERIMENTAL_GROUPED_GEMM) && DNNL_EXPERIMENTAL_GROUPED_GEMM)
  (void)ptr_A;
  (void)ptr_B_s4;
  (void)ptr_scales_permuted;
  (void)ptr_bias;
  (void)ptr_D;
  (void)expert_first_token_offset;
  (void)N;
  (void)K;
  (void)num_experts;
  TORCH_CHECK(
      false,
      "oneDNN grouped GEMM is not enabled in this build "
      "(DNNL_EXPERIMENTAL_GROUPED_MEMORY=0 / DNNL_EXPERIMENTAL_GROUPED_GEMM=0)");
#else
  // -----------------------------
  // Validate inputs
  // -----------------------------
  CHECK_DEVICE(ptr_A);
  CHECK_DEVICE(ptr_B_s4);
  CHECK_DEVICE(ptr_D);
  CHECK_DEVICE(expert_first_token_offset);

  TORCH_CHECK(ptr_A.dim() == 2, "ptr_A must be 2D [Total_M, K]");
  TORCH_CHECK(ptr_B_s4.dim() == 3, "ptr_B_s4 must be 3D [E, N, K/2]");
  TORCH_CHECK(ptr_D.dim() == 2, "ptr_D must be 2D [Total_M, N]");
  TORCH_CHECK(
      expert_first_token_offset.dim() == 1,
      "expert_first_token_offset must be 1D [E+1]");

  CHECK_CONTIGUOUS(ptr_A);
  CHECK_CONTIGUOUS(ptr_B_s4);
  CHECK_CONTIGUOUS(ptr_D);
  CHECK_CONTIGUOUS(expert_first_token_offset);

  auto A_dtype = ptr_A.scalar_type();
  TORCH_CHECK(
      is_fp16_or_bf16(A_dtype),
      "ptr_A must be fp16 or bf16 for oneDNN w4a16 grouped GEMM");
  TORCH_CHECK(
      ptr_D.scalar_type() == A_dtype,
      "ptr_D dtype must match ptr_A dtype for oneDNN w4a16 grouped GEMM");

  TORCH_CHECK(
      ptr_B_s4.scalar_type() == at::ScalarType::Byte,
      "ptr_B_s4 must be uint8 (packed s4) for oneDNN w4a16 grouped GEMM");

  TORCH_CHECK(
      expert_first_token_offset.scalar_type() == at::ScalarType::Long,
      "expert_first_token_offset must be int64");

  const int64_t total_M = ptr_A.size(0);
  TORCH_CHECK(
      total_M <= std::numeric_limits<int>::max(),
      "oneDNN grouped_gemm_w4a16_prepacked: total_M exceeds int32 range (total_M=",
      total_M,
      ")");
  TORCH_CHECK(
      ptr_A.size(1) == K,
      "ptr_A.size(1) must match K (expected K=",
      K,
      ", got ",
      ptr_A.size(1),
      ")");

  TORCH_CHECK(
      ptr_B_s4.size(0) == num_experts,
      "ptr_B_s4.size(0) must match num_experts");
  TORCH_CHECK(ptr_B_s4.size(1) == N, "ptr_B_s4.size(1) must match N");
  TORCH_CHECK(
      ptr_B_s4.size(2) * 2 == K,
      "ptr_B_s4.size(2) must be K/2 bytes for packed s4 weights");

  TORCH_CHECK(
      ptr_D.size(0) == total_M,
      "ptr_D.size(0) must match ptr_A.size(0)");
  TORCH_CHECK(ptr_D.size(1) == N, "ptr_D.size(1) must match N");

  TORCH_CHECK(
      expert_first_token_offset.numel() == (num_experts + 1),
      "expert_first_token_offset must have length num_experts+1");

  // Scales are required for w4a16 (already permuted to [E, G, N]).
  TORCH_CHECK(ptr_scales_permuted.has_value(), "w4a16 grouped GEMM must have scales");
  const at::Tensor& scales = *ptr_scales_permuted;
  CHECK_DEVICE(scales);
  CHECK_CONTIGUOUS(scales);
  TORCH_CHECK(scales.dim() == 3, "ptr_scales_permuted must be 3D [E, G, N]");
  TORCH_CHECK(
      scales.size(0) == num_experts, "ptr_scales_permuted.size(0) must match num_experts");
  TORCH_CHECK(scales.size(2) == N, "ptr_scales_permuted.size(2) must match N");
  TORCH_CHECK(
      scales.size(1) > 0, "ptr_scales_permuted.size(1) (group_num) must be > 0");
  TORCH_CHECK(
      K % scales.size(1) == 0,
      "ptr_scales_permuted.size(1) (group_num) must divide K");
  const int64_t group_num = scales.size(1);
  const int64_t group_size = K / group_num;

  // Grouped GPU int4 micro-kernel requires K and N even for WEI=s4/u4.
  TORCH_CHECK(
      (K % 2) == 0,
      "oneDNN grouped_gemm_w4a16_prepacked requires K even for int4 weights (K=",
      K,
      ")");
  TORCH_CHECK(
      (N % 2) == 0,
      "oneDNN grouped_gemm_w4a16_prepacked requires N even for int4 weights (N=",
      N,
      ")");

  // Optional bias: [E, N].
  if (ptr_bias.has_value()) {
    const at::Tensor& bias = *ptr_bias;
    CHECK_DEVICE(bias);
    CHECK_CONTIGUOUS(bias);
    TORCH_CHECK(bias.dim() == 2, "ptr_bias must be 2D [E, N]");
    TORCH_CHECK(bias.size(0) == num_experts, "ptr_bias.size(0) must match E");
    TORCH_CHECK(bias.size(1) == N, "ptr_bias.size(1) must match N");
  }

  // -----------------------------
  // oneDNN descriptors
  // -----------------------------
  const auto src_dt = to_onednn_type(A_dtype);
  const auto dst_dt = src_dt;
  const auto scales_dt = to_onednn_type(scales.scalar_type());

  // expert_first_token_offset is int64 in vLLM; oneDNN grouped encoding expects s32.
  auto expert_first_token_offset_i32 =
      (expert_first_token_offset.scalar_type() == at::ScalarType::Int)
      ? expert_first_token_offset
      : expert_first_token_offset.to(at::ScalarType::Int);
  CHECK_DEVICE(expert_first_token_offset_i32);
  CHECK_CONTIGUOUS(expert_first_token_offset_i32);

  // oneDNN expects cumulative ends: ends[e] = offsets[e+1].
  torch::Tensor expert_ends_onednn_i32 = torch::empty(
      {num_experts + 1},
      expert_first_token_offset_i32.options().dtype(at::ScalarType::Int));
  CHECK_DEVICE(expert_ends_onednn_i32);
  CHECK_CONTIGUOUS(expert_ends_onednn_i32);
  expert_ends_onednn_i32.narrow(0, 0, num_experts)
      .copy_(expert_first_token_offset_i32.narrow(0, 1, num_experts));
  expert_ends_onednn_i32.narrow(0, num_experts, 1).fill_(static_cast<int>(total_M));

  // Compute max group size from expert offsets for dispatch hint.
  // oneDNN reads this via host_ptr(), so it must be in CPU memory.
  int32_t max_group_size_val = 0;
  {
    auto offsets_cpu = expert_first_token_offset_i32.cpu();
    const int32_t* optr = offsets_cpu.data_ptr<int32_t>();
    for (int64_t e = 0; e < num_experts; ++e) {
      int32_t g = optr[e + 1] - optr[e];
      if (g > max_group_size_val) max_group_size_val = g;
    }
  }

  // Grouped encoding on dim0 (M) for src/dst.
  const dnnl::memory::dim ngroups = static_cast<dnnl::memory::dim>(num_experts);
  auto src_md = dnnl::memory::desc::grouped(
      {total_M, K},
      src_dt,
      /*variable_dim_idx=*/0,
      /*group_count=*/ngroups,
      /*offsets_dt=*/dnnl::memory::data_type::s32);
  auto dst_md = dnnl::memory::desc::grouped(
      {total_M, N},
      dst_dt,
      /*variable_dim_idx=*/0,
      /*group_count=*/ngroups,
      /*offsets_dt=*/dnnl::memory::data_type::s32);

  // Weights are dense logical [E, K, N] with dtype s4.
  // Physical buffer is [E, N, K/2] bytes (two int4 per byte).
  // Describe as tag `acb` so oneDNN indexes as (e-major, n-major, k-contig).
  auto wei_md = dnnl::memory::desc(
      {num_experts, K, N}, dnnl::memory::data_type::s4, dnnl::memory::format_tag::acb);

  // Scales are [E, K/group, N] (contiguous, already permuted).
  auto scales_md = dnnl::memory::desc(
      {num_experts, group_num, N}, scales_dt, dnnl::memory::format_tag::abc);

  // Optional bias: [E, N].
  dnnl::memory::desc bias_md;
  if (ptr_bias.has_value()) {
    const at::Tensor& bias = *ptr_bias;
    bias_md = dnnl::memory::desc(
        {num_experts, N},
        to_onednn_type(bias.scalar_type()),
        {/*stride_e=*/N, /*stride_n=*/1});
  }

  // Configure weight scales.
  dnnl::primitive_attr attr;
  // weights dims are [E, K, N]. mask 0/1/2 => per-(E, K-group, N).
  // groups correspond to (K, N) dims; K is grouped by group_size, N is not.
  attr.set_scales(
      DNNL_ARG_WEIGHTS,
      /*mask=*/(1 << 0) | (1 << 1) | (1 << 2),
      /*groups=*/{group_size, 1},
      scales_dt);

  // -----------------------------
  // Execute oneDNN matmul
  // -----------------------------
  const at::Device cur_device = ptr_A.device();
  const int device_id = cur_device.index();
  auto& engine = oneDNN::GpuEngineManager::Instance().get_engine(cur_device);
  auto& stream = oneDNN::GpuStreamManager::Instance().get_stream(device_id);

  grouped_gemm_primitive_key_t cache_key{};
  cache_key.device_id = device_id;
  cache_key.src_dtype = static_cast<int64_t>(src_dt);
  cache_key.wei_dtype = static_cast<int64_t>(dnnl::memory::data_type::s4);
  cache_key.dst_dtype = static_cast<int64_t>(dst_dt);
  cache_key.src_scales_dtype =
      static_cast<int64_t>(dnnl::memory::data_type::undef);
  cache_key.wei_scales_dtype = static_cast<int64_t>(scales_dt);
  cache_key.bias_dtype = ptr_bias.has_value()
      ? static_cast<int64_t>(to_onednn_type(ptr_bias.value().scalar_type()))
      : static_cast<int64_t>(dnnl::memory::data_type::undef);
  cache_key.requested_dst_dtype = static_cast<int64_t>(dst_dt);
  cache_key.total_m = total_M;
  cache_key.n = N;
  cache_key.k = K;
  cache_key.num_experts = num_experts;
  cache_key.group_num = group_num;
  cache_key.group_size = group_size;
  cache_key.has_bias = ptr_bias.has_value() ? 1 : 0;

  auto& primitive_cache = get_grouped_gemm_primitive_cache(device_id);
  auto iter = primitive_cache.find(cache_key);
  if (iter == primitive_cache.end()) {
    dnnl::matmul::primitive_desc pd;
    try {
      if (ptr_bias.has_value()) {
        pd = dnnl::matmul::primitive_desc(engine, src_md, wei_md, bias_md, dst_md, attr);
      } else {
        pd = dnnl::matmul::primitive_desc(engine, src_md, wei_md, dst_md, attr);
      }
    } catch (const dnnl::error& e) {
      TORCH_WARN(
          "oneDNN grouped_gemm_w4a16_prepacked: matmul primitive_desc creation failed: ",
          e.what());
      auto ocl_icd_vendors = vllm::xpu::getEnv("OCL_ICD_VENDORS");
      if (ocl_icd_vendors.has_value()) {
        TORCH_CHECK(
            false,
            "oneDNN grouped_gemm_w4a16_prepacked: matmul primitive_desc creation failed on GPU while "
            "OCL_ICD_VENDORS is set to '",
            ocl_icd_vendors.value(),
            "'. CPU fallback is disabled. Detail: ",
            e.what());
      }
      throw;
    }

    dnnl::matmul prim;
    try {
      prim = dnnl::matmul(pd);
    } catch (const sycl::exception& e) {
      TORCH_WARN(
          "oneDNN grouped_gemm_w4a16_prepacked: SYCL exception during primitive construction: ",
          e.what());
      throw;
    } catch (const dnnl::error& e) {
      TORCH_WARN("oneDNN grouped_gemm_w4a16_prepacked: matmul primitive creation failed: ", e.what());
      throw;
    } catch (const std::exception& e) {
      TORCH_WARN(
          "oneDNN grouped_gemm_w4a16_prepacked: std::exception during primitive construction: ",
          e.what());
      if (message_contains(e.what(), "Named barriers not yet implemented")) {
        TORCH_CHECK(
            false,
            "oneDNN grouped_gemm_w4a16_prepacked: GPU primitive construction failed because named barriers "
            "are not implemented in this environment. CPU fallback is disabled. Detail: ",
            e.what());
      }
      throw;
    }

    iter = primitive_cache
               .insert(
                   {cache_key,
                    grouped_gemm_cached_primitive_t{std::move(pd), std::move(prim)}})
               .first;
  }

  // Grouped src/dst memories carry both (values, offsets) handles.
  auto src_mem = dnnl::sycl_interop::make_memory(
      src_md,
      engine,
      dnnl::sycl_interop::memory_kind::usm,
      std::vector<void*>{ptr_A.data_ptr(), expert_ends_onednn_i32.data_ptr()});
  auto dst_mem = dnnl::sycl_interop::make_memory(
      dst_md,
      engine,
      dnnl::sycl_interop::memory_kind::usm,
      std::vector<void*>{ptr_D.data_ptr(), expert_ends_onednn_i32.data_ptr()});

  // Dense weights/scales/bias — already prepacked, use directly.
  auto wei_mem = oneDNN::make_onednn_memory(wei_md, engine, ptr_B_s4.data_ptr());
  auto scales_mem =
      oneDNN::make_onednn_memory(scales_md, engine, scales.data_ptr());

  std::unordered_map<int, dnnl::memory> args;
  args.emplace(DNNL_ARG_SRC, std::move(src_mem));
  args.emplace(DNNL_ARG_WEIGHTS, std::move(wei_mem));
  args.emplace(DNNL_ARG_DST, std::move(dst_mem));
  args.emplace(DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, std::move(scales_mem));
  // Max group size hint for oneDNN grouped dispatch.
  // grouped_micro_gemm dereferences this on the host before kernel launch,
  // so it must be shared USM (host + device accessible).
  auto sycl_queue = dnnl::sycl_interop::get_queue(stream);
  int32_t* hint_usm = sycl::malloc_shared<int32_t>(1, sycl_queue);
  hint_usm[0] = max_group_size_val;
  auto hint_md = dnnl::memory::desc(
      {1}, dnnl::memory::data_type::s32, dnnl::memory::format_tag::a);
  auto hint_mem = dnnl::sycl_interop::make_memory(
      hint_md, engine, dnnl::sycl_interop::memory_kind::usm, hint_usm);
  args.emplace(DNNL_ARG_HINT_MAX_GROUP_SIZE, std::move(hint_mem));
  if (ptr_bias.has_value()) {
    const at::Tensor& bias = *ptr_bias;
    auto bias_mem = oneDNN::make_onednn_memory(bias_md, engine, bias.data_ptr());
    args.emplace(DNNL_ARG_BIAS, std::move(bias_mem));
  }

  try {
    (void)dnnl::sycl_interop::execute(iter->second.prim, stream, args);
    sycl::free(hint_usm, sycl_queue);
  } catch (const sycl::exception& e) {
    TORCH_WARN(
        "oneDNN grouped_gemm_w4a16_prepacked: SYCL exception during execute: ",
        e.what());
    throw;
  } catch (const dnnl::error& e) {
    TORCH_WARN(
        "oneDNN grouped_gemm_w4a16_prepacked: oneDNN exception during execute: ",
        e.what());
    throw;
  } catch (const std::exception& e) {
    TORCH_WARN(
        "oneDNN grouped_gemm_w4a16_prepacked: std::exception during execute: ",
        e.what());
    throw;
  }
  return ptr_D;
#endif
}

}  // namespace oneDNN
