#include "xpu/onednn/grouped_gemm_w4a16.h"

#include "utils.h"
#include "xpu/onednn/onednn_grouped_gemm_cache.h"
#include "xpu/onednn/onednn_runtime.h"

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

static inline bool message_contains(std::string_view haystack, std::string_view needle) {
  return haystack.find(needle) != std::string_view::npos;
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
    bool is_B_mxfp4,
    int64_t max_expert_size) {
#if !(defined(DNNL_EXPERIMENTAL_GROUPED_MEMORY) && DNNL_EXPERIMENTAL_GROUPED_MEMORY) && \
    !(defined(DNNL_EXPERIMENTAL_GROUPED_GEMM) && DNNL_EXPERIMENTAL_GROUPED_GEMM)
  (void)ptr_A; (void)ptr_B; (void)ptr_scales; (void)ptr_bias; (void)ptr_D;
  (void)expert_first_token_offset; (void)N; (void)K; (void)num_experts;
  (void)is_B_int4; (void)is_B_mxfp4; (void)max_expert_size;
  TORCH_CHECK(
      false,
      "oneDNN grouped GEMM is not enabled in this build "
      "(DNNL_EXPERIMENTAL_GROUPED_MEMORY=0 / DNNL_EXPERIMENTAL_GROUPED_GEMM=0)");
#else
  TORCH_CHECK(is_B_int4, "oneDNN grouped GEMM path only supports is_B_int4");
  TORCH_CHECK(!is_B_mxfp4, "oneDNN grouped GEMM path does not support is_B_mxfp4");

  CHECK_DEVICE(ptr_A); CHECK_DEVICE(ptr_B); CHECK_DEVICE(ptr_D);
  CHECK_DEVICE(expert_first_token_offset);
  CHECK_CONTIGUOUS(ptr_A); CHECK_CONTIGUOUS(ptr_B); CHECK_CONTIGUOUS(ptr_D);
  CHECK_CONTIGUOUS(expert_first_token_offset);

  TORCH_CHECK(ptr_A.dim() == 2, "ptr_A must be 2D [Total_M, K]");
  TORCH_CHECK(ptr_B.dim() == 3, "ptr_B must be 3D [E, N, K/2]");
  TORCH_CHECK(ptr_D.dim() == 2, "ptr_D must be 2D [Total_M, N]");
  TORCH_CHECK(expert_first_token_offset.dim() == 1,
              "expert_first_token_offset must be 1D [E+1]");

  auto A_dtype = ptr_A.scalar_type();
  TORCH_CHECK(is_fp16_or_bf16(A_dtype), "ptr_A must be fp16 or bf16");
  TORCH_CHECK(ptr_D.scalar_type() == A_dtype, "ptr_D dtype must match ptr_A");
  TORCH_CHECK(ptr_B.scalar_type() == at::ScalarType::Byte,
              "ptr_B must be uint8 (pre-converted packed s4)");
  TORCH_CHECK(expert_first_token_offset.scalar_type() == at::ScalarType::Long ||
              expert_first_token_offset.scalar_type() == at::ScalarType::Int,
              "expert_first_token_offset must be int64 or int32");

  const int64_t total_M = ptr_A.size(0);
  TORCH_CHECK(total_M <= std::numeric_limits<int>::max(), "total_M exceeds int32 range");

  TORCH_CHECK(ptr_A.size(1) == K, "ptr_A.size(1) must match K");
  TORCH_CHECK(ptr_B.size(0) == num_experts, "ptr_B.size(0) must match num_experts");
  TORCH_CHECK(ptr_B.size(1) == N, "ptr_B.size(1) must match N");
  TORCH_CHECK(ptr_B.size(2) * 2 == K, "ptr_B.size(2) must be K/2");
  TORCH_CHECK(ptr_D.size(0) == total_M, "ptr_D.size(0) must match total_M");
  TORCH_CHECK(ptr_D.size(1) == N, "ptr_D.size(1) must match N");
  TORCH_CHECK(expert_first_token_offset.numel() == (num_experts + 1),
              "expert_first_token_offset must have length E+1");

  const bool has_scales = ptr_scales.has_value() && ptr_scales->defined();
  TORCH_CHECK(has_scales, "w4a16 grouped GEMM must have scales");
  const at::Tensor& scales = *ptr_scales;
  CHECK_DEVICE(scales); CHECK_CONTIGUOUS(scales);
  TORCH_CHECK(scales.dim() == 3, "ptr_scales must be 3D [E, G, N] (pre-permuted)");
  TORCH_CHECK(scales.size(0) == num_experts, "scales.size(0) must match num_experts");
  TORCH_CHECK(scales.size(2) == N, "scales.size(2) must match N");
  TORCH_CHECK(scales.size(1) > 0, "scales group_num must be > 0");
  TORCH_CHECK(K % scales.size(1) == 0, "group_num must divide K");
  const int64_t group_num = scales.size(1);
  const int64_t group_size = K / group_num;

  TORCH_CHECK((K % 2) == 0, "K must be even for int4 weights");
  TORCH_CHECK((N % 2) == 0, "N must be even for int4 weights");

  const bool has_bias = ptr_bias.has_value() && ptr_bias->defined();

  if (has_bias) {
    const at::Tensor& bias = *ptr_bias;
    CHECK_DEVICE(bias); CHECK_CONTIGUOUS(bias);
    TORCH_CHECK(bias.dim() == 2, "ptr_bias must be 2D [E, N]");
    TORCH_CHECK(bias.size(0) == num_experts, "bias.size(0) must match E");
    TORCH_CHECK(bias.size(1) == N, "bias.size(1) must match N");
  }

  auto expert_first_token_offset_i32 =
      (expert_first_token_offset.scalar_type() == at::ScalarType::Int)
      ? expert_first_token_offset
      : expert_first_token_offset.to(at::ScalarType::Int);

  const int32_t max_expert_size_val = static_cast<int32_t>(max_expert_size);

  const auto src_dt = to_onednn_type(A_dtype);
  const auto dst_dt = src_dt;
  const auto scales_dt = to_onednn_type(scales.scalar_type());

  const dnnl::memory::dim ngroups = static_cast<dnnl::memory::dim>(num_experts);
  auto src_md = dnnl::memory::desc::grouped(
      {total_M, K}, src_dt, 0, ngroups, dnnl::memory::data_type::s32);
  auto dst_md = dnnl::memory::desc::grouped(
      {total_M, N}, dst_dt, 0, ngroups, dnnl::memory::data_type::s32);
  auto wei_md = dnnl::memory::desc(
      {num_experts, K, N}, dnnl::memory::data_type::s4, dnnl::memory::format_tag::acb);
  auto scales_md = dnnl::memory::desc(
      {num_experts, group_num, N}, scales_dt, dnnl::memory::format_tag::abc);

  dnnl::memory::desc bias_md;
  if (has_bias) {
    const at::Tensor& bias = *ptr_bias;
    bias_md = dnnl::memory::desc(
        {num_experts, N}, to_onednn_type(bias.scalar_type()), {N, 1});
  }

  dnnl::primitive_attr attr;
  attr.set_scales(DNNL_ARG_WEIGHTS,
                  (1 << 0) | (1 << 1) | (1 << 2),
                  {group_size, 1},
                  scales_dt);
  const at::Device cur_device = ptr_A.device();
  const int device_id = cur_device.index();
  auto& engine = oneDNN::GpuEngineManager::Instance().get_engine(cur_device);
  auto& stream = oneDNN::GpuStreamManager::Instance().get_stream(device_id);

  grouped_gemm_primitive_key_t cache_key{};
  cache_key.device_id = device_id;
  cache_key.src_dtype = static_cast<int64_t>(src_dt);
  cache_key.wei_dtype = static_cast<int64_t>(dnnl::memory::data_type::s4);
  cache_key.dst_dtype = static_cast<int64_t>(dst_dt);
  cache_key.src_scales_dtype = static_cast<int64_t>(dnnl::memory::data_type::undef);
  cache_key.wei_scales_dtype = static_cast<int64_t>(scales_dt);
  cache_key.bias_dtype = has_bias
      ? static_cast<int64_t>(to_onednn_type(ptr_bias->scalar_type()))
      : static_cast<int64_t>(dnnl::memory::data_type::undef);
  cache_key.requested_dst_dtype = static_cast<int64_t>(dst_dt);
  cache_key.total_m = total_M;
  cache_key.n = N;
  cache_key.k = K;
  cache_key.num_experts = num_experts;
  cache_key.group_num = group_num;
  cache_key.group_size = group_size;
  cache_key.has_bias = has_bias ? 1 : 0;
  cache_key.max_expert_size = max_expert_size;
  cache_key.token_centric_prefill_tune =
      vllm::xpu::env_flag_enabled(
          "VLLM_XPU_ONEDNN_TOKEN_CENTRIC_PREFILL_TUNE")
      ? 1
      : 0;

  auto& primitive_cache = get_grouped_gemm_primitive_cache(device_id);
  auto iter = primitive_cache.find(cache_key);
  if (iter == primitive_cache.end()) {
    dnnl::matmul::primitive_desc pd;
    try {
      if (has_bias) {
        pd = dnnl::matmul::primitive_desc(engine, src_md, wei_md, bias_md, dst_md, attr);
      } else {
        pd = dnnl::matmul::primitive_desc(engine, src_md, wei_md, dst_md, attr);
      }
    } catch (const dnnl::error& e) {
      TORCH_WARN("oneDNN grouped_gemm_w4a16: primitive_desc creation failed: ", e.what());
      auto ocl_icd_vendors = vllm::xpu::getEnv("OCL_ICD_VENDORS");
      if (ocl_icd_vendors.has_value()) {
        TORCH_CHECK(false,
                    "oneDNN grouped_gemm_w4a16: failed on GPU while OCL_ICD_VENDORS='",
                    ocl_icd_vendors.value(), "'. Detail: ", e.what());
      }
      throw;
    }

    dnnl::matmul prim;
    try {
      prim = dnnl::matmul(pd);
    } catch (const std::exception& e) {
      TORCH_WARN("oneDNN grouped_gemm_w4a16: primitive construction failed: ", e.what());
      if (message_contains(e.what(), "Named barriers not yet implemented")) {
        TORCH_CHECK(false,
                    "oneDNN grouped_gemm_w4a16: named barriers not implemented. Detail: ",
                    e.what());
      }
      throw;
    }

    grouped_gemm_cached_primitive_t entry{std::move(pd), std::move(prim)};
    entry.src_md = std::move(src_md);
    entry.dst_md = std::move(dst_md);
    entry.wei_md = std::move(wei_md);
    entry.wei_scales_md = std::move(scales_md);
    entry.bias_md = std::move(bias_md);
    iter = primitive_cache.insert({cache_key, std::move(entry)}).first;
  }

  // oneDNN grouped-memory expects expert-end offsets [num_experts].
  torch::Tensor expert_ends_i32 =
      expert_first_token_offset_i32.narrow(0, 1, num_experts);

  auto& cached = iter->second;

  if (cached.hint_usm == nullptr) {
    auto sycl_queue = dnnl::sycl_interop::get_queue(stream);
    cached.hint_usm = sycl::malloc_shared<int32_t>(1, sycl_queue);
  }
  cached.hint_usm[0] = max_expert_size_val;

  if (!cached.memories_built) {
    auto hint_md = dnnl::memory::desc(
        {1}, dnnl::memory::data_type::s32, dnnl::memory::format_tag::a);
    cached.src_mem = dnnl::sycl_interop::make_memory(
        cached.src_md, engine, dnnl::sycl_interop::memory_kind::usm,
        std::vector<void*>{ptr_A.data_ptr(), expert_ends_i32.data_ptr()});
    cached.dst_mem = dnnl::sycl_interop::make_memory(
        cached.dst_md, engine, dnnl::sycl_interop::memory_kind::usm,
        std::vector<void*>{ptr_D.data_ptr(), expert_ends_i32.data_ptr()});
    cached.wei_mem = oneDNN::make_onednn_memory(
        cached.wei_md, engine, ptr_B.data_ptr());
    cached.wei_scales_mem = oneDNN::make_onednn_memory(
        cached.wei_scales_md, engine, scales.data_ptr());
    cached.hint_mem = dnnl::sycl_interop::make_memory(
        hint_md, engine, dnnl::sycl_interop::memory_kind::usm, cached.hint_usm);

    cached.args.emplace(DNNL_ARG_SRC, cached.src_mem);
    cached.args.emplace(DNNL_ARG_WEIGHTS, cached.wei_mem);
    cached.args.emplace(DNNL_ARG_DST, cached.dst_mem);
    cached.args.emplace(DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS,
        cached.wei_scales_mem);
    cached.args.emplace(DNNL_ARG_HINT_MAX_GROUP_SIZE, cached.hint_mem);
    if (has_bias) {
      const at::Tensor& bias = *ptr_bias;
      cached.bias_mem = oneDNN::make_onednn_memory(
          cached.bias_md, engine, bias.data_ptr());
      cached.args.emplace(DNNL_ARG_BIAS, cached.bias_mem);
    }
    cached.memories_built = true;
    cached.exec_handle = dnnl::sycl_interop::execute_handle(
        cached.prim, stream, cached.args);
  } else {
    cached.src_mem.set_data_handle(ptr_A.data_ptr(), 0);
    cached.src_mem.set_data_handle(expert_ends_i32.data_ptr(), 1);
    cached.dst_mem.set_data_handle(ptr_D.data_ptr(), 0);
    cached.dst_mem.set_data_handle(expert_ends_i32.data_ptr(), 1);
    cached.wei_mem.set_data_handle(ptr_B.data_ptr());
    cached.wei_scales_mem.set_data_handle(scales.data_ptr());
    if (has_bias) {
      cached.bias_mem.set_data_handle(ptr_bias->data_ptr());
    }
  }

  if (cached.use_fast_path) {
    try {
      (void)dnnl::sycl_interop::execute_fast(cached.exec_handle);
    } catch (const dnnl::error& fast_err) {
      if (fast_err.status == dnnl_unimplemented) {
        cached.use_fast_path = false;
      } else {
        TORCH_CHECK(false, "oneDNN grouped_gemm_w4a16: execute_fast failed: ",
                    fast_err.what());
      }
    } catch (...) {
      throw;
    }
  }
  if (!cached.use_fast_path) {
    try {
      cached.prim.execute(stream, cached.args);
    } catch (const dnnl::error& e) {
      TORCH_CHECK(false, "oneDNN grouped_gemm_w4a16: execute slow failed: ", e.what());
    } catch (...) {
      throw;
    }
  }

  return ptr_D;
#endif
}

}  // namespace oneDNN
