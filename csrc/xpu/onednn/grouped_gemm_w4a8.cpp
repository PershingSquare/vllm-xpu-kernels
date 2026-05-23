#include "xpu/onednn/grouped_gemm_w4a8.h"

#include "utils.h"
#include "xpu/onednn/onednn_grouped_gemm_cache.h"
#include "xpu/onednn/onednn_runtime.h"

#include <cstdint>
#include <exception>
#include <limits>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <oneapi/dnnl/dnnl_debug.h>

#ifndef DNNL_ARG_HINT_MAX_GROUP_SIZE
#define DNNL_ARG_HINT_MAX_GROUP_SIZE 384
#endif

namespace oneDNN {

namespace {

static inline bool message_contains(std::string_view haystack, std::string_view needle) {
  return haystack.find(needle) != std::string_view::npos;
}

static inline dnnl::memory::data_type to_onednn_type(at::ScalarType t) {
  switch (t) {
    case at::ScalarType::Half:      return dnnl::memory::data_type::f16;
    case at::ScalarType::BFloat16:  return dnnl::memory::data_type::bf16;
    case at::ScalarType::Float:     return dnnl::memory::data_type::f32;
    case at::ScalarType::Int:       return dnnl::memory::data_type::s32;
    default: break;
  }
  TORCH_CHECK(false, "Unsupported dtype in oneDNN grouped GEMM: ", t);
  return dnnl::memory::data_type::undef;
}

static inline bool is_fp16_bf16_or_fp32(at::ScalarType t) {
  return t == at::ScalarType::Half || t == at::ScalarType::BFloat16 ||
         t == at::ScalarType::Float;
}

static inline bool is_fp16_or_bf16(at::ScalarType t) {
  return t == at::ScalarType::Half || t == at::ScalarType::BFloat16;
}

}  // namespace

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
    int64_t max_expert_size) {
#if !(                                           \
    defined(DNNL_EXPERIMENTAL_GROUPED_MEMORY) && \
    DNNL_EXPERIMENTAL_GROUPED_MEMORY) &&         \
    !(defined(DNNL_EXPERIMENTAL_GROUPED_GEMM) && \
      DNNL_EXPERIMENTAL_GROUPED_GEMM)
  (void)A_q; (void)A_scale; (void)A_zp; (void)B_packed_u4; (void)B_scales;
  (void)bias; (void)D; (void)expert_first_token_offset; (void)N; (void)K; (void)num_experts; (void)max_expert_size;
  TORCH_CHECK(
      false,
      "oneDNN grouped GEMM is not enabled in this build "
      "(DNNL_EXPERIMENTAL_GROUPED_MEMORY=0 / DNNL_EXPERIMENTAL_GROUPED_GEMM=0)");
#else
  CHECK_DEVICE(A_q); CHECK_DEVICE(A_scale); CHECK_DEVICE(A_zp);
  CHECK_DEVICE(B_packed_u4); CHECK_DEVICE(B_scales); CHECK_DEVICE(D);
  CHECK_DEVICE(expert_first_token_offset);

  CHECK_CONTIGUOUS(A_q); CHECK_CONTIGUOUS(A_scale); CHECK_CONTIGUOUS(A_zp);
  CHECK_CONTIGUOUS(B_packed_u4); CHECK_CONTIGUOUS(B_scales); CHECK_CONTIGUOUS(D);
  CHECK_CONTIGUOUS(expert_first_token_offset);

  TORCH_CHECK(A_q.dim() == 2, "A_q must be 2D [Total_M, K]");
  TORCH_CHECK(B_packed_u4.dim() == 3, "B_packed_u4 must be 3D [E, N, K/2]");
  TORCH_CHECK(B_scales.dim() == 3, "B_scales must be 3D [E, N, K/group_size]");
  TORCH_CHECK(D.dim() == 2, "D must be 2D [Total_M, N]");
  TORCH_CHECK(expert_first_token_offset.dim() == 1,
              "expert_first_token_offset must be 1D [E+1]");

  TORCH_CHECK(A_q.scalar_type() == at::ScalarType::Byte,
              "A_q must be uint8 for oneDNN grouped_gemm_w4a8");
  TORCH_CHECK(is_fp16_bf16_or_fp32(A_scale.scalar_type()),
              "A_scale must be fp16, bf16, or fp32");
  TORCH_CHECK(A_scale.is_contiguous(),
              "A_scale must be contiguous to skip per-call reshape");
  TORCH_CHECK(A_zp.scalar_type() == at::ScalarType::Byte,
              "A_zp must be uint8; convert at the call site to skip per-call dtype conversion");
  TORCH_CHECK(A_zp.is_contiguous(),
              "A_zp must be contiguous to skip per-call reshape");
  TORCH_CHECK(B_packed_u4.scalar_type() == at::ScalarType::Byte,
              "B_packed_u4 must be uint8 (pre-converted packed s4)");
  TORCH_CHECK(is_fp16_bf16_or_fp32(B_scales.scalar_type()),
              "B_scales must be fp16, bf16, or fp32");
  TORCH_CHECK(is_fp16_or_bf16(D.scalar_type()),
              "D must be fp16 or bf16");
  TORCH_CHECK(expert_first_token_offset.scalar_type() == at::ScalarType::Long ||
              expert_first_token_offset.scalar_type() == at::ScalarType::Int,
              "expert_first_token_offset must be int64 or int32");

  const int64_t total_M = A_q.size(0);
  TORCH_CHECK(A_q.size(1) == K, "A_q.size(1) must match K");
  TORCH_CHECK(D.size(0) == total_M, "D.size(0) must match total_M");
  TORCH_CHECK(D.size(1) == N, "D.size(1) must match N");
  TORCH_CHECK(B_packed_u4.size(0) == num_experts, "B_packed_u4.size(0) must match num_experts");
  TORCH_CHECK(B_packed_u4.size(1) == N, "B_packed_u4.size(1) must match N");
  TORCH_CHECK(B_packed_u4.size(2) * 2 == K, "B_packed_u4.size(2) must be K/2");
  TORCH_CHECK(B_scales.size(0) == num_experts, "B_scales.size(0) must match num_experts");
  TORCH_CHECK(B_scales.size(1) > 0, "B_scales group_num must be > 0");
  TORCH_CHECK(B_scales.size(2) == N, "B_scales.size(2) must match N (pre-permuted [E,G,N])");
  TORCH_CHECK(K % B_scales.size(1) == 0, "group_num must divide K");
  TORCH_CHECK(expert_first_token_offset.numel() == (num_experts + 1),
              "expert_first_token_offset must have length E+1");

  if (bias.has_value()) {
    const at::Tensor& b = *bias;
    CHECK_DEVICE(b); CHECK_CONTIGUOUS(b);
    TORCH_CHECK(b.dim() == 2, "bias must be 2D [E, N]");
    TORCH_CHECK(b.size(0) == num_experts, "bias.size(0) must match E");
    TORCH_CHECK(b.size(1) == N, "bias.size(1) must match N");
    TORCH_CHECK(is_fp16_bf16_or_fp32(b.scalar_type()), "bias must be fp16, bf16, or fp32");
  }

  TORCH_CHECK((K % 2) == 0, "K must be even for int4 weights");
  TORCH_CHECK((N % 2) == 0, "N must be even for int4 weights");
  TORCH_CHECK(total_M <= std::numeric_limits<int>::max(), "total_M exceeds int32 range");

  const bool per_row_A_scale =
      (A_scale.dim() == 1 && A_scale.size(0) == total_M) ||
      (A_scale.dim() == 2 && A_scale.size(0) == total_M && A_scale.size(1) == 1);
  const bool per_row_A_zp =
      (A_zp.dim() == 1 && A_zp.size(0) == total_M) ||
      (A_zp.dim() == 2 && A_zp.size(0) == total_M && A_zp.size(1) == 1);
  TORCH_CHECK(per_row_A_scale, "A_scale must be [total_M] or [total_M,1]; got ", A_scale.sizes());
  TORCH_CHECK(per_row_A_zp, "A_zp must be [total_M] or [total_M,1]; got ", A_zp.sizes());

  const int64_t group_num = B_scales.size(1);
  const int64_t group_size = K / group_num;

  auto expert_first_token_offset_i32 =
      (expert_first_token_offset.scalar_type() == at::ScalarType::Int)
          ? expert_first_token_offset
          : expert_first_token_offset.to(at::ScalarType::Int);

  const int32_t max_expert_size_val = static_cast<int32_t>(max_expert_size);

  const auto src_dt = dnnl::memory::data_type::u8;
  const auto requested_dst_dt = to_onednn_type(D.scalar_type());
  const auto src_scales_dt = to_onednn_type(A_scale.scalar_type());
  const auto wei_scales_dt = to_onednn_type(B_scales.scalar_type());
  const auto src_zp_dt = dnnl::memory::data_type::u8;

  const at::Device cur_device = A_q.device();
  const int device_id = cur_device.index();
  auto& engine = oneDNN::GpuEngineManager::Instance().get_engine(cur_device);
  auto& stream = oneDNN::GpuStreamManager::Instance().get_stream(device_id);

  grouped_gemm_primitive_key_t cache_key{};
  cache_key.device_id = device_id;
  cache_key.src_dtype = static_cast<int64_t>(src_dt);
  cache_key.wei_dtype = static_cast<int64_t>(dnnl::memory::data_type::s4);
  cache_key.dst_dtype = static_cast<int64_t>(requested_dst_dt);
  cache_key.src_scales_dtype = static_cast<int64_t>(src_scales_dt);
  cache_key.wei_scales_dtype = static_cast<int64_t>(wei_scales_dt);
  cache_key.bias_dtype = bias.has_value()
      ? static_cast<int64_t>(to_onednn_type(bias.value().scalar_type()))
      : static_cast<int64_t>(dnnl::memory::data_type::undef);
  cache_key.requested_dst_dtype = static_cast<int64_t>(requested_dst_dt);
  cache_key.total_m = total_M;
  cache_key.n = N;
  cache_key.k = K;
  cache_key.num_experts = num_experts;
  cache_key.group_num = group_num;
  cache_key.group_size = group_size;
  cache_key.has_bias = bias.has_value() ? 1 : 0;
  cache_key.max_expert_size = max_expert_size;

  auto& primitive_cache = get_grouped_gemm_primitive_cache(device_id);
  auto iter = primitive_cache.find(cache_key);

  if (iter == primitive_cache.end()) {
    const dnnl::memory::dim ngroups = static_cast<dnnl::memory::dim>(num_experts);
    auto src_md = dnnl::memory::desc::grouped(
        {total_M, K}, src_dt, 0, ngroups, dnnl::memory::data_type::s32);
    auto dst_md = dnnl::memory::desc::grouped(
        {total_M, N}, requested_dst_dt, 0, ngroups, dnnl::memory::data_type::s32);
    auto wei_md = dnnl::memory::desc({num_experts, K, N},
        dnnl::memory::data_type::s4, dnnl::memory::format_tag::acb);
    auto wei_scales_md = dnnl::memory::desc(
        {num_experts, static_cast<dnnl::memory::dim>(group_num),
         static_cast<dnnl::memory::dim>(N)},
        wei_scales_dt, dnnl::memory::format_tag::abc);
    auto src_scales_md = dnnl::memory::desc(
        {total_M}, src_scales_dt, dnnl::memory::format_tag::a);
    auto src_zp_md = dnnl::memory::desc(
        {total_M}, src_zp_dt, dnnl::memory::format_tag::a);
    dnnl::memory::desc bias_md;
    if (bias.has_value()) {
      bias_md = dnnl::memory::desc(
          {num_experts, N}, to_onednn_type(bias->scalar_type()), {N, 1});
    }

    dnnl::primitive_attr attr;
    attr.set_scales(DNNL_ARG_SRC, (1 << 0), {}, src_scales_dt);
    attr.set_zero_points(DNNL_ARG_SRC, (1 << 0), {}, src_zp_dt);
    attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 0) | (1 << 1) | (1 << 2),
        {group_size, 1}, wei_scales_dt);

    dnnl::matmul::primitive_desc pd;
    try {
      if (bias.has_value()) {
        pd = dnnl::matmul::primitive_desc(engine, src_md, wei_md, bias_md, dst_md, attr);
      } else {
        pd = dnnl::matmul::primitive_desc(engine, src_md, wei_md, dst_md, attr);
      }
    } catch (const dnnl::error& e) {
      TORCH_CHECK(false,
                  "oneDNN grouped_gemm_w4a8: primitive_desc creation failed: ", e.what());
    }
    dnnl::matmul prim(pd);
    grouped_gemm_cached_primitive_t entry{std::move(pd), std::move(prim)};
    entry.src_md = std::move(src_md);
    entry.dst_md = std::move(dst_md);
    entry.wei_md = std::move(wei_md);
    entry.wei_scales_md = std::move(wei_scales_md);
    entry.src_scales_md = std::move(src_scales_md);
    entry.src_zp_md = std::move(src_zp_md);
    entry.bias_md = std::move(bias_md);
    iter = primitive_cache.insert({cache_key, std::move(entry)}).first;
  }

  // oneDNN grouped-memory expects expert-end offsets [num_experts].
  // expert_first_token_offset_i32 has shape [num_experts+1] with
  // [0, end_e0, ..., total_M], so a view at offset 1 is the ends array.
  torch::Tensor expert_ends_i32 =
      expert_first_token_offset_i32.narrow(0, 1, num_experts);

  auto& cached = iter->second;

  // Allocate hint USM lazily.
  if (cached.hint_usm == nullptr) {
    auto sycl_queue = dnnl::sycl_interop::get_queue(stream);
    cached.hint_usm = sycl::malloc_shared<int32_t>(1, sycl_queue);
  }
  cached.hint_usm[0] = max_expert_size_val;

  // Build the dnnl::memory objects + args map ONCE per cache entry. After
  // the first call, the data pointers are updated cheaply via
  // set_data_handle() instead of constructing fresh memories every call.
  if (!cached.memories_built) {
    auto hint_md = dnnl::memory::desc(
        {1}, dnnl::memory::data_type::s32, dnnl::memory::format_tag::a);
    cached.src_mem = dnnl::sycl_interop::make_memory(
        cached.src_md, engine, dnnl::sycl_interop::memory_kind::usm,
        std::vector<void*>{A_q.data_ptr(), expert_ends_i32.data_ptr()});
    cached.dst_mem = dnnl::sycl_interop::make_memory(
        cached.dst_md, engine, dnnl::sycl_interop::memory_kind::usm,
        std::vector<void*>{D.data_ptr(), expert_ends_i32.data_ptr()});
    cached.wei_mem = oneDNN::make_onednn_memory(
        cached.wei_md, engine, B_packed_u4.data_ptr());
    cached.wei_scales_mem = oneDNN::make_onednn_memory(
        cached.wei_scales_md, engine, B_scales.data_ptr());
    cached.src_scales_mem = oneDNN::make_onednn_memory(
        cached.src_scales_md, engine, A_scale.data_ptr());
    cached.src_zp_mem = oneDNN::make_onednn_memory(
        cached.src_zp_md, engine, A_zp.data_ptr());
    cached.hint_mem = dnnl::sycl_interop::make_memory(
        hint_md, engine, dnnl::sycl_interop::memory_kind::usm,
        cached.hint_usm);
    cached.args.emplace(DNNL_ARG_SRC, cached.src_mem);
    cached.args.emplace(DNNL_ARG_WEIGHTS, cached.wei_mem);
    cached.args.emplace(DNNL_ARG_DST, cached.dst_mem);
    cached.args.emplace(DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, cached.src_scales_mem);
    cached.args.emplace(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC, cached.src_zp_mem);
    cached.args.emplace(DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, cached.wei_scales_mem);
    cached.args.emplace(DNNL_ARG_HINT_MAX_GROUP_SIZE, cached.hint_mem);
    if (bias.has_value()) {
      cached.bias_mem = oneDNN::make_onednn_memory(
          cached.bias_md, engine, bias->data_ptr());
      cached.args.emplace(DNNL_ARG_BIAS, cached.bias_mem);
    }
    cached.memories_built = true;
  } else {
    cached.src_mem.set_data_handle(A_q.data_ptr(), 0);
    cached.src_mem.set_data_handle(expert_ends_i32.data_ptr(), 1);
    cached.dst_mem.set_data_handle(D.data_ptr(), 0);
    cached.dst_mem.set_data_handle(expert_ends_i32.data_ptr(), 1);
    cached.src_scales_mem.set_data_handle(A_scale.data_ptr());
    cached.src_zp_mem.set_data_handle(A_zp.data_ptr());
    cached.wei_mem.set_data_handle(B_packed_u4.data_ptr());
    cached.wei_scales_mem.set_data_handle(B_scales.data_ptr());
    if (bias.has_value()) {
      cached.bias_mem.set_data_handle(bias->data_ptr());
    }
  }

  try {
    (void)dnnl::sycl_interop::execute(cached.prim, stream, cached.args);
  } catch (const dnnl::error& e) {
    TORCH_CHECK(false, "oneDNN grouped_gemm_w4a8: execute failed: ", e.what());
  }

  return D;
#endif
}

}  // namespace oneDNN
