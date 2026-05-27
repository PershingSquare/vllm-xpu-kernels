#include "xpu/onednn/grouped_gemm_w4a8.h"

#include "utils.h"
#include "xpu/onednn/onednn_grouped_gemm_cache.h"
#include "xpu/onednn/onednn_runtime.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <oneapi/dnnl/dnnl_debug.h>

#ifndef DNNL_ARG_HINT_MAX_GROUP_SIZE
#define DNNL_ARG_HINT_MAX_GROUP_SIZE 384
#endif

namespace oneDNN {

namespace {

using W4A8ProfileClock = std::chrono::steady_clock;

enum W4A8ProfileStage : int {
  kW4A8ProfileTotal = 0,
  kW4A8ProfileOffsetCast,
  kW4A8ProfileCacheLookup,
  kW4A8ProfileCacheBuild,
  kW4A8ProfileOffsetView,
  kW4A8ProfileHint,
  kW4A8ProfileMemoryBuild,
  kW4A8ProfileSetDataHandle,
  kW4A8ProfileExecuteFast,
  kW4A8ProfileExecuteSlow,
  kW4A8ProfileStageCount,
};

struct W4A8ProfileStageStats {
  std::atomic<uint64_t> calls{0};
  std::atomic<uint64_t> ns{0};
  std::atomic<uint64_t> max_ns{0};
};

struct W4A8ProfileStats {
  W4A8ProfileStageStats stage[kW4A8ProfileStageCount];
  std::atomic<uint64_t> total_calls{0};
  std::atomic<uint64_t> cache_misses{0};
  std::atomic<uint64_t> offset_i32_copies{0};
  std::atomic<uint64_t> memory_builds{0};
  std::atomic<uint64_t> fast_calls{0};
  std::atomic<uint64_t> fast_unimplemented{0};
  std::atomic<uint64_t> slow_calls{0};
};

static inline W4A8ProfileStats& w4a8_profile_stats() {
  static W4A8ProfileStats stats;
  return stats;
}

static inline bool w4a8_profile_enabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("VLLM_XPU_ONEDNN_W4A8_PROFILE");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
  }();
  return enabled;
}

static inline uint64_t w4a8_profile_dump_every() {
  static const uint64_t every = [] {
    const char* value = std::getenv("VLLM_XPU_ONEDNN_W4A8_PROFILE_EVERY");
    if (value == nullptr || value[0] == '\0') {
      return static_cast<uint64_t>(100);
    }
    char* end = nullptr;
    const auto parsed = std::strtoull(value, &end, 10);
    if (end == value || parsed == 0) {
      return static_cast<uint64_t>(100);
    }
    return static_cast<uint64_t>(parsed);
  }();
  return every;
}

static inline W4A8ProfileClock::time_point w4a8_profile_now() {
  return W4A8ProfileClock::now();
}

static inline uint64_t w4a8_profile_elapsed_ns(
    W4A8ProfileClock::time_point start) {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          W4A8ProfileClock::now() - start)
          .count());
}

static inline const char* w4a8_profile_stage_name(int stage) {
  switch (stage) {
    case kW4A8ProfileTotal: return "total";
    case kW4A8ProfileOffsetCast: return "offset_cast";
    case kW4A8ProfileCacheLookup: return "cache_lookup";
    case kW4A8ProfileCacheBuild: return "cache_build";
    case kW4A8ProfileOffsetView: return "offset_view";
    case kW4A8ProfileHint: return "hint";
    case kW4A8ProfileMemoryBuild: return "memory_build";
    case kW4A8ProfileSetDataHandle: return "set_data_handle";
    case kW4A8ProfileExecuteFast: return "execute_fast";
    case kW4A8ProfileExecuteSlow: return "execute_slow";
    default: return "unknown";
  }
}

static inline void w4a8_profile_record_stage(
    W4A8ProfileStage stage,
    uint64_t ns) {
  auto& stat = w4a8_profile_stats().stage[stage];
  stat.calls.fetch_add(1, std::memory_order_relaxed);
  stat.ns.fetch_add(ns, std::memory_order_relaxed);
  auto max_ns = stat.max_ns.load(std::memory_order_relaxed);
  while (ns > max_ns &&
         !stat.max_ns.compare_exchange_weak(
             max_ns, ns, std::memory_order_relaxed)) {
  }
}

static inline void w4a8_profile_dump(uint64_t calls) {
  static std::mutex dump_mutex;
  std::lock_guard<std::mutex> guard(dump_mutex);
  auto& stats = w4a8_profile_stats();
  std::cerr << "[onednn_w4a8_profile] calls=" << calls
            << " cache_misses=" << stats.cache_misses.load(std::memory_order_relaxed)
            << " offset_i32_copies=" << stats.offset_i32_copies.load(std::memory_order_relaxed)
            << " memory_builds=" << stats.memory_builds.load(std::memory_order_relaxed)
            << " fast_calls=" << stats.fast_calls.load(std::memory_order_relaxed)
            << " fast_unimplemented=" << stats.fast_unimplemented.load(std::memory_order_relaxed)
            << " slow_calls=" << stats.slow_calls.load(std::memory_order_relaxed)
            << '\n';
  for (int i = 0; i < kW4A8ProfileStageCount; ++i) {
    const auto& stage = stats.stage[i];
    const uint64_t stage_calls = stage.calls.load(std::memory_order_relaxed);
    const uint64_t total_ns = stage.ns.load(std::memory_order_relaxed);
    const uint64_t max_ns = stage.max_ns.load(std::memory_order_relaxed);
    const double avg_us = stage_calls == 0
        ? 0.0
        : static_cast<double>(total_ns) / static_cast<double>(stage_calls) / 1000.0;
    std::cerr << "[onednn_w4a8_profile] stage=" << w4a8_profile_stage_name(i)
              << " calls=" << stage_calls
              << " avg_us=" << avg_us
              << " max_us=" << static_cast<double>(max_ns) / 1000.0
              << '\n';
  }
}

static inline void w4a8_profile_maybe_dump(uint64_t calls) {
  const uint64_t every = w4a8_profile_dump_every();
  static std::atomic<uint64_t> last_dump{0};
  auto previous = last_dump.load(std::memory_order_relaxed);
  while (calls > previous && calls - previous >= every) {
    if (last_dump.compare_exchange_weak(
            previous, calls, std::memory_order_relaxed)) {
      w4a8_profile_dump(calls);
      return;
    }
  }
}


static inline bool w4a8_profile_should_sample(uint64_t calls) {
  const uint64_t every = w4a8_profile_dump_every();
  return calls == 1 || (every != 0 && calls % every == 0);
}

static inline uint64_t pointer_mod(const void* ptr, uint64_t modulus) {
  if (ptr == nullptr) {
    return 0;
  }
  return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ptr) % modulus);
}

static inline void append_pointer_alignment(
    std::ostream& os,
    const char* name,
    const void* ptr) {
  os << ' ' << name << "_mod64=" << pointer_mod(ptr, 64)
     << ' ' << name << "_mod128=" << pointer_mod(ptr, 128)
     << ' ' << name << "_mod4096=" << pointer_mod(ptr, 4096);
}

static inline int64_t scalar_type_id(at::ScalarType type) {
  return static_cast<int64_t>(type);
}

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
  const bool profile = w4a8_profile_enabled();
  const auto total_profile_start = profile
      ? w4a8_profile_now()
      : W4A8ProfileClock::time_point{};

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

  const bool has_bias = bias.has_value() && bias->defined();

  if (has_bias) {
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

  const bool offset_needs_i32_copy =
      expert_first_token_offset.scalar_type() != at::ScalarType::Int;
  const auto offset_cast_profile_start = profile
      ? w4a8_profile_now()
      : W4A8ProfileClock::time_point{};
  auto expert_first_token_offset_i32 = offset_needs_i32_copy
      ? expert_first_token_offset.to(at::ScalarType::Int)
      : expert_first_token_offset;
  if (profile) {
    w4a8_profile_record_stage(
        kW4A8ProfileOffsetCast,
        w4a8_profile_elapsed_ns(offset_cast_profile_start));
    if (offset_needs_i32_copy) {
      w4a8_profile_stats().offset_i32_copies.fetch_add(
          1, std::memory_order_relaxed);
    }
  }

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
  cache_key.bias_dtype = has_bias
      ? static_cast<int64_t>(to_onednn_type(bias->scalar_type()))
      : static_cast<int64_t>(dnnl::memory::data_type::undef);
  cache_key.requested_dst_dtype = static_cast<int64_t>(requested_dst_dt);
  cache_key.total_m = total_M;
  cache_key.n = N;
  cache_key.k = K;
  cache_key.num_experts = num_experts;
  cache_key.group_num = group_num;
  cache_key.group_size = group_size;
  cache_key.has_bias = has_bias ? 1 : 0;
  cache_key.max_expert_size = max_expert_size;

  const auto cache_lookup_profile_start = profile
      ? w4a8_profile_now()
      : W4A8ProfileClock::time_point{};
  auto& primitive_cache = get_grouped_gemm_primitive_cache(device_id);
  auto iter = primitive_cache.find(cache_key);
  const bool cache_hit = iter != primitive_cache.end();
  if (profile) {
    w4a8_profile_record_stage(
        kW4A8ProfileCacheLookup,
        w4a8_profile_elapsed_ns(cache_lookup_profile_start));
  }

  if (!cache_hit) {
    if (profile) {
      w4a8_profile_stats().cache_misses.fetch_add(
          1, std::memory_order_relaxed);
    }
    const auto cache_build_profile_start = profile
        ? w4a8_profile_now()
        : W4A8ProfileClock::time_point{};
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
    if (has_bias) {
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
      if (has_bias) {
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
    if (profile) {
      w4a8_profile_record_stage(
          kW4A8ProfileCacheBuild,
          w4a8_profile_elapsed_ns(cache_build_profile_start));
    }
  }

  // oneDNN grouped-memory expects expert-end offsets [num_experts].
  // expert_first_token_offset_i32 has shape [num_experts+1] with
  // [0, end_e0, ..., total_M], so a view at offset 1 is the ends array.
  const auto offset_view_profile_start = profile
      ? w4a8_profile_now()
      : W4A8ProfileClock::time_point{};
  torch::Tensor expert_ends_i32 =
      expert_first_token_offset_i32.narrow(0, 1, num_experts);
  if (profile) {
    w4a8_profile_record_stage(
        kW4A8ProfileOffsetView,
        w4a8_profile_elapsed_ns(offset_view_profile_start));
  }

  auto& cached = iter->second;
  const bool memories_built_before = cached.memories_built;

  const auto hint_profile_start = profile
      ? w4a8_profile_now()
      : W4A8ProfileClock::time_point{};
  // max_expert_size is part of the cache key, so the hint value is invariant
  // for this cached oneDNN primitive.
  if (cached.hint_usm == nullptr) {
    auto sycl_queue = dnnl::sycl_interop::get_queue(stream);
    cached.hint_usm = sycl::malloc_shared<int32_t>(1, sycl_queue);
    cached.hint_usm[0] = max_expert_size_val;
  }
  if (profile) {
    w4a8_profile_record_stage(
        kW4A8ProfileHint,
        w4a8_profile_elapsed_ns(hint_profile_start));
  }

  // Build the dnnl::memory objects + args map ONCE per cache entry. After
  // the first call, the data pointers are updated cheaply via
  // set_data_handle() instead of constructing fresh memories every call.
  if (!cached.memories_built) {
    if (profile) {
      w4a8_profile_stats().memory_builds.fetch_add(
          1, std::memory_order_relaxed);
    }
    const auto memory_build_profile_start = profile
        ? w4a8_profile_now()
        : W4A8ProfileClock::time_point{};
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
    if (has_bias) {
      cached.bias_mem = oneDNN::make_onednn_memory(
          cached.bias_md, engine, bias->data_ptr());
      cached.args.emplace(DNNL_ARG_BIAS, cached.bias_mem);
    }
    cached.memories_built = true;
    cached.exec_handle = dnnl::sycl_interop::execute_handle(
        cached.prim, stream, cached.args);
    if (profile) {
      w4a8_profile_record_stage(
          kW4A8ProfileMemoryBuild,
          w4a8_profile_elapsed_ns(memory_build_profile_start));
    }
  } else {
    const auto set_data_handle_profile_start = profile
        ? w4a8_profile_now()
        : W4A8ProfileClock::time_point{};
    cached.src_mem.set_data_handle(A_q.data_ptr(), 0);
    cached.src_mem.set_data_handle(expert_ends_i32.data_ptr(), 1);
    cached.dst_mem.set_data_handle(D.data_ptr(), 0);
    cached.dst_mem.set_data_handle(expert_ends_i32.data_ptr(), 1);
    cached.src_scales_mem.set_data_handle(A_scale.data_ptr());
    cached.src_zp_mem.set_data_handle(A_zp.data_ptr());
    cached.wei_mem.set_data_handle(B_packed_u4.data_ptr());
    cached.wei_scales_mem.set_data_handle(B_scales.data_ptr());
    if (has_bias) {
      cached.bias_mem.set_data_handle(bias->data_ptr());
    }
    if (profile) {
      w4a8_profile_record_stage(
          kW4A8ProfileSetDataHandle,
          w4a8_profile_elapsed_ns(set_data_handle_profile_start));
    }
  }

  const bool use_fast_path_before = cached.use_fast_path;
  bool fast_unimplemented_this_call = false;
  bool slow_call_this_call = false;

  if (cached.use_fast_path) {
    if (profile) {
      w4a8_profile_stats().fast_calls.fetch_add(
          1, std::memory_order_relaxed);
    }
    const auto execute_fast_profile_start = profile
        ? w4a8_profile_now()
        : W4A8ProfileClock::time_point{};
    try {
      (void)dnnl::sycl_interop::execute_fast(cached.exec_handle);
    } catch (const dnnl::error& fast_err) {
      if (fast_err.status == dnnl_unimplemented) {
        fast_unimplemented_this_call = true;
        if (profile) {
          w4a8_profile_stats().fast_unimplemented.fetch_add(
              1, std::memory_order_relaxed);
        }
        cached.use_fast_path = false;
      } else {
        TORCH_CHECK(false, "oneDNN grouped_gemm_w4a8: execute_fast failed: ",
                    fast_err.what());
      }
    } catch (...) {
      throw;
    }
    if (profile) {
      w4a8_profile_record_stage(
          kW4A8ProfileExecuteFast,
          w4a8_profile_elapsed_ns(execute_fast_profile_start));
    }
  }
  if (!cached.use_fast_path) {
    slow_call_this_call = true;
    if (profile) {
      w4a8_profile_stats().slow_calls.fetch_add(
          1, std::memory_order_relaxed);
    }
    const auto execute_slow_profile_start = profile
        ? w4a8_profile_now()
        : W4A8ProfileClock::time_point{};
    // Slow path: regular execute() for unsupported shapes (token-centric etc.)
    try {
      cached.prim.execute(stream, cached.args);
    } catch (const dnnl::error& e) {
      TORCH_CHECK(false, "oneDNN grouped_gemm_w4a8: execute slow failed: ", e.what());
    } catch (...) {
      throw;
    }
    if (profile) {
      w4a8_profile_record_stage(
          kW4A8ProfileExecuteSlow,
          w4a8_profile_elapsed_ns(execute_slow_profile_start));
    }
  }

  if (profile) {
    w4a8_profile_record_stage(
        kW4A8ProfileTotal,
        w4a8_profile_elapsed_ns(total_profile_start));
    const uint64_t calls = w4a8_profile_stats().total_calls.fetch_add(
        1, std::memory_order_relaxed) + 1;
    if (w4a8_profile_should_sample(calls)) {
      std::ostringstream row;
      row << "[onednn_w4a8_profile_call] call_id=" << calls
          << " device_id=" << device_id
          << " total_M=" << total_M
          << " N=" << N
          << " K=" << K
          << " num_experts=" << num_experts
          << " group_num=" << group_num
          << " group_size=" << group_size
          << " max_expert_size=" << max_expert_size_val
          << " has_bias=" << (has_bias ? 1 : 0)
          << " decode_like=" << (total_M < num_experts ? 1 : 0)
          << " cache_hit=" << (cache_hit ? 1 : 0)
          << " offset_needs_i32_copy=" << (offset_needs_i32_copy ? 1 : 0)
          << " memories_built_before=" << (memories_built_before ? 1 : 0)
          << " use_fast_path_before=" << (use_fast_path_before ? 1 : 0)
          << " fast_unimplemented_this_call=" << (fast_unimplemented_this_call ? 1 : 0)
          << " slow_call_this_call=" << (slow_call_this_call ? 1 : 0)
          << " A_q_dtype=" << scalar_type_id(A_q.scalar_type())
          << " A_scale_dtype=" << scalar_type_id(A_scale.scalar_type())
          << " A_zp_dtype=" << scalar_type_id(A_zp.scalar_type())
          << " B_packed_u4_dtype=" << scalar_type_id(B_packed_u4.scalar_type())
          << " B_scales_dtype=" << scalar_type_id(B_scales.scalar_type())
          << " D_dtype=" << scalar_type_id(D.scalar_type())
          << " bias_dtype=" << (has_bias ? scalar_type_id(bias->scalar_type()) : -1)
          << " src_onednn_dtype=" << static_cast<int>(src_dt)
          << " dst_onednn_dtype=" << static_cast<int>(requested_dst_dt)
          << " src_scales_onednn_dtype=" << static_cast<int>(src_scales_dt)
          << " wei_scales_onednn_dtype=" << static_cast<int>(wei_scales_dt)
          << " bias_onednn_dtype=" << cache_key.bias_dtype;
      append_pointer_alignment(row, "A_q", A_q.data_ptr());
      append_pointer_alignment(row, "A_scale", A_scale.data_ptr());
      append_pointer_alignment(row, "A_zp", A_zp.data_ptr());
      append_pointer_alignment(row, "B_packed_u4", B_packed_u4.data_ptr());
      append_pointer_alignment(row, "B_scales", B_scales.data_ptr());
      append_pointer_alignment(row, "D", D.data_ptr());
      append_pointer_alignment(row, "bias", has_bias ? bias->data_ptr() : nullptr);
      append_pointer_alignment(row, "offset_i32", expert_first_token_offset_i32.data_ptr());
      append_pointer_alignment(row, "expert_ends", expert_ends_i32.data_ptr());
      append_pointer_alignment(row, "hint_usm", cached.hint_usm);
      row << '\n';
      std::cerr << row.str();
    }
    w4a8_profile_maybe_dump(calls);
  }

  return D;
#endif
}

}  // namespace oneDNN
