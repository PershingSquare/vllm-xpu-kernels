#pragma once

#include <ATen/ATen.h>
#include <ATen/native/mkldnn/xpu/detail/LRUCache.h>

#include <array>
#include <cstdint>
#include <functional>
#include <utility>

#include <dnnl.hpp>

namespace oneDNN {

struct grouped_gemm_primitive_key_t {
  int64_t device_id = 0;

  int64_t src_dtype = 0;
  int64_t wei_dtype = 0;
  int64_t dst_dtype = 0;
  int64_t src_scales_dtype = 0;
  int64_t wei_scales_dtype = 0;
  int64_t bias_dtype = 0;
  int64_t requested_dst_dtype = 0;

  int64_t total_m = 0;
  int64_t n = 0;
  int64_t k = 0;
  int64_t num_experts = 0;
  int64_t group_num = 0;
  int64_t group_size = 0;

  int64_t has_bias = 0;

  bool operator==(const grouped_gemm_primitive_key_t& other) const {
    return device_id == other.device_id && src_dtype == other.src_dtype &&
        wei_dtype == other.wei_dtype && dst_dtype == other.dst_dtype &&
        src_scales_dtype == other.src_scales_dtype &&
        wei_scales_dtype == other.wei_scales_dtype &&
        bias_dtype == other.bias_dtype &&
        requested_dst_dtype == other.requested_dst_dtype &&
        total_m == other.total_m && n == other.n && k == other.k &&
        num_experts == other.num_experts && group_num == other.group_num &&
        group_size == other.group_size && has_bias == other.has_bias;
  }
};

}  // namespace oneDNN

namespace std {

template <>
struct hash<oneDNN::grouped_gemm_primitive_key_t> {
  size_t operator()(const oneDNN::grouped_gemm_primitive_key_t& key) const {
    size_t seed = 0;
    auto hash_combine = [&](int64_t value) {
      seed ^= std::hash<int64_t>{}(value) + 0x9e3779b9 + (seed << 6) +
          (seed >> 2);
    };

    hash_combine(key.device_id);
    hash_combine(key.src_dtype);
    hash_combine(key.wei_dtype);
    hash_combine(key.dst_dtype);
    hash_combine(key.src_scales_dtype);
    hash_combine(key.wei_scales_dtype);
    hash_combine(key.bias_dtype);
    hash_combine(key.requested_dst_dtype);

    hash_combine(key.total_m);
    hash_combine(key.n);
    hash_combine(key.k);
    hash_combine(key.num_experts);
    hash_combine(key.group_num);
    hash_combine(key.group_size);

    hash_combine(key.has_bias);
    return seed;
  }
};

}  // namespace std

namespace oneDNN {

struct grouped_gemm_cached_primitive_t {
  dnnl::matmul::primitive_desc pd;
  dnnl::matmul prim;
};

using grouped_gemm_primitive_cache = at::native::onednn::lru_cache<
    grouped_gemm_primitive_key_t,
    grouped_gemm_cached_primitive_t>;

inline grouped_gemm_primitive_cache& get_grouped_gemm_primitive_cache(
    int device_id) {
  static constexpr int max_cache_capacity = 512;
  static constexpr int max_device_count = 16;
  TORCH_CHECK(
      device_id >= 0 && device_id < max_device_count,
      "Unsupported XPU device index for grouped GEMM cache: ",
      device_id);

  static thread_local std::array<grouped_gemm_primitive_cache, max_device_count>
      mappings;
  auto& mapping = mappings[device_id];
  if (mapping.max_size() == 0) {
    mapping.resize(max_cache_capacity);
  }
  return mapping;
}

template <typename CreateFn>
inline grouped_gemm_cached_primitive_t& grouped_gemm_primitive_create_and_cache(
    const int device_id,
    const grouped_gemm_primitive_key_t& key,
    CreateFn&& create_fn) {
  auto& cache = get_grouped_gemm_primitive_cache(device_id);
  auto iter = cache.find(key);
  if (iter == cache.end()) {
    return cache
        .insert(
            {key,
             std::forward<CreateFn>(create_fn)()})
        .first->second;
  }
  return iter->second;
}

}  // namespace oneDNN
