#include <ATen/ATen.h>
#include <ATen/DeviceGuard.h>
#include <ATen/xpu/XPUContext.h>
#include <sycl/sycl.hpp>

#include "dispatch_utils.h"
#include "ops.h"
#include "utils.h"

#include "quantization/fp8/quant_utils.h"
#include "quantization/int8/silu_and_mul_quant.h"

namespace vllm {

template <typename T>
[[intel::device_indirectly_callable]] inline __attribute__((always_inline)) float
compute_swigluoai(float gate, float up, float alpha, float limit) {
  float const cg = gate > limit ? limit : gate;
  float const cu = up > limit ? limit : (up < -limit ? -limit : up);
  float const sigmoid = 1.0f / (1.0f + sycl::exp(-cg * alpha));
  return (cu + 1.0f) * (cg * sigmoid);
}

template <typename scalar_t, int CACHED_D>
class swigluoai_and_mul_quant_int8_asym_kernel {
 private:
  uint8_t* out_q;
  scalar_t* out_scale;
  uint8_t* out_zp;
  scalar_t const* input;
  int d;
  float alpha;
  float limit;

 public:
  swigluoai_and_mul_quant_int8_asym_kernel(
      uint8_t* out_q_,
      scalar_t* out_scale_,
      uint8_t* out_zp_,
      scalar_t const* input_,
      int d_,
      float alpha_,
      float limit_)
      : out_q(out_q_),
        out_scale(out_scale_),
        out_zp(out_zp_),
        input(input_),
        d(d_),
        alpha(alpha_),
        limit(limit_) {}

  void operator()(sycl::nd_item<1> item) const {
    int const tid = item.get_local_id(0);
    int const local_range = item.get_local_range(0);
    int64_t const token_idx = item.get_group(0);

    int64_t const in_offset = static_cast<int64_t>(token_idx) * 2 * d;
    int64_t const out_offset = static_cast<int64_t>(token_idx) * d;
    scalar_t const* token_input = input + in_offset;
    uint8_t* token_output_q = out_q + out_offset;

    auto& shared = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[2]>(item.get_group());

    using vec_t = fp8::vec4_t<scalar_t>;
    using out_vec_t = fp8::dtypex4_t<uint8_t>;
    constexpr int OUT_VEC = 4;
    constexpr int IN_VEC = 4;

    bool const can_vec = (d % OUT_VEC == 0)
        && ((reinterpret_cast<uintptr_t>(token_input) &
             (IN_VEC * sizeof(scalar_t) - 1)) == 0)
        && ((reinterpret_cast<uintptr_t>(token_output_q) & (OUT_VEC - 1)) == 0);

    float thread_min = std::numeric_limits<float>::infinity();
    float thread_max = -std::numeric_limits<float>::infinity();

    if constexpr (CACHED_D > 0) {
      auto& act_cache =
          *sycl::ext::oneapi::group_local_memory_for_overwrite<float[CACHED_D]>(
              item.get_group());
      if (can_vec) {
        vec_t const* in_vec = reinterpret_cast<vec_t const*>(token_input);
        int const n_out_blocks = d / OUT_VEC;
#pragma unroll 2
        for (int i = tid; i < n_out_blocks; i += local_range) {
          vec_t v0 = in_vec[2 * i];
          vec_t v1 = in_vec[2 * i + 1];
          float const a0 = compute_swigluoai<scalar_t>(
              static_cast<float>(v0.x), static_cast<float>(v0.y), alpha, limit);
          float const a1 = compute_swigluoai<scalar_t>(
              static_cast<float>(v0.z), static_cast<float>(v0.w), alpha, limit);
          float const a2 = compute_swigluoai<scalar_t>(
              static_cast<float>(v1.x), static_cast<float>(v1.y), alpha, limit);
          float const a3 = compute_swigluoai<scalar_t>(
              static_cast<float>(v1.z), static_cast<float>(v1.w), alpha, limit);
          act_cache[OUT_VEC * i + 0] = a0;
          act_cache[OUT_VEC * i + 1] = a1;
          act_cache[OUT_VEC * i + 2] = a2;
          act_cache[OUT_VEC * i + 3] = a3;
          thread_min = sycl::min(thread_min,
              sycl::min(sycl::min(a0, a1), sycl::min(a2, a3)));
          thread_max = sycl::max(thread_max,
              sycl::max(sycl::max(a0, a1), sycl::max(a2, a3)));
        }
      } else {
        for (int i = tid; i < d; i += local_range) {
          float const act = compute_swigluoai<scalar_t>(
              static_cast<float>(token_input[2 * i]),
              static_cast<float>(token_input[2 * i + 1]), alpha, limit);
          act_cache[i] = act;
          thread_min = sycl::min(thread_min, act);
          thread_max = sycl::max(thread_max, act);
        }
      }

      float const block_min = sycl::reduce_over_group(
          item.get_group(), thread_min, sycl::minimum<float>());
      float const block_max = sycl::reduce_over_group(
          item.get_group(), thread_max, sycl::maximum<float>());

      if (tid == 0) {
        float const range = block_max - block_min;
        float const scale = sycl::max(range / 255.0f, 1e-10f);
        float const zp_f = sycl::rint(-block_min / scale);
        float const zp_clamped = sycl::clamp(zp_f, 0.0f, 255.0f);
        uint8_t const zp_u8 = static_cast<uint8_t>(zp_clamped);
        shared[0] = scale;
        shared[1] = static_cast<float>(zp_u8);
        out_scale[token_idx] = static_cast<scalar_t>(scale);
        out_zp[token_idx] = zp_u8;
      }
      sycl::group_barrier(item.get_group());
      float const scale = shared[0];
      float const zp = shared[1];
      float const inv_scale = 1.0f / scale;

      if (can_vec) {
        out_vec_t* out_vec = reinterpret_cast<out_vec_t*>(token_output_q);
        int const n_out_blocks = d / OUT_VEC;
        auto quant = [&](float a) -> uint8_t {
          return static_cast<uint8_t>(
              sycl::clamp(sycl::rint(a * inv_scale + zp), 0.0f, 255.0f));
        };
#pragma unroll 2
        for (int i = tid; i < n_out_blocks; i += local_range) {
          out_vec_t o;
          o.x = quant(act_cache[OUT_VEC * i + 0]);
          o.y = quant(act_cache[OUT_VEC * i + 1]);
          o.z = quant(act_cache[OUT_VEC * i + 2]);
          o.w = quant(act_cache[OUT_VEC * i + 3]);
          out_vec[i] = o;
        }
      } else {
        for (int i = tid; i < d; i += local_range) {
          float const q = sycl::rint(act_cache[i] * inv_scale + zp);
          token_output_q[i] = static_cast<uint8_t>(sycl::clamp(q, 0.0f, 255.0f));
        }
      }
    } else {
      if (can_vec) {
        vec_t const* in_vec = reinterpret_cast<vec_t const*>(token_input);
        int const n_out_blocks = d / OUT_VEC;
#pragma unroll 2
        for (int i = tid; i < n_out_blocks; i += local_range) {
          vec_t v0 = in_vec[2 * i];
          vec_t v1 = in_vec[2 * i + 1];
          float const a0 = compute_swigluoai<scalar_t>(
              static_cast<float>(v0.x), static_cast<float>(v0.y), alpha, limit);
          float const a1 = compute_swigluoai<scalar_t>(
              static_cast<float>(v0.z), static_cast<float>(v0.w), alpha, limit);
          float const a2 = compute_swigluoai<scalar_t>(
              static_cast<float>(v1.x), static_cast<float>(v1.y), alpha, limit);
          float const a3 = compute_swigluoai<scalar_t>(
              static_cast<float>(v1.z), static_cast<float>(v1.w), alpha, limit);
          thread_min = sycl::min(thread_min,
              sycl::min(sycl::min(a0, a1), sycl::min(a2, a3)));
          thread_max = sycl::max(thread_max,
              sycl::max(sycl::max(a0, a1), sycl::max(a2, a3)));
        }
      } else {
        for (int i = tid; i < d; i += local_range) {
          float const act = compute_swigluoai<scalar_t>(
              static_cast<float>(token_input[2 * i]),
              static_cast<float>(token_input[2 * i + 1]), alpha, limit);
          thread_min = sycl::min(thread_min, act);
          thread_max = sycl::max(thread_max, act);
        }
      }

      float const block_min = sycl::reduce_over_group(
          item.get_group(), thread_min, sycl::minimum<float>());
      float const block_max = sycl::reduce_over_group(
          item.get_group(), thread_max, sycl::maximum<float>());

      if (tid == 0) {
        float const range = block_max - block_min;
        float const scale = sycl::max(range / 255.0f, 1e-10f);
        float const zp_f = sycl::rint(-block_min / scale);
        float const zp_clamped = sycl::clamp(zp_f, 0.0f, 255.0f);
        uint8_t const zp_u8 = static_cast<uint8_t>(zp_clamped);
        shared[0] = scale;
        shared[1] = static_cast<float>(zp_u8);
        out_scale[token_idx] = static_cast<scalar_t>(scale);
        out_zp[token_idx] = zp_u8;
      }
      sycl::group_barrier(item.get_group());
      float const scale = shared[0];
      float const zp = shared[1];
      float const inv_scale = 1.0f / scale;

      if (can_vec) {
        vec_t const* in_vec = reinterpret_cast<vec_t const*>(token_input);
        out_vec_t* out_vec = reinterpret_cast<out_vec_t*>(token_output_q);
        int const n_out_blocks = d / OUT_VEC;
        auto quant = [&](float a) -> uint8_t {
          return static_cast<uint8_t>(
              sycl::clamp(sycl::rint(a * inv_scale + zp), 0.0f, 255.0f));
        };
#pragma unroll 2
        for (int i = tid; i < n_out_blocks; i += local_range) {
          vec_t v0 = in_vec[2 * i];
          vec_t v1 = in_vec[2 * i + 1];
          out_vec_t o;
          o.x = quant(compute_swigluoai<scalar_t>(
              static_cast<float>(v0.x), static_cast<float>(v0.y), alpha, limit));
          o.y = quant(compute_swigluoai<scalar_t>(
              static_cast<float>(v0.z), static_cast<float>(v0.w), alpha, limit));
          o.z = quant(compute_swigluoai<scalar_t>(
              static_cast<float>(v1.x), static_cast<float>(v1.y), alpha, limit));
          o.w = quant(compute_swigluoai<scalar_t>(
              static_cast<float>(v1.z), static_cast<float>(v1.w), alpha, limit));
          out_vec[i] = o;
        }
      } else {
        for (int i = tid; i < d; i += local_range) {
          float const act = compute_swigluoai<scalar_t>(
              static_cast<float>(token_input[2 * i]),
              static_cast<float>(token_input[2 * i + 1]), alpha, limit);
          float const q = sycl::rint(act * inv_scale + zp);
          token_output_q[i] = static_cast<uint8_t>(sycl::clamp(q, 0.0f, 255.0f));
        }
      }
    }
  }
};

}  // namespace vllm

void swigluoai_and_mul_quant_int8_asym(
    torch::Tensor& out_q,
    torch::Tensor& out_scale,
    torch::Tensor& out_zp,
    torch::Tensor const& input,
    double alpha,
    double limit) {
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(out_q.is_contiguous(), "out_q must be contiguous");
  TORCH_CHECK(out_scale.is_contiguous(), "out_scale must be contiguous");
  TORCH_CHECK(out_zp.is_contiguous(), "out_zp must be contiguous");
  TORCH_CHECK(
      input.scalar_type() == at::ScalarType::Half ||
          input.scalar_type() == at::ScalarType::BFloat16,
      "input must be fp16 or bf16");
  TORCH_CHECK(out_q.scalar_type() == at::ScalarType::Byte, "out_q must be uint8");
  TORCH_CHECK(
      out_scale.scalar_type() == input.scalar_type(),
      "out_scale must match input dtype");
  TORCH_CHECK(out_zp.scalar_type() == at::ScalarType::Byte, "out_zp must be uint8");
  TORCH_CHECK(input.size(-1) % 2 == 0, "input.size(-1) must be even (2*d)");

  int const two_d = static_cast<int>(input.size(-1));
  int const d = two_d / 2;
  int64_t const num_tokens = input.numel() / two_d;
  TORCH_CHECK(out_q.numel() == num_tokens * d, "out_q numel must be num_tokens*d");
  TORCH_CHECK(
      out_scale.numel() == num_tokens, "out_scale numel must equal num_tokens");
  TORCH_CHECK(
      out_zp.numel() == num_tokens, "out_zp numel must equal num_tokens");

  if (num_tokens == 0) {
    return;
  }

  constexpr int WG_MAX = 256;
  constexpr int WG_MIN = 16;
  constexpr int SG_SIZE = 16;
  int const n_out = d / 4;
  int wg_size = ((std::min(n_out / 3, WG_MAX) + SG_SIZE - 1) / SG_SIZE) * SG_SIZE;
  wg_size = std::max(wg_size, WG_MIN);
  sycl::range<1> grid(num_tokens);
  sycl::range<1> block(wg_size);

  at::DeviceGuard const device_guard(input.device());
  auto& queue = vllm::xpu::vllmGetQueue();

  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "swigluoai_and_mul_quant_int8_asym", [&] {
        queue.submit([&](sycl::handler& cgh) {
          if (d <= 768) {
            auto kernel =
                vllm::swigluoai_and_mul_quant_int8_asym_kernel<scalar_t, 768>(
                    out_q.data_ptr<uint8_t>(),
                    out_scale.data_ptr<scalar_t>(),
                    out_zp.data_ptr<uint8_t>(),
                    input.data_ptr<scalar_t>(),
                    d,
                    static_cast<float>(alpha),
                    static_cast<float>(limit));
            cgh.parallel_for(sycl::nd_range<1>(grid * block, block), kernel);
          } else {
            auto kernel =
                vllm::swigluoai_and_mul_quant_int8_asym_kernel<scalar_t, 0>(
                    out_q.data_ptr<uint8_t>(),
                    out_scale.data_ptr<scalar_t>(),
                    out_zp.data_ptr<uint8_t>(),
                    input.data_ptr<scalar_t>(),
                    d,
                    static_cast<float>(alpha),
                    static_cast<float>(limit));
            cgh.parallel_for(sycl::nd_range<1>(grid * block, block), kernel);
          }
        });
      });
}
