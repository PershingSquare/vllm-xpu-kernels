#include <ATen/ATen.h>
#include <ATen/DeviceGuard.h>
#include <ATen/xpu/XPUContext.h>
#include <sycl/sycl.hpp>

#include "dispatch_utils.h"
#include "ops.h"
#include "utils.h"

#include "quantization/fp8/quant_utils.h"
#include "quantization/int8/int8_quant.h"

namespace vllm {

template <typename scalar_t>
class dynamic_per_token_quant_int8_asym_kernel {
 private:
  uint8_t* out_q;
  scalar_t* out_scale;
  uint8_t* out_zp;
  scalar_t const* input;
  int hidden_size;

 public:
  dynamic_per_token_quant_int8_asym_kernel(
      uint8_t* out_q_,
      scalar_t* out_scale_,
      uint8_t* out_zp_,
      scalar_t const* input_,
      int hidden_size_)
      : out_q(out_q_),
        out_scale(out_scale_),
        out_zp(out_zp_),
        input(input_),
        hidden_size(hidden_size_) {}

  void operator()(sycl::nd_item<1> item) const {
    int const tid = item.get_local_id(0);
    int const local_range = item.get_local_range(0);
    int64_t const token_idx = item.get_group(0);

    int64_t const offset = static_cast<int64_t>(token_idx) * hidden_size;
    scalar_t const* token_input = input + offset;
    uint8_t* token_output_q = out_q + offset;

    auto& shared = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[2]>(item.get_group());

    using vec_t = fp8::vec4_t<scalar_t>;
    using out_vec_t = fp8::dtypex4_t<uint8_t>;
    constexpr int VEC = 4;
    bool const can_vec = (hidden_size % VEC == 0)
        && ((reinterpret_cast<uintptr_t>(token_input) &
             (VEC * sizeof(scalar_t) - 1)) == 0)
        && ((reinterpret_cast<uintptr_t>(token_output_q) & (VEC - 1)) == 0);

    float thread_min = std::numeric_limits<float>::infinity();
    float thread_max = -std::numeric_limits<float>::infinity();

    if (can_vec) {
      vec_t const* in_vec = reinterpret_cast<vec_t const*>(token_input);
      int const n_vec = hidden_size / VEC;
#pragma unroll 4
      for (int i = tid; i < n_vec; i += local_range) {
        vec_t v = in_vec[i];
        float const vx = static_cast<float>(v.x);
        float const vy = static_cast<float>(v.y);
        float const vz = static_cast<float>(v.z);
        float const vw = static_cast<float>(v.w);
        thread_min = sycl::min(thread_min,
            sycl::min(sycl::min(vx, vy), sycl::min(vz, vw)));
        thread_max = sycl::max(thread_max,
            sycl::max(sycl::max(vx, vy), sycl::max(vz, vw)));
      }
    } else {
      for (int i = tid; i < hidden_size; i += local_range) {
        float const x = static_cast<float>(token_input[i]);
        thread_min = sycl::min(thread_min, x);
        thread_max = sycl::max(thread_max, x);
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
      int const n_vec = hidden_size / VEC;
#pragma unroll 4
      for (int i = tid; i < n_vec; i += local_range) {
        vec_t v = in_vec[i];
        out_vec_t o;
        o.x = static_cast<uint8_t>(sycl::clamp(
            sycl::rint(static_cast<float>(v.x) * inv_scale + zp), 0.0f, 255.0f));
        o.y = static_cast<uint8_t>(sycl::clamp(
            sycl::rint(static_cast<float>(v.y) * inv_scale + zp), 0.0f, 255.0f));
        o.z = static_cast<uint8_t>(sycl::clamp(
            sycl::rint(static_cast<float>(v.z) * inv_scale + zp), 0.0f, 255.0f));
        o.w = static_cast<uint8_t>(sycl::clamp(
            sycl::rint(static_cast<float>(v.w) * inv_scale + zp), 0.0f, 255.0f));
        out_vec[i] = o;
      }
    } else {
      for (int i = tid; i < hidden_size; i += local_range) {
        float const x = static_cast<float>(token_input[i]);
        float const q = sycl::rint(x * inv_scale + zp);
        token_output_q[i] = static_cast<uint8_t>(sycl::clamp(q, 0.0f, 255.0f));
      }
    }
  }
};



}  // namespace vllm

void dynamic_per_token_quant_int8_asym(
    torch::Tensor& out_q,
    torch::Tensor& out_scale,
    torch::Tensor& out_zp,
    torch::Tensor const& input) {
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(out_q.is_contiguous(), "out_q must be contiguous");
  TORCH_CHECK(out_scale.is_contiguous(), "out_scale must be contiguous");
  TORCH_CHECK(out_zp.is_contiguous(), "out_zp must be contiguous");
  TORCH_CHECK(
      input.scalar_type() == at::ScalarType::Half ||
          input.scalar_type() == at::ScalarType::BFloat16,
      "input must be fp16 or bf16, got ",
      input.scalar_type());
  TORCH_CHECK(out_q.scalar_type() == at::ScalarType::Byte, "out_q must be uint8");
  TORCH_CHECK(
      out_scale.scalar_type() == input.scalar_type(),
      "out_scale must match input dtype");
  TORCH_CHECK(out_zp.scalar_type() == at::ScalarType::Byte, "out_zp must be uint8");

  int const hidden_size = static_cast<int>(input.size(-1));
  int64_t const num_tokens = input.numel() / hidden_size;
  TORCH_CHECK(out_q.numel() == input.numel(), "out_q numel must match input");
  TORCH_CHECK(
      out_scale.numel() == num_tokens, "out_scale numel must equal num_tokens");
  TORCH_CHECK(
      out_zp.numel() == num_tokens, "out_zp numel must equal num_tokens");

  TORCH_CHECK(hidden_size % 4 == 0, "hidden_size must be divisible by 4, got ", hidden_size);

  if (num_tokens == 0) {
    return;
  }

  constexpr int WG_MAX = 256;
  constexpr int WG_MIN = 16;
  constexpr int SG_SIZE = 16;
  int const n_vec = hidden_size / 4;
  int wg_size = ((std::min(n_vec / 3, WG_MAX) + SG_SIZE - 1) / SG_SIZE) * SG_SIZE;
  wg_size = std::max(wg_size, WG_MIN);
  sycl::range<1> grid(num_tokens);
  sycl::range<1> block(wg_size);

  at::DeviceGuard const device_guard(input.device());
  auto& queue = vllm::xpu::vllmGetQueue();

  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "dynamic_per_token_quant_int8_asym", [&] {
        queue.submit([&](sycl::handler& cgh) {
          auto kernel = vllm::dynamic_per_token_quant_int8_asym_kernel<scalar_t>(
              out_q.data_ptr<uint8_t>(),
              out_scale.data_ptr<scalar_t>(),
              out_zp.data_ptr<uint8_t>(),
              input.data_ptr<scalar_t>(),
              hidden_size);
          cgh.parallel_for(sycl::nd_range<1>(grid * block, block), kernel);
        });
      });
}
