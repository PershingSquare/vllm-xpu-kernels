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

namespace oneDNN {

namespace {

static inline bool
message_contains(std::string_view haystack, std::string_view needle) {
  return haystack.find(needle) != std::string_view::npos;
}

static inline dnnl::memory::data_type to_onednn_type(at::ScalarType t) {
  switch (t) {
    case at::ScalarType::Half:
      return dnnl::memory::data_type::f16;
    case at::ScalarType::BFloat16:
      return dnnl::memory::data_type::bf16;
    case at::ScalarType::Float:
      return dnnl::memory::data_type::f32;
    case at::ScalarType::Int:
      return dnnl::memory::data_type::s32;
    default:
      break;
  }
  TORCH_CHECK(false, "Unsupported dtype in oneDNN grouped GEMM: ", t);
  return dnnl::memory::data_type::undef;
}

static inline const char* torch_dtype_name(at::ScalarType t) {
  switch (t) {
    case at::ScalarType::Half:
      return "torch.float16";
    case at::ScalarType::BFloat16:
      return "torch.bfloat16";
    case at::ScalarType::Float:
      return "torch.float32";
    case at::ScalarType::Char:
      return "torch.int8";
    case at::ScalarType::Byte:
      return "torch.uint8";
    case at::ScalarType::Int:
      return "torch.int32";
    case at::ScalarType::Long:
      return "torch.int64";
    default:
      return "unknown";
  }
}

static inline bool is_fp16_bf16_or_fp32(at::ScalarType t) {
  return t == at::ScalarType::Half || t == at::ScalarType::BFloat16 ||
         t == at::ScalarType::Float;
}

static inline bool is_fp16_or_bf16(at::ScalarType t) {
  return t == at::ScalarType::Half || t == at::ScalarType::BFloat16;
}

static torch::Tensor grouped_gemm_w4a8_fallback_dequant_cpu(
    torch::Tensor A_q,
    torch::Tensor A_scale,
    torch::Tensor A_zp,
    torch::Tensor B_packed_u4,
    torch::Tensor B_scales,
    const c10::optional<at::Tensor>& bias,
    torch::Tensor D,
    torch::Tensor expert_first_token_offset_i32,
    int64_t N,
    int64_t K,
    int64_t num_experts,
    int64_t group_num,
    int64_t group_size) {
  (void)group_num;

  auto A_q_cpu = A_q.cpu().contiguous();
  auto A_scale_cpu =
      A_scale.cpu().to(at::kFloat).reshape({A_q.size(0)}).contiguous();
  auto A_zp_cpu = A_zp.cpu().to(at::kInt).reshape({A_q.size(0)}).contiguous();
  auto B_cpu = B_packed_u4.cpu().contiguous();
  auto B_scales_cpu = B_scales.cpu().to(at::kFloat).contiguous();
  auto offsets_cpu = expert_first_token_offset_i32.cpu().contiguous();

  auto A_deq_cpu = torch::empty(
      {A_q.size(0), K},
      torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat));
  auto A_q_acc = A_q_cpu.accessor<int8_t, 2>();
  auto A_scale_acc = A_scale_cpu.accessor<float, 1>();
  auto A_zp_acc = A_zp_cpu.accessor<int, 1>();
  auto A_deq_acc = A_deq_cpu.accessor<float, 2>();
  for (int64_t m = 0; m < A_q.size(0); ++m) {
    const float scale = A_scale_acc[m];
    const int zp = A_zp_acc[m];
    for (int64_t k_idx = 0; k_idx < K; ++k_idx) {
      A_deq_acc[m][k_idx] = static_cast<float>(A_q_acc[m][k_idx] - zp) * scale;
    }
  }

  auto W_cpu = torch::empty(
      {num_experts, K, N},
      torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat));
  auto B_acc = B_cpu.accessor<uint8_t, 3>();
  auto S_acc = B_scales_cpu.accessor<float, 3>();
  auto W_acc = W_cpu.accessor<float, 3>();
  const int64_t K_half = K / 2;
  for (int64_t e = 0; e < num_experts; ++e) {
    for (int64_t n_idx = 0; n_idx < N; ++n_idx) {
      for (int64_t j = 0; j < K_half; ++j) {
        const uint8_t byte = B_acc[e][n_idx][j];
        const int8_t lo =
            static_cast<int8_t>(static_cast<int>(byte & 0x0F) - 8);
        const int8_t hi =
            static_cast<int8_t>(static_cast<int>((byte >> 4) & 0x0F) - 8);
        const int64_t k0 = 2 * j;
        const int64_t k1 = k0 + 1;
        const float s0 = S_acc[e][n_idx][k0 / group_size];
        const float s1 = S_acc[e][n_idx][k1 / group_size];
        W_acc[e][k0][n_idx] = static_cast<float>(lo) * s0;
        W_acc[e][k1][n_idx] = static_cast<float>(hi) * s1;
      }
    }
  }

  auto out_cpu = torch::empty(
      {A_q.size(0), N},
      torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat));
  auto offsets_ptr = offsets_cpu.data_ptr<int>();
  for (int64_t e = 0; e < num_experts; ++e) {
    const int64_t start = offsets_ptr[e];
    const int64_t end = offsets_ptr[e + 1];
    if (end <= start) continue;
    auto A_slice = A_deq_cpu.slice(0, start, end);
    auto W_slice = W_cpu[e];
    auto out_slice = torch::matmul(A_slice, W_slice);
    if (bias.has_value()) {
      out_slice.add_(bias.value().cpu().to(at::kFloat)[e]);
    }
    out_cpu.slice(0, start, end).copy_(out_slice);
  }

  D.copy_(out_cpu.to(D.device()).to(D.scalar_type()));
  return D;
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
    int64_t num_experts) {
#if !(                                           \
    defined(DNNL_EXPERIMENTAL_GROUPED_MEMORY) && \
    DNNL_EXPERIMENTAL_GROUPED_MEMORY) &&         \
    !(defined(DNNL_EXPERIMENTAL_GROUPED_GEMM) && \
      DNNL_EXPERIMENTAL_GROUPED_GEMM)
  (void)A_q;
  (void)A_scale;
  (void)A_zp;
  (void)B_packed_u4;
  (void)B_scales;
  (void)bias;
  (void)D;
  (void)expert_first_token_offset;
  (void)N;
  (void)K;
  (void)num_experts;
  TORCH_CHECK(
      false,
      "oneDNN grouped GEMM is not enabled in this build "
      "(DNNL_EXPERIMENTAL_GROUPED_MEMORY=0 / "
      "DNNL_EXPERIMENTAL_GROUPED_GEMM=0)");
#else
  // -----------------------------
  // Validate inputs
  // -----------------------------
  CHECK_DEVICE(A_q);
  CHECK_DEVICE(A_scale);
  CHECK_DEVICE(A_zp);
  CHECK_DEVICE(B_packed_u4);
  CHECK_DEVICE(B_scales);
  CHECK_DEVICE(D);
  CHECK_DEVICE(expert_first_token_offset);

  CHECK_CONTIGUOUS(A_q);
  CHECK_CONTIGUOUS(A_scale);
  CHECK_CONTIGUOUS(A_zp);
  CHECK_CONTIGUOUS(B_packed_u4);
  CHECK_CONTIGUOUS(B_scales);
  CHECK_CONTIGUOUS(D);
  CHECK_CONTIGUOUS(expert_first_token_offset);

  TORCH_CHECK(A_q.dim() == 2, "A_q must be 2D [Total_M, K]");
  TORCH_CHECK(B_packed_u4.dim() == 3, "B_packed_u4 must be 3D [E, N, K/2]");
  TORCH_CHECK(B_scales.dim() == 3, "B_scales must be 3D [E, N, K/group_size]");
  TORCH_CHECK(D.dim() == 2, "D must be 2D [Total_M, N]");
  TORCH_CHECK(
      expert_first_token_offset.dim() == 1,
      "expert_first_token_offset must be 1D [E+1]");

  TORCH_CHECK(
      A_q.scalar_type() == at::ScalarType::Byte,
      "A_q must be uint8 (torch.uint8) for oneDNN grouped_gemm_w4a8");
  TORCH_CHECK(
      is_fp16_bf16_or_fp32(A_scale.scalar_type()),
      "A_scale must be fp16, bf16, or fp32 for oneDNN grouped_gemm_w4a8");
  TORCH_CHECK(
      A_zp.scalar_type() == at::ScalarType::Byte ||
          A_zp.scalar_type() == at::ScalarType::Int ||
          A_zp.scalar_type() == at::ScalarType::Long,
      "A_zp must be uint8, int32, or int64 for oneDNN grouped_gemm_w4a8");
  TORCH_CHECK(
      B_packed_u4.scalar_type() == at::ScalarType::Byte,
      "B_packed_u4 must be uint8 (packed int4) for oneDNN grouped_gemm_w4a8");
  TORCH_CHECK(
      is_fp16_bf16_or_fp32(B_scales.scalar_type()),
      "B_scales must be fp16, bf16, or fp32 for oneDNN grouped_gemm_w4a8");
  TORCH_CHECK(
      is_fp16_or_bf16(D.scalar_type()),
      "D must be fp16 or bf16 for oneDNN grouped_gemm_w4a8");

  TORCH_CHECK(
      expert_first_token_offset.scalar_type() == at::ScalarType::Long ||
          expert_first_token_offset.scalar_type() == at::ScalarType::Int,
      "expert_first_token_offset must be int64 or int32");

  const int64_t total_M = A_q.size(0);
  TORCH_CHECK(
      A_q.size(1) == K,
      "A_q.size(1) must match K (expected K=",
      K,
      ", got ",
      A_q.size(1),
      ")");
  TORCH_CHECK(
      D.size(0) == total_M,
      "D.size(0) must match A_q.size(0) (expected ",
      total_M,
      ", got ",
      D.size(0),
      ")");
  TORCH_CHECK(
      D.size(1) == N,
      "D.size(1) must match N (expected N=",
      N,
      ", got ",
      D.size(1),
      ")");

  TORCH_CHECK(
      B_packed_u4.size(0) == num_experts,
      "B_packed_u4.size(0) must match num_experts");
  TORCH_CHECK(B_packed_u4.size(1) == N, "B_packed_u4.size(1) must match N");
  TORCH_CHECK(
      B_packed_u4.size(2) * 2 == K,
      "B_packed_u4.size(2) must be K/2 bytes for packed int4 weights");

  TORCH_CHECK(
      B_scales.size(0) == num_experts,
      "B_scales.size(0) must match num_experts");
  TORCH_CHECK(B_scales.size(1) == N, "B_scales.size(1) must match N");
  TORCH_CHECK(B_scales.size(2) > 0, "B_scales.size(2) (group_num) must be > 0");
  TORCH_CHECK(
      K % B_scales.size(2) == 0, "B_scales.size(2) (group_num) must divide K");
  const int64_t group_num = B_scales.size(2);
  const int64_t group_size = K / group_num;

  TORCH_CHECK(
      expert_first_token_offset.numel() == (num_experts + 1),
      "expert_first_token_offset must have length E+1 (E=num_experts)");

  // Optional bias: [E, N]
  if (bias.has_value()) {
    const at::Tensor& b = *bias;
    CHECK_DEVICE(b);
    CHECK_CONTIGUOUS(b);
    TORCH_CHECK(b.dim() == 2, "bias must be 2D [E, N]");
    TORCH_CHECK(b.size(0) == num_experts, "bias.size(0) must match E");
    TORCH_CHECK(b.size(1) == N, "bias.size(1) must match N");
    TORCH_CHECK(
        is_fp16_bf16_or_fp32(b.scalar_type()),
        "bias must be fp16, bf16, or fp32 for oneDNN grouped_gemm_w4a8");
  }

  // Grouped GPU int4 micro-kernel requires K and N even for WEI=s4/u4.
  TORCH_CHECK(
      (K % 2) == 0,
      "oneDNN grouped_gemm_w4a8 requires K even for int4 weights (K=",
      K,
      ")");
  TORCH_CHECK(
      (N % 2) == 0,
      "oneDNN grouped_gemm_w4a8 requires N even for int4 weights (N=",
      N,
      ")");
  TORCH_CHECK(
      total_M <= std::numeric_limits<int>::max(),
      "oneDNN grouped_gemm_w4a8: total_M exceeds int32 range (total_M=",
      total_M,
      ")");

  // Updated oneDNN grouped GPU kernels require row-wise SRC scales for this
  // int8xint4 grouped path. This wrapper also uses row-wise asymmetric source
  // zero-points to match the w4a8 activation quantization contract.
  const bool per_row_A_scale =
      (A_scale.dim() == 1 && A_scale.size(0) == total_M) ||
      (A_scale.dim() == 2 && A_scale.size(0) == total_M &&
       A_scale.size(1) == 1);
  const bool per_row_A_zp =
      (A_zp.dim() == 1 && A_zp.size(0) == total_M) ||
      (A_zp.dim() == 2 && A_zp.size(0) == total_M && A_zp.size(1) == 1);
  TORCH_CHECK(
      per_row_A_scale,
      "A_scale must be [total_M] or [total_M,1] (row-wise); got shape ",
      A_scale.sizes());
  TORCH_CHECK(
      per_row_A_zp,
      "A_zp must be [total_M] or [total_M,1] (row-wise); got shape ",
      A_zp.sizes());

  // -----------------------------
  // Grouped offsets: vLLM -> oneDNN
  // -----------------------------
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
  expert_ends_onednn_i32.narrow(0, num_experts, 1)
      .fill_(static_cast<int>(total_M));

  // -----------------------------
  // oneDNN descriptors
  // -----------------------------
  const auto src_dt = dnnl::memory::data_type::u8;
  const auto requested_dst_dt = to_onednn_type(D.scalar_type());
  const auto src_scales_dt = to_onednn_type(A_scale.scalar_type());
  const auto wei_scales_dt = to_onednn_type(B_scales.scalar_type());
  const auto src_zp_dt = dnnl::memory::data_type::u8;

  const dnnl::memory::dim ngroups = static_cast<dnnl::memory::dim>(num_experts);
  auto src_md = dnnl::memory::desc::grouped(
      {total_M, K},
      src_dt,
      /*variable_dim_idx=*/0,
      /*group_count=*/ngroups,
      /*offsets_dt=*/dnnl::memory::data_type::s32);
  auto make_dst_md = [&](dnnl::memory::data_type dt) {
    return dnnl::memory::desc::grouped(
        {total_M, N},
        dt,
        /*variable_dim_idx=*/0,
        /*group_count=*/ngroups,
        /*offsets_dt=*/dnnl::memory::data_type::s32);
  };

  // Weights: logical [E, K, N] s4, physical buffer is vLLM [E, N, K/2] bytes.
  // Describe as tag `acb` so oneDNN indexes as (e-major, n-major, k-contig).
  auto wei_md = dnnl::memory::desc(
      {num_experts, K, N},
      dnnl::memory::data_type::s4,
      dnnl::memory::format_tag::acb);

  // Weight scales: vLLM provides [E, N, G], oneDNN expects [E, G, N].
  const dnnl::memory::dim G = static_cast<dnnl::memory::dim>(group_num);
  const dnnl::memory::dim NN = static_cast<dnnl::memory::dim>(N);
  torch::Tensor wei_scales_onednn = B_scales.permute({0, 2, 1}).contiguous();
  CHECK_DEVICE(wei_scales_onednn);
  CHECK_CONTIGUOUS(wei_scales_onednn);
  auto wei_scales_md = dnnl::memory::desc(
      {num_experts, G, NN}, wei_scales_dt, dnnl::memory::format_tag::abc);

  // SRC scales: row-wise [M]. SRC zero-points: row-wise [M].
  torch::Tensor src_scales_onednn = A_scale.reshape({total_M}).contiguous();
  CHECK_DEVICE(src_scales_onednn);
  CHECK_CONTIGUOUS(src_scales_onednn);
  auto src_scales_md =
      dnnl::memory::desc({total_M}, src_scales_dt, dnnl::memory::format_tag::a);
  torch::Tensor src_zp_onednn = (A_zp.scalar_type() == at::ScalarType::Byte
                                     ? A_zp
                                     : A_zp.to(at::ScalarType::Byte))
                                    .reshape({total_M})
                                    .contiguous();
  CHECK_DEVICE(src_zp_onednn);
  CHECK_CONTIGUOUS(src_zp_onednn);
  auto src_zp_md =
      dnnl::memory::desc({total_M}, src_zp_dt, dnnl::memory::format_tag::a);

  // Optional bias: [E, N]
  dnnl::memory::desc bias_md;
  if (bias.has_value()) {
    const at::Tensor& b = *bias;
    bias_md = dnnl::memory::desc(
        {num_experts, N},
        to_onednn_type(b.scalar_type()),
        {/*stride_e=*/N, /*stride_n=*/1});
  }

  // -----------------------------
  // Configure scales
  // -----------------------------
  dnnl::primitive_attr attr;
  // Row-wise asymmetric activation quantization.
  attr.set_scales(
      DNNL_ARG_SRC,
      /*mask=*/(1 << 0),
      /*groups=*/{},
      src_scales_dt);
  attr.set_zero_points(
      DNNL_ARG_SRC,
      /*mask=*/(1 << 0),
      /*groups=*/{},
      src_zp_dt);

  // Weight scales: weights dims are [E, K, N].
  // Use the same convention as grouped_gemm_w4a16: mask=7 with
  // groups={group_size,1}.
  attr.set_scales(
      DNNL_ARG_WEIGHTS,
      /*mask=*/(1 << 0) | (1 << 1) | (1 << 2),
      /*groups=*/{group_size, 1},
      wei_scales_dt);

  // -----------------------------
  // Execute oneDNN matmul
  // -----------------------------
  const at::Device cur_device = A_q.device();
  const int device_id = cur_device.index();
  auto& engine = oneDNN::GpuEngineManager::Instance().get_engine(cur_device);
  auto& stream = oneDNN::GpuStreamManager::Instance().get_stream(device_id);

  // Convert vLLM packed u4(zp=8) to packed s4 (two's complement) for oneDNN.
  torch::Tensor B_for_onednn = (B_packed_u4 ^ 0x88).contiguous();

  auto wei_mem =
      oneDNN::make_onednn_memory(wei_md, engine, B_for_onednn.data_ptr());
  auto wei_scales_mem = oneDNN::make_onednn_memory(
      wei_scales_md, engine, wei_scales_onednn.data_ptr());
  auto src_scales_mem = oneDNN::make_onednn_memory(
      src_scales_md, engine, src_scales_onednn.data_ptr());
  auto src_zp_mem =
      oneDNN::make_onednn_memory(src_zp_md, engine, src_zp_onednn.data_ptr());

  // Grouped src memories carry both (values, offsets) handles.
  auto make_src_mem = [&]() {
    return dnnl::sycl_interop::make_memory(
        src_md,
        engine,
        dnnl::sycl_interop::memory_kind::usm,
        std::vector<void*>{A_q.data_ptr(), expert_ends_onednn_i32.data_ptr()});
  };

  auto run_int8 = [&](dnnl::memory::data_type dst_dt) {
    auto dst_md = make_dst_md(dst_dt);
    try {
      grouped_gemm_primitive_key_t cache_key{};
      cache_key.device_id = device_id;
      cache_key.src_dtype = static_cast<int64_t>(src_dt);
      cache_key.wei_dtype = static_cast<int64_t>(dnnl::memory::data_type::s4);
      cache_key.dst_dtype = static_cast<int64_t>(dst_dt);
      cache_key.src_scales_dtype = static_cast<int64_t>(src_scales_dt);
      cache_key.wei_scales_dtype = static_cast<int64_t>(wei_scales_dt);
      cache_key.bias_dtype =
          bias.has_value()
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

      auto& primitive_cache = get_grouped_gemm_primitive_cache(device_id);
      auto iter = primitive_cache.find(cache_key);
      if (iter == primitive_cache.end()) {
        dnnl::matmul::primitive_desc pd;
        try {
          if (bias.has_value()) {
            pd = dnnl::matmul::primitive_desc(
                engine, src_md, wei_md, bias_md, dst_md, attr);
          } else {
            pd = dnnl::matmul::primitive_desc(
                engine, src_md, wei_md, dst_md, attr);
          }
        } catch (const dnnl::error& e) {
          if (e.status == dnnl_unimplemented ||
              message_contains(e.what(), "unsupported scales configuration")) {
            TORCH_CHECK(
                false,
                "oneDNN grouped_gemm_w4a8: matmul primitive_desc creation "
                "failed on GPU. "
                "CPU fallback is disabled. Detail: ",
                e.what());
          }
          throw;
        }
        dnnl::matmul prim(pd);
        iter = primitive_cache
                   .insert(
                       {cache_key,
                        grouped_gemm_cached_primitive_t{
                            std::move(pd), std::move(prim)}})
                   .first;
      }

      auto src_mem = make_src_mem();
      auto dst_mem = dnnl::sycl_interop::make_memory(
          dst_md,
          engine,
          dnnl::sycl_interop::memory_kind::usm,
          std::vector<void*>{D.data_ptr(), expert_ends_onednn_i32.data_ptr()});

      std::unordered_map<int, dnnl::memory> args;
      args.emplace(DNNL_ARG_SRC, std::move(src_mem));
      args.emplace(DNNL_ARG_WEIGHTS, wei_mem);
      args.emplace(DNNL_ARG_DST, std::move(dst_mem));
      args.emplace(DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, src_scales_mem);
      args.emplace(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC, src_zp_mem);
      args.emplace(DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, wei_scales_mem);
      if (bias.has_value()) {
        const at::Tensor& b = *bias;
        auto bias_mem =
            oneDNN::make_onednn_memory(bias_md, engine, b.data_ptr());
        args.emplace(DNNL_ARG_BIAS, std::move(bias_mem));
      }

      (void)dnnl::sycl_interop::execute(iter->second.prim, stream, args);
    } catch (const dnnl::error& e) {
      if (e.status == dnnl_unimplemented ||
          message_contains(e.what(), "unsupported scales configuration")) {
        TORCH_CHECK(
            false,
            "oneDNN grouped_gemm_w4a8: GPU execution failed. CPU fallback is "
            "disabled. Detail: ",
            e.what());
      }
      if (e.status == dnnl_unimplemented) {
        TORCH_CHECK(
            false,
            "oneDNN grouped_gemm_w4a8: native int8xint4 grouped matmul is "
            "unimplemented. ",
            "scale dtypes: A_scale=",
            torch_dtype_name(A_scale.scalar_type()),
            " (oneDNN=",
            dnnl_dt2str(static_cast<dnnl_data_type_t>(src_scales_dt)),
            "), B_scales=",
            torch_dtype_name(B_scales.scalar_type()),
            " (oneDNN=",
            dnnl_dt2str(static_cast<dnnl_data_type_t>(wei_scales_dt)),
            "); zero-point dtypes: A_zp=",
            dnnl_dt2str(static_cast<dnnl_data_type_t>(src_zp_dt)),
            " (row-wise), B_zp=u4 (implicit 8, converted to packed s4 "
            "weights). ",
            "status=",
            static_cast<int>(e.status),
            " (",
            dnnl_status2str(e.status),
            "), detail=",
            e.what());
      }
      TORCH_WARN(
          "oneDNN grouped_gemm_w4a8: int8 primitive creation/execute failed "
          "(dst_dt=",
          static_cast<int>(dst_dt),
          "): status=",
          static_cast<int>(e.status),
          " (",
          dnnl_status2str(e.status),
          ") ",
          e.what());
      throw;
    } catch (const std::exception& e) {
      if (message_contains(e.what(), "unsupported scales configuration")) {
        TORCH_CHECK(
            false,
            "oneDNN grouped_gemm_w4a8: GPU execution failed with an "
            "unsupported scales "
            "configuration. CPU fallback is disabled. Detail: ",
            e.what());
      }
      throw;
    }
    return D;
  };

  return run_int8(requested_dst_dt);
#endif
}

}  // namespace oneDNN
