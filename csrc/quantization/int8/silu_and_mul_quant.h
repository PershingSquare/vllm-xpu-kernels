#pragma once

#include <torch/all.h>

void swigluoai_and_mul_quant_int8_asym(
    torch::Tensor& out_q,
    torch::Tensor& out_scale,
    torch::Tensor& out_zp,
    torch::Tensor const& input,
    double alpha,
    double limit);
