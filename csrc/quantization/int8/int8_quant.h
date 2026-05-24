#pragma once

#include <torch/all.h>

void dynamic_per_token_quant_int8_asym(
    torch::Tensor& out_q,
    torch::Tensor& out_scale,
    torch::Tensor& out_zp,
    torch::Tensor const& input);
