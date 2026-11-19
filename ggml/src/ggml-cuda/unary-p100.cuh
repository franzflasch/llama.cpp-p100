#pragma once

#include "common.cuh"

bool ggml_cuda_op_add_softplus_mul(ggml_backend_cuda_context & ctx, ggml_tensor * add, ggml_tensor * sp, ggml_tensor * mul);
