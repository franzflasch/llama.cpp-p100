#pragma once

#include "common.cuh"

// cuBLAS-GEMM flash attention for pre-Volta NVIDIA.
// See fattn-gemm.cu for why this path exists.
bool ggml_cuda_flash_attn_ext_gemm_supported(const ggml_tensor * dst);
void ggml_cuda_flash_attn_ext_gemm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
