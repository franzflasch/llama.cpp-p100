#pragma once

#include "common.cuh"

// sm_60-only builds compile fattn-tile-p100.cuh / fattn-vec-p100.cuh in place of the upstream headers
#if defined(__CUDA_ARCH_LIST__) && __CUDA_ARCH_LIST__ == 600
#define GGML_CUDA_P100_FATTN 1
#else
#define GGML_CUDA_P100_FATTN 0
#endif

// cuBLAS-GEMM attention, q4_0 decode/verify kernel and compact causal mask (fattn-p100.cu).
// Each returns true if it handled the call.
bool ggml_cuda_p100_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_cuda_p100_flash_attn_ext_get_alloc_size(int device, const ggml_tensor * dst, size_t & size);
bool ggml_cuda_p100_flash_attn_ext_supported(int device, const ggml_tensor * dst, bool & supported);

// the tile kernel reads a q4_0 K/V cache directly (fattn-tile-p100.cuh)
bool ggml_cuda_p100_fattn_tile_q4_0_direct(const ggml_tensor * dst);
