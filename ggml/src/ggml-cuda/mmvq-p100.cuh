#pragma once

#include "common.cuh"

// sm_60-only builds run mmvq-p100.cu instead of the mmvq.cu kernels
#if defined(__CUDA_ARCH_LIST__) && __CUDA_ARCH_LIST__ == 600
#define GGML_CUDA_P100_MMVQ 1
#else
#define GGML_CUDA_P100_MMVQ 0
#endif

void ggml_cuda_p100_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
    const ggml_cuda_mm_fusion_args_host * fusion = nullptr);

void ggml_cuda_p100_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);

// Pascal: 17..28 columns of q6_K/q5_K weights as fp16 mat-vec column chunks
bool ggml_cuda_mmvq_chunked_ok(int cc, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);
bool ggml_cuda_mul_mat_vec_q_chunked(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
// strided (view) quantized src0 without a non-contiguous converter: mat-vec column chunks instead of cuBLAS
bool ggml_cuda_mul_mat_vec_q_chunked_strided(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
