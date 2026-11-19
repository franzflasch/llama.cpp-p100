// included by ggml-cuda.cu, uses its static helpers
// Mirrors ggml_cuda_mul_mat's routing: true iff it would reach ggml_cuda_mul_mat_cublas with an f16
// compute type (where the fold GEMM is tried). Used to decide the SwiGLU -> fold prescale fusion.
static bool ggml_cuda_mul_mat_is_cublas_f16(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                                            const ggml_tensor * dst) {
    if (ggml_get_op_params_i32(dst, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
        return false;
    }
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    const bool bad_padding_clear = ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE
        && ggml_nbytes(src0) != ggml_backend_buffer_get_alloc_size(src0->buffer, src0) && src0->view_src;
    if (!(bad_padding_clear || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32)) {
        const int     warp_size = ggml_cuda_info().devices[ctx.device].warp_size;
        const int64_t ne11      = src1->ne[1];
        if (ggml_cuda_should_use_mmvf(src0, cc, warp_size, ne11)) {
            return false;
        }
        if (src0->ne[1] == 1 && ne11 > MMVF_MAX_BATCH_SIZE && dst->ne[2] == 1 && dst->ne[3] == 1 && src0->type == GGML_TYPE_F32) {
            return false; // (the transposed-vector case; conservatively without its remaining conditions)
        }
        if (ggml_cuda_should_use_mmf(src0->type, cc, warp_size, src0->ne, src0->nb, ne11, /*mul_mat_id =*/ false) ||
            ggml_cuda_should_use_mmq_few(src0, src1, nullptr, cc) || ggml_cuda_should_use_mmvq(src0->type, cc, ne11)) {
            return false;
        }
        if (src0->type == GGML_TYPE_F8_E4M3) {
            return false;
        }
        // the 9..16-column fp16 matvec route ggml_cuda_mul_mat takes before MMQ and cuBLAS; without this a SwiGLU
        // planned for the fold GEMM was skipped and never computed (GGML_ASSERT(!ggml_cuda_gemm_fold_glu_pending()))
        if (ne11 > MMVQ_MAX_BATCH_SIZE && !ggml_cuda_gemm_fold_wants_f32(ctx, src0, src1, dst) &&
            ggml_cuda_mmvq_f16_takes(src0, src1, dst, ne11)) {
            return false;
        }
        if (ggml_cuda_should_use_mmq(src0->type, cc, ne11, /*n_experts =*/ 0)) {
            return false;
        }
    }
    return ggml_cuda_mul_mat_cublas_compute_type(src0, src1, dst, cc, ggml_cuda_info().devices[ctx.device].prefer_fp32_acc) == GGML_TYPE_F16;
}
