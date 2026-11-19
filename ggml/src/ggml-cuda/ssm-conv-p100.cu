#include "common.cuh"
#include "ssm-conv-p100.cuh"
#include "gated_delta_net.cuh"
#include "unary.cuh"

// Single-token decode of the short conv in recurrent layers: CONCAT(conv state, new column) -> CPY of the
// window's last d_conv - 1 columns back into the state -> SSM_CONV -> SILU, as one kernel with one thread
// per channel. Each thread reads its channel's state before writing it, so the in-place state update
// (state read straight from the cache) is race-free. The convolution is ssm_conv_f32's first-token
// expression (same products in the same order, the same + bias of 0, the same silu), so the result is
// bit-identical; the concat output is written too, for any other reader.
template <int d_conv>
static __global__ void ssm_conv_decode_fused_f32(
        const char * st, const int64_t st_nb0, const int64_t st_nb1, const int32_t * g_idx, const int64_t g_row_bytes,
        const char * __restrict__ xn, const int64_t xn_nb1,
        const float * __restrict__ w, const int64_t w_nb1,
        float * cat, float * st_out, float * __restrict__ y, const int64_t C) {
    const int64_t c = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= C) {
        return;
    }
    // the state may be the cache row itself (st_out aliases it): no __restrict__ on st / st_out / cat
    if (g_idx != nullptr) {
        st += (int64_t) g_idx[0]*g_row_bytes;   // gathered in place: the sequence's cache row
    }
    float x[d_conv];
#pragma unroll
    for (int j = 0; j < d_conv - 1; ++j) {
        x[j] = *(const float *) (st + c*st_nb1 + j*st_nb0);
    }
    x[d_conv - 1] = *(const float *) (xn + c*xn_nb1);
    float wr[d_conv];
#pragma unroll
    for (int j = 0; j < d_conv; ++j) {
        wr[j] = w[c*(w_nb1/sizeof(float)) + j];
    }
#pragma unroll
    for (int j = 0; j < d_conv; ++j) {
        cat[c*d_conv + j] = x[j];
    }
#pragma unroll
    for (int j = 0; j < d_conv - 1; ++j) {
        st_out[c*(d_conv - 1) + j] = x[j + 1];
    }
    float sumf = 0.0f;
#pragma unroll
    for (int j = 0; j < d_conv; ++j) {
        sumf += x[j] * wr[j];
    }
    sumf += 0.0f; // ssm_conv_f32's bias of 0 (turns a -0 into +0, as there)
    y[c] = ggml_cuda_op_silu_single(sumf);
}

bool ggml_cuda_ssm_conv_decode_fused_ok(const ggml_tensor * concat, const ggml_tensor * cpy, const ggml_tensor * conv,
                                        const ggml_tensor * silu) {
    const ggml_tensor * st = concat->src[0];   // [d_conv - 1, C, 1] conv state (may be read in place from the cache)
    const ggml_tensor * xn = concat->src[1];   // [1, C, 1] new column
    const ggml_tensor * w  = conv->src[1];     // [d_conv, C]
    const ggml_tensor * tail = cpy->src[0];
    const ggml_tensor * sd   = cpy->src[1];
    const int64_t C = concat->ne[1];
    if (ggml_get_op_params_i32(concat, 0) != 0 || concat->ne[0] != 4 || st->ne[0] != 3 || xn->ne[0] != 1 ||
            concat->ne[2] != 1 || concat->ne[3] != 1 || st->ne[1] != C || xn->ne[1] != C ||
            concat->type != GGML_TYPE_F32 || st->type != GGML_TYPE_F32 || xn->type != GGML_TYPE_F32 || w->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(concat) || tail->view_src != concat || tail->view_offs != sizeof(float) || tail->ne[0] != 3 ||
            tail->ne[1] != C || tail->ne[2] != 1 || sd->type != GGML_TYPE_F32 || !ggml_is_contiguous(sd) || ggml_nelements(sd) != 3*C ||
            conv->src[0] != concat || w->ne[0] != 4 || w->ne[1] != C || w->nb[0] != sizeof(float) ||
            conv->ne[0] != C || conv->ne[1] != 1 || conv->ne[2] != 1 || silu->src[0] != conv ||
            silu->type != GGML_TYPE_F32 || !ggml_is_contiguous(silu) || ggml_nelements(silu) != C) {
        return false;
    }
    return true;
}

bool ggml_cuda_op_ssm_conv_decode_fused(ggml_backend_cuda_context & ctx, const ggml_tensor * concat, const ggml_tensor * cpy,
                                        const ggml_tensor * conv, const ggml_tensor * silu) {
    if (!ggml_cuda_ssm_conv_decode_fused_ok(concat, cpy, conv, silu)) {
        return false;
    }
    const ggml_tensor * st = concat->src[0];
    const ggml_tensor * xn = concat->src[1];
    const ggml_tensor * w  = conv->src[1];
    const ggml_tensor * sd = cpy->src[1];
    const int64_t C = concat->ne[1];
    const int nt = 256;
    // the conv state's GET_ROWS may have been skipped, the concat reading the cache row in place
    // (gated_delta_net.cu, ggml_cuda_gdn_gather_*): read it from there too
    const float *   g_base = nullptr;
    const int32_t * g_idx  = nullptr;
    int64_t         g_row  = 0;
    const bool gathered = ggml_cuda_gdn_gather_lookup(concat, &g_base, &g_idx, &g_row);
    ssm_conv_decode_fused_f32<4><<<(unsigned) ((C + nt - 1)/nt), nt, 0, ctx.stream()>>>(
        gathered ? (const char *) g_base : (const char *) st->data, st->nb[0], st->nb[1],
        gathered ? g_idx : nullptr, g_row*(int64_t) sizeof(float), (const char *) xn->data, xn->nb[1],
        (const float *) w->data, w->nb[1], (float *) concat->data, (float *) sd->data, (float *) silu->data, C);
    CUDA_CHECK(cudaGetLastError());
    return true;
}
