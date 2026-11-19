#include "unary-p100.cuh"

// same expression as unary.cu's op_softplus
static __device__ __forceinline__ float op_softplus_p100(float x) {
    return (x > 20.0f) ? x : log1pf(expf(x));
}

/* fused ADD (row bias) -> SOFTPLUS -> MUL (row scale): the delta net's gate, softplus(alpha + dt) * a.
   Each value goes through exactly the unfused kernels' operations (op_add, op_softplus, op_mul), so the
   result is bit-identical. */

static __global__ void add_softplus_mul_f32(const float * x, const float * b, const float * m, float * dst,
                                            const int ne0, const int nrows, const int64_t sx, const int64_t sd) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= ne0*nrows) {
        return;
    }
    const int r = i / ne0, c = i % ne0;
    const float t = x[r*sx + c] + b[c];
    dst[r*sd + c] = op_softplus_p100(t) * m[c];
}

bool ggml_cuda_op_add_softplus_mul(ggml_backend_cuda_context & ctx, ggml_tensor * add, ggml_tensor * sp, ggml_tensor * mul) {
    if (ggml_get_unary_op(sp) != GGML_UNARY_OP_SOFTPLUS) {
        return false;
    }
    const int64_t ne0 = add->ne[0], nrows = ggml_nrows(add);
    auto is_row_vec = [&](const ggml_tensor * t) {
        return t->type == GGML_TYPE_F32 && t->ne[0] == ne0 && ggml_nelements(t) == ne0 && ggml_is_contiguous(t);
    };
    const ggml_tensor * x = add->src[0], * b = add->src[1];
    if (!ggml_are_same_shape(x, add)) {
        std::swap(x, b);
    }
    const ggml_tensor * m = mul->src[0] == sp ? mul->src[1] : mul->src[0];
    // a 2-d view of rows: x and mul contiguous within rows, rows of equal stride
    auto rows2d = [&](const ggml_tensor * t) {
        return t->type == GGML_TYPE_F32 && t->nb[0] == sizeof(float) && ggml_are_same_shape(t, add) &&
            t->nb[2] == t->ne[1]*t->nb[1] && t->nb[3] == t->ne[2]*t->nb[2];
    };
    if (add->type != GGML_TYPE_F32 || !rows2d(x) || !rows2d(mul) || !is_row_vec(b) || !is_row_vec(m) ||
            ne0*nrows > INT_MAX) {
        return false;
    }
    const int n = (int) (ne0*nrows);
    add_softplus_mul_f32<<<(n + 255)/256, 256, 0, ctx.stream()>>>((const float *) x->data, (const float *) b->data,
        (const float *) m->data, (float *) mul->data, (int) ne0, (int) nrows, x->nb[1]/sizeof(float), mul->nb[1]/sizeof(float));
    CUDA_CHECK(cudaGetLastError());
    return true;
}

