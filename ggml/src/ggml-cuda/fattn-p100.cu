#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-p100.cuh"
#include "fattn-tile.cuh"
#include "fattn.cuh"

#if GGML_CUDA_P100_FATTN
#include "fattn-gemm.cuh"
#include "fattn-q4p.cuh"

bool ggml_cuda_p100_fattn_tile_q4_0_direct(const ggml_tensor * dst) {
    return ggml_cuda_fattn_tile_q4_0_direct(dst);
}

// cuBLAS-GEMM attention for pre-Volta. On Pascal the tile kernel runs at 18.6% of peak while
// cuBLAS reaches 13-15 TFLOPS at these same shapes; attention is ~86% of prefill at 262144 context.
static bool fattn_gemm_ok(const ggml_tensor * dst, const int device) {
    return ggml_cuda_flash_attn_ext_gemm_supported(dst) && ggml_cuda_info().devices[device].cc < GGML_CUDA_CC_VOLTA;
}
#else
bool ggml_cuda_p100_fattn_tile_q4_0_direct(const ggml_tensor *) {
    return false;
}
#endif // GGML_CUDA_P100_FATTN

// Compact causal mask (I32 [nq], ggml_flash_attn_ext): query t keeps its first L_t keys. Every
// kernel choice and size check is made on the equivalent f16 [nkv x nq] mask descriptor, so
// they answer exactly as for the full mask. The GEMM path reads the compact form directly
// (fattn-gemm.cu); the other kernels get the full mask expanded into pool scratch, which is small
// for them (few queries, or few keys).
static bool fattn_mask_is_compact(const ggml_tensor * dst) {
    return dst->src[3] != nullptr && dst->src[3]->type == GGML_TYPE_I32;
}

static void fattn_mask_f16_view(const ggml_tensor * dst, ggml_tensor & d2, ggml_tensor & m2, void * data) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    d2 = *dst;
    m2 = *dst->src[3];
    m2.type  = GGML_TYPE_F16;
    m2.ne[0] = K->ne[1];
    m2.ne[1] = Q->ne[1];
    m2.ne[2] = 1;
    m2.ne[3] = 1;
    m2.nb[0] = sizeof(half);
    m2.nb[1] = m2.nb[0]*m2.ne[0];
    m2.nb[2] = m2.nb[1]*m2.ne[1];
    m2.nb[3] = m2.nb[2];
    m2.data  = data;
    d2.src[3] = &m2;
}

static __global__ void fattn_mask_expand(const int32_t * __restrict__ L, half * __restrict__ M, const int64_t nkv) {
    const int64_t t = blockIdx.y;
    const int32_t l = L[t];
    for (int64_t j = (int64_t) blockIdx.x*blockDim.x + threadIdx.x; j < nkv; j += (int64_t) gridDim.x*blockDim.x) {
        M[t*nkv + j] = j < l ? __float2half(0.0f) : __float2half(-INFINITY);
    }
}

bool ggml_cuda_p100_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (fattn_mask_is_compact(dst)) {
        ggml_tensor d2, m2;
        fattn_mask_f16_view(dst, d2, m2, nullptr);
#if GGML_CUDA_P100_FATTN
        if (fattn_gemm_ok(&d2, ggml_cuda_get_device())) {
            ggml_cuda_flash_attn_ext_gemm(ctx, dst);   // reads the compact mask itself
            return true;
        }
#endif // GGML_CUDA_P100_FATTN
        const int64_t nkv = m2.ne[0];
        const int64_t nq  = m2.ne[1];
        ggml_cuda_pool_alloc<half> full(ctx.pool(), nkv*nq);
        fattn_mask_expand<<<dim3((unsigned) std::min<int64_t>((nkv + 255)/256, 256), (unsigned) nq), 256, 0, ctx.stream()>>>(
            (const int32_t *) dst->src[3]->data, full.get(), nkv);
        CUDA_CHECK(cudaGetLastError());
        m2.data = full.get();
        ggml_cuda_flash_attn_ext(ctx, &d2);
        return true;
    }

#if GGML_CUDA_P100_FATTN
    if (fattn_gemm_ok(dst, ggml_cuda_get_device())) {
        ggml_cuda_flash_attn_ext_gemm(ctx, dst);
        return true;
    }

    // Few-token attention over a q4_0 cache on pre-Volta (decode and MTP verify).
    if (ggml_cuda_fattn_q4p_supported(dst)) {
        ggml_cuda_flash_attn_ext_q4p(ctx, dst);
        return true;
    }
#endif // GGML_CUDA_P100_FATTN

    return false;
}

bool ggml_cuda_p100_flash_attn_ext_get_alloc_size(int device, const ggml_tensor * dst, size_t & size) {
    if (fattn_mask_is_compact(dst)) {
        ggml_tensor d2, m2;
        fattn_mask_f16_view(dst, d2, m2, nullptr);
        size = ggml_cuda_flash_attn_ext_get_alloc_size(device, &d2);
        return true;
    }

#if GGML_CUDA_P100_FATTN
    // The GEMM path dequantizes K/V one chunk at a time out of its own pool scratch, so it
    // needs none of the whole-cache f16 staging. That staging is 512 MiB per GPU at 262144
    // context (2 of 4 KV heads x 256 dim x 262144 positions x 2 bytes, K and V), which is ~58k
    // tokens of q4_0 KV cache -- so this early-out is the point of the path as much as the speed is.
    if (fattn_gemm_ok(dst, device)) {
        size = ggml_nbytes(dst);
        return true;
    }

    // The q4_0 decode/verify kernel reads the quantized cache directly, so it needs no staging.
    if (ggml_cuda_fattn_q4p_supported(dst)) {
        size = ggml_nbytes(dst);
        return true;
    }
#endif // GGML_CUDA_P100_FATTN

    return false;
}

bool ggml_cuda_p100_flash_attn_ext_supported(int device, const ggml_tensor * dst, bool & supported) {
    if (fattn_mask_is_compact(dst)) {
        ggml_tensor d2, m2;
        fattn_mask_f16_view(dst, d2, m2, nullptr);
        supported = ggml_cuda_flash_attn_ext_supported(device, &d2);
        return true;
    }
    return false;
}
