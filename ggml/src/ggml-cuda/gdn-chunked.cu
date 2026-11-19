// Alternative gated delta net paths for prefill (scalar gate, S_v = 128), kept out of
// gated_delta_net.cu so a change here rebuilds one small translation unit.
// Chunked form (gdn-chunked.cuh), all fp32.
#include "common.cuh"

#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>

namespace gdn_f32 {
#include "gdn-chunked.cuh"
}

struct ggml_cuda_gdn_alt_args {
    const float * q; const float * k; const float * v; const float * g; const float * beta;
    const float * curr_state; float * dst; float * state;
    int64_t H, n_tokens, n_seqs, neqk1, rq3;
    int64_t sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3;
    float scale; int64_t state_slot_stride; int K; const int32_t * s_idx; int64_t s_row;
};

// returns false when the shipped kernel should run
bool ggml_cuda_gdn_alt(ggml_backend_cuda_context & ctx, const ggml_cuda_gdn_alt_args & a) {
    cudaStream_t stream = ctx.stream();
    const bool keep = a.K > 1;
    const int64_t L = keep ? a.n_tokens - (a.K - 1) : a.n_tokens;
    if (L < GCH) {
        return false;
    }
    // float4 loads need 16-byte aligned rows
    if ((((uintptr_t) a.q | (uintptr_t) a.k | (uintptr_t) a.v) % 16) != 0 ||
        ((a.sq1 | a.sq2 | a.sq3 | a.sv1 | a.sv2 | a.sv3) % 4) != 0) {
        return false;
    }
    const size_t n = gdn_f32::gdn_ch_scratch_floats(a.H, a.neqk1, a.n_seqs, L);
    ggml_cuda_pool_alloc<float> scratch(ctx.pool(), n);
    gdn_f32::gdn_ch_params p = {};
    p.q = a.q; p.k = a.k; p.v = a.v; p.g = a.g; p.beta = a.beta;
    p.curr_state = a.curr_state; p.dst = a.dst; p.state = a.state;
    p.H = a.H; p.n_tokens = a.n_tokens; p.n_seqs = a.n_seqs; p.neqk1 = a.neqk1; p.rq3 = a.rq3; p.K = a.K;
    p.sq1 = a.sq1; p.sq2 = a.sq2; p.sq3 = a.sq3; p.sv1 = a.sv1; p.sv2 = a.sv2; p.sv3 = a.sv3;
    p.sb1 = a.sb1; p.sb2 = a.sb2; p.sb3 = a.sb3;
    p.scale = a.scale; p.state_slot_stride = a.state_slot_stride; p.s_idx = a.s_idx; p.s_row = a.s_row;
    p.keep = keep;
    gdn_f32::gdn_ch_launch(p, scratch.get(), stream);
    CUDA_CHECK(cudaGetLastError());
    return true;
}
