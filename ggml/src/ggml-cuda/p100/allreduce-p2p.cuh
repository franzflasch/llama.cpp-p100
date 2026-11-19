// included by ggml-cuda.cu, uses its static helpers
// Two-GPU AllReduce of a small f32 tensor in one kernel per GPU over direct P2P loads/stores
// (the approach of ik_llama.cpp's reduce.cu). GPU j owns half j of the elements: it reads both
// partials, adds them as t0 + t1 in fp32 -- the same sum, bit for bit, as the butterfly's
// ADD(own, peer copy) on either GPU, since fp32 addition commutes -- and writes the sum into both
// GPUs' tensors. No peer memcpy, no ADD subgraph: two launches and four events per exchange
// instead of two copies, two copy-stream waits and two one-node graphs. The halves are disjoint,
// so the only ordering needed is "both partials done" before and "both halves written" after.
// Measured on 2x P100-PCIe (PHB): a 5-token exchange costs ~0 us over a local-only kernel, the
// memcpyPeer + ADD path ~10 us. GGML_CUDA_AR_P2P=0 turns it off.
static __global__ void k_ar_p2p_f32(float * __restrict__ t0, float * __restrict__ t1, const int64_t lo, const int64_t hi) {
    const int64_t i = lo + (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i < hi) {
        const float s = t0[i] + t1[i];
        t0[i] = s;
        t1[i] = s;
    }
}

// own[i] = own[i] + (float) peer[i]: the butterfly's widen + ADD(own, tmp) in one pass, bit for bit
static __global__ void k_add_f16_into_f32(float * __restrict__ own, const half * __restrict__ peer, const int64_t n) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) {
        own[i] = own[i] + __half2float(peer[i]);
    }
}

static bool ggml_cuda_peer_copy_compressible(const ggml_tensor * src, const ggml_tensor * dst, int cc, bool prefer_fp32_acc);

// Large (prefill) two-GPU exchange whose source matmul ran in token chunks (xchg, common.cuh): each
// chunk is narrowed to f16 and sent as soon as its event fires, on the copy stream, while the compute
// stream runs the next chunk. Same wire format and same sum as the butterfly's compressed peer copy
// + ADD; only the overlap differs.
static bool ggml_cuda_allreduce_chunked(ggml_backend_cuda_context * ctx[2], struct ggml_tensor ** tensors, const int64_t ne) {
    for (int j = 0; j < 2; ++j) {
        const auto & xc = ctx[j]->xchg;
        const int cc = ggml_cuda_info().devices[ctx[j]->device].cc;
        if (xc.data != tensors[j]->data || xc.n < 2 || ctx[j]->peer_f16_ok != 1 ||
                xc.n != ctx[1 - j]->xchg.n || xc.rows*xc.col[xc.n] != ne ||
                !ggml_cuda_peer_copy_compressible(tensors[j], tensors[j], cc, ggml_cuda_info().devices[ctx[j]->device].prefer_fp32_acc)) {
            return false;
        }
        for (int c = 0; c <= xc.n; ++c) {
            if (xc.col[c] != ctx[1 - j]->xchg.col[c]) {
                return false;
            }
        }
    }
    const size_t nb16 = ne*sizeof(half);
    half * stage_out[2];
    half * stage_in[2];
    for (int j = 0; j < 2; ++j) {
        ggml_cuda_set_device(ctx[j]->device);
        stage_out[j] = (half *) ctx[j]->peer_stage_get(ggml_backend_cuda_context::PEER_STAGE_OUT, nb16);
        stage_in[j]  = (half *) ctx[j]->peer_stage_get(ggml_backend_cuda_context::PEER_STAGE_IN,  nb16);
        if (ctx[j]->peer_stage_free == nullptr) {
            CUDA_CHECK(cudaEventCreateWithFlags(&ctx[j]->peer_stage_free, cudaEventDisableTiming));
        }
        if (ctx[j]->copy_event == nullptr) {
            CUDA_CHECK(cudaEventCreateWithFlags(&ctx[j]->copy_event, cudaEventDisableTiming));
        }
    }
    for (int j = 0; j < 2; ++j) {
        const int o = 1 - j;
        const auto & xc = ctx[j]->xchg;
        const int pj = ggml_cuda_get_physical_device(ctx[j]->device);
        const int po = ggml_cuda_get_physical_device(ctx[o]->device);
        ggml_cuda_set_device(ctx[j]->device);
        cudaStream_t cs = ctx[j]->peer_copy_stream();
        // the peer's landing buffer must have been consumed by its previous exchange
        CUDA_CHECK(cudaStreamWaitEvent(cs, ctx[o]->peer_stage_free, 0));
        const float * src = (const float *) tensors[j]->data;
        for (int c = 0; c < xc.n; ++c) {
            const int64_t off = xc.col[c]*xc.rows;
            const int64_t cnt = (xc.col[c + 1] - xc.col[c])*xc.rows;
            CUDA_CHECK(cudaStreamWaitEvent(cs, xc.ev[c], 0));
            ggml_get_to_fp16_cuda(GGML_TYPE_F32)(src + off, stage_out[j] + off, cnt, cs);
            CUDA_CHECK(cudaMemcpyPeerAsync(stage_in[o] + off, po, stage_out[j] + off, pj, cnt*sizeof(half), cs));
        }
        CUDA_CHECK(cudaEventRecord(ctx[j]->copy_event, cs));
    }
    for (int j = 0; j < 2; ++j) {
        ggml_cuda_set_device(ctx[j]->device);
        cudaStream_t st = ctx[j]->stream();
        ggml_cuda_gemm_fold_prefetch(*ctx[j]);                             // fill the wait (gemm-fold.cuh)
        CUDA_CHECK(cudaStreamWaitEvent(st, ctx[1 - j]->copy_event, 0));   // the peer's partial has landed
        CUDA_CHECK(cudaStreamWaitEvent(st, ctx[j]->copy_event, 0));       // our own partial has been read
        k_add_f16_into_f32<<<(unsigned) ((ne + 255)/256), 256, 0, st>>>((float *) tensors[j]->data, stage_in[j], ne);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(ctx[j]->peer_stage_free, st));
        ctx[j]->record_work();
        ctx[j]->xchg.data = nullptr;
    }
    return true;
}

struct ggml_cuda_ar_p2p_state {
    int         ok = -1; // -1 unknown, 0 unusable, 1 usable
    cudaEvent_t ready[2] = {nullptr, nullptr};
    cudaEvent_t done[2]  = {nullptr, nullptr};
};

static bool ggml_backend_cuda_comm_try_allreduce_p2p(
        ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors) {
    static ggml_cuda_ar_p2p_state st; // one comm context per process in practice; devices checked below
    if (comm_ctx->backends.size() != 2) {
        return false;
    }
    ggml_backend_cuda_context * ctx[2] = {
        (ggml_backend_cuda_context *) comm_ctx->backends[0]->context,
        (ggml_backend_cuda_context *) comm_ctx->backends[1]->context,
    };
    if (st.ok < 0) {
        st.ok = 0;
        const int p0 = ggml_cuda_get_physical_device(ctx[0]->device);
        const int p1 = ggml_cuda_get_physical_device(ctx[1]->device);
        int can01 = 0, can10 = 0;
        if (p0 != p1) {
            CUDA_CHECK(cudaDeviceCanAccessPeer(&can01, p0, p1));
            CUDA_CHECK(cudaDeviceCanAccessPeer(&can10, p1, p0));
        }
        // peer access is only enabled with GGML_CUDA_P2P (ggml_cuda_init)
        if (getenv("GGML_CUDA_P2P") != nullptr && p0 != p1 && can01 && can10) {
            for (int j = 0; j < 2; ++j) {
                ggml_cuda_set_device(ctx[j]->device);
                CUDA_CHECK(cudaEventCreateWithFlags(&st.ready[j], cudaEventDisableTiming));
                CUDA_CHECK(cudaEventCreateWithFlags(&st.done[j],  cudaEventDisableTiming));
            }
            st.ok = 1;
            GGML_LOG_INFO("%s: small tensor-parallel exchanges use the one-kernel P2P AllReduce\n", __func__);
        }
    }
    if (st.ok != 1) {
        return false;
    }

    const int64_t ne = ggml_nelements(tensors[0]);
    for (int j = 0; j < 2; ++j) {
        const ggml_tensor * t = tensors[j];
        if (t == nullptr || t->type != GGML_TYPE_F32 || ggml_nelements(t) != ne || !ggml_is_contiguous(t) ||
                (t->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            return false;
        }
    }
    // large (prefill) exchanges: chunked and overlapped when the source matmul was chunked, else the
    // butterfly with its f16 wire format
    if (ne == 0) {
        return false;
    }
    if (ne*sizeof(float) > 256*1024) {
        return ggml_cuda_allreduce_chunked(ctx, tensors, ne);
    }

    float * d[2] = {(float *) tensors[0]->data, (float *) tensors[1]->data};
    const int64_t half = (ne + 1) / 2;

    for (int j = 0; j < 2; ++j) {
        ggml_cuda_set_device(ctx[j]->device);
        CUDA_CHECK(cudaEventRecord(st.ready[j], ctx[j]->stream()));
    }
    for (int j = 0; j < 2; ++j) {
        ggml_cuda_set_device(ctx[j]->device);
        CUDA_CHECK(cudaStreamWaitEvent(ctx[j]->stream(), st.ready[1 - j], 0));
        const int64_t lo = j == 0 ? 0 : half;
        const int64_t hi = j == 0 ? half : ne;
        if (hi > lo) {
            const int nb = (int) ((hi - lo + 255) / 256);
            k_ar_p2p_f32<<<nb, 256, 0, ctx[j]->stream()>>>(d[0], d[1], lo, hi);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaEventRecord(st.done[j], ctx[j]->stream()));
    }
    for (int j = 0; j < 2; ++j) {
        ggml_cuda_set_device(ctx[j]->device);
        CUDA_CHECK(cudaStreamWaitEvent(ctx[j]->stream(), st.done[1 - j], 0));
        // the peer copy path orders its copy stream against work_event; keep it current
        ctx[j]->record_work();
    }
    return true;
}
