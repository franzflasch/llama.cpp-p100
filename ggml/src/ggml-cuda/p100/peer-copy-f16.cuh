// included by ggml-cuda.cu, uses its static helpers
// Sets *flag if any element is not exactly representable in f16. Used once, at the first
// tensor-parallel exchange, to decide whether the partial sums may be shipped as f16.
static __global__ void k_probe_f16_exact(const float * __restrict__ x, const int64_t n, int * __restrict__ flag) {
    const int64_t i = blockIdx.x*(int64_t)blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const float v = x[i];
    // NaN also trips this (NaN != NaN), which is the conservative direction
    if (__half2float(__float2half(v)) != v) {
        *flag = 1;
    }
}

// Is this copy a tensor-parallel partial sum that we know came out of the f16 cuBLAS path?
// On pre-Volta the mul_mat epilogue writes f16 and widens it to f32, so such a tensor holds
// exactly-f16 values and can be shipped over PCIe at half the bytes with no loss at all.
// Narrow batches go through mul_mat_vec_q instead, which produces genuine f32, hence the
// ne[1] threshold -- it matches the condition under which the cuBLAS f16 path is taken.
// The compute type is checked per exchange: the one-time probe below only sees the first
// exchange, and a matmul whose weights are F32 or BF16, or that requests GGML_PREC_F32 (GLM4
// does for these very projections), produces genuine f32 even when the first one did not.
static bool ggml_cuda_peer_copy_compressible(const ggml_tensor * src, const ggml_tensor * dst, int cc, bool prefer_fp32_acc) {
    return cc < GGML_CUDA_CC_VOLTA
        && src->op   == GGML_OP_MUL_MAT
        && src->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32
        && src->ne[1] >= 512
        && ggml_cuda_mul_mat_cublas_compute_type(src->src[0], src->src[1], src, cc, prefer_fp32_acc) == GGML_TYPE_F16
        && ggml_is_contiguous(src) && ggml_is_contiguous(dst)
        && ggml_nelements(src) == ggml_nelements(dst)
        && ggml_nelements(src) % 2 == 0
        && fast_fp16_available(cc);
}

#ifndef GGML_CUDA_NO_PEER_COPY
// peer copy on a dedicated copy stream, sent as f16 when the partials are f16-exact
static bool ggml_cuda_p100_peer_copy(ggml_backend_cuda_context * cuda_ctx_src, ggml_backend_cuda_context * cuda_ctx_dst,
        const ggml_tensor * src, ggml_tensor * dst, const int src_physical, const int dst_physical) {
    // Issue the peer copy on a dedicated stream rather than the compute stream, so that
    // opposing copies of a tensor-parallel all-reduce overlap instead of serialising --
    // see the comment on copy_stream in ggml_backend_cuda_context. The copy stream waits
    // on work_event (everything already enqueued on the src compute stream) instead of on
    // the compute stream itself, which is what keeps it independent of waits installed
    // there by the other direction of the exchange.
    ggml_cuda_set_device(cuda_ctx_src->device);

    if (!cuda_ctx_src->copy_event) {
        CUDA_CHECK(cudaEventCreateWithFlags(&cuda_ctx_src->copy_event, cudaEventDisableTiming));
    }

    cudaStream_t copy_stream = cuda_ctx_src->peer_copy_stream();

    if (cuda_ctx_src->work_event) {
        CUDA_CHECK(cudaStreamWaitEvent(copy_stream, cuda_ctx_src->work_event, 0));
    } else {
        // nothing has been recorded yet: fall back to ordering against the compute stream
        cuda_ctx_src->record_work();
        CUDA_CHECK(cudaStreamWaitEvent(copy_stream, cuda_ctx_src->work_event, 0));
    }

    const int  cc_src  = ggml_cuda_info().devices[cuda_ctx_src->device].cc;
    const bool eligible = ggml_cuda_peer_copy_compressible(src, dst, cc_src, ggml_cuda_info().devices[cuda_ctx_src->device].prefer_fp32_acc);

    // One-time probe, on the first eligible exchange. The probe copy itself still goes
    // uncompressed, so a model whose partials are not f16-exact never sees a lossy copy.
    if (eligible && cuda_ctx_src->peer_f16_ok < 0) {
        const int64_t ne = ggml_nelements(src);
        int * flag = (int *) cuda_ctx_src->peer_stage_get(
                ggml_backend_cuda_context::PEER_STAGE_OUT, sizeof(int));
        CUDA_CHECK(cudaMemsetAsync(flag, 0, sizeof(int), copy_stream));
        const int64_t nblocks = (ne + 255) / 256;
        k_probe_f16_exact<<<nblocks, 256, 0, copy_stream>>>((const float *) src->data, ne, flag);
        int host_flag = 1;
        CUDA_CHECK(cudaMemcpyAsync(&host_flag, flag, sizeof(int), cudaMemcpyDeviceToHost, copy_stream));
        CUDA_CHECK(cudaStreamSynchronize(copy_stream));
        cuda_ctx_src->peer_f16_ok = host_flag == 0 ? 1 : 0;
        cuda_ctx_src->peer_stage_cap[ggml_backend_cuda_context::PEER_STAGE_OUT] = 0;
        GGML_LOG_INFO("%s: tensor-parallel partials are %sf16-exact; peer copies %s\n", __func__,
                cuda_ctx_src->peer_f16_ok ? "" : "not ",
                cuda_ctx_src->peer_f16_ok ? "will be sent as f16" : "stay f32");
    }

    if (eligible && cuda_ctx_src->peer_f16_ok == 1) {
        // half the PCIe bytes; the local narrow/widen passes cost ~0.25ms against ~2.1ms
        // of PCIe saved, and are exact in both directions for f16-valued data
        const int64_t ne     = ggml_nelements(src);
        const size_t  nb_f16 = ne * sizeof(half);

        half * stage_src = (half *) cuda_ctx_src->peer_stage_get(
                ggml_backend_cuda_context::PEER_STAGE_OUT, nb_f16);
        ggml_cuda_set_device(cuda_ctx_dst->device);
        half * stage_dst = (half *) cuda_ctx_dst->peer_stage_get(
                ggml_backend_cuda_context::PEER_STAGE_IN, nb_f16);
        if (cuda_ctx_dst->peer_stage_free == nullptr) {
            CUDA_CHECK(cudaEventCreateWithFlags(&cuda_ctx_dst->peer_stage_free, cudaEventDisableTiming));
        }
        ggml_cuda_set_device(cuda_ctx_src->device);

        // do not refill the peer's landing buffer until it has widened the last delivery
        CUDA_CHECK(cudaStreamWaitEvent(copy_stream, cuda_ctx_dst->peer_stage_free, 0));

        ggml_get_to_fp16_cuda(GGML_TYPE_F32)(src->data, stage_src, ne, copy_stream);
        CUDA_CHECK(cudaMemcpyPeerAsync(stage_dst, dst_physical, stage_src, src_physical, nb_f16, copy_stream));
        CUDA_CHECK(cudaEventRecord(cuda_ctx_src->copy_event, copy_stream));

        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx_src->stream(), cuda_ctx_src->copy_event, 0));
        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx_dst->stream(), cuda_ctx_src->copy_event, 0));

        ggml_cuda_set_device(cuda_ctx_dst->device);
        // Deliberately no record_work() here: the widen is enqueued on the dst compute
        // stream behind the wait on copy_event, so advancing dst's work marker would make
        // the *other* direction's copy stream wait on this direction's copy -- reinstating
        // exactly the serialisation that the dedicated copy stream exists to avoid. The
        // widened buffer is consumed by the ADD on the same stream (so ordered), and the
        // next graph compute records a marker that covers it.
        ggml_get_to_fp32_cuda(GGML_TYPE_F16)(stage_dst, (float *) dst->data, ne, cuda_ctx_dst->stream());
        CUDA_CHECK(cudaEventRecord(cuda_ctx_dst->peer_stage_free, cuda_ctx_dst->stream()));
        return true;
    }

    // The destination may still have a reader pending on its compute stream. The tensor-parallel
    // all-reduce lands every exchange in the same per-device reduction buffer
    // (ggml-backend-meta.cpp, bcj.bufs[i_buf]) and folds it in with an ADD queued on the
    // destination stream. This copy stream is otherwise ordered against the source only, so a
    // source that had raced ahead to the next layer could overwrite that buffer before the
    // destination's ADD of the previous layer had read it. The compressed path above guards the
    // same reuse with peer_stage_free.
    // Waiting on the destination's existing work marker is enough: graph_compute records it after
    // every graph, the meta backend's ADD included, and that ADD is enqueued before the next
    // layer's copies are issued. Re-recording it here would also capture this exchange's wait on
    // the opposite copy and serialise the two directions again.
    if (cuda_ctx_dst->work_event == nullptr) {
        ggml_cuda_set_device(cuda_ctx_dst->device);
        cuda_ctx_dst->record_work();
        ggml_cuda_set_device(cuda_ctx_src->device);
    }
    CUDA_CHECK(cudaStreamWaitEvent(copy_stream, cuda_ctx_dst->work_event, 0));

    CUDA_CHECK(cudaMemcpyPeerAsync(dst->data, dst_physical, src->data, src_physical, ggml_nbytes(dst), copy_stream));
    CUDA_CHECK(cudaEventRecord(cuda_ctx_src->copy_event, copy_stream));

    // the src compute stream must not overwrite the source buffer while the copy is in
    // flight -- with the copy on the compute stream this was implicit
    CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx_src->stream(), cuda_ctx_src->copy_event, 0));

    // wait on dst stream for the copy to complete
    CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx_dst->stream(), cuda_ctx_src->copy_event, 0));

    return true;
}
#endif // GGML_CUDA_NO_PEER_COPY
