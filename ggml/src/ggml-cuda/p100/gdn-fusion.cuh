// included by ggml-cuda.cu, uses its static helpers
// get_rows of one recurrent state row that only feeds the state of a single-sequence gated_delta_net decode step;
// the plain recurrence then reads the row from the cache itself (the chunked prefill form needs n_tokens >= 64)
static bool ggml_cuda_gdn_state_rows_ok(const ggml_tensor * rows, const ggml_tensor * gdn) {
    const ggml_tensor * s = gdn->src[5];
    return rows->op == GGML_OP_GET_ROWS && rows->type == GGML_TYPE_F32 && rows->src[0]->type == GGML_TYPE_F32 &&
        rows->src[1]->type == GGML_TYPE_I32 && ggml_nrows(rows) == 1 && rows->src[0]->nb[0] == sizeof(float) &&
        gdn->op == GGML_OP_GATED_DELTA_NET && gdn->type == GGML_TYPE_F32 && s->view_src == rows && s->view_offs == 0 &&
        ggml_is_contiguous(s) && ggml_nelements(s) == ggml_nelements(rows) && gdn->src[2]->ne[3] == 1 &&
        gdn->src[2]->ne[2] <= 32;
}

static void ggml_cuda_gdn_set_state_rows(ggml_cuda_gated_delta_net_fused_cache & cache, const ggml_tensor * rows) {
    cache.state_src        = (const float *) rows->src[0]->data;
    cache.state_ids        = (const int *) rows->src[1]->data;
    cache.state_row_stride = rows->src[0]->nb[1] / sizeof(float);
}

// true if nodes[idx] is not a graph output and all its uses are by nodes[lo..hi]
static bool ggml_cuda_uses_within(const ggml_cgraph * cgraph, int idx, int lo, int hi) {
    const ggml_tensor * t = cgraph->nodes[idx];
    if (t->flags & GGML_TENSOR_FLAG_OUTPUT) {
        return false;
    }
    int n = 0;
    for (int j = lo; j <= hi; ++j) {
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            n += cgraph->nodes[j]->src[s] == t;
        }
    }
    return n == ggml_node_get_use_count(cgraph, idx);
}

// indices of the next n nodes after node_idx that are not views/no-ops (in a small window)
static int ggml_cuda_next_real_nodes(const ggml_cgraph * cgraph, int node_idx, int * idx, int n) {
    int found = 0;
    for (int j = node_idx + 1; j < cgraph->n_nodes && j <= node_idx + 32 && found < n; ++j) {
        if (!ggml_cuda_is_view_or_noop(cgraph->nodes[j])) {
            idx[found++] = j;
        }
    }
    return found;
}

// the memory of a and b overlaps
static bool ggml_cuda_memory_overlaps(const ggml_tensor * a, const ggml_tensor * b) {
    const char * a0 = (const char *) a->data;
    const char * b0 = (const char *) b->data;
    return a0 < b0 + ggml_nbytes(b) && b0 < a0 + ggml_nbytes(a);
}

// The decode pre-ops of gated_delta_net (S_v = 128, one sequence, up to 32 tokens) run inside its kernel:
//   q, k: scale(rms_norm(x)) (build_gdn_l2_norm), state = get_rows(cache)
// RMS_NORM(q) SCALE RMS_NORM(k) SCALE GET_ROWS GATED_DELTA_NET, views between. The gate and beta are computed before
// the chain on their usual path (the models build them first, where the gate/beta mat-vec fusion takes both), so the
// only arithmetic moved into the kernel is the l2 norm, done in the order of the rms_norm kernel (bit-identical).
//
// The submitted version also folded the gate and sigmoid(beta) and ran the alpha and b mat-vecs ahead of the plan.
// The allocator had put b in alpha's buffer (free again after the skipped ADD), so b overwrote alpha before the
// kernel read it: KLD 0.09-0.22 at decode, invisible to prefill perplexity and to a test whose alpha and b are inputs.
// The kernel output still runs ahead of the plan, so it must not overlap anything the kernel reads.
static int ggml_cuda_try_gdn_fusion(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {
    if (cgraph->nodes[i]->op != GGML_OP_RMS_NORM || ggml_cuda_info().devices[cuda_ctx->device].warp_size != 32) {
        return 0;
    }
    constexpr int n_win = 8;
    int idx[n_win];
    idx[0] = i;
    const int n = 1 + ggml_cuda_next_real_nodes(cgraph, i, idx + 1, n_win - 1);
    auto at = [&](int p) { return p < n ? cgraph->nodes[idx[p]] : nullptr; };

    const ggml_tensor * rq   = at(0);
    const ggml_tensor * sq   = at(1);
    const ggml_tensor * rk   = at(2);
    const ggml_tensor * sk   = at(3);
    const ggml_tensor * rows = at(4);
    ggml_tensor *       gdn  = at(5);
    if (!gdn || !rows) {
        return 0;
    }
    const int last = idx[5];

    if (!ggml_cuda_gdn_state_rows_ok(rows, gdn) || gdn->src[2]->ne[0] != 128 || gdn->src[3]->ne[0] != 1) {
        return 0;
    }

    // q and k: scale(rms_norm(x), 1/sqrt(128)) over 128 columns, same eps and scale
    const ggml_tensor * q = rq->src[0];
    const ggml_tensor * k = rk->src[0];
    if (rk->op != GGML_OP_RMS_NORM || sq->op != GGML_OP_SCALE || sk->op != GGML_OP_SCALE || sq->src[0] != rq ||
        sk->src[0] != rk || gdn->src[0] != sq || gdn->src[1] != sk || rq->type != GGML_TYPE_F32 ||
        rk->type != GGML_TYPE_F32 || sq->type != GGML_TYPE_F32 || sk->type != GGML_TYPE_F32 ||
        q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F32 || q->ne[0] != 128 || !ggml_are_same_shape(q, k) ||
        !ggml_are_same_stride(q, k) || !ggml_is_contiguous_rows(q) ||
        ggml_get_op_params_f32(rq, 0) != ggml_get_op_params_f32(rk, 0) ||
        ggml_get_op_params_f32(sq, 0) != ggml_get_op_params_f32(sk, 0) ||
        ggml_get_op_params_f32(sq, 1) != 0.0f || ggml_get_op_params_f32(sk, 1) != 0.0f) {
        return 0;
    }

    for (int j = i; j < last; ++j) {
        if (!ggml_cuda_uses_within(cgraph, j, i + 1, last)) {
            return 0;
        }
    }

    // the kernel output is written ahead of the plan: disjoint from q, k, v, g and beta
    for (int r = 0; r < 5; ++r) {
        const ggml_tensor * t = r == 0 ? q : r == 1 ? k : gdn->src[r];
        if (ggml_cuda_memory_overlaps(gdn, t)) {
            return 0;
        }
    }

    ggml_cuda_gated_delta_net_fused_cache cache;
    const int skip = ggml_cuda_try_gdn_cache_fusion(cgraph, last, cache);
    ggml_cuda_gdn_set_state_rows(cache, rows);
    cache.q        = q;
    cache.k        = k;
    cache.l2_eps   = ggml_get_op_params_f32(rq, 0);
    cache.l2_scale = ggml_get_op_params_f32(sq, 0);
    ggml_cuda_op_gated_delta_net_fused_cache(*cuda_ctx, gdn, cache);
    return last - i + skip;
}
