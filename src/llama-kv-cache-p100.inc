// included by llama-kv-cache.cpp

bool llama_kv_cache::kq_mask_compact_static() const {
    return n_seq_max == 1 && swa_type == LLAMA_SWA_TYPE_NONE && !hparams.use_alibi && v_cells.size() == 1;
}

bool llama_kv_cache::kq_mask_prefix(const llama_ubatch * ubatch, uint32_t n_kv, bool causal_attn, int32_t * L) const {
    if (!causal_attn && hparams.non_causal_type == LLAMA_NON_CAUSAL_TYPE_SWA_ONLY) {
        causal_attn = swa_type == LLAMA_SWA_TYPE_NONE;
    }
    if (!kq_mask_compact_static() || !causal_attn || ubatch->n_tokens == 0) {
        return false;
    }
    const auto & cells = v_cells[0];
    if (n_kv > cells.size()) {
        return false;
    }
    const llama_pos * cpos = cells.pos_data();
    const bool is_2d = ubatch->is_pos_2d();

    // The mask keeps cell j for token (p, x, y) iff pos_j >= 0 && pos_j <= p, and with M-RoPE not
    // (pos_j == p && (y_j, x_j) > (y, x)) (set_input_kq_mask_impl). If the used cells come first, in
    // non-decreasing position and, within one position, strictly increasing (y, x), every kept set
    // is a prefix of the cells.
    uint32_t n_used = 0;
    while (n_used < n_kv && cpos[n_used] >= 0) {
        if (n_used > 0 && cpos[n_used] < cpos[n_used - 1]) {
            return false;
        }
        if (n_used > 0 && is_2d && cpos[n_used] == cpos[n_used - 1]) {
            const auto & a = cells.ext_get(n_used - 1);
            if (!cells.ext_get(n_used).is_2d_gt(a.x, a.y)) {
                return false;
            }
        }
        n_used++;
    }
    for (uint32_t j = n_used; j < n_kv; ++j) {
        if (cpos[j] >= 0) {
            return false;
        }
    }

    const uint32_t n_tokens = ubatch->n_tokens;
    int64_t L0 = 0;
    for (uint32_t t = 0; t < n_tokens; ++t) {
        if (ubatch->n_seq_id[t] != 1) {
            return false;
        }
        const llama_pos p  = ubatch->pos[t];
        const llama_pos * lo = std::lower_bound(cpos, cpos + n_used, p);
        const llama_pos * hi = std::upper_bound(cpos, cpos + n_used, p);
        int64_t Lt = hi - cpos;
        if (is_2d && hi > lo) {
            const llama_pos py = ubatch->pos[t + n_tokens];
            const llama_pos px = ubatch->pos[t + n_tokens*2];
            Lt = lo - cpos;
            while (cpos + Lt < hi && !cells.ext_get((uint32_t) Lt).is_2d_gt(px, py)) {
                Lt++;
            }
        }
        if (t == 0) {
            L0 = Lt;
        } else if (Lt != L0 + t) {
            return false;
        }
        if (L != nullptr) {
            L[t] = (int32_t) Lt;
        }
    }
    return true;
}
