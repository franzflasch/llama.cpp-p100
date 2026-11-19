// P100 test cases, included by test-backend-ops.cpp

// GGML_OP_ADD + GGML_OP_RMS_NORM + GGML_OP_MUL, the sum also an output (the residual stream; fused
// on CUDA by add_rms_norm_mul_f32)
struct test_add_rms_norm_mul_p100 : public test_case {
    const std::array<int64_t, 2> ne;
    const float eps;

    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return "ADD_RMS_NORM_MUL";
    }

    bool run_whole_graph() override { return true; }

    std::string vars() override {
        return VARS_TO_STR2(ne, eps);
    }

    test_add_rms_norm_mul_p100(std::array<int64_t, 2> ne = {5120, 5}, float eps = 1e-6f) : ne(ne), eps(eps) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne[0], ne[1]);
        ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne[0], ne[1]);
        ggml_tensor * w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne[0]);
        ggml_set_name(a, "a");
        ggml_set_name(b, "b");
        ggml_set_name(w, "w");
        ggml_tensor * sum = ggml_add(ctx, a, b);
        ggml_set_name(sum, "sum");
        ggml_set_output(sum);
        ggml_tensor * out = ggml_mul(ctx, ggml_rms_norm(ctx, sum, eps), w);
        ggml_set_name(out, "out");
        return out;
    }
};

// the gated delta net's gate: softplus(x + dt) * a, dt and a row vectors (fused on CUDA)
struct test_add_softplus_mul : public test_case {
    const std::array<int64_t, 2> ne;

    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return "ADD_SOFTPLUS_MUL";
    }

    bool run_whole_graph() override { return true; }

    std::string vars() override {
        return VARS_TO_STR1(ne);
    }

    test_add_softplus_mul(std::array<int64_t, 2> ne = {24, 5}) : ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * x  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne[0], ne[1]);
        ggml_tensor * dt = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne[0]);
        ggml_tensor * a  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne[0]);
        ggml_tensor * out = ggml_mul(ctx, ggml_softplus(ctx, ggml_add(ctx, x, dt)), a);
        ggml_set_name(out, "out");
        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            init_tensor_uniform(t, -30.f, 30.f); // both softplus branches
        }
    }
};

// the gated delta net's alpha/beta projections and gates (one launch on Pascal, mmvq-f16.cu):
// gate = softplus(W_a x + dt) * a, beta = sigmoid(W_b x)
struct test_gdn_gate : public test_case {
    const int64_t rows, n, k;

    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return "GDN_GATE";
    }

    bool run_whole_graph() override { return true; }

    std::string vars() override {
        return VARS_TO_STR3(rows, n, k);
    }

    double max_nmse_err() override {
        return 5e-4;
    }

    test_gdn_gate(int64_t rows = 24, int64_t n = 5, int64_t k = 5120) : rows(rows), n(n), k(k) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * wa = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, k, rows);
        ggml_tensor * wb = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, k, rows);
        ggml_tensor * x  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n);
        ggml_tensor * dt = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, rows);
        ggml_tensor * a  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, rows);
        ggml_tensor * alpha = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, wa, x), rows, n, 1);
        ggml_tensor * gate  = ggml_mul(ctx, ggml_softplus(ctx, ggml_add(ctx, alpha, dt)), a);
        ggml_tensor * beta  = ggml_sigmoid(ctx, ggml_reshape_3d(ctx, ggml_mul_mat(ctx, wb, x), rows, n, 1));
        ggml_tensor * out = ggml_concat(ctx, gate, beta, 0);
        ggml_set_name(out, "out");
        return out;
    }
};

// the gated delta net's q and k l2 norms, two views of one tensor (one launch on CUDA)
struct test_rms_norm_scale2 : public test_case {
    const int64_t d, h, t;

    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return "RMS_NORM_SCALE2";
    }

    bool run_whole_graph() override { return true; }

    std::string vars() override {
        return VARS_TO_STR3(d, h, t);
    }

    test_rms_norm_scale2(int64_t d = 128, int64_t h = 8, int64_t t = 5) : d(d), h(h), t(t) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d*h*3, t); // q, k, v per token
        const float sc = 1.0f/sqrtf((float) d);
        ggml_tensor * q = ggml_view_3d(ctx, x, d, h, t, d*sizeof(float), x->nb[1], 0);
        ggml_tensor * k = ggml_view_3d(ctx, x, d, h, t, d*sizeof(float), x->nb[1], d*h*sizeof(float));
        q = ggml_scale(ctx, ggml_rms_norm(ctx, q, 1e-6f/d), sc);
        k = ggml_scale(ctx, ggml_rms_norm(ctx, k, 1e-6f/d), sc);
        ggml_tensor * out = ggml_add(ctx, q, k);
        ggml_set_name(out, "out");
        return out;
    }
};

// the conv state update: CONCAT(state, x^T) along dim 0, CPY of its last columns (fused on CUDA)
struct test_concat_cpy : public test_case {
    const int64_t kc, nt, ch;

    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return "CONCAT_CPY";
    }

    bool run_whole_graph() override { return true; }

    std::string vars() override {
        return VARS_TO_STR3(kc, nt, ch);
    }

    test_concat_cpy(int64_t kc = 3, int64_t nt = 5, int64_t ch = 512) : kc(kc), nt(nt), ch(ch) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * st = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kc, ch, 1);
        ggml_tensor * x  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ch, nt);
        ggml_tensor * c  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kc*ch, 1);
        ggml_tensor * conc = ggml_concat(ctx, st, ggml_transpose(ctx, x), 0);
        ggml_tensor * last = ggml_view_3d(ctx, conc, kc, ch, 1, conc->nb[1], conc->nb[2], ggml_row_size(conc->type, nt));
        ggml_tensor * t = ggml_cpy(ctx, last, c);
        ggml_tensor * out = ggml_concat(ctx, ggml_reshape_1d(ctx, t, kc*ch), ggml_reshape_1d(ctx, conc, (kc + nt)*ch), 0);
        ggml_set_name(out, "out");
        return out;
    }
};

// qwen35's gated norm: RMS_NORM -> MUL(w), then MUL_MAT(z) -> SILU -> MUL (one norm launch on CUDA)
struct test_norm_gate : public test_case {
    const int64_t d, h, t;

    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return "NORM_GATE";
    }

    bool run_whole_graph() override { return true; }

    std::string vars() override {
        return VARS_TO_STR3(d, h, t);
    }

    test_norm_gate(int64_t d = 128, int64_t h = 24, int64_t t = 5) : d(d), h(h), t(t) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * x  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d, h, t);
        ggml_tensor * w  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d);
        ggml_tensor * wz = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, d*h);
        ggml_tensor * a  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, t);
        ggml_tensor * nrm = ggml_mul(ctx, ggml_rms_norm(ctx, x, 1e-6f), w);
        ggml_tensor * z   = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, wz, a), d, h, t);
        ggml_tensor * out = ggml_mul(ctx, nrm, ggml_silu(ctx, z));
        ggml_set_name(out, "out");
        return out;
    }
};

// the gated delta net's l2 norm: RMS_NORM -> SCALE (fused on CUDA)
struct test_rms_norm_scale : public test_case {
    const std::array<int64_t, 4> ne;
    const float scale;

    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return "RMS_NORM_SCALE";
    }

    bool run_whole_graph() override { return true; }

    std::string vars() override {
        return VARS_TO_STR2(ne, scale);
    }

    test_rms_norm_scale(std::array<int64_t, 4> ne = {128, 8, 5, 1}, float scale = 0.0883883f) : ne(ne), scale(scale) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * x = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne.data());
        ggml_tensor * out = ggml_scale(ctx, ggml_rms_norm(ctx, x, 1e-6f/ne[0]), scale);
        ggml_set_name(out, "out");
        return out;
    }
};


// one gated delta net decode layer as qwen35 builds it (conv state, conv, l2 norms, gate, state read, gated norm),
// in the same node order, so the CUDA backend can fuse the chain; both caches are read at row `row` and written at row `head`
struct test_gdn_decode_fusion : public test_case {
    const int64_t H_k;
    const int64_t H_v;
    const int64_t S;
    const int64_t n_t;
    const int64_t n_seqs;
    const int     row;
    const int     head;
    const ggml_type mm_type;
    const int64_t n_pend; // rs_replay: pending tokens per cell (0: no replay)
    const bool    keep;

    ggml_tensor * conv_cpy  = nullptr;
    ggml_tensor * state_cpy = nullptr;
    ggml_tensor * out       = nullptr;
    ggml_tensor * pend_cpy[4] = {};

    static constexpr int64_t d_conv  = 4;
    static constexpr int64_t n_cells = 3;

    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return "GDN_DECODE_FUSION";
    }

    bool run_whole_graph() override { return true; }
    std::vector<ggml_tensor *> fusion_test_nodes() override {
        std::vector<ggml_tensor *> nodes = { out, conv_cpy, state_cpy };
        for (ggml_tensor * t : pend_cpy) {
            if (t) {
                nodes.push_back(t);
            }
        }
        return nodes;
    }

    // quantized alpha/b mat-vecs: CPU and CUDA quantize the activations differently
    double max_nmse_err() override { return mm_type == GGML_TYPE_COUNT || mm_type == GGML_TYPE_F32 ? 1e-7 : 5e-4; }

    std::string vars() override {
        return VARS_TO_STR7(H_k, H_v, S, n_t, n_seqs, row, head) + (mm_type == GGML_TYPE_COUNT ? std::string() : ",mm=" + std::string(ggml_type_name(mm_type))) +
            (n_pend == 0 ? std::string() : ",n_pend=" + std::to_string(n_pend) + ",keep=" + std::to_string(keep));
    }

    test_gdn_decode_fusion(int64_t H_k = 4, int64_t H_v = 12, int64_t S = 128, int64_t n_t = 1, int64_t n_seqs = 1,
            int row = 1, int head = 1, ggml_type mm_type = GGML_TYPE_COUNT, int64_t n_pend = 0, bool keep = false)
        : H_k(H_k), H_v(H_v), S(S), n_t(n_t), n_seqs(n_seqs), row(row), head(head), mm_type(mm_type), n_pend(n_pend), keep(keep) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        const int64_t nr = 2*H_k*S + H_v*S; // conv channels
        const float eps = 1e-6f;

        ggml_tensor * x          = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, nr, n_t, n_seqs);
        ggml_tensor * conv_w     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_conv, nr);
        ggml_tensor * conv_cache = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (d_conv - 1)*nr, n_cells);
        ggml_tensor * ssm_cache  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, S*S*H_v, n_cells);
        ggml_tensor * ids        = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_seqs);
        ggml_tensor * alpha;
        ggml_tensor * b;
        if (mm_type != GGML_TYPE_COUNT) {
            // as in the models: alpha and b are mat-vecs of the layer input
            const int64_t n_embd = 256;
            ggml_tensor * cur = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_t*n_seqs);
            ggml_tensor * w_a = ggml_new_tensor_2d(ctx, mm_type, n_embd, H_v);
            ggml_tensor * w_b = ggml_new_tensor_2d(ctx, mm_type, n_embd, H_v);
            ggml_set_name(cur, "cur");
            b     = ggml_reshape_4d(ctx, ggml_mul_mat(ctx, w_b, cur), 1, H_v, n_t, n_seqs);
            alpha = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, w_a, cur), H_v, n_t, n_seqs);
        } else {
            alpha = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, H_v, n_t, n_seqs);
            b     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H_v, n_t, n_seqs);
        }
        ggml_tensor * dt         = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H_v);
        ggml_tensor * A          = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H_v);
        ggml_tensor * z          = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H_v, n_t, n_seqs);
        ggml_tensor * norm_w     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, S);
        ggml_set_name(x, "x");
        ggml_set_name(conv_w, "conv_w");
        ggml_set_name(conv_cache, "conv_cache");
        ggml_set_name(ssm_cache, "ssm_cache");
        ggml_set_name(ids, "ids");
        ggml_set_name(alpha, "alpha");
        ggml_set_name(b, "b");
        ggml_set_name(dt, "dt");
        ggml_set_name(A, "A");
        ggml_set_name(z, "z");
        ggml_set_name(norm_w, "norm_w");

        // conv state: gather, append the new tokens, store the last d_conv - 1 columns, conv + silu
        ggml_tensor * conv_states = ggml_reshape_3d(ctx, ggml_get_rows(ctx, conv_cache, ids), d_conv - 1, nr, n_seqs);
        ggml_tensor * conv_input  = ggml_concat(ctx, conv_states, ggml_transpose(ctx, x), 0);
        ggml_tensor * conv_last   = ggml_view_3d(ctx, conv_input, d_conv - 1, nr, n_seqs, conv_input->nb[1], conv_input->nb[2],
                ggml_row_size(conv_input->type, n_t));
        conv_cpy = ggml_cpy(ctx, conv_last, ggml_view_2d(ctx, conv_cache, (d_conv - 1)*nr, n_seqs, conv_cache->nb[1], head*conv_cache->nb[1]));
        if (mode == MODE_TEST) {
            ggml_build_forward_expand(gf, conv_cpy);
        }
        ggml_tensor * conv = ggml_silu(ctx, ggml_ssm_conv(ctx, conv_input, conv_w));
        if (mode == MODE_TEST) {
            ggml_build_forward_expand(gf, conv);
        }

        const size_t nb1 = ggml_row_size(conv->type, nr);
        ggml_tensor * q = ggml_view_4d(ctx, conv, S, H_k, n_t, n_seqs, ggml_row_size(conv->type, S), nb1, nb1*n_t, 0);
        ggml_tensor * k = ggml_view_4d(ctx, conv, S, H_k, n_t, n_seqs, ggml_row_size(conv->type, S), nb1, nb1*n_t,
                ggml_row_size(conv->type, H_k*S));
        ggml_tensor * v = ggml_view_4d(ctx, conv, S, H_v, n_t, n_seqs, ggml_row_size(conv->type, S), nb1, nb1*n_t,
                ggml_row_size(conv->type, 2*H_k*S));

        // l2 norms as build_gdn_l2_norm, gate and beta
        q = ggml_scale(ctx, ggml_rms_norm(ctx, q, eps/S), 1.0f/sqrtf(S));
        k = ggml_scale(ctx, ggml_rms_norm(ctx, k, eps/S), 1.0f/sqrtf(S));
        ggml_tensor * g = ggml_mul(ctx, ggml_softplus(ctx, ggml_add(ctx, alpha, dt)), A);
        g = ggml_reshape_4d(ctx, g, 1, H_v, n_t, n_seqs);
        ggml_tensor * beta = ggml_sigmoid(ctx, b);
        if (mode == MODE_TEST) {
            ggml_build_forward_expand(gf, g); // the models build gate and beta first
            ggml_build_forward_expand(gf, beta);
            ggml_build_forward_expand(gf, q);
            ggml_build_forward_expand(gf, k);
        }

        // state read right before the delta net, new state stored back
        ggml_tensor * s   = ggml_reshape_4d(ctx, ggml_get_rows(ctx, ssm_cache, ids), S, S, H_v, n_seqs);

        // rs_replay as build_recurrent_attn_replay: pending k, v, g/beta per cell, gathered after the state
        ggml_tensor * ck = nullptr, * cv = nullptr, * cgb = nullptr;
        ggml_tensor * pk = nullptr, * pv = nullptr, * pgb = nullptr, * rep_n = nullptr;
        if (n_pend > 0) {
            ggml_tensor * spk  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, S*H_k*n_pend, n_cells);
            ggml_tensor * spv  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, S*H_v*n_pend, n_cells);
            ggml_tensor * spgb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H_v*2*n_pend, n_cells);
            rep_n = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_seqs);
            ggml_set_name(spk, "spk");
            ggml_set_name(spv, "spv");
            ggml_set_name(spgb, "spgb");
            ggml_set_name(rep_n, "rep_n");
            ck  = ggml_reshape_4d(ctx, spk,  S, H_k, n_pend,   n_cells);
            cv  = ggml_reshape_4d(ctx, spv,  S, H_v, n_pend,   n_cells);
            cgb = ggml_reshape_4d(ctx, spgb, 1, H_v, 2*n_pend, n_cells);
            pk  = ggml_reshape_4d(ctx, ggml_get_rows(ctx, spk,  ids), S, H_k, n_pend,   n_seqs);
            pv  = ggml_reshape_4d(ctx, ggml_get_rows(ctx, spv,  ids), S, H_v, n_pend,   n_seqs);
            pgb = ggml_reshape_4d(ctx, ggml_get_rows(ctx, spgb, ids), 1, H_v, 2*n_pend, n_seqs);
        }

        ggml_tensor * gdn = ggml_gated_delta_net_ext(ctx, q, k, v, g, beta, s, pk, pv, pgb, rep_n, keep);
        ggml_tensor * o   = ggml_view_4d(ctx, gdn, S, H_v, n_t, n_seqs, ggml_row_size(gdn->type, S),
                ggml_row_size(gdn->type, S*H_v), ggml_row_size(gdn->type, S*H_v*n_t), 0);
        ggml_tensor * new_state = ggml_view_2d(ctx, gdn, S*S*H_v, n_seqs, ggml_row_size(gdn->type, S*S*H_v),
                ggml_row_size(gdn->type, S*H_v*n_t*n_seqs));
        state_cpy = ggml_cpy(ctx, new_state, ggml_view_2d(ctx, ssm_cache, S*S*H_v, n_seqs, ssm_cache->nb[1], head*ssm_cache->nb[1]));
        if (mode == MODE_TEST) {
            ggml_build_forward_expand(gf, state_cpy);
        }

        // store the inputs of the new tokens as pending (the first n_pend of them)
        if (n_pend > 0) {
            const int64_t n_st = std::min(n_t, n_pend);
            ggml_tensor * xs[4] = { k, v, g, beta };
            ggml_tensor * cs[4] = { ck, cv, cgb, cgb };
            for (int i = 0; i < 4; ++i) {
                ggml_tensor * x  = xs[i];
                ggml_tensor * c  = cs[i];
                const int64_t t0 = i == 3 ? n_pend : 0;
                ggml_tensor * x_st = ggml_view_4d(ctx, x, x->ne[0], x->ne[1], n_st, n_seqs, x->nb[1], x->nb[2], x->nb[3], 0);
                ggml_tensor * dst  = ggml_view_4d(ctx, c, x->ne[0], x->ne[1], n_st, n_seqs, c->nb[1], c->nb[2], c->nb[3],
                        head*c->nb[3] + t0*c->nb[2]);
                pend_cpy[i] = ggml_cpy(ctx, x_st, dst);
                if (mode == MODE_TEST) {
                    ggml_build_forward_expand(gf, pend_cpy[i]);
                }
            }
        }

        // gated norm
        out = ggml_mul(ctx, ggml_mul(ctx, ggml_rms_norm(ctx, o, eps), norm_w), ggml_silu(ctx, z));
        ggml_set_name(out, "out");
        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
            if (ggml_is_view_op(t->op)) { continue; }
            if (strcmp(t->name, "ids") == 0) {
                std::vector<int32_t> data(n_seqs);
                for (int64_t i = 0; i < n_seqs; i++) {
                    data[i] = (row + i) % n_cells;
                }
                ggml_backend_tensor_set(t, data.data(), 0, n_seqs * sizeof(int32_t));
            } else if (strcmp(t->name, "rep_n") == 0) {
                std::vector<int32_t> data(n_seqs);
                for (int64_t i = 0; i < n_seqs; i++) {
                    data[i] = (int32_t) ((n_pend - i) % (n_pend + 1));
                }
                ggml_backend_tensor_set(t, data.data(), 0, n_seqs * sizeof(int32_t));
            } else if (strcmp(t->name, "A") == 0) {
                init_tensor_uniform(t, -2.0f, -0.1f);
            } else if (strcmp(t->name, "ssm_cache") == 0) {
                init_tensor_uniform(t, -0.1f, 0.1f);
            } else {
                init_tensor_uniform(t);
            }
        }
    }
};

struct ggml_type_pair { ggml_type a, b; };

static void p100_add_test_cases_eval(std::vector<std::unique_ptr<test_case>> & test_cases) {
    for (int64_t n : {1024, 5120, 5121, 8192}) {
        for (int64_t r : {1, 5, 7}) {
            test_cases.emplace_back(new test_add_rms_norm_mul_p100({n, r}));
        }
    }
    for (int64_t n : {1, 2, 3, 5, 8, 10, 16}) {
        test_cases.emplace_back(new test_gdn_gate(24, n, 5120));
        test_cases.emplace_back(new test_gdn_gate(20, n, 3072));
    }
    for (int64_t r : {1, 5, 300}) {
        test_cases.emplace_back(new test_add_softplus_mul({24, r}));
        test_cases.emplace_back(new test_add_softplus_mul({48, r}));
    }
    for (int64_t t : {1, 5, 7}) {
        test_cases.emplace_back(new test_norm_gate(128, 24, t));
        test_cases.emplace_back(new test_norm_gate(64, 3, t));
    }
    for (int64_t nt : {1, 3, 5, 8}) {
        test_cases.emplace_back(new test_concat_cpy(3, nt, 512));
    }
    for (int64_t t : {1, 5, 7}) {
        test_cases.emplace_back(new test_rms_norm_scale2(128, 8, t));
        test_cases.emplace_back(new test_rms_norm_scale2(64, 3, t));
    }
    for (int64_t n : {64, 128, 1000}) {
        test_cases.emplace_back(new test_rms_norm_scale({n, 8, 5, 1}));
        test_cases.emplace_back(new test_rms_norm_scale({n, 3, 2, 2}, 2.0f));
    }

    // cc 6.0 fp16 expert GEMM (GGML_CUDA_P60_MOE_GEMM=1) at the shapes of a 512-wide expert split over 2 devices
    // (gate/up m 256 k 2048, src1 broadcast; down m 2048 k 256): 8, 32 and 128 pairs per expert reach the small-batch
    // kernel, the 32-pair tile and the 64-pair x 128-row tile
    for (ggml_type type_a : {GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K}) {
        for (int n : {64, 256, 1024}) {
            test_cases.emplace_back(new test_mul_mat_id(type_a, GGML_TYPE_F32, 64, 8, true,   256, n, 2048));
            test_cases.emplace_back(new test_mul_mat_id(type_a, GGML_TYPE_F32, 64, 8, false, 2048, n,  256));
        }
    }

    // prefill attention of this model per GPU under -sm tensor: D 256, 2 KV heads, GQA 6, q4_0 cache
    for (int64_t nb : {128, 256, 320, 384, 448, 512}) {
        for (int64_t kv : {256, 512, 768, 1024}) {
            if (kv < nb) {
                continue;
            }
            test_cases.emplace_back(new test_flash_attn_ext(256, 256, 2, {6, 1}, kv, nb, true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0));
            test_cases.emplace_back(new test_flash_attn_ext(256, 256, 2, {6, 1}, kv, nb, true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0));
        }
    }

    // Correctness for the 2x P100 operating shape: D=256, 2 KV heads, GQA 6, q4_0 cache.
    // This shape had NO eval coverage, so the tile kernel's q4_0 path and its half2 VKQ
    // accumulator were never checked against the CPU reference. kv is swept so that
    // accumulation error against context length is visible rather than assumed.
    // 262144 is the real operating context, so the sweep runs all the way out to it: the
    // claim that the error stays flat once accumulation is bounded is worth measuring at
    // the depth actually used rather than extrapolating from 65536.
    for (int kv : {512, 4096, 16384, 65536, 131072, 262144}) {
        test_cases.emplace_back(new test_flash_attn_ext(256, 256, 2, {6, 1}, kv, 1, true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0));
    }

    // Speculative-verify widths at the same shape: MTP verifies up to n_draft+1 = 5 tokens, and
    // on Pascal these take the q4p kernel. kv 768 and 1280 leave the last KV chunk partly filled.
    for (int kv : {256, 768, 1280, 4096, 32768}) {
        for (int nb : {2, 3, 4, 5}) {
            test_cases.emplace_back(new test_flash_attn_ext(256, 256, 2, {6, 1}, kv, nb, true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0));
        }
    }
    for (int kv : {768, 1280}) {
        test_cases.emplace_back(new test_flash_attn_ext(256, 256, 2, {6, 1}, kv, 1, true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0));
    }
    // the same decode/verify shapes with a q8_0 K cache (q4p's q8_0-K variant on Pascal)
    for (int kv : {256, 768, 1280, 4096, 32768, 262144}) {
        for (int nb : {1, 2, 3, 4, 5}) {
            test_cases.emplace_back(new test_flash_attn_ext(256, 256, 2, {6, 1}, kv, nb, true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0));
        }
    }
    test_cases.emplace_back(new test_flash_attn_ext(256, 256, 2, {6, 1}, 4096, 5, false, false, 0, 0, GGML_PREC_F32, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0));
    // and without a mask, and with a second sequence
    test_cases.emplace_back(new test_flash_attn_ext(256, 256, 2, {6, 1}, 4096, 1, false, false, 0, 0, GGML_PREC_F32, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0));
    test_cases.emplace_back(new test_flash_attn_ext(256, 256, 2, {6, 1}, 4096, 5, false, false, 0, 0, GGML_PREC_F32, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0));
    test_cases.emplace_back(new test_flash_attn_ext(256, 256, 2, {6, 2}, 4096, 3, true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0));

    // q6_K matvec at the model's shapes and the speculative-verify widths. The m=16 cases elsewhere
    // take the small-rows path on Pascal and never reach the multi-column kernel. 4360 rows leaves a
    // partial block, 17408 is the down projection's K.
    for (int n : {1, 2, 3, 4, 5, 6, 7}) {
        test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q6_K, GGML_TYPE_F32, 1024, n, 5120, {1, 1}, {1, 1}));
        test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q6_K, GGML_TYPE_F32, 4360, n, 5120, {1, 1}, {1, 1}));
        test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q6_K, GGML_TYPE_F32, 2560, n, 17408, {1, 1}, {1, 1}));
    }
    // the fp16 verify path on Pascal (mmvq-f16.cu) takes >= 3072 rows at 2..5 columns;
    // 6150 rows leaves a partial block of rows
    for (int m : {3072, 6144, 6150, 8704}) {
        for (int n : {2, 3, 4, 5}) {
            test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q6_K, GGML_TYPE_F32, m, n, 5120, {1, 1}, {1, 1}));
        }
    }
    for (int k : {3072, 8704}) {
        for (int n : {2, 5}) {
            test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q6_K, GGML_TYPE_F32, 5120, n, k, {1, 1}, {1, 1}));
        }
    }
    // below 3072 rows (wk/wv at 512 rows, the delta-net alpha/beta at 24); 300 and 20 rows leave
    // partial blocks
    for (int m : {512, 300, 24, 20}) {
        for (int n : {2, 3, 5}) {
            test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q6_K, GGML_TYPE_F32, m, n, 5120, {1, 1}, {1, 1}));
        }
        test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q6_K, GGML_TYPE_F32, m, 5, 8704, {1, 1}, {1, 1}));
    }
    // several parallel sequences verified together: 6..12 columns per launch, wider ones split in two
    // (13 -> 7 + 6, 16 -> 8 + 8); 9..16 would otherwise take MMQ
    for (int m : {8704, 6150, 3072, 512, 300, 24, 20}) {
        for (int n : {6, 8, 10, 12, 13, 16}) {
            test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q6_K, GGML_TYPE_F32, m, n, 5120, {1, 1}, {1, 1}));
        }
    }
    for (int n : {10, 16}) {
        test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q6_K, GGML_TYPE_F32, 5120, n, 8704, {1, 1}, {1, 1}));
    }

    // the fp16-product / fp32-accumulation prefill GEMM on Pascal (gemm-fold.cu) takes >= 1024 rows
    // past the mmvq widths; 1028 rows and 130 columns leave partial tiles
    for (int m : {1024, 1028}) {
        for (int n : {9, 130}) {
            test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q6_K, GGML_TYPE_F32, m, n, 5120, {1, 1}, {1, 1}));
        }
    }
    // short prompts on Pascal: q6_K/q5_K at 17..28 columns run as fp16 matvec column chunks (mmvq.cu),
    // 29..64 columns on the fold GEMM's 32/64-column tiles (gemm-fold.cu); 6150 rows leaves a partial
    // block of rows, 1028 a partial tile, and 300 rows stays below the fold GEMM
    for (ggml_type type_a : {GGML_TYPE_Q6_K, GGML_TYPE_Q5_K}) {
        for (int m : {6150, 1028, 300}) {
            for (int n : {17, 24, 28, 29, 32, 44, 64}) {
                test_cases.emplace_back(new test_mul_mat(type_a, GGML_TYPE_F32, m, n, 5120, {1, 1}, {1, 1}));
            }
        }
    }
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q4_0, GGML_TYPE_F32, 2048, 64, 4096, {1, 1}, {1, 1}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F16,  GGML_TYPE_F32, 1024, 130, 4096, {1, 1}, {1, 1}));

    // Prefill-shaped correctness at the same operating shape. The sweep above is nb=1, which
    // never reaches the cuBLAS-GEMM attention path: that path fires only at Q->ne[1] >= 128 and
    // K->ne[1] >= 4096, i.e. on essentially every long-context prefill ubatch, and it is ON by
    // default. These cases compare it against the CPU fp32 reference. Run once with
    // GGML_CUDA_FA_GEMM=1 and once with =0 (tile kernel) at the same shapes to see what the
    // GEMM path's fp16 accumulators cost.
    for (int kv : {4096, 16384, 65536}) {
        for (int nb : {512, 2048}) {
            test_cases.emplace_back(new test_flash_attn_ext(256, 256, 2, {6, 1}, kv, nb, true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0));
        }
    }

    // Same prefill shapes for the fold path's other K/V types (q8_0, and f16 with GGML_CUDA_FA_FOLD=2),
    // at both head sizes the fold kernels run (256: Qwen3.5/3.8 hybrids, 128: Qwen3-8B)
    for (int kv : {4096, 16384}) {
        for (int nb : {512, 2048}) {
            for (ggml_type_pair tp : { ggml_type_pair{GGML_TYPE_Q8_0, GGML_TYPE_Q8_0}, ggml_type_pair{GGML_TYPE_Q8_0, GGML_TYPE_F16},
                                       ggml_type_pair{GGML_TYPE_F16,  GGML_TYPE_Q8_0}, ggml_type_pair{GGML_TYPE_Q4_0, GGML_TYPE_Q8_0} }) {
                test_cases.emplace_back(new test_flash_attn_ext(256, 256, 2, {6, 1}, kv, nb, true, false, 0, 0, GGML_PREC_F32, tp.a, tp.b));
                test_cases.emplace_back(new test_flash_attn_ext(128, 128, 8, {4, 1}, kv, nb, true, false, 0, 0, GGML_PREC_F32, tp.a, tp.b));
            }
        }
    }

    // the generic fp16 multi-column matvec (mmvq-f16.cu) for every other type it takes: 2..16
    // columns, a partial row block (4100), K not a multiple of the 1024-value window (5376), and the
    // fused gate+up+SWIGLU
    for (ggml_type t : {GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0, GGML_TYPE_IQ4_NL,
                        GGML_TYPE_IQ4_XS, GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K,
                        GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S,
                        GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M, GGML_TYPE_MXFP4}) {
        for (int n : {2, 3, 4, 5, 8, 9, 10, 13, 16}) {
            test_cases.emplace_back(new test_mul_mat(t, GGML_TYPE_F32, 4100, n, 5376, {1, 1}, {1, 1}));
        }
        test_cases.emplace_back(new test_mul_mat(t, GGML_TYPE_F32, 300, 4, 3072, {1, 1}, {1, 1}));
        for (int64_t m_batch : {2, 5, 10}) {
            test_cases.emplace_back(new test_mul_mat_fusion(t, GGML_GLU_OP_SWIGLU, m_batch, 3072, 5376,
                false, 1, 1, false, false, true, false, {1, 1}));
        }
    }
    // the fp16 q6_K gate+up+SWIGLU verify kernel (mmvq-f16.cu): >= 3072 rows, K a multiple of 512
    for (int64_t m_batch : { 2, 3, 4, 5, 6, 10, 13, 16 }) {
        for (int64_t n_rows : { 3072, 8704 }) {
            test_cases.emplace_back(new test_mul_mat_fusion(GGML_TYPE_Q6_K, GGML_GLU_OP_SWIGLU, m_batch, n_rows, 5120,
                false, 1, 1, false, false, true, false, {1, 1}));
        }
    }

    // prefill widths of this model per GPU under -sm tensor: 8 k heads x 128, v_repeat 3 (OPTLOG multi-slot)
    for (int64_t nt : {64, 128, 170, 256, 320, 384, 448, 512}) {
        test_cases.emplace_back(new test_gated_delta_net(GGML_TYPE_F32, 8, 128, nt, 1, 3));
    }
    test_cases.emplace_back(new test_gated_delta_net(GGML_TYPE_F32, 8, 128, 170, 3, 3));

    // decode chain of a qwen35 delta net layer (the CUDA backend fuses it for one sequence and up to 32 tokens)
    for (int64_t n_t : { 1, 2, 4, 8, 33, 70 }) {
        test_cases.emplace_back(new test_gdn_decode_fusion(4, 12, 128, n_t));
    }
    test_cases.emplace_back(new test_gdn_decode_fusion(16, 48, 128, 1));
    // the same chain with rs_replay (pending gathers between the state rows and the gdn, fused pending stores)
    for (int64_t n_t : { 1, 2, 4, 8 }) {
        for (bool keep : { false, true }) {
            if (keep && n_t < 2) {
                continue;
            }
            test_cases.emplace_back(new test_gdn_decode_fusion(4, 12, 128, n_t, 1, 1, 1, GGML_TYPE_COUNT, 4, keep));
        }
    }
    test_cases.emplace_back(new test_gdn_decode_fusion(16, 48, 128, 1, 1, 1, 1, GGML_TYPE_COUNT, 4, false));
    test_cases.emplace_back(new test_gdn_decode_fusion(4, 12, 128, 2, 1, 1, 1, GGML_TYPE_COUNT, 1, true));
    for (ggml_type t : { GGML_TYPE_F32, GGML_TYPE_Q8_0, GGML_TYPE_Q6_K }) {
        test_cases.emplace_back(new test_gdn_decode_fusion(16, 48, 128, 1, 1, 1, 1, t));
        test_cases.emplace_back(new test_gdn_decode_fusion(16, 32, 128, 4, 1, 1, 1, t));
    }
    test_cases.emplace_back(new test_gdn_decode_fusion(16, 32, 128, 1, 1, /*row=*/0, /*head=*/1));
    test_cases.emplace_back(new test_gdn_decode_fusion(16, 32, 128, 3, 1, /*row=*/2, /*head=*/0));
    test_cases.emplace_back(new test_gdn_decode_fusion(4, 12, 128, 1, /*n_seqs=*/2));
    test_cases.emplace_back(new test_gdn_decode_fusion(4, 12, 128, 4, /*n_seqs=*/2));
    test_cases.emplace_back(new test_gdn_decode_fusion(4, 8, 64, 1));
    test_cases.emplace_back(new test_gdn_decode_fusion(4, 8, 64, 5));
    test_cases.emplace_back(new test_gdn_decode_fusion(8, 8, 32, 1));
}

static void p100_add_test_cases_perf(std::vector<std::unique_ptr<test_case>> & test_cases) {
    // q8_0 mat-vec: 27B shapes and their 1/4 tensor-split slices (m, k)
    for (auto mk : std::vector<std::pair<int64_t, int64_t>>{{17408, 5120}, {5120, 17408}, {10240, 5120}, {6144, 5120}, {12288, 5120}, {5120, 6144},
                                                           {4352, 5120}, {5120, 4352}, {2560, 5120}, {1536, 5120}, {5120, 1536}, {3072, 5120}}) {
        for (int bs : {1, 2, 3, 4, 5, 6, 7, 8}) {
            test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q8_0, GGML_TYPE_F32, mk.first, bs, mk.second, {1, 1}, {1, 1}));
        }
    }

    // IQ mat-vec: 27B shapes and a 1/4 tensor-split slice (m, k)
    for (ggml_type type_a : {GGML_TYPE_IQ4_XS, GGML_TYPE_IQ4_NL, GGML_TYPE_IQ3_S, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S, GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M}) {
        for (auto mk : std::vector<std::pair<int64_t, int64_t>>{{17408, 5120}, {5120, 17408}, {6144, 5120}, {4352, 5120}, {1536, 5120}}) {
            for (int bs : {1, 2, 3, 4, 5, 8}) {
                test_cases.emplace_back(new test_mul_mat(type_a, GGML_TYPE_F32, mk.first, bs, mk.second, {1, 1}, {1, 1}));
            }
        }
    }

    // Long-context prefill attention, exactly as one GPU sees it for qwen3.5-27B under
    // -sm tensor -ctk q4_0 -ctv q4_0 -ub 2048: 2 KV heads, GQA 6 (12 Q heads), D 256.
    // 16 of the 65 layers carry a growing KV cache (full_attention_interval 4), so a
    // prefill batch costs 16x one of these. Benchmarking the op directly avoids the ~30
    // minutes llama-bench spends rebuilding 262144 tokens of context to time one batch.
    // nb=2048 is a prefill batch; nb=1 is decode, which takes the VEC kernel instead.
    for (int kv : {32768, 65536, 131072, 262144}) {
        for (int nb : {2048, 1024, 512, 16, 8, 7, 6, 5, 4, 3, 2, 1}) {
            test_cases.emplace_back(new test_flash_attn_ext(256, 256, 2, {6, 1}, kv, nb, true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0));
            // the same with a q8_0 K cache (q4p and fold q8_0-K variants)
            test_cases.emplace_back(new test_flash_attn_ext(256, 256, 2, {6, 1}, kv, nb, true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0));
        }
    }
    // decode and verify widths at shallower depths, where fixed per-launch costs start to show
    for (int kv : {2048, 4096, 8192, 16384}) {
        for (int nb : {5, 4, 3, 2, 1}) {
            test_cases.emplace_back(new test_flash_attn_ext(256, 256, 2, {6, 1}, kv, nb, true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0));
        }
    }
    // The q6_K matvecs of one decode/verify pass on each GPU under -sm tensor (m rows x k), by width.
    for (int64_t r : {1, 2, 5}) {
        test_cases.emplace_back(new test_add_rms_norm_mul_p100({5120, r}));
    }
    for (auto mk : std::vector<std::pair<int, int>>{{8704, 5120}, {5120, 8704}, {5120, 5120}, {3072, 5120}, {5120, 3072}, {6144, 5120}, {512, 5120}, {24, 5120}}) {
        for (int n : {1, 2, 3, 4, 5, 6, 8, 10, 12, 15}) {
            test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q6_K, GGML_TYPE_F32, mk.first, n, mk.second, {1, 1}, {1, 1}));
        }
    }
    // The same pass for every other weight type (mmvq-f16.cu's generic kernel against the integer
    // path: GGML_CUDA_MMVQ_F16=0)
    for (ggml_type t : {GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0, GGML_TYPE_IQ4_NL,
                        GGML_TYPE_IQ4_XS, GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K,
                        GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S,
                        GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M, GGML_TYPE_MXFP4}) {
        for (auto mk : std::vector<std::pair<int, int>>{{8704, 5120}, {5120, 8704}, {6144, 5120}, {512, 5120}, {24, 5120}}) {
            for (int n : {1, 2, 3, 4, 5, 6, 8, 10, 16}) {
                test_cases.emplace_back(new test_mul_mat(t, GGML_TYPE_F32, mk.first, n, mk.second, {1, 1}, {1, 1}));
            }
        }
    }
    // Same shapes with an f16 cache. The tile kernel is launched with need_f16_K/V, so a
    // quantized cache is dequantized in full by launch_fattn on every call; these isolate
    // how much of the tile kernel's cost is that conversion rather than attention itself.
    for (int nb : {8, 6, 5, 4, 1}) {
        test_cases.emplace_back(new test_flash_attn_ext(256, 256, 2, {6, 1}, 262144, nb, true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_F16, GGML_TYPE_F16));
    }
    // Decode shape across the 32-value block quants, to separate "bytes moved" from "work per
    // value" without a profiler (nvprof metrics need NVreg_RestrictProfilingToAdminUsers=0).
    // All of these hold 32 values per block, so a call touches the same number of blocks and
    // values; only the block size and the dequant differ:
    //     q4_0 18 B (0.5625 B/val)  q4_1 20 B  q5_0 22 B  q5_1 24 B  q8_0 34 B (1.0625)  f16 2 B/val
    // If time tracks bytes, the kernel is bandwidth-bound. If it is flat across these while
    // f16 is faster than all of them, the cost is per-value unpacking, not traffic. q8_0 is
    // the key point: ~2x the bytes of q4_0 with the cheapest dequant (no nibble unpack).
    for (ggml_type tkv : {GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0}) {
        test_cases.emplace_back(new test_flash_attn_ext(256, 256, 2, {6, 1}, 262144, 1, true, false, 0, 0, GGML_PREC_F32, tkv, tkv));
    }
}
