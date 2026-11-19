// included by ggml-cuda.cu, uses its static helpers
// GET_ROWS that gathers recurrent states whose only consumer, through reshape views, is a gated delta
// net's state input: skip it and let the kernel read the rows in place (gated_delta_net.cuh).
static bool ggml_cuda_try_gdn_state_gather(const ggml_cgraph * cgraph, int i) {
    const ggml_tensor * node = cgraph->nodes[i];
    if (node->op != GGML_OP_GET_ROWS || (node->flags & GGML_TENSOR_FLAG_OUTPUT) ||
            node->type != GGML_TYPE_F32 || node->src[0]->type != GGML_TYPE_F32 || node->src[1]->type != GGML_TYPE_I32 ||
            node->src[0]->nb[0] != sizeof(float) || !ggml_is_contiguous(node) || !ggml_is_contiguous(node->src[1]) ||
            node->ne[2] != 1 || node->ne[3] != 1 || node->src[0]->ne[2] != 1 || node->src[0]->ne[3] != 1 ||
            node->src[1]->ne[0] != node->ne[1]) {
        return false;
    }
    // one sequence only: with several, the delta net's blocks for one sequence can read a state row
    // that another sequence's blocks write (the gather's source rows are cache cells, and the new states
    // go back into the cache), which the chunked kernel does. Multi-slot prefill read wrong states:
    // KLD 0.044 against one sequence at a time, 0.0019 with the copy (OPTLOG multi-slot).
    if (node->ne[1] != 1) {
        return false;
    }
    const ggml_tensor * aliases[8] = { node };
    int n_alias = 1;
    auto is_alias = [&](const ggml_tensor * t) {
        for (int a = 0; a < n_alias; ++a) {
            if (t == aliases[a]) {
                return true;
            }
        }
        return false;
    };
    const ggml_tensor * gdn = nullptr;
    int gdn_j = -1;
    for (int j = i + 1; j < cgraph->n_nodes; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        bool uses = false;
        for (int k = 0; k < GGML_MAX_SRC; ++k) {
            if (n->src[k] && is_alias(n->src[k])) {
                uses = true;
            }
        }
        if (n->view_src && is_alias(n->view_src)) {
            uses = true;
        }
        // the states_all gather (llama-graph build_rs) also has an empty view for the extra states of a
        // single-sequence ubatch, and a copy of it: those read nothing
        if (!uses || ggml_nelements(n) == 0) {
            continue;
        }
        if ((n->op == GGML_OP_RESHAPE || n->op == GGML_OP_VIEW) && n->view_offs == 0 &&
                ggml_nbytes(n) == ggml_nbytes(node) && ggml_is_contiguous(n) && !(n->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            if (n_alias == 8) {
                return false;
            }
            aliases[n_alias++] = n;
            continue;
        }
        if (n->op == GGML_OP_CONCAT && gdn == nullptr && is_alias(n->src[0]) && !is_alias(n->src[1]) &&
                ggml_get_op_params_i32(n, 0) == 0 && n->ne[0] <= 64 && n->type == GGML_TYPE_F32) {
            gdn = n; // the narrow dim-0 concat kernel reads src0 rows through the index (concat.cu)
            gdn_j = j;
            continue;
        }
        if (n->op == GGML_OP_GATED_DELTA_NET && gdn == nullptr && is_alias(n->src[5])) {
            for (int k = 0; k < 5; ++k) {
                if (n->src[k] && is_alias(n->src[k])) {
                    return false;
                }
            }
            gdn = n;
            gdn_j = j;
            continue;
        }
        return false; // any other consumer needs the gathered copy
    }
    if (gdn == nullptr) {
        return false;
    }
    // The consumer reads the index tensor when it runs, not when this GET_ROWS would have. The graph allocator frees
    // the index after its last consumer in the graph, which is this GET_ROWS for the last recurrent layer: the memory
    // is then reused before the consumer runs and it reads a garbage row (illegal address, 4claps' Qwen3.6-35B-A3B
    // decode at depth >= 1024). Skip the gather only while a later node still uses the index; otherwise copy as usual.
    {
        const ggml_tensor * ids = node->src[1];
        bool live = false;
        for (int k = gdn_j + 1; k < cgraph->n_nodes && !live; ++k) {
            const ggml_tensor * n = cgraph->nodes[k];
            if (n->view_src == ids) {
                live = true;
            }
            for (int s = 0; s < GGML_MAX_SRC && !live; ++s) {
                if (n->src[s] == ids || (n->src[s] && n->src[s]->view_src == ids)) {
                    live = true;
                }
            }
        }
        if (!live) {
            return false;
        }
    }
    ggml_cuda_gdn_gather_register(gdn, (const float *) node->src[0]->data, (const int32_t *) node->src[1]->data,
                                  node->src[0]->nb[1] / sizeof(float));
    return true;
}
