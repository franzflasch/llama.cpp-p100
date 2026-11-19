// included by ggml-cuda.cu

// Pascal supports CUDA graphs; upstream gates them on architecture alone. Profiling the MTP path
// shows ~95 ms of a 148 ms verify pass is GPU idle across ~3000 kernel launches.
static bool ggml_cuda_p100_graphs_pre_volta() {
    static const bool on = [] {
        const char * s = getenv("GGML_CUDA_GRAPHS_PRE_VOLTA");
        return s && atoi(s) != 0;
    }();
    return on;
}

static bool ggml_cuda_p100_graph_allowed(const ggml_cgraph * cgraph) {
    // GGML_CUDA_GRAPHS_PRE_VOLTA=3: graphs only for single-token graphs (MTP draft steps, plain
    // decode). Not for 1-node graphs: the tensor-parallel backend reuses one auxiliary graph object
    // for every exchange's ADD, whose shape alternates, so it never stabilizes. Not for multi-token
    // graphs: a fixed-width verify's real-token count (active-tokens.cuh) is read at enqueue time and
    // would be frozen into a captured graph.
    static const bool single_token_only = [] {
        const char * s = getenv("GGML_CUDA_GRAPHS_PRE_VOLTA");
        return s && atoi(s) == 3;
    }();
    if (single_token_only) {
        if (cgraph->n_nodes <= 1) {
            return false;
        }
        for (int i = 0; i < cgraph->n_nodes; i++) {
            const ggml_tensor * node = cgraph->nodes[i];
            // The K/V Hadamard rotation is a MUL_MAT over all heads (src1 ne[1] = n_head * n_tokens),
            // so it is not a token count: judge the attention subgraph by FLASH_ATTN_EXT below.
            // Without this, every single-token attention subgraph ran eagerly.
            if (node->op == GGML_OP_MUL_MAT && ggml_get_op_params_i32(node, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                continue;
            }
            if ((node->op == GGML_OP_MUL_MAT || node->op == GGML_OP_MUL_MAT_ID) && node->src[1] && node->src[1]->ne[1] > 1) {
                return false;
            }
            if (node->op == GGML_OP_FLASH_ATTN_EXT && node->src[0]->ne[1] > 1) {
                return false;
            }
        }
    }
    return true;
}
