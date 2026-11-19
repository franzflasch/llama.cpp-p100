// included by ggml-cuda.cu, uses its static helpers
// p100-series: the fused top-k MoE router rounds differently from the unfused softmax / argsort / get_rows / norm chain, and for
// batches of more than TOPK_MOE_ROWS_PER_BLOCK rows the fusion was refused whenever its outputs overlapped a live input -- so which
// path ran depended on where the allocator put the router logits. Flash-Next's output then changed with buffer placement (first
// request after load != later ones; upstream #31243 moved the layout). Instead of falling back, stage the outputs in pool memory and
// copy them into place after the kernel has read everything: always fused, placement-independent.
static bool ggml_cuda_topk_moe_fused_or_staged(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int node_idx, int n_ops,
                                               const int * out_nodes, const ggml_tensor * logits, ggml_tensor * weights, ggml_tensor * ids,
                                               const ggml_tensor * clamp, const ggml_tensor * scale, const ggml_tensor * bias,
                                               const ggml_cuda_topk_moe_args & args) {
    if (ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, n_ops, out_nodes, 2, /*is_topk_moe=*/true)) {
        ggml_cuda_op_topk_moe(ctx, logits, weights, ids, clamp, scale, bias, args);
        return true;
    }
    // ids is a view into the argsort-shaped buffer (row stride n_expert): copy its whole span, which the unfused argsort writes too
    const size_t nb_w = ggml_nbytes(weights);
    const size_t nb_i = ggml_nbytes(ids);
    ggml_cuda_pool_alloc<char> w_tmp(ctx.pool(), nb_w);
    ggml_cuda_pool_alloc<char> i_tmp(ctx.pool(), nb_i);
    ggml_tensor w_staged = *weights;
    ggml_tensor i_staged = *ids;
    w_staged.data = w_tmp.ptr;
    i_staged.data = i_tmp.ptr;
    ggml_cuda_op_topk_moe(ctx, logits, &w_staged, &i_staged, clamp, scale, bias, args);
    CUDA_CHECK(cudaMemcpyAsync(weights->data, w_tmp.ptr, nb_w, cudaMemcpyDeviceToDevice, ctx.stream()));
    CUDA_CHECK(cudaMemcpyAsync(ids->data,     i_tmp.ptr, nb_i, cudaMemcpyDeviceToDevice, ctx.stream()));
    return true;
}
