// included by fusion.cu, uses its static helpers

#include "../active-tokens.cuh"
#include "../concat.cuh"
#include "../cpy-batch.cuh"
#include "../mmvq-f16.cuh"
#include "../norm-p100.cuh"
#include "../ssm-conv-p100.cuh"
#include "../unary-p100.cuh"

#include <unordered_map>

// defined in ggml-cuda.cu
bool ggml_cuda_compute_forward(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst);
std::unordered_map<const ggml_tensor *, std::pair<const ggml_tensor *, const ggml_tensor *>> & ggml_cuda_conv_decode_pending();

// the fused P100 norm kernels launch ne[2] x ne[3] blocks, CUDA caps gridDim.y/z at 65535
static bool ggml_cuda_norm_too_wide(const ggml_tensor * t) {
    return t->ne[2] > 65535 || t->ne[3] > 65535;
}

// runs first in ggml_cuda_try_fuse: returns the number of nodes to skip, or -1 to continue
static int ggml_cuda_p100_try_fuse_pre(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {
    // past the grid limit leave the nodes to the plain ops
    {
        const ggml_tensor * n0 = cgraph->nodes[i];
        if ((n0->op == GGML_OP_NORM || n0->op == GGML_OP_RMS_NORM || n0->op == GGML_OP_L2_NORM ||
             n0->op == GGML_OP_ADD) && ggml_cuda_norm_too_wide(n0)) {
            return 0;
        }
    }

    // consecutive same-shape copies between the same two tensors (delta-net conv-state snapshots)
    if (cgraph->nodes[i]->op == GGML_OP_CPY) {
        const int n_skip = ggml_cuda_try_cpy_batch(*cuda_ctx, cgraph, i);
        if (n_skip > 0) {
            return n_skip;
        }
    }

    ggml_tensor * node = cgraph->nodes[i];

    if (node->op == GGML_OP_RMS_NORM) {
        const int nodes_to_skip = ggml_cuda_try_gdn_fusion(cuda_ctx, cgraph, i);
        if (nodes_to_skip > 0) {
            return nodes_to_skip;
        }
    }

    // residual ADD -> RMS_NORM -> MUL (the next pre-norm). The sum stays a graph output of its own
    // (it is the next residual), so this is not a plain chain for ggml_can_fuse: the norm and the
    // multiply must be its only consumers of the intermediate norm.
    if (node->op == GGML_OP_ADD && i + 2 < cgraph->n_nodes) {
        ggml_tensor * rms = cgraph->nodes[i + 1];
        ggml_tensor * mul = cgraph->nodes[i + 2];
        if (rms->op == GGML_OP_RMS_NORM && rms->src[0] == node &&
                mul->op == GGML_OP_MUL && (mul->src[0] == rms || mul->src[1] == rms) &&
                (rms->flags & GGML_TENSOR_FLAG_COMPUTE) && (mul->flags & GGML_TENSOR_FLAG_COMPUTE) &&
                !(rms->flags & GGML_TENSOR_FLAG_OUTPUT) && ggml_node_get_use_count(cgraph, i + 1) == 1 &&
                !(i + 3 < cgraph->n_nodes && cgraph->nodes[i + 3]->op == GGML_OP_ADD &&
                  (cgraph->nodes[i + 3]->src[0] == mul || cgraph->nodes[i + 3]->src[1] == mul)) &&
                ggml_cuda_op_add_rms_norm_mul(*cuda_ctx, node, rms, mul)) {
            return 2;
        }
    }

    return -1;
}

// runs before the upstream RMS_NORM + MUL fusion: returns the number of nodes to skip, or 0 to continue
static int ggml_cuda_p100_try_fuse_post(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {
    ggml_tensor * node = cgraph->nodes[i];

    // RMS_NORM -> MUL(w), then MUL_MAT(z) -> SILU -> MUL(norm): run the matmul first, then the norm
    // with the silu gate in its epilogue (qwen35's gated norm).
    if (node->op == GGML_OP_RMS_NORM && ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL }, {})) {
        const int n = cgraph->n_nodes;
        auto next = [&](int j) {
            for (++j; j < n; ++j) {
                if (!ggml_cuda_is_view_or_noop(cgraph->nodes[j])) {
                    return j;
                }
            }
            return -1;
        };
        ggml_tensor * mulw = cgraph->nodes[i + 1];
        const int jm = next(i + 1), js = jm < 0 ? -1 : next(jm), jo = js < 0 ? -1 : next(js);
        if (jo > 0) {
            ggml_tensor * mm = cgraph->nodes[jm], * sl = cgraph->nodes[js], * mo = cgraph->nodes[jo];
            auto base = [](const ggml_tensor * t) { return t->view_src ? t->view_src : t; };
            // The fused order runs the matmul before the norm, but the allocator planned memory for the
            // graph order: once the norm has read x, x's buffer is free and the matmul's output may be put
            // there. Running the matmul first would then overwrite x before the norm reads it. Which
            // buffers alias depends on the tensor sizes, so it broke only some ubatch sizes (256..384
            // tokens: PPL ~10^4). Likewise the output may take x's buffer: fine in place with the same
            // layout (each block reads its whole row before writing it), not otherwise.
            auto overlap = [](const ggml_tensor * a, const ggml_tensor * b) {
                const char * a0 = (const char *) a->data, * b0 = (const char *) b->data;
                return a0 < b0 + ggml_nbytes(b) && b0 < a0 + ggml_nbytes(a);
            };
            const ggml_tensor * xin = node->src[0];
            auto same = [](const ggml_tensor * a, const ggml_tensor * b) {
                return a->data == b->data && ggml_are_same_shape(a, b) && ggml_are_same_stride(a, b);
            };
            const bool alias_ok = !overlap(mm, xin) &&
                (!overlap(mo, mm)  || same(mo, mm)) &&
                (!overlap(mo, xin) || same(mo, xin));
            if (mm->op == GGML_OP_MUL_MAT && sl->op == GGML_OP_UNARY && ggml_get_unary_op(sl) == GGML_UNARY_OP_SILU &&
                    mo->op == GGML_OP_MUL && (mm->flags & GGML_TENSOR_FLAG_COMPUTE) && (sl->flags & GGML_TENSOR_FLAG_COMPUTE) &&
                    (mo->flags & GGML_TENSOR_FLAG_COMPUTE) &&
                    base(sl->src[0]) == mm && sl->src[0]->view_offs == 0 && ggml_is_contiguous(sl->src[0]) &&
                    ((mo->src[0] == sl && mo->src[1] == mulw) || (mo->src[1] == sl && mo->src[0] == mulw)) &&
                    base(mm->src[0]) != node && base(mm->src[1]) != node && base(mm->src[0]) != mulw && base(mm->src[1]) != mulw &&
                    ggml_node_get_use_count(cgraph, i + 1) == 1 && ggml_node_get_use_count(cgraph, js) == 1 &&
                    !(mulw->flags & GGML_TENSOR_FLAG_OUTPUT) && !(sl->flags & GGML_TENSOR_FLAG_OUTPUT) &&
                    alias_ok && ggml_cuda_op_rms_norm_mul_silu_gate(*cuda_ctx, node, mulw, sl->src[0], mo, true)) {
                GGML_ASSERT(ggml_cuda_compute_forward(*cuda_ctx, mm));
                ggml_cuda_op_rms_norm_mul_silu_gate(*cuda_ctx, node, mulw, sl->src[0], mo, false);
                return jo - i;
            }
        }
    }

    // The FFN's gate and up matvecs and their SWIGLU, in one launch (verify widths, fp16 path):
    //   MUL_MAT(up) -> MUL_MAT(gate) -> GLU(SWIGLU, gate, up), in either matvec order, views between.
    if (node->op == GGML_OP_MUL_MAT && ggml_is_quantized(node->src[0]->type) && node->ne[0] >= 3072 &&
            node->ne[1] >= 2 && node->ne[1] <= MMVQ_F16_MAX_COLS) {
        const int n = cgraph->n_nodes;
        auto next = [&](int j) {
            for (++j; j < n; ++j) {
                if (!ggml_cuda_is_view_or_noop(cgraph->nodes[j]) && (cgraph->nodes[j]->flags & GGML_TENSOR_FLAG_COMPUTE)) {
                    return j;
                }
            }
            return -1;
        };
        const int j2 = next(i), jg = j2 < 0 ? -1 : next(j2);
        if (jg > 0) {
            ggml_tensor * mm2 = cgraph->nodes[j2], * glu = cgraph->nodes[jg];
            if (mm2->op == GGML_OP_MUL_MAT && glu->op == GGML_OP_GLU && ggml_get_glu_op(glu) == GGML_GLU_OP_SWIGLU &&
                    ggml_get_op_params_i32(glu, 1) == 0 && glu->src[1] != nullptr &&
                    ((glu->src[0] == node && glu->src[1] == mm2) || (glu->src[0] == mm2 && glu->src[1] == node)) &&
                    mm2->src[1] == node->src[1] && ggml_are_same_shape(node, mm2) &&
                    ggml_node_get_use_count(cgraph, i) == 1 && ggml_node_get_use_count(cgraph, j2) == 1 &&
                    !(node->flags & GGML_TENSOR_FLAG_OUTPUT) && !(mm2->flags & GGML_TENSOR_FLAG_OUTPUT)) {
                const ggml_tensor * mm_gate = glu->src[0], * mm_up = glu->src[1];
                if (ggml_cuda_mmvq_f16_glu(*cuda_ctx, mm_gate, mm_up, glu, ggml_cuda_active_cols(node->src[1]->ne[1]))) {
                    return jg - i;
                }
            }
        }
    }

    // The delta net's alpha/beta matvecs and what follows them, in one launch:
    //   MUL_MAT(alpha) -> ADD(dt) -> SOFTPLUS -> MUL(a) = gate,  MUL_MAT(beta) -> SIGMOID = beta
    // (views between them allowed). Every intermediate must be consumed only inside the group.
    if (node->op == GGML_OP_MUL_MAT && node->src[0]->type == GGML_TYPE_Q6_K && node->ne[0] < 256) {
        const int n = cgraph->n_nodes;
        auto next = [&](int j) {
            for (++j; j < n; ++j) {
                if (!ggml_cuda_is_view_or_noop(cgraph->nodes[j]) && (cgraph->nodes[j]->flags & GGML_TENSOR_FLAG_COMPUTE)) {
                    return j;
                }
            }
            return -1;
        };
        auto same_mem = [](const ggml_tensor * v, const ggml_tensor * t) { // v: t, or a contiguous view of all of t
            return v == t || (v->data == t->data && ggml_is_contiguous(v) && ggml_nelements(v) == ggml_nelements(t) &&
                              v->type == t->type);
        };
        const int ja = next(i), jsp = ja < 0 ? -1 : next(ja), jm = jsp < 0 ? -1 : next(jsp), jb = jm < 0 ? -1 : next(jm),
                  js = jb < 0 ? -1 : next(jb);
        if (js > 0) {
            ggml_tensor * add = cgraph->nodes[ja], * sp = cgraph->nodes[jsp], * mul = cgraph->nodes[jm];
            ggml_tensor * mmb = cgraph->nodes[jb], * sg = cgraph->nodes[js];
            if (add->op == GGML_OP_ADD && sp->op == GGML_OP_UNARY && ggml_get_unary_op(sp) == GGML_UNARY_OP_SOFTPLUS &&
                    mul->op == GGML_OP_MUL && mmb->op == GGML_OP_MUL_MAT && sg->op == GGML_OP_UNARY &&
                    ggml_get_unary_op(sg) == GGML_UNARY_OP_SIGMOID && sp->src[0] == add &&
                    (mul->src[0] == sp || mul->src[1] == sp) && same_mem(sg->src[0], mmb) &&
                    (same_mem(add->src[0], node) || same_mem(add->src[1], node))) {
                const ggml_tensor * bvec = same_mem(add->src[0], node) ? add->src[1] : add->src[0];
                const ggml_tensor * mvec = mul->src[0] == sp ? mul->src[1] : mul->src[0];
                // intermediates (and their views) consumed only within (i, js], never outputs
                const ggml_tensor * inter[4] = { node, add, sp, mmb };
                auto is_inter = [&](const ggml_tensor * t) {
                    for (; t; t = t->view_src) {
                        for (const ggml_tensor * x : inter) {
                            if (t == x) {
                                return true;
                            }
                        }
                    }
                    return false;
                };
                bool ok = ggml_are_same_shape(add, sp) && ggml_are_same_shape(sp, mul) && ggml_nelements(add) == ggml_nelements(node);
                for (int j = 0; j < n && ok; ++j) {
                    const ggml_tensor * t = cgraph->nodes[j];
                    if (j >= i && j <= js && t != mul && t != sg && is_inter(t) && (t->flags & GGML_TENSOR_FLAG_OUTPUT)) {
                        ok = false;
                    }
                    if (j > i && j <= js) {
                        continue;
                    }
                    for (int k = 0; k < GGML_MAX_SRC && ok; ++k) {
                        if (t->src[k] && is_inter(t->src[k])) {
                            ok = false;
                        }
                    }
                    if (t->view_src && is_inter(t->view_src) && j > js) {
                        ok = false;
                    }
                }
                if (ok && ggml_cuda_mmvq_f16_gdn_gate(*cuda_ctx, node, mmb, bvec, mvec, mul, sg,
                                                      ggml_cuda_active_cols(node->src[1]->ne[1]))) {
                    return js - i;
                }
            }
        }
    }

    // single-token decode: CONCAT -> CPY of its tail (the conv state update) ... SSM_CONV -> SILU, one launch.
    // The state ops of the delta net sit between the CPY and the SSM_CONV, so CONCAT and CPY are skipped
    // here and the fused kernel runs at the SSM_CONV (where its output buffer is live); nothing in between
    // may read the concat or touch the conv-state cache.
    if (node->op == GGML_OP_CONCAT && node->ne[0] == 4 && node->ne[2] == 1 && !(node->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        int j = i + 1;
        while (j < cgraph->n_nodes && ggml_cuda_is_view_or_noop(cgraph->nodes[j])) {
            ++j;
        }
        ggml_tensor * cpy = j < cgraph->n_nodes ? cgraph->nodes[j] : nullptr;
        if (cpy && cpy->op == GGML_OP_CPY && (cpy->flags & GGML_TENSOR_FLAG_COMPUTE) && cpy->src[0]->view_src == node) {
            const auto root = [](const ggml_tensor * t) { while (t && t->view_src) { t = t->view_src; } return t; };
            const ggml_tensor * cache = root(cpy->src[1]);
            int k = -1;
            for (int q = j + 1; q < cgraph->n_nodes && q < j + 64; ++q) {
                const ggml_tensor * n = cgraph->nodes[q];
                if (n->op == GGML_OP_SSM_CONV && n->src[0] == node) {
                    k = q;
                    break;
                }
                bool touches = root(n) == cache || root(n) == node;
                for (int sidx = 0; sidx < GGML_MAX_SRC && !touches; ++sidx) {
                    touches = n->src[sidx] && (root(n->src[sidx]) == node || root(n->src[sidx]) == cache);
                }
                if (touches && !ggml_cuda_is_view_or_noop(n)) {
                    break;
                }
            }
            if (k > 0 && k + 1 < cgraph->n_nodes &&
                    ggml_cuda_can_fuse(cgraph, k, { GGML_OP_SSM_CONV, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU }) &&
                    ggml_cuda_ssm_conv_decode_fused_ok(node, cpy, cgraph->nodes[k], cgraph->nodes[k + 1])) {
                ggml_cuda_conv_decode_pending()[cgraph->nodes[k]] = { node, cpy };
                return j - i;
            }
        }
    }
    if (node->op == GGML_OP_SSM_CONV) {
        auto & pending = ggml_cuda_conv_decode_pending();
        auto it = pending.find(node);
        if (it != pending.end()) {
            const auto [cat, cpy] = it->second;
            pending.erase(it);
            GGML_ASSERT(ggml_cuda_op_ssm_conv_decode_fused(*cuda_ctx, cat, cpy, node, cgraph->nodes[i + 1]));
            return 1;
        }
    }

    // CONCAT -> CPY of its tail (the conv state update, views between): one launch.
    if (node->op == GGML_OP_CONCAT) {
        int j = i + 1;
        while (j < cgraph->n_nodes && ggml_cuda_is_view_or_noop(cgraph->nodes[j])) {
            ++j;
        }
        if (j < cgraph->n_nodes && cgraph->nodes[j]->op == GGML_OP_CPY && (cgraph->nodes[j]->flags & GGML_TENSOR_FLAG_COMPUTE) &&
                ggml_cuda_op_concat_cpy(*cuda_ctx, node, cgraph->nodes[j])) {
            return j - i;
        }
    }

    // ADD (row bias) -> SOFTPLUS -> MUL (row scale): the delta net's gate.
    if (node->op == GGML_OP_ADD && i + 2 < cgraph->n_nodes && cgraph->nodes[i + 1]->op == GGML_OP_UNARY &&
            cgraph->nodes[i + 2]->op == GGML_OP_MUL) {
        const enum ggml_op ops_g[3] = { GGML_OP_ADD, GGML_OP_UNARY, GGML_OP_MUL };
        if (ggml_can_fuse(cgraph, i, ops_g, 3) &&
                ggml_cuda_op_add_softplus_mul(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2])) {
            return 2;
        }
    }

    // RMS_NORM -> SCALE (the gated delta net's l2 norm of q and k).
    if (node->op == GGML_OP_RMS_NORM && i + 1 < cgraph->n_nodes && cgraph->nodes[i + 1]->op == GGML_OP_SCALE) {
        const enum ggml_op ops_ns[2] = { GGML_OP_RMS_NORM, GGML_OP_SCALE };
        // the q and k l2 norms back to back (views between them): one launch for both
        if (ggml_can_fuse(cgraph, i, ops_ns, 2)) {
            int j = i + 2;
            while (j < cgraph->n_nodes && ggml_cuda_is_view_or_noop(cgraph->nodes[j])) {
                ++j;
            }
            if (j + 1 < cgraph->n_nodes && cgraph->nodes[j]->op == GGML_OP_RMS_NORM && cgraph->nodes[j + 1]->op == GGML_OP_SCALE &&
                    cgraph->nodes[j]->src[0] != cgraph->nodes[i + 1] && cgraph->nodes[j]->src[0]->view_src != cgraph->nodes[i + 1] &&
                    ggml_can_fuse(cgraph, j, ops_ns, 2) &&
                    ggml_cuda_op_rms_norm_scale2(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[j], cgraph->nodes[j + 1])) {
                return j + 1 - i;
            }
        }
        if (ggml_can_fuse(cgraph, i, ops_ns, 2) &&
                ggml_cuda_op_rms_norm_scale(*cuda_ctx, node, cgraph->nodes[i + 1])) {
            return 1;
        }
    }

    return 0;
}
