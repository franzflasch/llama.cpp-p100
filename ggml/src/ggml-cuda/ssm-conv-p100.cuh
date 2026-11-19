#pragma once

#include "common.cuh"

// single-token CONCAT -> CPY(state tail) -> SSM_CONV -> SILU in one launch (bit-identical); false if the shapes do not fit
bool ggml_cuda_ssm_conv_decode_fused_ok(const ggml_tensor * concat, const ggml_tensor * cpy, const ggml_tensor * conv,
                                        const ggml_tensor * silu);
bool ggml_cuda_op_ssm_conv_decode_fused(ggml_backend_cuda_context & ctx, const ggml_tensor * concat, const ggml_tensor * cpy,
                                        const ggml_tensor * conv, const ggml_tensor * silu);
