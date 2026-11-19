// included by sampling.cpp

// Block verification, the decision itself (sampling.h). P[i]: target distribution after X^i, i = 0..G;
// dists[i]: draft distribution the i-th draft token was drawn from; ids[i]: a token drawn from P[i].
// Returns tau (accepted drafts) and the correction/bonus token y; keep[i] is p_i of the paper.
size_t common_spec_block_verify(const std::vector<std::vector<llama_token_data>> & P,
                                const std::vector<std::vector<llama_token_data>> & dists,
                                const llama_tokens & draft, const std::vector<llama_token> & ids,
                                std::mt19937 & rng, llama_token & y, std::vector<double> & keep) {
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    const size_t G = draft.size();
    const auto prob = [](const std::vector<llama_token_data> & d, llama_token t) {
        for (const auto & c : d) {
            if (c.id == t) {
                return (double) c.p;
            }
        }
        return 0.0;
    };
    keep.assign(G + 1, 1.0); // p_i: probability the prefix X^i is kept
    size_t tau = 0;
    for (size_t i = 1; i <= G; ++i) {
        const double eta = unif(rng);
        const double qx  = prob(dists[i - 1], draft[i - 1]);
        keep[i] = qx > 0.0 ? std::min(keep[i - 1] * prob(P[i - 1], draft[i - 1]) / qx, 1.0) : 0.0;
        double h = keep[i];
        if (i < G) {
            double S = 0.0;
            for (const auto & c : P[i]) {
                S += std::max(0.0, keep[i] * c.p - prob(dists[i], c.id));
            }
            const double den = S + 1.0 - keep[i];
            h = den > 0.0 ? S / den : 0.0;
        }
        if (eta <= h) {
            tau = i;
        }
    }
    y = ids[tau];
    if (tau < G) {
        // residual max(p_tau * M_b - M_s, 0) at X^tau
        double sum_r = 0.0;
        for (const auto & c : P[tau]) {
            sum_r += std::max(0.0, keep[tau] * c.p - prob(dists[tau], c.id));
        }
        if (sum_r > 0.0) {
            double t = unif(rng) * sum_r;
            for (const auto & c : P[tau]) {
                const double r = std::max(0.0, keep[tau] * c.p - prob(dists[tau], c.id));
                if (r > 0.0) {
                    y = c.id;
                    t -= r;
                    if (t < 0.0) {
                        break;
                    }
                }
            }
        }
    }
    return tau;
}

std::vector<llama_token> common_sampler_sample_and_accept_n_dist(struct common_sampler * gsmpl, struct llama_context * ctx, const std::vector<int> & idxs, const llama_tokens & draft,
        const std::vector<std::vector<llama_token_data>> & dists, std::mt19937 & rng) {
    GGML_ASSERT(idxs.size() == draft.size() + 1 && "idxs.size() must be draft.size() + 1");
    GGML_ASSERT(dists.size() >= draft.size());

    llama_synchronize(ctx);
    if (llama_get_sampled_token_ith(ctx, idxs[0]) != LLAMA_TOKEN_NULL) {
        // a backend sampler picked the tokens; cur_p is not the distribution they came from
        return common_sampler_sample_and_accept_n(gsmpl, ctx, idxs, draft);
    }

    std::uniform_real_distribution<double> unif(0.0, 1.0);

    std::vector<llama_token> result;
    result.reserve(idxs.size());

    // Block verification (Sun et al., arXiv:2403.10444, Algorithm 2)
    // instead of the per-token rule. Same output distribution (their Theorem 1), and never fewer
    // accepted tokens in expectation (Theorem 2): a rejection at position i does not end the block,
    // a later position can still accept the longer prefix, and the correction token is drawn from a
    // residual scaled by the prefix's keep probability. It needs every position's target
    // distribution up front, so only for a stateless chain: no grammar, penalties, DRY, mirostat
    // or reasoning budget (their outputs would depend on the tokens accepted before them).
    const auto & sp = gsmpl->params;
    if (!gsmpl->grmr && !gsmpl->rbudget && sp.penalty_repeat == 1.0f && sp.penalty_freq == 0.0f &&
            sp.penalty_present == 0.0f && sp.dry_multiplier == 0.0f && sp.mirostat == 0) {
        const size_t G = draft.size();
        // target distributions after X^i (i = 0..G) and a token drawn from each
        std::vector<std::vector<llama_token_data>> P(G + 1);
        std::vector<llama_token> ids(G + 1);
        for (size_t i = 0; i <= G; ++i) {
            ids[i] = common_sampler_sample(gsmpl, ctx, idxs[i]);
            P[i].assign(gsmpl->cur_p.data, gsmpl->cur_p.data + gsmpl->cur_p.size);
        }
        std::vector<double> keep;
        llama_token y;
        const size_t tau = common_spec_block_verify(P, dists, draft, ids, rng, y, keep);
        for (size_t i = 0; i < tau; ++i) {
            common_sampler_accept(gsmpl, draft[i], true);
            result.push_back(draft[i]);
        }
        common_sampler_accept(gsmpl, y, true);
        result.push_back(y);
        return result;
    }

    for (size_t i = 0; i < draft.size(); i++) {
        const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i]);

        if (grammar_should_apply(gsmpl)) {
            // grammar rejection sampling does not draw from cur_p
            common_sampler_accept(gsmpl, id, true);
            result.push_back(id);
            if (draft[i] != id) {
                return result;
            }
            continue;
        }

        const llama_token_data_array & cur_p = gsmpl->cur_p;
        const std::vector<llama_token_data> & q = dists[i];

        const auto q_of = [&](llama_token t) {
            for (const auto & c : q) {
                if (c.id == t) {
                    return (double) c.p;
                }
            }
            return 0.0;
        };

        double p_x = 0.0;
        for (size_t j = 0; j < cur_p.size; ++j) {
            if (cur_p.data[j].id == draft[i]) {
                p_x = cur_p.data[j].p;
                break;
            }
        }
        const double q_x = q_of(draft[i]);

        if (q_x > 0.0 && unif(rng) * q_x < p_x) {
            common_sampler_accept(gsmpl, draft[i], true);
            result.push_back(draft[i]);
            continue;
        }

        // rejected: draw from the residual max(p - q, 0)
        double sum_r = 0.0;
        for (size_t j = 0; j < cur_p.size; ++j) {
            sum_r += std::max(0.0, (double) cur_p.data[j].p - q_of(cur_p.data[j].id));
        }

        llama_token y = id; // only if rounding left no residual mass
        if (sum_r > 0.0) {
            double t = unif(rng) * sum_r;
            for (size_t j = 0; j < cur_p.size; ++j) {
                const double r = std::max(0.0, (double) cur_p.data[j].p - q_of(cur_p.data[j].id));
                if (r > 0.0) {
                    y = cur_p.data[j].id;
                    t -= r;
                    if (t < 0.0) {
                        break;
                    }
                }
            }
        }

        common_sampler_accept(gsmpl, y, true);
        result.push_back(y);

        return result;
    }

    const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[draft.size()]);

    common_sampler_accept(gsmpl, id, true);

    result.push_back(id);

    return result;
}
