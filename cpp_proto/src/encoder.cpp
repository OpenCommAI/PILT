// pilt/encoder.cpp -- worker-side PILT encoder.

#include "pilt/encoder.hpp"
#include "pilt/wire.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <cstring>
#include <cassert>

namespace pilt {

PILTEncoder::PILTEncoder(std::vector<uint32_t> layer_sizes,
                         EncoderConfig cfg)
    : layer_sizes_(std::move(layer_sizes)),
      cfg_(cfg),
      v_(layer_sizes_.size(), 0.0),
      eps_(layer_sizes_.size(),
           cfg.E_total)              // initial: uniform at E_total
{
    if (cfg_.beta < 0.0f || cfg_.beta >= 1.0f) {
        throw PiltError("EncoderConfig.beta must lie in [0, 1)");
    }
    if (cfg_.E_total <= 0.0f || cfg_.E_total > 1.0f) {
        throw PiltError("EncoderConfig.E_total must lie in (0, 1]");
    }
    if (cfg_.eps_min < 0.0f || cfg_.eps_min > 1.0f) {
        throw PiltError("EncoderConfig.eps_min must lie in [0, 1]");
    }

    residual_.resize(layer_sizes_.size());
    last_sent_.resize(layer_sizes_.size());
    for (size_t l = 0; l < layer_sizes_.size(); ++l) {
        residual_[l].assign(layer_sizes_[l], 0.0f);
        last_sent_[l].assign(layer_sizes_[l], 0.0f);
    }
}

// Rescale eps_l so Sigma eps_l * S_l <= E_total * Sigma S_l; second
// pass handles the eps_min floor.
void PILTEncoder::enforce_budget_() {
    const size_t L = layer_sizes_.size();
    long double sum_S = 0.0L;
    for (uint32_t s : layer_sizes_) sum_S += static_cast<long double>(s);
    long double total_budget = static_cast<long double>(cfg_.E_total) * sum_S;
    if (total_budget <= 0.0L) return;

    auto current_load = [&]() {
        long double cur = 0.0L;
        for (size_t l = 0; l < L; ++l) {
            cur += static_cast<long double>(eps_[l]) *
                   static_cast<long double>(layer_sizes_[l]);
        }
        return cur;
    };

    long double cur = current_load();
    if (cur > total_budget) {
        const float scale = static_cast<float>(total_budget / cur);
        for (size_t l = 0; l < L; ++l) {
            float v = eps_[l] * scale;
            if (v < cfg_.eps_min) v = cfg_.eps_min;
            else if (v > 1.0f)    v = 1.0f;
            eps_[l] = v;
        }
        // Floor may have lifted us back above the budget; one more pass
        // (without re-flooring) brings us back monotonically.
        cur = current_load();
        if (cur > total_budget) {
            const float scale2 = static_cast<float>(total_budget / cur);
            for (size_t l = 0; l < L; ++l) eps_[l] *= scale2;
        }
    }
}

RatioVec PILTEncoder::compute_ratios() {
    if (!ratios_dirty_) return eps_;

    const size_t L = layer_sizes_.size();
    if (L == 0) return eps_;

    // Stable rank by descending v_, ties broken by layer index.
    // Same convention as PILTImportanceTracker.rank().
    std::vector<size_t> order(L);
    std::iota(order.begin(), order.end(), size_t{0});
    std::stable_sort(order.begin(), order.end(),
                     [&](size_t a, size_t b) { return v_[a] > v_[b]; });

    std::vector<int> rank(L);
    for (size_t pos = 0; pos < L; ++pos) rank[order[pos]] = static_cast<int>(pos + 1);

    // c_l = d * ((L+1)/2 - rank_l); eps_l <- clip(eps_l + c_l).
    const float center = 0.5f * (static_cast<float>(L) + 1.0f);
    for (size_t l = 0; l < L; ++l) {
        const float c = cfg_.d * (center - static_cast<float>(rank[l]));
        float v = eps_[l] + c;
        if (v < cfg_.eps_min) v = cfg_.eps_min;
        else if (v > 1.0f)    v = 1.0f;
        eps_[l] = v;
    }
    enforce_budget_();
    ratios_dirty_ = false;
    return eps_;
}

ByteBuf PILTEncoder::encode(uint32_t round_num,
                            uint16_t worker_id,
                            const LayerSet& grads,
                            EncodeStats* out_stats) {
    const size_t L = layer_sizes_.size();
    if (grads.size() != L) {
        throw PiltError("PILTEncoder::encode: grads.size() != n_layers");
    }
    // Make sure ratios for this round are materialised; harmless if the
    // caller already did this explicitly.
    compute_ratios();

    std::vector<IndexVec> idx_per_layer(L);
    std::vector<LayerVec> val_per_layer(L);
    std::vector<uint32_t> sizes(L);

    for (size_t l = 0; l < L; ++l) {
        const uint32_t S_l = layer_sizes_[l];
        sizes[l] = S_l;
        if (grads[l].size() != S_l) {
            throw PiltError(
                "PILTEncoder::encode: layer " + std::to_string(l) +
                " grad has " + std::to_string(grads[l].size()) +
                " elems, expected " + std::to_string(S_l));
        }
        if (S_l == 0) {
            // Defensive empty-layer support (matches Python encoder).
            continue;
        }

        // g_total = g + residual_l   (EF accumulator)
        LayerVec g_total(S_l);
        const float* g_in = grads[l].data();
        const float* g_res = residual_[l].data();
        for (uint32_t i = 0; i < S_l; ++i) g_total[i] = g_in[i] + g_res[i];

        const float eps_l = eps_[l];
        uint32_t k_l = static_cast<uint32_t>(std::lround(static_cast<double>(eps_l) * S_l));
        if (k_l < 1u) k_l = 1u;
        if (k_l > S_l) k_l = S_l;

        // Build mask = top-k by |g_total|.  argpartition-style via
        // std::nth_element on (|g|, idx) descending.
        IndexVec idx;
        idx.reserve(k_l);

        if (k_l >= S_l) {
            idx.resize(S_l);
            std::iota(idx.begin(), idx.end(), uint32_t{0});
        } else {
            std::vector<uint32_t> all(S_l);
            std::iota(all.begin(), all.end(), uint32_t{0});
            // Partition so the last k_l elements are the largest |g_total|.
            const auto cmp = [&](uint32_t a, uint32_t b) {
                return std::fabs(g_total[a]) < std::fabs(g_total[b]);
            };
            std::nth_element(all.begin(), all.end() - k_l, all.end(), cmp);
            idx.assign(all.end() - k_l, all.end());
            // Sorted indices keep wire output deterministic and aid the
            // receiver's sequential merge (smaller branches in inner loop).
            std::sort(idx.begin(), idx.end());
        }

        LayerVec values(idx.size());
        for (size_t i = 0; i < idx.size(); ++i) values[i] = g_total[idx[i]];

        // Update last_sent_ (EF "historical retention"): start from the
        // previous broadcast, overwrite the sent positions.
        LayerVec eff = last_sent_[l];      // copy prior
        for (size_t i = 0; i < idx.size(); ++i) eff[idx[i]] = values[i];
        last_sent_[l] = std::move(eff);

        // Update residual_: keep g_total in unsent positions, zero the rest.
        for (size_t i = 0; i < idx.size(); ++i) g_total[idx[i]] = 0.0f;
        residual_[l] = std::move(g_total);

        idx_per_layer[l] = std::move(idx);
        val_per_layer[l] = std::move(values);
    }

    if (out_stats) {
        out_stats->k_sent.assign(L, 0);
        out_stats->layer_size = sizes;
        for (size_t l = 0; l < L; ++l) {
            out_stats->k_sent[l] = static_cast<uint32_t>(idx_per_layer[l].size());
        }
    }

    // Mark next-round ratios dirty.  Workers normally call update_importance
    // after the PS broadcast, which already sets this flag, but explicit
    // safety here avoids a stale ratio if the protocol skips a round.
    ratios_dirty_ = true;

    return encode_message(worker_id, round_num, sizes,
                          idx_per_layer, val_per_layer);
}

void PILTEncoder::update_importance(const LayerSet& avg_effective_grads) {
    const size_t L = layer_sizes_.size();
    if (avg_effective_grads.size() != L) {
        throw PiltError(
            "update_importance: layer count mismatch (" +
            std::to_string(avg_effective_grads.size()) + " vs " +
            std::to_string(L) + ")");
    }

    std::vector<double> new_norm(L, 0.0);
    for (size_t l = 0; l < L; ++l) {
        const uint32_t S_l = layer_sizes_[l];
        const auto& g = avg_effective_grads[l];
        if (g.size() != S_l) {
            throw PiltError(
                "update_importance: layer " + std::to_string(l) +
                " has " + std::to_string(g.size()) +
                " elems, expected " + std::to_string(S_l));
        }
        if (S_l == 0) continue;
        long double acc = 0.0L;
        for (uint32_t i = 0; i < S_l; ++i) {
            const double x = static_cast<double>(g[i]);
            acc += static_cast<long double>(x * x);
        }
        new_norm[l] = std::sqrt(static_cast<double>(acc)) /
                      std::sqrt(static_cast<double>(std::max<uint32_t>(1u, S_l)));
    }

    if (!initialized_v_) {
        v_ = new_norm;
        initialized_v_ = true;
    } else {
        const double b  = static_cast<double>(cfg_.beta);
        const double mb = 1.0 - b;
        for (size_t l = 0; l < L; ++l) {
            v_[l] = b * v_[l] + mb * new_norm[l];
        }
    }
    ratios_dirty_ = true;     // next compute_ratios() recomputes from new v_
}

}  // namespace pilt
