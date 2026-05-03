// pilt/encoder.hpp -- worker-side PILT encoder.
//
// Per-round flow:
//   (1) v_l   <- beta * v_l + (1-beta) * ||g_l||_2 / sqrt(S_l)
//   (2) c_l   <- d * ((L+1)/2 - rank_l)                    (zero-sum)
//   (3) eps_l <- clip(eps_l + c_l, eps_min, 1), then rescale to budget
//   (4) g_tilde_l = g_l (*) M_l + g_prev_l (*) (1-M_l)     (top-|g| mask)
//                   with local residual:
//                     g_total       = g_l + residual_l
//                     residual_l   <- g_total (*) (1-M_l)
//
// Usage:
//
//   PILTEncoder enc(layer_sizes);
//   auto eps   = enc.compute_ratios();
//   auto bytes = enc.encode(round, worker_id, gradients);
//   // PS averages all workers' eff tensors, feeds the average back:
//   enc.update_importance(avg_grads_after_aggregation);
//
// Single-threaded; instantiate one per worker.

#pragma once

#include "pilt/types.hpp"

#include <cstdint>
#include <vector>
#include <cstddef>

namespace pilt {

struct EncoderConfig {
    float beta    = 0.9f;        // EMA decay for v_l
    float d       = 0.05f;       // rank-update step
    float E_total = 0.5f;        // global transmission budget E_total in (0, 1]
    float eps_min = 0.05f;       // per-layer floor on eps_l
};

// Per-encoded layer side-product, useful for logging / debugging /
// downstream analytics.  k_sent[l] is k_l, the number of (idx, val) pairs
// actually emitted for layer l this round.
struct EncodeStats {
    std::vector<uint32_t> k_sent;     // length L
    std::vector<uint32_t> layer_size; // length L (S_l)
};

class PILTEncoder {
public:
    PILTEncoder(std::vector<uint32_t> layer_sizes,
                EncoderConfig cfg = EncoderConfig{});

    // Number of layer groups L the encoder was constructed with.
    size_t n_layers() const noexcept { return layer_sizes_.size(); }

    // Layer sizes (S_l) the encoder was constructed with.
    const std::vector<uint32_t>& layer_sizes() const noexcept {
        return layer_sizes_;
    }

    // Produce next-round per-layer transmission ratios. Must be called
    // once per round before encode(); idempotent (cached) within a round.
    RatioVec compute_ratios();

    // Encode one worker's per-layer gradients into a wire buffer ready
    // for the network layer. `grads.size()` must equal `n_layers()` and
    // each `grads[l].size()` must equal `layer_sizes()[l]`.
    ByteBuf encode(uint32_t round_num,
                   uint16_t worker_id,
                   const LayerSet& grads,
                   EncodeStats* out_stats = nullptr);

    // Consume the PS-broadcast averaged gradient and update v_l.
    void update_importance(const LayerSet& avg_effective_grads);

    // Read-only views of internal state for instrumentation.
    const std::vector<double>& importance() const noexcept { return v_; }
    const RatioVec&            ratios()     const noexcept { return eps_; }

private:
    void enforce_budget_();

    std::vector<uint32_t> layer_sizes_;
    EncoderConfig cfg_;
    bool initialized_v_ = false;
    std::vector<double>   v_;
    RatioVec              eps_;
    bool ratios_dirty_ = true;

    std::vector<LayerVec> residual_;
    std::vector<LayerVec> last_sent_;
};

}  // namespace pilt
