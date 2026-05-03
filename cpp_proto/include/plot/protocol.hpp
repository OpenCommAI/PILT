// plot/protocol.hpp -- PLOT C++ implementation.
//
// Per-layer Layer-LTT (Loss-Tolerance Threshold). On the first send
// each layer's delivered fraction is observed; layers whose loss
// exceeds their LTT trigger a second, retx-only round. The aggregator
// merges the retx pieces into the per-layer accumulator.
//
// Wire format reuses the dense gradient codec; the retx round encodes
// only the layers in `retx_layers_per_worker[k]`.

#pragma once

#include "common/grad_codec.hpp"
#include "common/types.hpp"

#include <cstdint>
#include <random>
#include <vector>

namespace plot {

struct LayerLttConfig {
    // Per-layer LTT (delivered_fraction below this triggers retx).
    float default_ltt = 0.7f;
    // Optional per-layer override; size must equal layer count.
    std::vector<float> per_layer_ltt;
    uint64_t seed = 0xBADC0DEull;
};

class Encoder {
public:
    explicit Encoder(std::vector<uint32_t> layer_sizes)
        : layer_sizes_(std::move(layer_sizes)) {}

    // Initial round: encode the entire gradient (dense).
    proto::ByteBuf encode(uint32_t round_num,
                          uint16_t worker_id,
                          const proto::LayerSet& grads) const {
        if (grads.size() != layer_sizes_.size()) {
            throw proto::ProtocolError("plot::Encoder: layer count mismatch");
        }
        for (size_t l = 0; l < layer_sizes_.size(); ++l) {
            if (grads[l].size() != layer_sizes_[l]) {
                throw proto::ProtocolError("plot::Encoder: layer size mismatch");
            }
        }
        return proto::grad_encode(worker_id, round_num, grads);
    }

    // Retx round: encode only the layers in `retx_layers`; non-retx
    // layers are emitted as size-0 placeholders so the receiver can match
    // by index without an extra side-channel of "which layers retx'd".
    proto::ByteBuf encode_retx(uint32_t round_num,
                               uint16_t worker_id,
                               const proto::LayerSet& grads,
                               const std::vector<uint8_t>& retx_layers) const {
        if (grads.size() != layer_sizes_.size() ||
            retx_layers.size() != layer_sizes_.size()) {
            throw proto::ProtocolError("plot::Encoder: retx mask size mismatch");
        }
        proto::LayerSet retx_payload(layer_sizes_.size());
        for (size_t l = 0; l < layer_sizes_.size(); ++l) {
            if (retx_layers[l]) {
                retx_payload[l] = grads[l];
            } else {
                retx_payload[l].clear();          // empty -> 4-byte header only
            }
        }
        return proto::grad_encode(worker_id, round_num, retx_payload);
    }

    const std::vector<uint32_t>& layer_sizes() const noexcept { return layer_sizes_; }

private:
    std::vector<uint32_t> layer_sizes_;
};

// Result of the first-pass loss-detection step: which layers (per worker)
// must be retransmitted, and which can keep the bubble-filled estimate.
struct RetxDecision {
    // [K][L] bool: 1 if (worker k, layer l) goes into retx.
    std::vector<std::vector<uint8_t>> retx_mask;
    // [K]: total bytes the second NS-3 / TCP / RDMA pass would need to
    // carry per worker.  Useful for a bandwidth estimate without doing
    // the second round.
    std::vector<uint64_t> retx_bytes_per_worker;
};

class Aggregator {
public:
    Aggregator(uint32_t n_workers,
               std::vector<uint32_t> layer_sizes,
               LayerLttConfig cfg = LayerLttConfig{})
        : n_workers_(n_workers),
          layer_sizes_(std::move(layer_sizes)),
          cfg_(std::move(cfg)),
          rng_(cfg_.seed)
    {
        if (cfg_.per_layer_ltt.empty()) {
            cfg_.per_layer_ltt.assign(layer_sizes_.size(), cfg_.default_ltt);
        } else if (cfg_.per_layer_ltt.size() != layer_sizes_.size()) {
            throw proto::ProtocolError(
                "plot::Aggregator: per_layer_ltt size mismatch");
        }
    }

    void begin_round(uint32_t round) {
        round_num_ = round;
        sum_.assign(layer_sizes_.size(), proto::LayerVec{});
        cnt_.assign(layer_sizes_.size(), 0);
        for (size_t l = 0; l < layer_sizes_.size(); ++l) {
            sum_[l].assign(layer_sizes_[l], 0.0f);
        }
        n_received_ = 0;
        per_worker_layer_arrived_.assign(
            n_workers_, std::vector<float>(layer_sizes_.size(), 0.0f));
    }

    // First pass: feed (encoded bytes, per-layer arrived fraction) for one
    // worker.  Bubble-fills layers under their LTT and accumulates them.
    // Layers above their LTT are stashed (count_l increments) but kept
    // pending for the retx round.
    void add_first_pass(const proto::ByteBuf& bytes,
                        const std::vector<float>& delivered_per_layer) {
        auto msg = proto::grad_decode(bytes);
        if (msg.round != round_num_) {
            throw proto::ProtocolError("plot::Aggregator: round mismatch");
        }
        if (msg.layers.size() != layer_sizes_.size() ||
            delivered_per_layer.size() != layer_sizes_.size()) {
            throw proto::ProtocolError("plot::Aggregator: layer count mismatch");
        }
        if (msg.worker_id >= n_workers_) {
            throw proto::ProtocolError("plot::Aggregator: worker id out of range");
        }
        per_worker_layer_arrived_[msg.worker_id] = delivered_per_layer;

        for (size_t l = 0; l < layer_sizes_.size(); ++l) {
            const float frac = std::max(0.0f, std::min(1.0f, delivered_per_layer[l]));
            if (frac >= cfg_.per_layer_ltt[l]) {
                // Above LTT -> bubble-fill and accept this round.
                merge_layer_with_bubble_fill_(l, msg.layers[l], frac);
                cnt_[l] += 1;
            }
            // Below LTT: defer; aggregator will get this layer via retx.
        }
        ++n_received_;
    }

    // After all K workers' first-pass contributions are in, this returns
    // the per-worker retx mask and projected retx byte budget.  The mask
    // is empty (all zeros) when no layers exceeded their LTT.
    RetxDecision plan_retx() const {
        RetxDecision out;
        out.retx_mask.assign(n_workers_, std::vector<uint8_t>(layer_sizes_.size(), 0));
        out.retx_bytes_per_worker.assign(n_workers_, 0);
        for (uint32_t k = 0; k < n_workers_; ++k) {
            for (size_t l = 0; l < layer_sizes_.size(); ++l) {
                const float frac = per_worker_layer_arrived_[k][l];
                if (frac < cfg_.per_layer_ltt[l]) {
                    out.retx_mask[k][l] = 1;
                    out.retx_bytes_per_worker[k] +=
                        4ull * static_cast<uint64_t>(layer_sizes_[l]);
                }
            }
        }
        return out;
    }

    // Second pass: feed the retx-round bytes from one worker.  The
    // encoder must have produced this with `Encoder::encode_retx(...)`,
    // which leaves non-retx layers as 0-length placeholders.
    void add_retx_pass(const proto::ByteBuf& bytes,
                       const std::vector<float>& delivered_per_layer) {
        auto msg = proto::grad_decode(bytes);
        if (msg.round != round_num_) {
            throw proto::ProtocolError("plot::Aggregator: round mismatch");
        }
        if (msg.worker_id >= n_workers_) {
            throw proto::ProtocolError("plot::Aggregator: worker id out of range");
        }
        for (size_t l = 0; l < layer_sizes_.size(); ++l) {
            const auto& src = msg.layers[l];
            if (src.empty()) continue;       // not a retx layer
            if (src.size() != layer_sizes_[l]) {
                throw proto::ProtocolError("plot::Aggregator: retx layer size mismatch");
            }
            // Retx may itself be partially delivered; bubble-fill again.
            const float frac = std::max(0.0f, std::min(1.0f, delivered_per_layer[l]));
            merge_layer_with_bubble_fill_(l, src, frac);
            cnt_[l] += 1;
        }
    }

    proto::LayerSet finalize_average() {
        proto::LayerSet out(layer_sizes_.size());
        for (size_t l = 0; l < layer_sizes_.size(); ++l) {
            out[l] = std::move(sum_[l]);
            const uint32_t c = cnt_[l];
            if (c > 0) {
                const float inv = 1.0f / static_cast<float>(c);
                for (auto& x : out[l]) x *= inv;
            }
        }
        n_received_ = 0;
        return out;
    }

    size_t messages_received() const noexcept { return n_received_; }
    uint32_t n_workers() const noexcept { return n_workers_; }
    const std::vector<uint32_t>& layer_sizes() const noexcept { return layer_sizes_; }

private:
    void merge_layer_with_bubble_fill_(size_t l,
                                       const proto::LayerVec& src,
                                       float frac) {
        const uint32_t S_l = layer_sizes_[l];
        if (S_l == 0) return;
        if (frac >= 1.0f - 1e-6f) {
            for (uint32_t i = 0; i < S_l; ++i) sum_[l][i] += src[i];
            return;
        }
        if (frac <= 0.0f) return;

        uint32_t k = static_cast<uint32_t>(frac * S_l);
        if (k == 0) k = 1;
        if (k > S_l) k = S_l;

        std::vector<uint32_t> idx(S_l);
        for (uint32_t i = 0; i < S_l; ++i) idx[i] = i;
        for (uint32_t i = 0; i < k; ++i) {
            std::uniform_int_distribution<uint32_t> d(i, S_l - 1);
            std::swap(idx[i], idx[d(rng_)]);
            sum_[l][idx[i]] += src[idx[i]];
        }
        std::uniform_int_distribution<uint32_t> pick(0, k - 1);
        for (uint32_t j = k; j < S_l; ++j) {
            sum_[l][idx[j]] += src[idx[pick(rng_)]];
        }
    }

    uint32_t n_workers_;
    std::vector<uint32_t> layer_sizes_;
    LayerLttConfig cfg_;
    std::mt19937_64 rng_;
    uint32_t round_num_ = 0;
    size_t n_received_ = 0;
    proto::LayerSet sum_;
    std::vector<uint32_t> cnt_;
    std::vector<std::vector<float>> per_worker_layer_arrived_;
};

}  // namespace plot
