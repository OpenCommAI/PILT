// ltp/protocol.hpp -- Loss-Tolerant Protocol C++ implementation.
//
// LTP transmits the full gradient on a lossy channel and decides at the
// receiver, based on how much has arrived by the LT-threshold time,
// whether to close the flow early or wait for the deadline. Positions
// that never arrived are filled by a Random-K bubble-fill from the
// delivered subset.
//
// Public API mirrors libpilt and libdctcp:
//
//   ltp::Encoder enc(layer_sizes);
//   auto bytes = enc.encode(round, worker_id, grads);
//   ltp::Aggregator ps(K, layer_sizes, ec_config);
//   ps.begin_round(round);
//   for (k in workers) ps.add_with_loss(bytes_k, delivered_fraction_k);
//   auto avg = ps.finalize_average();

#pragma once

#include "common/grad_codec.hpp"
#include "common/types.hpp"

#include <cstdint>
#include <random>
#include <vector>

namespace ltp {

struct EarlyCloseConfig {
    // If by the LT-threshold the arrived fraction is at least
    // `min_percent`, close the flow early; otherwise wait until the
    // deadline. Concrete millisecond values are computed by the caller
    // from its FCT model.
    float min_percent  = 0.50f;     // fraction of bytes needed at LT
    uint64_t seed      = 0xC0FFEEull;  // deterministic bubble-fill draw
};

enum class CloseReason : uint8_t {
    Full     = 0,    // the flow finished before LT-threshold
    Early    = 1,    // arrived_fraction >= min_percent at LT-threshold
    Deadline = 2,    // hit deadline without reaching min_percent
};

class Encoder {
public:
    explicit Encoder(std::vector<uint32_t> layer_sizes)
        : layer_sizes_(std::move(layer_sizes)) {}

    proto::ByteBuf encode(uint32_t round_num,
                          uint16_t worker_id,
                          const proto::LayerSet& grads) const {
        if (grads.size() != layer_sizes_.size()) {
            throw proto::ProtocolError("ltp::Encoder: layer count mismatch");
        }
        for (size_t l = 0; l < layer_sizes_.size(); ++l) {
            if (grads[l].size() != layer_sizes_[l]) {
                throw proto::ProtocolError("ltp::Encoder: layer size mismatch");
            }
        }
        // LTP transmits the entire dense gradient; loss-tolerance kicks
        // in at the aggregator side (Early-Close + bubble-fill).  Wire
        // format is the shared dense codec.
        return proto::grad_encode(worker_id, round_num, grads);
    }

    const std::vector<uint32_t>& layer_sizes() const noexcept { return layer_sizes_; }

private:
    std::vector<uint32_t> layer_sizes_;
};

// Aggregator: feeds in (frame, delivered_fraction) per worker, where
// `delivered_fraction in [0,1]` is what the network actually carried by
// the time the PS closed the flow.  When < 1, the aggregator applies
// Random-K bubble-fill on the missing positions of EACH layer (matching
// of the LTP algorithm).
class Aggregator {
public:
    Aggregator(uint32_t n_workers,
               std::vector<uint32_t> layer_sizes,
               EarlyCloseConfig ec = EarlyCloseConfig{})
        : n_workers_(n_workers),
          layer_sizes_(std::move(layer_sizes)),
          ec_(ec),
          rng_(ec.seed) {}

    void begin_round(uint32_t round) {
        round_num_ = round;
        sum_.assign(layer_sizes_.size(), proto::LayerVec{});
        for (size_t l = 0; l < layer_sizes_.size(); ++l) {
            sum_[l].assign(layer_sizes_[l], 0.0f);
        }
        n_received_ = 0;
    }

    // Per-worker semantics for `arrived_fraction`:
    //   1.0  -> flow closed normally ("Full"); use entire dense gradient.
    //   <1   -> early-close OR deadline; bubble-fill missing positions.
    //   ==0  -> the flow delivered nothing this round (worker dropped).
    void add_with_loss(const proto::ByteBuf& bytes,
                       float arrived_fraction) {
        if (arrived_fraction < 0.0f) arrived_fraction = 0.0f;
        if (arrived_fraction > 1.0f) arrived_fraction = 1.0f;

        auto msg = proto::grad_decode(bytes);
        if (msg.round != round_num_) {
            throw proto::ProtocolError("ltp::Aggregator: round mismatch");
        }
        if (msg.layers.size() != layer_sizes_.size()) {
            throw proto::ProtocolError("ltp::Aggregator: layer count mismatch");
        }

        // Apply per-layer truncation + bubble-fill.  For the byte-rate
        // arrival model used in the Python sim (uniform constant bitrate),
        // each layer receives the same fraction.
        for (size_t l = 0; l < layer_sizes_.size(); ++l) {
            const auto& src = msg.layers[l];
            auto& dst = sum_[l];
            const uint32_t S_l = static_cast<uint32_t>(src.size());
            if (S_l == 0) continue;

            uint32_t k = static_cast<uint32_t>(arrived_fraction * S_l);
            if (k == 0 && arrived_fraction > 0.0f) k = 1;     // at least one
            if (k > S_l) k = S_l;

            if (k == S_l) {
                for (uint32_t i = 0; i < S_l; ++i) dst[i] += src[i];
                continue;
            }
            if (k == 0) continue;     // worker contributed nothing

            // Random-K positions delivered this round.  Reservoir of
            // indices [0, S_l) -> sample k without replacement.
            std::vector<uint32_t> idx(S_l);
            for (uint32_t i = 0; i < S_l; ++i) idx[i] = i;
            // Partial Fisher-Yates: pick k from the prefix.
            for (uint32_t i = 0; i < k; ++i) {
                std::uniform_int_distribution<uint32_t> d(i, S_l - 1);
                std::swap(idx[i], idx[d(rng_)]);
                dst[idx[i]] += src[idx[i]];
            }

            // Bubble-fill: for the (S_l - k) missing positions sample
            // with replacement from the k delivered values; preserves
            // the expected per-element magnitude.
            std::uniform_int_distribution<uint32_t> pick(0, k - 1);
            for (uint32_t j = k; j < S_l; ++j) {
                const uint32_t donor_pos = idx[pick(rng_)];
                dst[idx[j]] += src[donor_pos];
            }
        }
        ++n_received_;
    }

    size_t messages_received() const noexcept { return n_received_; }

    proto::LayerSet finalize_average() {
        proto::LayerSet out(layer_sizes_.size());
        if (n_received_ == 0) {
            for (size_t l = 0; l < layer_sizes_.size(); ++l) {
                out[l].assign(layer_sizes_[l], 0.0f);
            }
            return out;
        }
        const float invK = 1.0f / static_cast<float>(n_received_);
        for (size_t l = 0; l < layer_sizes_.size(); ++l) {
            out[l] = std::move(sum_[l]);
            for (auto& x : out[l]) x *= invK;
        }
        n_received_ = 0;
        return out;
    }

    uint32_t n_workers() const noexcept { return n_workers_; }
    const std::vector<uint32_t>& layer_sizes() const noexcept { return layer_sizes_; }

private:
    uint32_t n_workers_;
    std::vector<uint32_t> layer_sizes_;
    EarlyCloseConfig ec_;
    std::mt19937_64 rng_;
    uint32_t round_num_ = 0;
    size_t n_received_ = 0;
    proto::LayerSet sum_;
};

// Convenience: stateless Early-Close decision function.
// Inputs are the wall-clock arrival fractions at LT-threshold and at
// deadline (caller computes them from its own FCT model).  Returns the
// "arrived" fraction the aggregator should be told to use.
struct EarlyCloseDecision {
    CloseReason reason = CloseReason::Deadline;
    float arrived = 0.0f;
};

inline EarlyCloseDecision decide_close(float at_lt,
                                       float at_deadline,
                                       float full_completion_or_neg,
                                       const EarlyCloseConfig& cfg) {
    EarlyCloseDecision out;
    if (full_completion_or_neg >= 0.0f) {
        out.reason = CloseReason::Full;
        out.arrived = 1.0f;
        return out;
    }
    if (at_lt >= cfg.min_percent) {
        out.reason = CloseReason::Early;
        out.arrived = at_lt;
        return out;
    }
    out.reason = CloseReason::Deadline;
    out.arrived = at_deadline;
    return out;
}

}  // namespace ltp
