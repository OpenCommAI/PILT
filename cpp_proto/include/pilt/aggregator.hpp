// pilt/aggregator.hpp -- PS-side aggregator for PILT uploads.
//
// As each worker's encoded message arrives (via TCP, RDMA, gRPC, ...)
// the PS calls add() with the raw byte buffer; the aggregator validates
// the wire frame, decodes it, and accumulates a per-layer running sum
// plus per-element receive counter. finalize_average() returns the
// per-layer dense averaged tensor.
//
// Aggregation semantics:
//   n[l][i] = workers that transmitted element (l, i) this round
//   s[l][i] = sum of those workers' values
//   avg[l][i] = s[l][i] / n[l][i] if n[l][i] > 0, else 0.0
// Missing positions are not imputed to zero downstream; the PS reuses
// the prior global value.

#pragma once

#include "pilt/types.hpp"
#include "pilt/wire.hpp"

#include <cstdint>
#include <vector>
#include <atomic>
#include <mutex>

namespace pilt {

class PILTAggregator {
public:
    // `expected_layer_sizes` MUST match what the workers' encoders were
    // built with.  Mismatches are flagged on add() with PiltError.
    PILTAggregator(uint32_t n_workers,
                   std::vector<uint32_t> expected_layer_sizes);

    // Number of distinct workers' messages accepted so far this round.
    size_t messages_received() const noexcept { return n_received_; }

    // Round number expected (set on first add() of a fresh round, or via
    // begin_round() ahead of time).  Late / duplicated workers are
    // rejected.
    void begin_round(uint32_t round_num);

    // Accept one worker's wire buffer.  Thread-safe (mutex-guarded so
    // multi-connection PS loops can dispatch from a worker pool).
    // Throws PiltError on bad frame, layer-size mismatch, duplicate
    // worker_id, or wrong round_num.
    void add(const uint8_t* data, size_t len);
    void add(const ByteBuf& buf) { add(buf.data(), buf.size()); }

    // Same as above but for callers that already decoded the frame.
    void add_decoded(const DecodedMessage& msg);

    // Materialise the per-layer dense averaged gradient and reset internal
    // accumulators for the next round.  Returns vectors of size
    // `expected_layer_sizes[l]` per layer.
    LayerSet finalize_average();

    // Drop pending state without producing an output (use on round abort).
    void abort_round();

    // Read-only views for diagnostics.
    uint32_t round() const noexcept { return current_round_; }
    uint32_t n_workers() const noexcept { return n_workers_; }
    const std::vector<uint32_t>& layer_sizes() const noexcept {
        return layer_sizes_;
    }

private:
    void reset_buffers_();
    bool worker_already_seen_(uint16_t wid) const;

    uint32_t n_workers_;
    std::vector<uint32_t> layer_sizes_;

    std::mutex mu_;
    bool round_open_ = false;
    uint32_t current_round_ = 0;
    size_t   n_received_    = 0;
    std::vector<bool> seen_;                 // per-worker dedup, length n_workers_
    std::vector<std::vector<float>>    sum_;     // per-layer Σ values
    std::vector<std::vector<uint32_t>> count_;   // per-layer Σ workers/elem
};

}  // namespace pilt
