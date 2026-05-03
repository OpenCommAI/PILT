// pilt/aggregator.cpp -- PS-side aggregator for PILT uploads.

#include "pilt/aggregator.hpp"

#include <stdexcept>

namespace pilt {

PILTAggregator::PILTAggregator(uint32_t n_workers,
                               std::vector<uint32_t> expected_layer_sizes)
    : n_workers_(n_workers),
      layer_sizes_(std::move(expected_layer_sizes)),
      seen_(n_workers, false),
      sum_(layer_sizes_.size()),
      count_(layer_sizes_.size())
{
    if (n_workers == 0) {
        throw PiltError("PILTAggregator: n_workers must be > 0");
    }
    reset_buffers_();
}

void PILTAggregator::reset_buffers_() {
    for (size_t l = 0; l < layer_sizes_.size(); ++l) {
        sum_[l].assign(layer_sizes_[l], 0.0f);
        count_[l].assign(layer_sizes_[l], 0u);
    }
    seen_.assign(n_workers_, false);
    n_received_ = 0;
    round_open_ = true;
}

void PILTAggregator::begin_round(uint32_t round_num) {
    std::lock_guard<std::mutex> lock(mu_);
    current_round_ = round_num;
    reset_buffers_();
}

bool PILTAggregator::worker_already_seen_(uint16_t wid) const {
    return wid < seen_.size() && seen_[wid];
}

void PILTAggregator::add(const uint8_t* data, size_t len) {
    DecodedMessage msg = decode_message(data, len);
    add_decoded(msg);
}

void PILTAggregator::add_decoded(const DecodedMessage& msg) {
    std::lock_guard<std::mutex> lock(mu_);

    if (!round_open_) {
        // First add() of a fresh round; latch round_num from the message.
        current_round_ = msg.round_num;
        reset_buffers_();
    }
    if (msg.round_num != current_round_) {
        throw PiltError(
            "aggregator.add: round mismatch (got " +
            std::to_string(msg.round_num) +
            ", expected " + std::to_string(current_round_) + ")");
    }
    if (msg.worker_id >= n_workers_) {
        throw PiltError("aggregator.add: worker_id out of range");
    }
    if (worker_already_seen_(msg.worker_id)) {
        throw PiltError(
            "aggregator.add: duplicate worker_id " +
            std::to_string(msg.worker_id) + " in round " +
            std::to_string(current_round_));
    }
    if (msg.layers.size() != layer_sizes_.size()) {
        throw PiltError("aggregator.add: layer count mismatch");
    }

    for (size_t l = 0; l < msg.layers.size(); ++l) {
        const auto& lay = msg.layers[l];
        if (lay.layer_size != layer_sizes_[l]) {
            throw PiltError(
                "aggregator.add: layer " + std::to_string(l) +
                " size mismatch (got " + std::to_string(lay.layer_size) +
                ", expected " + std::to_string(layer_sizes_[l]) + ")");
        }
        const uint32_t S_l = layer_sizes_[l];
        if (S_l == 0) continue;
        auto& dst_sum = sum_[l];
        auto& dst_cnt = count_[l];
        const size_t k = lay.indices.size();
        for (size_t i = 0; i < k; ++i) {
            const uint32_t idx = lay.indices[i];
            if (idx >= S_l) {
                throw PiltError(
                    "aggregator.add: index " + std::to_string(idx) +
                    " >= S_l=" + std::to_string(S_l) +
                    " in layer " + std::to_string(l));
            }
            dst_sum[idx] += lay.values[i];
            dst_cnt[idx] += 1u;
        }
    }

    seen_[msg.worker_id] = true;
    ++n_received_;
}

LayerSet PILTAggregator::finalize_average() {
    std::lock_guard<std::mutex> lock(mu_);
    LayerSet out(layer_sizes_.size());
    for (size_t l = 0; l < layer_sizes_.size(); ++l) {
        const uint32_t S_l = layer_sizes_[l];
        out[l].assign(S_l, 0.0f);
        if (S_l == 0) continue;
        const auto& s = sum_[l];
        const auto& c = count_[l];
        for (uint32_t i = 0; i < S_l; ++i) {
            out[l][i] = (c[i] > 0u)
                            ? s[i] / static_cast<float>(c[i])
                            : 0.0f;
        }
    }
    // Hand the round back to the caller cleanly: future add() will be
    // rejected until begin_round() is called for the next round (or until
    // the next add() implicitly opens a fresh round, see add_decoded()).
    round_open_ = false;
    n_received_ = 0;
    seen_.assign(n_workers_, false);
    return out;
}

void PILTAggregator::abort_round() {
    std::lock_guard<std::mutex> lock(mu_);
    round_open_ = false;
    n_received_ = 0;
    seen_.assign(n_workers_, false);
}

}  // namespace pilt
