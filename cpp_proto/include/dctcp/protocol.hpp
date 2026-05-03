// dctcp/protocol.hpp -- "DCTCP-style" reliable full-gradient transport.
//
// The DCTCP-specific parts of this protocol live entirely in the kernel
// (ECN marking, reduced-cwnd reaction).  At the application layer DCTCP
// is just a reliable, in-order, congestion-controlled byte stream -- so
// the encoder/aggregator are a thin wrapper around the dense gradient
// codec in <common/grad_codec.hpp> and a TCP socket created with
// `proto::tcp::try_set_dctcp(fd)` enabled where the kernel allows it.
//
// Why expose this at all?  Because it gives every protocol the same C++
// API surface (encode -> bytes / aggregate -> dense LayerSet), so the
// benchmark harness in `bench/compare_protocols.cpp` can swap protocols
// transparently.

#pragma once

#include "common/grad_codec.hpp"
#include "common/types.hpp"

#include <cstdint>

namespace dctcp {

class Encoder {
public:
    explicit Encoder(std::vector<uint32_t> layer_sizes)
        : layer_sizes_(std::move(layer_sizes)) {}

    proto::ByteBuf encode(uint32_t round_num,
                          uint16_t worker_id,
                          const proto::LayerSet& grads) const {
        if (grads.size() != layer_sizes_.size()) {
            throw proto::ProtocolError("dctcp::Encoder: layer count mismatch");
        }
        for (size_t l = 0; l < layer_sizes_.size(); ++l) {
            if (grads[l].size() != layer_sizes_[l]) {
                throw proto::ProtocolError(
                    "dctcp::Encoder: layer size mismatch");
            }
        }
        return proto::grad_encode(worker_id, round_num, grads);
    }

    const std::vector<uint32_t>& layer_sizes() const noexcept { return layer_sizes_; }

private:
    std::vector<uint32_t> layer_sizes_;
};

class Aggregator {
public:
    Aggregator(uint32_t n_workers, std::vector<uint32_t> layer_sizes)
        : n_workers_(n_workers), layer_sizes_(std::move(layer_sizes)) {}

    void begin_round(uint32_t round_num) {
        round_num_ = round_num;
        msgs_.clear();
        msgs_.reserve(n_workers_);
    }

    void add(const proto::ByteBuf& buf) {
        auto m = proto::grad_decode(buf);
        if (m.round != round_num_) {
            throw proto::ProtocolError("dctcp::Aggregator: round mismatch");
        }
        if (m.layers.size() != layer_sizes_.size()) {
            throw proto::ProtocolError("dctcp::Aggregator: layer count mismatch");
        }
        msgs_.push_back(std::move(m));
    }

    size_t messages_received() const noexcept { return msgs_.size(); }

    proto::LayerSet finalize_average() {
        auto out = proto::grad_average(msgs_, layer_sizes_);
        msgs_.clear();
        return out;
    }

    uint32_t n_workers() const noexcept { return n_workers_; }
    const std::vector<uint32_t>& layer_sizes() const noexcept { return layer_sizes_; }

private:
    uint32_t n_workers_;
    std::vector<uint32_t> layer_sizes_;
    uint32_t round_num_ = 0;
    std::vector<proto::DecodedGrad> msgs_;
};

}  // namespace dctcp
