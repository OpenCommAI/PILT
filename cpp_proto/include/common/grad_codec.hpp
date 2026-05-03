// common/grad_codec.hpp -- shared helpers for serialising / deserialising
// dense per-layer gradient buffers (the building block used by DCTCP, LTP,
// and PLOT, all of which transmit the entire gradient).
//
// Wire format (per message):
//
//     uint32 magic      "GRAD"
//     uint16 worker_id
//     uint16 reserved
//     uint32 round
//     uint16 num_layers
//     uint16 reserved
//     for each layer:
//         uint32 layer_size       -- S_l (element count)
//         float32 values[S_l]
//
// All multi-byte fields are little-endian.  No CRC -- the dense codec is
// used by reliable (DCTCP) and packet-level lossy (LTP / PLOT) protocols
// alike; loss models live in the protocol-specific code that consumes
// these buffers, not in the codec.

#pragma once

#include "common/types.hpp"

#include <cstdint>
#include <cstring>

namespace proto {

constexpr uint32_t kGradMagic = 0x44415247u;   // 'GRAD' little-endian
constexpr size_t   kGradHeaderBytes = 16;

// Returns the exact byte size of a serialised dense gradient.
inline size_t grad_encoded_size(const LayerSet& grads) noexcept {
    size_t n = kGradHeaderBytes;
    for (const auto& l : grads) n += 4 + 4ull * l.size();
    return n;
}

inline ByteBuf grad_encode(uint16_t worker_id,
                           uint32_t round,
                           const LayerSet& grads) {
    if (grads.size() > UINT16_MAX) {
        throw ProtocolError("grad_encode: too many layers");
    }
    ByteBuf out(grad_encoded_size(grads));
    uint8_t* p = out.data();

    auto put_u16 = [&](uint16_t v) {
        p[0] = static_cast<uint8_t>(v);
        p[1] = static_cast<uint8_t>(v >> 8);
        p += 2;
    };
    auto put_u32 = [&](uint32_t v) {
        p[0] = static_cast<uint8_t>(v);
        p[1] = static_cast<uint8_t>(v >> 8);
        p[2] = static_cast<uint8_t>(v >> 16);
        p[3] = static_cast<uint8_t>(v >> 24);
        p += 4;
    };

    put_u32(kGradMagic);
    put_u16(worker_id);
    put_u16(0);
    put_u32(round);
    put_u16(static_cast<uint16_t>(grads.size()));
    put_u16(0);

    for (const auto& layer : grads) {
        put_u32(static_cast<uint32_t>(layer.size()));
        if (!layer.empty()) {
            std::memcpy(p, layer.data(), 4ull * layer.size());
            p += 4ull * layer.size();
        }
    }
    return out;
}

struct DecodedGrad {
    uint16_t worker_id = 0;
    uint32_t round     = 0;
    LayerSet layers;
};

inline DecodedGrad grad_decode(const uint8_t* data, size_t len) {
    if (len < kGradHeaderBytes) {
        throw ProtocolError("grad_decode: buffer too small");
    }
    auto get_u16 = [&](const uint8_t*& p) -> uint16_t {
        uint16_t v = static_cast<uint16_t>(p[0]) |
                     (static_cast<uint16_t>(p[1]) << 8);
        p += 2; return v;
    };
    auto get_u32 = [&](const uint8_t*& p) -> uint32_t {
        uint32_t v = static_cast<uint32_t>(p[0]) |
                     (static_cast<uint32_t>(p[1]) << 8) |
                     (static_cast<uint32_t>(p[2]) << 16) |
                     (static_cast<uint32_t>(p[3]) << 24);
        p += 4; return v;
    };

    const uint8_t* p   = data;
    const uint8_t* end = data + len;
    if (get_u32(p) != kGradMagic) {
        throw ProtocolError("grad_decode: bad magic");
    }
    DecodedGrad out;
    out.worker_id = get_u16(p);
    (void)get_u16(p);
    out.round     = get_u32(p);
    uint16_t L    = get_u16(p);
    (void)get_u16(p);

    out.layers.resize(L);
    for (uint16_t l = 0; l < L; ++l) {
        if (p + 4 > end) throw ProtocolError("grad_decode: truncated layer header");
        uint32_t S_l = get_u32(p);
        if (p + 4ull * S_l > end) throw ProtocolError("grad_decode: truncated layer values");
        out.layers[l].resize(S_l);
        if (S_l) {
            std::memcpy(out.layers[l].data(), p, 4ull * S_l);
            p += 4ull * S_l;
        }
    }
    if (p != end) throw ProtocolError("grad_decode: trailing bytes");
    return out;
}

inline DecodedGrad grad_decode(const ByteBuf& buf) {
    return grad_decode(buf.data(), buf.size());
}

// Convenience: per-element averaged dense LayerSet from K decoded msgs.
inline LayerSet grad_average(const std::vector<DecodedGrad>& msgs,
                             const std::vector<uint32_t>& expected_sizes) {
    LayerSet out(expected_sizes.size());
    if (msgs.empty()) {
        for (size_t l = 0; l < expected_sizes.size(); ++l) {
            out[l].assign(expected_sizes[l], 0.0f);
        }
        return out;
    }
    for (const auto& m : msgs) {
        if (m.layers.size() != expected_sizes.size()) {
            throw ProtocolError("grad_average: layer count mismatch");
        }
        for (size_t l = 0; l < expected_sizes.size(); ++l) {
            if (m.layers[l].size() != expected_sizes[l]) {
                throw ProtocolError("grad_average: layer size mismatch");
            }
        }
    }
    const float invK = 1.0f / static_cast<float>(msgs.size());
    for (size_t l = 0; l < expected_sizes.size(); ++l) {
        out[l].assign(expected_sizes[l], 0.0f);
        for (const auto& m : msgs) {
            const auto& src = m.layers[l];
            auto& dst = out[l];
            for (size_t i = 0; i < dst.size(); ++i) dst[i] += src[i];
        }
        for (auto& x : out[l]) x *= invK;
    }
    return out;
}

}  // namespace proto
