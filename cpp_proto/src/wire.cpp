// pilt/wire.cpp -- on-the-wire serialisation for PILT.

#include "pilt/wire.hpp"

#include <array>
#include <cstring>

namespace pilt {

namespace {

// CRC-32/IEEE (zlib) table, populated lazily on first use.
struct CrcTable {
    std::array<uint32_t, 256> t{};
    CrcTable() {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int j = 0; j < 8; ++j) {
                c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            }
            t[i] = c;
        }
    }
};
const CrcTable& crc_table() {
    static const CrcTable kTable;
    return kTable;
}

// Little-endian primitive writers.  We don't rely on host endianness
// because the protocol is wire-defined; the host might be big-endian.
inline void put_u16(uint8_t*& p, uint16_t v) noexcept {
    p[0] = static_cast<uint8_t>(v & 0xFFu);
    p[1] = static_cast<uint8_t>((v >> 8) & 0xFFu);
    p += 2;
}
inline void put_u32(uint8_t*& p, uint32_t v) noexcept {
    p[0] = static_cast<uint8_t>(v & 0xFFu);
    p[1] = static_cast<uint8_t>((v >> 8) & 0xFFu);
    p[2] = static_cast<uint8_t>((v >> 16) & 0xFFu);
    p[3] = static_cast<uint8_t>((v >> 24) & 0xFFu);
    p += 4;
}
inline void put_f32(uint8_t*& p, float v) noexcept {
    uint32_t bits;
    std::memcpy(&bits, &v, 4);
    put_u32(p, bits);
}

inline uint16_t get_u16(const uint8_t*& p) noexcept {
    uint16_t v = static_cast<uint16_t>(p[0]) |
                 (static_cast<uint16_t>(p[1]) << 8);
    p += 2;
    return v;
}
inline uint32_t get_u32(const uint8_t*& p) noexcept {
    uint32_t v = static_cast<uint32_t>(p[0]) |
                 (static_cast<uint32_t>(p[1]) << 8) |
                 (static_cast<uint32_t>(p[2]) << 16) |
                 (static_cast<uint32_t>(p[3]) << 24);
    p += 4;
    return v;
}
inline float get_f32(const uint8_t*& p) noexcept {
    uint32_t bits = get_u32(p);
    float v;
    std::memcpy(&v, &bits, 4);
    return v;
}

}  // namespace

uint32_t crc32_ieee(const uint8_t* data, size_t len) noexcept {
    const auto& tbl = crc_table().t;
    uint32_t c = 0xFFFFFFFFu;
    for (size_t i = 0; i < len; ++i) {
        c = tbl[(c ^ data[i]) & 0xFFu] ^ (c >> 8);
    }
    return c ^ 0xFFFFFFFFu;
}

size_t encoded_size(const std::vector<uint32_t>& k_per_layer) noexcept {
    size_t total = kHeaderBytes;
    for (uint32_t k : k_per_layer) {
        total += kPerLayerHdrBytes + 8ull * static_cast<size_t>(k);
    }
    total += kCrcBytes;
    return total;
}

ByteBuf encode_message(uint16_t worker_id,
                       uint32_t round_num,
                       const std::vector<uint32_t>& layer_sizes,
                       const std::vector<IndexVec>& indices_per_layer,
                       const std::vector<LayerVec>& values_per_layer)
{
    if (layer_sizes.size() != indices_per_layer.size() ||
        layer_sizes.size() != values_per_layer.size()) {
        throw PiltError("encode_message: layer count mismatch");
    }
    if (layer_sizes.size() > UINT16_MAX) {
        throw PiltError("encode_message: too many layers (>65535)");
    }

    std::vector<uint32_t> k_per_layer(layer_sizes.size());
    for (size_t l = 0; l < layer_sizes.size(); ++l) {
        if (indices_per_layer[l].size() != values_per_layer[l].size()) {
            throw PiltError("encode_message: idx/val length mismatch in layer");
        }
        if (indices_per_layer[l].size() > layer_sizes[l]) {
            throw PiltError("encode_message: k_l exceeds S_l");
        }
        k_per_layer[l] = static_cast<uint32_t>(indices_per_layer[l].size());
    }

    const size_t total = encoded_size(k_per_layer);
    ByteBuf out(total);
    uint8_t* p = out.data();

    put_u32(p, kPiltMagic);
    put_u16(p, kPiltVersion);
    put_u16(p, worker_id);
    put_u32(p, round_num);
    put_u16(p, static_cast<uint16_t>(layer_sizes.size()));
    put_u16(p, 0);                                 // reserved
    const uint32_t payload_len = static_cast<uint32_t>(total - kHeaderBytes - kCrcBytes);
    put_u32(p, payload_len);

    for (size_t l = 0; l < layer_sizes.size(); ++l) {
        put_u16(p, static_cast<uint16_t>(l));
        put_u16(p, 0);                             // reserved
        put_u32(p, layer_sizes[l]);
        put_u32(p, k_per_layer[l]);
        const uint32_t k = k_per_layer[l];
        for (uint32_t i = 0; i < k; ++i) put_u32(p, indices_per_layer[l][i]);
        for (uint32_t i = 0; i < k; ++i) put_f32(p, values_per_layer[l][i]);
    }

    const uint32_t crc = crc32_ieee(out.data(), total - kCrcBytes);
    put_u32(p, crc);

    if (p != out.data() + total) {
        throw PiltError("encode_message: internal length mismatch");
    }
    return out;
}

DecodedMessage decode_message(const uint8_t* data, size_t len) {
    if (len < kHeaderBytes + kCrcBytes) {
        throw PiltError("decode_message: buffer too small for header+crc");
    }
    const uint8_t* p = data;
    uint32_t magic = get_u32(p);
    if (magic != kPiltMagic) {
        throw PiltError("decode_message: bad magic");
    }
    uint16_t version    = get_u16(p);
    if (version != kPiltVersion) {
        throw PiltError("decode_message: unsupported version");
    }

    DecodedMessage msg;
    msg.version    = version;
    msg.worker_id  = get_u16(p);
    msg.round_num  = get_u32(p);
    uint16_t L     = get_u16(p);
    (void)get_u16(p);                              // reserved
    uint32_t payload_len = get_u32(p);

    if (payload_len + kHeaderBytes + kCrcBytes != len) {
        throw PiltError("decode_message: payload_len mismatches buffer size");
    }

    msg.layers.resize(L);
    for (uint16_t l = 0; l < L; ++l) {
        if (static_cast<size_t>(p - data) + kPerLayerHdrBytes > len - kCrcBytes) {
            throw PiltError("decode_message: truncated layer header");
        }
        uint16_t layer_id = get_u16(p);
        (void)get_u16(p);                          // reserved
        uint32_t layer_size = get_u32(p);
        uint32_t k          = get_u32(p);
        if (k > layer_size) {
            throw PiltError("decode_message: k_l exceeds S_l");
        }
        const size_t need = 8ull * k;
        if (static_cast<size_t>(p - data) + need > len - kCrcBytes) {
            throw PiltError("decode_message: truncated layer values");
        }
        DecodedLayer lay;
        lay.layer_id   = layer_id;
        lay.layer_size = layer_size;
        lay.indices.resize(k);
        lay.values.resize(k);
        for (uint32_t i = 0; i < k; ++i) lay.indices[i] = get_u32(p);
        for (uint32_t i = 0; i < k; ++i) lay.values[i]  = get_f32(p);
        msg.layers[l] = std::move(lay);
    }

    if (static_cast<size_t>(p - data) != len - kCrcBytes) {
        throw PiltError("decode_message: trailing bytes inside payload");
    }
    uint32_t expected = crc32_ieee(data, len - kCrcBytes);
    uint32_t actual   = get_u32(p);
    if (expected != actual) {
        throw PiltError("decode_message: CRC mismatch");
    }
    return msg;
}

}  // namespace pilt
