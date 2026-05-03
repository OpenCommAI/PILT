// pilt/wire.hpp -- on-the-wire framing for PILT per-worker uploads.
//
// One PILT message describes one worker's encoded gradient update for one
// BSP round.  The wire format is little-endian on every multi-byte field
// and version-tagged so future revisions can be detected without breaking
// existing receivers.
//
//   Offset  Size    Field
//   0       4       magic        "PLT1"
//   4       2       version      uint16  (currently 0x0001)
//   6       2       worker_id    uint16
//   8       4       round_num    uint32
//  12       2       num_layers   uint16
//  14       2       reserved     uint16  (0)
//  16       4       payload_len  uint32  -- total bytes from layer-0 header
//                                          through last layer's value array,
//                                          excluding the trailing CRC32
//  20       --      LAYER[0]
//   ...             LAYER[1] ... LAYER[num_layers-1]
//   ...     4       crc32         CRC32-IEEE over [magic .. last value]
//
// Per-layer record (variable size, 4-byte aligned to keep float reads aligned):
//
//   0       2       layer_id     uint16
//   2       2       reserved     uint16  (0)
//   4       4       layer_size   uint32   -- S_l, full element count
//   8       4       num_sent     uint32   -- k_l, count of (idx, val) pairs
//  12       4*k_l   indices      uint32[]
//  12+4k   4*k_l   values        float32[]   (IEEE 754 little-endian)
//
// k_l == 0 is allowed (degenerate empty layer) and consumes 12 bytes total.

#pragma once

#include "pilt/types.hpp"
#include <cstdint>
#include <cstring>

namespace pilt {

constexpr uint32_t kPiltMagic    = 0x31544C50u;   // "PLT1" in little-endian
constexpr uint16_t kPiltVersion  = 0x0001u;
constexpr size_t   kHeaderBytes  = 20;
constexpr size_t   kCrcBytes     = 4;
constexpr size_t   kPerLayerHdrBytes = 12;

// Decoded view of one layer's contribution from one worker.
struct DecodedLayer {
    uint16_t layer_id   = 0;
    uint32_t layer_size = 0;     // S_l  (sender-asserted; receiver checks)
    IndexVec indices;            // sorted by sender; receiver does not assume
    LayerVec values;             // same length as indices
};

// Decoded view of one worker's per-round upload.
struct DecodedMessage {
    uint16_t version    = kPiltVersion;
    uint16_t worker_id  = 0;
    uint32_t round_num  = 0;
    std::vector<DecodedLayer> layers;
};

// CRC-32/IEEE (polynomial 0xEDB88320, reflected, init 0xFFFFFFFF, xor-out
// 0xFFFFFFFF -- same as zlib).  Pure C++17, zero deps.
uint32_t crc32_ieee(const uint8_t* data, size_t len) noexcept;

// Compute the exact wire size of an upload that has the supplied k_l per
// layer.  Useful for caller-side buffer pre-allocation.
size_t encoded_size(const std::vector<uint32_t>& k_per_layer) noexcept;

// Serialise.  `indices_per_layer[l]` and `values_per_layer[l]` must have
// equal length (== k_l).  Throws PiltError on inconsistent lengths.
ByteBuf encode_message(uint16_t worker_id,
                       uint32_t round_num,
                       const std::vector<uint32_t>& layer_sizes,
                       const std::vector<IndexVec>& indices_per_layer,
                       const std::vector<LayerVec>& values_per_layer);

// Deserialise.  Throws PiltError on bad magic, bad CRC, truncation,
// unknown version, or layer-record overrun.  The returned struct owns its
// data so the input buffer can be freed afterwards.
DecodedMessage decode_message(const uint8_t* data, size_t len);

inline DecodedMessage decode_message(const ByteBuf& buf) {
    return decode_message(buf.data(), buf.size());
}

}  // namespace pilt
