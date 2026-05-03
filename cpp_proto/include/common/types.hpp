// common/types.hpp -- shared data primitives across all protocol libs.
//
// Each protocol library (libpilt, libdctcp, libltp, libplot) exposes its
// own `<proto>::` namespace and includes this header for the elementary
// container types and the PiltError-style exception.

#pragma once

#include <cstdint>
#include <vector>
#include <stdexcept>

namespace proto {

// One contiguous layer's flat float buffer (host memory).
using LayerVec = std::vector<float>;

// All L layer-vectors for a single tensor (per-worker per-round payload, or
// a global model snapshot, etc.).
using LayerSet = std::vector<LayerVec>;

// Bytes -- public binary transport unit returned by the encoders.
using ByteBuf = std::vector<uint8_t>;

// Indices into a layer's flat buffer.
using IndexVec = std::vector<uint32_t>;

// Uniform error type across all protocol libraries.  Inherits from
// std::runtime_error so callers can rely on the standard `.what()` API.
class ProtocolError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

}  // namespace proto
