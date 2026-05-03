// pilt/types.hpp -- shared types for libpilt encoder / aggregator.
//
// Header-light, C++17, dependency-free. Tensor data is exchanged as
//   std::vector<float>              -- a single layer's flat values
//   std::vector<std::vector<float>> -- per-layer values for one tensor
// keeping the public ABI memcpy-friendly.

#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <cstddef>
#include <stdexcept>

namespace pilt {

// One contiguous layer's flat float buffer (host memory).
using LayerVec = std::vector<float>;

// All L layer-vectors for a single tensor (per-worker per-round payload, or
// a global model snapshot, etc.).
using LayerSet = std::vector<LayerVec>;

// Per-layer transmission ratio epsilon_l in [eps_min, 1.0].
using RatioVec = std::vector<float>;

// Indices into a layer's flat buffer (uint32 = up to 4 G elements).
using IndexVec = std::vector<uint32_t>;

// Bytes -- public binary transport unit returned by the encoder.
using ByteBuf = std::vector<uint8_t>;

// All public functions throw this on protocol violations.
class PiltError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

}  // namespace pilt
