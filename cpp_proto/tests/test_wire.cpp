// Tests for the on-the-wire codec.

#include "pilt/wire.hpp"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

using namespace pilt;

#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); \
            std::abort();                                                      \
        }                                                                      \
    } while (0)

static void test_round_trip_small() {
    std::vector<uint32_t> sizes = {4, 7, 0};
    std::vector<IndexVec> idx = {
        {0, 2, 3},                 // layer 0: send 3 of 4
        {1, 4, 5, 6},              // layer 1: send 4 of 7
        {}                         // layer 2: empty
    };
    std::vector<LayerVec> val = {
        {1.0f, -2.0f, 3.5f},
        {0.1f, 0.2f, 0.3f, 0.4f},
        {}
    };
    auto buf = encode_message(/*worker_id=*/7, /*round=*/42, sizes, idx, val);
    auto msg = decode_message(buf);

    CHECK(msg.worker_id == 7);
    CHECK(msg.round_num == 42);
    CHECK(msg.layers.size() == 3);

    CHECK(msg.layers[0].layer_id   == 0);
    CHECK(msg.layers[0].layer_size == 4);
    CHECK(msg.layers[0].indices    == idx[0]);
    CHECK(msg.layers[0].values.size() == 3);
    CHECK(msg.layers[0].values[0]  == 1.0f);
    CHECK(msg.layers[0].values[1]  == -2.0f);
    CHECK(msg.layers[0].values[2]  == 3.5f);

    CHECK(msg.layers[1].layer_size == 7);
    CHECK(msg.layers[1].indices    == idx[1]);
    CHECK(msg.layers[1].values     == val[1]);

    CHECK(msg.layers[2].layer_size == 0);
    CHECK(msg.layers[2].indices.empty());
    CHECK(msg.layers[2].values.empty());
}

static void test_crc_detects_corruption() {
    auto buf = encode_message(0, 0, {3}, {{0, 1}}, {{0.0f, 0.0f}});
    CHECK(buf.size() > 4);
    buf[buf.size() / 2] ^= 0x55;
    bool threw = false;
    try {
        (void)decode_message(buf);
    } catch (const PiltError&) {
        threw = true;
    }
    CHECK(threw);
}

static void test_truncation_detected() {
    auto buf = encode_message(1, 1, {3}, {{0, 1}}, {{1.0f, 2.0f}});
    buf.resize(buf.size() - 4);                        // drop CRC
    bool threw = false;
    try { (void)decode_message(buf); } catch (const PiltError&) { threw = true; }
    CHECK(threw);

    // Header-only buffer: too small for even a header+crc.
    bool threw2 = false;
    std::vector<uint8_t> tiny(4, 0);
    try { (void)decode_message(tiny); } catch (const PiltError&) { threw2 = true; }
    CHECK(threw2);
}

static void test_size_inequalities() {
    // k_l > S_l must fail at encode time.
    bool threw = false;
    try {
        encode_message(0, 0, {2}, {{0, 1, 2}}, {{1.0f, 2.0f, 3.0f}});
    } catch (const PiltError&) { threw = true; }
    CHECK(threw);

    // mismatched idx/val lengths.
    bool threw2 = false;
    try {
        encode_message(0, 0, {3}, {{0, 1}}, {{1.0f, 2.0f, 3.0f}});
    } catch (const PiltError&) { threw2 = true; }
    CHECK(threw2);
}

int main() {
    test_round_trip_small();
    test_crc_detects_corruption();
    test_truncation_detected();
    test_size_inequalities();
    std::printf("test_wire: OK\n");
    return 0;
}
