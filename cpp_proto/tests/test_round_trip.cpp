// End-to-end test: K worker encoders -> wire bytes -> PS aggregator ->
// per-layer averaged dense gradient -- compared against a hand-rolled
// reference average.

#include "pilt/aggregator.hpp"
#include "pilt/encoder.hpp"
#include "pilt/wire.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace pilt;

#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); \
            std::abort();                                                      \
        }                                                                      \
    } while (0)

// Hand-computed reference: per-element average over only those workers that
// actually transmitted that element (matches the aggregator semantics).
static LayerSet reference_avg(uint32_t n_workers,
                              const std::vector<uint32_t>& sizes,
                              const std::vector<DecodedMessage>& msgs)
{
    LayerSet sum(sizes.size());
    std::vector<std::vector<uint32_t>> cnt(sizes.size());
    for (size_t l = 0; l < sizes.size(); ++l) {
        sum[l].assign(sizes[l], 0.0f);
        cnt[l].assign(sizes[l], 0u);
    }
    for (const auto& m : msgs) {
        for (size_t l = 0; l < m.layers.size(); ++l) {
            const auto& lay = m.layers[l];
            for (size_t i = 0; i < lay.indices.size(); ++i) {
                sum[l][lay.indices[i]] += lay.values[i];
                cnt[l][lay.indices[i]] += 1;
            }
        }
    }
    LayerSet out(sizes.size());
    for (size_t l = 0; l < sizes.size(); ++l) {
        out[l].assign(sizes[l], 0.0f);
        for (size_t i = 0; i < sizes[l]; ++i) {
            out[l][i] = (cnt[l][i] > 0u)
                          ? sum[l][i] / static_cast<float>(cnt[l][i])
                          : 0.0f;
        }
    }
    (void)n_workers;
    return out;
}

static void run_one_round(uint32_t K, std::vector<uint32_t> sizes,
                          uint64_t seed, int round) {
    EncoderConfig cfg;
    cfg.E_total = 0.5f;
    cfg.eps_min = 0.1f;
    cfg.beta    = 0.9f;
    cfg.d       = 0.05f;

    std::vector<PILTEncoder> encoders;
    encoders.reserve(K);
    for (uint32_t k = 0; k < K; ++k) {
        encoders.emplace_back(sizes, cfg);
    }

    PILTAggregator ps(K, sizes);
    ps.begin_round(static_cast<uint32_t>(round));

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    std::vector<DecodedMessage> msgs;
    msgs.reserve(K);

    for (uint32_t k = 0; k < K; ++k) {
        LayerSet grads(sizes.size());
        for (size_t l = 0; l < sizes.size(); ++l) {
            grads[l].resize(sizes[l]);
            for (auto& x : grads[l]) x = dist(rng);
        }
        auto bytes = encoders[k].encode(static_cast<uint32_t>(round),
                                        static_cast<uint16_t>(k),
                                        grads);
        // Decode once for the reference avg, then feed the SAME bytes to
        // the aggregator so we exercise the full (decode-inside-add) path.
        msgs.push_back(decode_message(bytes));
        ps.add(bytes);
    }

    auto got = ps.finalize_average();
    auto ref = reference_avg(K, sizes, msgs);

    CHECK(got.size() == sizes.size());
    for (size_t l = 0; l < sizes.size(); ++l) {
        CHECK(got[l].size() == sizes[l]);
        for (size_t i = 0; i < sizes[l]; ++i) {
            // bit-exact equality: aggregator and reference do exactly the
            // same arithmetic, in the same order, on the same float32 input.
            CHECK(got[l][i] == ref[l][i]);
        }
    }
    CHECK(ps.messages_received() == 0);   // finalize_average() resets state
}

static void test_aggregator_rejects_bad_inputs() {
    PILTAggregator ps(4, {3, 5});
    ps.begin_round(11);

    auto good = encode_message(0, 11, {3, 5}, {{0, 1}, {0, 1}}, {{1.0f, 2.0f}, {0.5f, 0.5f}});
    ps.add(good);
    // Duplicate worker_id
    bool threw = false;
    try { ps.add(good); } catch (const PiltError&) { threw = true; }
    CHECK(threw);

    // Wrong round
    auto wrong_round = encode_message(1, 12, {3, 5}, {{}, {}}, {{}, {}});
    bool threw2 = false;
    try { ps.add(wrong_round); } catch (const PiltError&) { threw2 = true; }
    CHECK(threw2);

    // Worker id out of range
    auto oor = encode_message(99, 11, {3, 5}, {{}, {}}, {{}, {}});
    bool threw3 = false;
    try { ps.add(oor); } catch (const PiltError&) { threw3 = true; }
    CHECK(threw3);

    // Layer-size mismatch
    auto bad_size = encode_message(2, 11, {3, 6}, {{0}, {0}}, {{1.0f}, {1.0f}});
    bool threw4 = false;
    try { ps.add(bad_size); } catch (const PiltError&) { threw4 = true; }
    CHECK(threw4);

    ps.abort_round();
}

int main() {
    // Multiple sizes / multiple rounds, deterministic seed.
    for (int round = 0; round < 5; ++round) {
        run_one_round(/*K=*/4,  {16,  9, 25}, /*seed=*/0xC0FFEEu + round, round);
        run_one_round(/*K=*/10, {1024, 256, 4096}, /*seed=*/0xBADC0DEu + round, round);
    }
    test_aggregator_rejects_bad_inputs();

    std::printf("test_round_trip: OK\n");
    return 0;
}
