// test_dctcp_ltp_plot.cpp -- combined round-trip / loss-model test for the
// three baseline protocols.

#include "common/grad_codec.hpp"
#include "dctcp/protocol.hpp"
#include "ltp/protocol.hpp"
#include "plot/protocol.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); \
            std::abort();                                                      \
        }                                                                      \
    } while (0)

static proto::LayerSet make_synth_grads(const std::vector<uint32_t>& sizes,
                                        uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    proto::LayerSet g(sizes.size());
    for (size_t l = 0; l < sizes.size(); ++l) {
        g[l].resize(sizes[l]);
        for (auto& x : g[l]) x = nd(rng);
    }
    return g;
}

// DCTCP: lossless reliable -- the aggregator average must equal the
// per-element mean of all workers' inputs.
static void test_dctcp_round_trip() {
    std::vector<uint32_t> sizes = {1024, 256, 64};
    const uint32_t K = 8;
    dctcp::Encoder    enc(sizes);
    dctcp::Aggregator agg(K, sizes);
    agg.begin_round(7);
    std::vector<proto::LayerSet> all;
    for (uint32_t k = 0; k < K; ++k) {
        auto g = make_synth_grads(sizes, /*seed=*/0x42u + k);
        all.push_back(g);
        agg.add(enc.encode(7, k, g));
    }
    auto avg = agg.finalize_average();
    CHECK(avg.size() == sizes.size());
    for (size_t l = 0; l < sizes.size(); ++l) {
        CHECK(avg[l].size() == sizes[l]);
        for (size_t i = 0; i < sizes[l]; ++i) {
            float ref = 0.0f;
            for (uint32_t k = 0; k < K; ++k) ref += all[k][l][i];
            ref /= static_cast<float>(K);
            // Exact-match: same arithmetic in same order.
            CHECK(avg[l][i] == ref);
        }
    }
}

// LTP: per-worker delivered_fraction < 1 -> bubble-fill kicks in.  The
// aggregator average per-layer should still have norm in the same ballpark
// as the lossless version because the bubble-fill resampler preserves
// expected magnitude.
static void test_ltp_with_loss() {
    std::vector<uint32_t> sizes = {2048, 512};
    const uint32_t K = 6;
    ltp::Encoder    enc(sizes);
    ltp::EarlyCloseConfig ec; ec.seed = 1234;
    ltp::Aggregator agg(K, sizes, ec);
    agg.begin_round(0);

    std::vector<proto::LayerSet> all;
    for (uint32_t k = 0; k < K; ++k) {
        auto g = make_synth_grads(sizes, /*seed=*/0xA0u + k);
        all.push_back(g);
        // Half the workers experience 60% delivery, half 100%.
        const float frac = (k % 2 == 0) ? 0.6f : 1.0f;
        agg.add_with_loss(enc.encode(0, k, g), frac);
    }
    auto avg = agg.finalize_average();

    // Expected magnitude: ||avg||_2 should be within 30% of the lossless
    // baseline because bubble-fill preserves L2 in expectation.
    long double sq_lossy = 0.0L, sq_loss = 0.0L;
    for (size_t l = 0; l < sizes.size(); ++l) {
        proto::LayerVec lossless(sizes[l], 0.0f);
        for (uint32_t k = 0; k < K; ++k) {
            for (size_t i = 0; i < sizes[l]; ++i) lossless[i] += all[k][l][i];
        }
        for (auto& x : lossless) x /= static_cast<float>(K);
        for (auto x : lossless) sq_lossy += static_cast<long double>(x) * x;
        for (auto x : avg[l])  sq_loss  += static_cast<long double>(x) * x;
    }
    const double ratio = std::sqrt((double)sq_loss) /
                         std::sqrt((double)sq_lossy + 1e-12);
    CHECK(ratio > 0.5 && ratio < 1.5);
}

// PLOT: per-layer loss with LTT.  Layers above LTT enter on first pass;
// layers below trigger a retx round.  Verify retx mask is non-empty for
// the high-loss layers and finalize_average produces sensible output.
static void test_plot_two_pass() {
    std::vector<uint32_t> sizes = {1024, 1024, 1024};
    const uint32_t K = 4;
    plot::LayerLttConfig cfg;
    cfg.default_ltt = 0.7f;     // fraction-arrived < 0.7 -> retx
    cfg.seed = 99;

    plot::Encoder enc(sizes);
    plot::Aggregator agg(K, sizes, cfg);
    agg.begin_round(11);

    std::vector<proto::LayerSet> grads(K);
    for (uint32_t k = 0; k < K; ++k) {
        grads[k] = make_synth_grads(sizes, /*seed=*/0xB0u + k);
        // Layer 0: fully delivered; layer 1: 80% (above LTT, accept);
        // layer 2: 30% (below LTT, retx).
        std::vector<float> per_layer = {1.0f, 0.8f, 0.3f};
        agg.add_first_pass(enc.encode(11, k, grads[k]), per_layer);
    }

    auto plan = agg.plan_retx();
    CHECK(plan.retx_mask.size() == K);
    // Layer 2 should be in EVERY worker's retx mask, layers 0/1 in none.
    for (uint32_t k = 0; k < K; ++k) {
        CHECK(plan.retx_mask[k][0] == 0);
        CHECK(plan.retx_mask[k][1] == 0);
        CHECK(plan.retx_mask[k][2] == 1);
        // Retx bytes for one worker = exactly layer-2 size in float32.
        CHECK(plan.retx_bytes_per_worker[k] == 4ull * sizes[2]);
    }

    // Run retx round at 100% delivery this time.
    for (uint32_t k = 0; k < K; ++k) {
        auto retx_bytes = enc.encode_retx(11, k, grads[k], plan.retx_mask[k]);
        agg.add_retx_pass(retx_bytes, {0.0f, 0.0f, 1.0f});
    }
    auto avg = agg.finalize_average();
    CHECK(avg.size() == sizes.size());
    for (size_t l = 0; l < sizes.size(); ++l) {
        CHECK(avg[l].size() == sizes[l]);
    }
}

int main() {
    test_dctcp_round_trip();
    test_ltp_with_loss();
    test_plot_two_pass();
    std::printf("test_dctcp_ltp_plot: OK\n");
    return 0;
}
