// Encoder unit tests.

#include "pilt/encoder.hpp"
#include "pilt/wire.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>
#include <random>
#include <stdexcept>

using namespace pilt;

#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); \
            std::abort();                                                      \
        }                                                                      \
    } while (0)

// First-round behaviour: v_l == 0 for all l, so the rank update produces a
// symmetric (zero-sum) shift around E_total -- but the global-budget pass
// in compute_ratios() may then proportionally rescale.  We don't pin the
// exact eps_l value (that is exercised by test_rank_drives_ratios), we
// just check that (a) the budget constraint is honoured and (b) Top-|g|
// element selection picks exactly the largest k_l elements per layer.
static void test_first_round_top_g() {
    EncoderConfig cfg;
    cfg.E_total = 0.5f;
    cfg.eps_min = 0.05f;
    cfg.d       = 0.0f;        // no rank update -> eps stays at E_total
    PILTEncoder enc({10, 5}, cfg);

    LayerVec l0 = {0.1f, -0.2f, 5.0f, -4.0f, 3.0f, 0.05f, 0.1f, 6.0f, -0.1f, 7.0f};
    LayerVec l1 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};        // top-3 = {2,3,4}

    auto eps = enc.compute_ratios();
    CHECK(eps.size() == 2);
    CHECK(std::fabs(eps[0] - 0.5f) < 1e-6f);
    CHECK(std::fabs(eps[1] - 0.5f) < 1e-6f);

    EncodeStats stats;
    auto bytes = enc.encode(0, 0, {l0, l1}, &stats);
    auto msg   = decode_message(bytes);

    CHECK(msg.worker_id == 0);
    CHECK(msg.round_num == 0);
    CHECK(msg.layers.size() == 2);

    // k_0 = round(0.5 * 10) = 5.  k_1 = round(0.5 * 5) = 3 (banker's? no:
    // std::lround rounds half-to-even-away-from-zero -> 3).  We check the
    // exact values to catch off-by-one regressions.
    CHECK(stats.k_sent[0] == 5);
    CHECK(stats.k_sent[1] == 3);
    CHECK(msg.layers[0].indices.size() == 5);
    CHECK(msg.layers[1].indices.size() == 3);

    // Layer 0: top-5 |g| are at {2, 3, 4, 7, 9} (|g| = 5, 4, 3, 6, 7).
    std::vector<uint32_t> got0 = msg.layers[0].indices;
    std::sort(got0.begin(), got0.end());
    std::vector<uint32_t> expect0 = {2, 3, 4, 7, 9};
    CHECK(got0 == expect0);

    // Layer 1: top-3 |g| are at {2, 3, 4}.
    std::vector<uint32_t> got1 = msg.layers[1].indices;
    std::sort(got1.begin(), got1.end());
    std::vector<uint32_t> expect1 = {2, 3, 4};
    CHECK(got1 == expect1);
}

// Even *with* the rank update enabled, the budget constraint must hold
// after compute_ratios(): Sigma eps_l * S_l <= E_total * Sigma S_l.
static void test_budget_holds_first_round() {
    EncoderConfig cfg;
    cfg.E_total = 0.5f;
    cfg.eps_min = 0.05f;
    cfg.d       = 0.05f;
    PILTEncoder enc({10, 5}, cfg);

    auto eps = enc.compute_ratios();
    double load = eps[0] * 10.0 + eps[1] * 5.0;
    CHECK(load <= 0.5 * 15.0 + 1e-3);
    // And eps stays inside [eps_min, 1].
    for (float e : eps) {
        CHECK(e >= cfg.eps_min - 1e-6f);
        CHECK(e <= 1.0f + 1e-6f);
    }
}

// Across rounds, locally-accumulated residual eventually pushes a small but
// persistent gradient component into the top-k mask (error-feedback).
static void test_residual_eventually_sent() {
    EncoderConfig cfg;
    cfg.E_total = 0.2f;        // send only 20% per round
    cfg.eps_min = 0.2f;
    cfg.beta    = 0.99f;       // stable v_l
    cfg.d       = 0.0f;        // disable rank update for this test
    PILTEncoder enc({10}, cfg);

    // Element 0 has small but non-zero gradient every round.  Element 1
    // has a giant gradient; everyone else zero.  With k_l = 2 per round,
    // elements {1, 0} should be sent every round once residual builds up.
    LayerVec g = {0.05f, 10.0f, 0, 0, 0, 0, 0, 0, 0, 0};

    bool seen_zero = false;
    for (int r = 0; r < 30; ++r) {
        EncodeStats stats;
        auto bytes = enc.encode(r, 0, {g}, &stats);
        auto msg   = decode_message(bytes);
        // k_l = round(0.2 * 10) = 2 by config.
        CHECK(stats.k_sent[0] == 2);
        for (uint32_t idx : msg.layers[0].indices) {
            if (idx == 0) seen_zero = true;
        }
        // Mock PS broadcast: just feed the same magnitudes back.
        enc.update_importance({g});
    }
    CHECK(seen_zero);
}

// Importance EMA + rank should drive ε_l of a layer with large gradients
// upward and ε_l of a layer with tiny gradients downward, while the global
// budget Σ ε_l · S_l ≤ E_total · Σ S_l is still respected.
static void test_rank_drives_ratios() {
    EncoderConfig cfg;
    cfg.E_total = 0.5f;
    cfg.eps_min = 0.05f;
    cfg.beta    = 0.5f;
    cfg.d       = 0.1f;
    PILTEncoder enc({100, 100, 100}, cfg);

    LayerVec big(100,  10.0f);
    LayerVec mid(100,   1.0f);
    LayerVec sml(100,   0.01f);

    // Ten rounds is plenty for v_l to settle with beta=0.5.
    for (int r = 0; r < 10; ++r) {
        (void)enc.compute_ratios();
        (void)enc.encode(r, 0, {big, mid, sml});
        enc.update_importance({big, mid, sml});
    }
    auto eps = enc.ratios();
    // Layer 0 (largest grads) should have the highest ratio.
    CHECK(eps[0] > eps[1]);
    CHECK(eps[1] > eps[2]);

    // Budget check: Σ ε_l · S_l ≤ E_total · Σ S_l (within a small fp slack).
    double load = 0.0;
    for (size_t l = 0; l < 3; ++l) load += eps[l] * 100.0;
    CHECK(load <= 0.5 * 300.0 + 1e-3);
}

// Encoder must reject mismatched layer counts / sizes loudly.
static void test_input_validation() {
    PILTEncoder enc({4, 4});
    bool threw = false;
    try { (void)enc.encode(0, 0, {LayerVec(4, 0.0f)}); }
    catch (const PiltError&) { threw = true; }
    CHECK(threw);
    bool threw2 = false;
    try { (void)enc.encode(0, 0, {LayerVec(4, 0.0f), LayerVec(3, 0.0f)}); }
    catch (const PiltError&) { threw2 = true; }
    CHECK(threw2);
}

int main() {
    test_first_round_top_g();
    test_budget_holds_first_round();
    test_residual_eventually_sent();
    test_rank_drives_ratios();
    test_input_validation();
    std::printf("test_encoder: OK\n");
    return 0;
}
