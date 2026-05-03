// bench/compare_protocols.cpp -- DCTCP / LTP / PLOT / PILT head-to-head.
//
// Runs each protocol's encoder + aggregator against the same synthetic
// gradient on one machine (no NS-3, no real network), measuring:
//
//   * encode_ms            time to produce the wire bytes per worker
//   * aggregate_ms         time spent in the aggregator per worker
//   * bytes_sent           bytes the network would carry per round
//                          (sum_k frame_k for single-pass; plus retx for PLOT)
//   * goodput_ratio        fraction of bytes that contribute updates
//   * delivery_correlation Pearson correlation between the protocol's
//                          per-element average and the lossless per-element
//                          average (single-number quality score)
//
// Output: self-describing JSON on stdout (or --out path).
//
// Usage:
//   compare_protocols [--out file.json] [--K 10] [--seed 0]
//                     [--layers 1024,4096,256] [--loss_dctcp 0.0]
//                     [--loss_ltp 0.4] [--loss_plot 0.4] [--ltt 0.7]

#include "common/grad_codec.hpp"
#include "common/types.hpp"
#include "dctcp/protocol.hpp"
#include "ltp/protocol.hpp"
#include "pilt/aggregator.hpp"
#include "pilt/encoder.hpp"
#include "plot/protocol.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

using clock_t = std::chrono::steady_clock;

// True lossless per-element average across K workers.
proto::LayerSet true_average(const std::vector<proto::LayerSet>& g) {
    const auto& base = g.front();
    proto::LayerSet out(base.size());
    const float invK = 1.0f / static_cast<float>(g.size());
    for (size_t l = 0; l < base.size(); ++l) {
        out[l].assign(base[l].size(), 0.0f);
        for (const auto& gk : g) {
            for (size_t i = 0; i < base[l].size(); ++i) out[l][i] += gk[l][i];
        }
        for (auto& x : out[l]) x *= invK;
    }
    return out;
}

double pearson(const proto::LayerSet& a, const proto::LayerSet& b) {
    long double mean_a = 0, mean_b = 0;
    long double n = 0;
    for (size_t l = 0; l < a.size(); ++l) {
        n += a[l].size();
        for (size_t i = 0; i < a[l].size(); ++i) {
            mean_a += a[l][i]; mean_b += b[l][i];
        }
    }
    if (n == 0) return 0.0;
    mean_a /= n; mean_b /= n;
    long double sxy = 0, sxx = 0, syy = 0;
    for (size_t l = 0; l < a.size(); ++l) {
        for (size_t i = 0; i < a[l].size(); ++i) {
            const long double da = a[l][i] - mean_a;
            const long double db = b[l][i] - mean_b;
            sxy += da * db; sxx += da * da; syy += db * db;
        }
    }
    if (sxx <= 0 || syy <= 0) return 0.0;
    return static_cast<double>(sxy / std::sqrt(sxx * syy));
}

// Count non-zero positions in the averaged tensor (proxy for "elements
// that received any update from any worker").
size_t count_nonzero(const proto::LayerSet& a) {
    size_t n = 0;
    for (const auto& l : a) for (auto x : l) if (x != 0.0f) ++n;
    return n;
}
size_t total_elems(const proto::LayerSet& a) {
    size_t n = 0;
    for (const auto& l : a) n += l.size();
    return n;
}

struct Result {
    std::string proto;
    double encode_ms_total = 0;
    double agg_ms_total    = 0;
    uint64_t total_bytes_sent = 0;
    double delivery_corr   = 0;
    double coverage_pct    = 0;
    double goodput_ratio   = 0;
    int    rounds_simulated = 1;
};

void emit_json(const std::vector<Result>& rs, std::ostream& os,
               int K, const std::vector<uint32_t>& sizes,
               int seed) {
    os << "{\n";
    os << "  \"meta\": {\n";
    os << "    \"K\": " << K << ", \"seed\": " << seed << ",\n";
    os << "    \"layer_sizes\": [";
    for (size_t i = 0; i < sizes.size(); ++i) {
        if (i) os << ", ";
        os << sizes[i];
    }
    os << "]\n  },\n";
    os << "  \"protocols\": [\n";
    for (size_t i = 0; i < rs.size(); ++i) {
        const auto& r = rs[i];
        os << "    {\n";
        os << "      \"name\": \""        << r.proto << "\",\n";
        os << "      \"encode_ms\": "     << r.encode_ms_total << ",\n";
        os << "      \"aggregate_ms\": "  << r.agg_ms_total    << ",\n";
        os << "      \"bytes_sent\": "    << r.total_bytes_sent << ",\n";
        os << "      \"coverage_pct\": "  << r.coverage_pct    << ",\n";
        os << "      \"goodput_ratio\": " << r.goodput_ratio   << ",\n";
        os << "      \"delivery_correlation\": " << r.delivery_corr << "\n";
        os << "    }" << (i + 1 == rs.size() ? "" : ",") << "\n";
    }
    os << "  ]\n}\n";
}

// Generate K worker-grads with shared structure + per-worker noise.
std::vector<proto::LayerSet> gen_grads(uint32_t K,
                                       const std::vector<uint32_t>& sizes,
                                       uint64_t seed) {
    std::vector<proto::LayerSet> out(K);
    std::mt19937_64 base(seed);
    std::vector<proto::LayerSet> shared_signal(1);
    shared_signal[0].resize(sizes.size());
    for (size_t l = 0; l < sizes.size(); ++l) {
        shared_signal[0][l].resize(sizes[l]);
        std::normal_distribution<float> nd(0.0f, 2.0f);
        for (auto& x : shared_signal[0][l]) x = nd(base);
    }
    for (uint32_t k = 0; k < K; ++k) {
        std::mt19937_64 rng(seed + 1000ull + k);
        std::normal_distribution<float> nd(0.0f, 0.5f);
        out[k].resize(sizes.size());
        for (size_t l = 0; l < sizes.size(); ++l) {
            out[k][l].resize(sizes[l]);
            for (size_t i = 0; i < sizes[l]; ++i) {
                out[k][l][i] = shared_signal[0][l][i] + nd(rng);
            }
        }
    }
    return out;
}

double now_ms() {
    return std::chrono::duration<double, std::milli>(
               clock_t::now().time_since_epoch()).count();
}

}  // namespace

int main(int argc, char** argv) {
    int K     = 10;
    int seed  = 0;
    int rounds = 1;                   // optional repetition for stability
    std::vector<uint32_t> layers = {1024u, 4096u, 16384u, 1024u};
    std::string out_path;
    float loss_dctcp = 0.00f;         // DCTCP modelled as reliable
    float loss_ltp   = 0.40f;         // typical lossy delivery
    float loss_plot  = 0.40f;
    float ltt        = 0.70f;
    float pilt_E     = 0.50f;

    for (int i = 1; i < argc; ++i) {
        auto eq = [&](const char* s){ return std::strcmp(argv[i], s) == 0; };
        auto next = [&](float& dst){
            if (i + 1 >= argc) { std::fprintf(stderr,"missing arg\n"); std::exit(2);}
            dst = std::atof(argv[++i]);
        };
        auto nexti = [&](int& dst){
            if (i + 1 >= argc) { std::fprintf(stderr,"missing arg\n"); std::exit(2);}
            dst = std::atoi(argv[++i]);
        };
        if      (eq("--K"))             nexti(K);
        else if (eq("--seed"))          nexti(seed);
        else if (eq("--rounds"))        nexti(rounds);
        else if (eq("--out"))           { if (i+1>=argc) std::exit(2); out_path = argv[++i]; }
        else if (eq("--loss_dctcp"))    next(loss_dctcp);
        else if (eq("--loss_ltp"))      next(loss_ltp);
        else if (eq("--loss_plot"))     next(loss_plot);
        else if (eq("--ltt"))           next(ltt);
        else if (eq("--pilt_e_total")) next(pilt_E);
        else if (eq("--layers"))        {
            if (i+1>=argc) std::exit(2);
            layers.clear();
            std::string s = argv[++i]; std::stringstream ss(s); std::string tok;
            while (std::getline(ss, tok, ',')) layers.push_back(std::atoi(tok.c_str()));
        }
        else { std::fprintf(stderr, "unknown arg: %s\n", argv[i]); std::exit(2); }
    }

    auto grads = gen_grads(K, layers, static_cast<uint64_t>(seed));
    auto truth = true_average(grads);

    std::vector<Result> results;

    auto run_dctcp = [&]() {
        Result r; r.proto = "DCTCP";
        for (int rd = 0; rd < rounds; ++rd) {
            dctcp::Encoder    enc(layers);
            dctcp::Aggregator agg(K, layers);
            agg.begin_round(rd);
            const double t0 = now_ms();
            for (uint32_t k = 0; k < (uint32_t)K; ++k) {
                auto b = enc.encode(rd, k, grads[k]);
                r.total_bytes_sent += b.size();
                agg.add(b);
            }
            const double t1 = now_ms();
            auto avg = agg.finalize_average();
            const double t2 = now_ms();
            r.encode_ms_total += (t1 - t0);
            r.agg_ms_total    += (t2 - t1);
            r.delivery_corr   += pearson(avg, truth);
            r.coverage_pct    += 100.0 * (double)count_nonzero(avg) / total_elems(avg);
        }
        r.delivery_corr /= rounds;
        r.coverage_pct  /= rounds;
        r.goodput_ratio  = (1.0 - loss_dctcp);     // reliable
        results.push_back(std::move(r));
    };

    auto run_ltp = [&]() {
        Result r; r.proto = "LTP";
        for (int rd = 0; rd < rounds; ++rd) {
            ltp::Encoder    enc(layers);
            ltp::EarlyCloseConfig ec; ec.seed = seed * 1000ULL + 11;
            ltp::Aggregator agg(K, layers, ec);
            agg.begin_round(rd);
            const double t0 = now_ms();
            for (uint32_t k = 0; k < (uint32_t)K; ++k) {
                auto b = enc.encode(rd, k, grads[k]);
                r.total_bytes_sent += b.size();        // sender always sends
                agg.add_with_loss(b, 1.0f - loss_ltp);
            }
            const double t1 = now_ms();
            auto avg = agg.finalize_average();
            const double t2 = now_ms();
            r.encode_ms_total += (t1 - t0);
            r.agg_ms_total    += (t2 - t1);
            r.delivery_corr   += pearson(avg, truth);
            r.coverage_pct    += 100.0 * (double)count_nonzero(avg) / total_elems(avg);
        }
        r.delivery_corr /= rounds;
        r.coverage_pct  /= rounds;
        r.goodput_ratio  = 1.0 - loss_ltp;
        results.push_back(std::move(r));
    };

    auto run_plot = [&]() {
        Result r; r.proto = "PLOT";
        for (int rd = 0; rd < rounds; ++rd) {
            plot::LayerLttConfig cfg; cfg.default_ltt = ltt; cfg.seed = seed*1000ULL+13;
            plot::Encoder enc(layers);
            plot::Aggregator agg(K, layers, cfg);
            agg.begin_round(rd);
            const double t0 = now_ms();
            for (uint32_t k = 0; k < (uint32_t)K; ++k) {
                auto b = enc.encode(rd, k, grads[k]);
                r.total_bytes_sent += b.size();
                std::vector<float> per_layer(layers.size(), 1.0f - loss_plot);
                agg.add_first_pass(b, per_layer);
            }
            // Retx: send full retx; assume reliable second pass.
            auto plan = agg.plan_retx();
            for (uint32_t k = 0; k < (uint32_t)K; ++k) {
                if (std::all_of(plan.retx_mask[k].begin(),
                                plan.retx_mask[k].end(),
                                [](uint8_t x){ return x == 0; })) continue;
                auto rb = enc.encode_retx(rd, k, grads[k], plan.retx_mask[k]);
                r.total_bytes_sent += rb.size();
                std::vector<float> per_layer(layers.size(), 1.0f);
                agg.add_retx_pass(rb, per_layer);
            }
            const double t1 = now_ms();
            auto avg = agg.finalize_average();
            const double t2 = now_ms();
            r.encode_ms_total += (t1 - t0);
            r.agg_ms_total    += (t2 - t1);
            r.delivery_corr   += pearson(avg, truth);
            r.coverage_pct    += 100.0 * (double)count_nonzero(avg) / total_elems(avg);
        }
        r.delivery_corr /= rounds;
        r.coverage_pct  /= rounds;
        // Goodput = useful bytes / total bytes sent. First-pass and
        // retx layer bytes are both useful here.
        r.goodput_ratio = 1.0;
        results.push_back(std::move(r));
    };

    auto run_pilt = [&]() {
        Result r; r.proto = "PILT";
        // Use the final round's metrics; preceding rounds let v_l and
        // the EF residuals settle.
        const int warmup = std::max(0, rounds - 1);
        pilt::EncoderConfig cfg;
        cfg.E_total = pilt_E; cfg.eps_min = 0.05f; cfg.beta = 0.9f; cfg.d = 0.05f;
        std::vector<pilt::PILTEncoder> enc;
        for (int k = 0; k < K; ++k) enc.emplace_back(layers, cfg);

        for (int rd = 0; rd < rounds; ++rd) {
            pilt::PILTAggregator agg(K, layers);
            agg.begin_round(rd);
            const double t0 = now_ms();
            for (uint32_t k = 0; k < (uint32_t)K; ++k) {
                auto bytes = enc[k].encode(rd, k, grads[k]);
                if (rd >= warmup) r.total_bytes_sent += bytes.size();
                agg.add(bytes);
            }
            const double t1 = now_ms();
            auto avg = agg.finalize_average();
            const double t2 = now_ms();
            if (rd >= warmup) {
                r.encode_ms_total += (t1 - t0);
                r.agg_ms_total    += (t2 - t1);
                r.delivery_corr   += pearson(avg, truth);
                r.coverage_pct    += 100.0 * (double)count_nonzero(avg) / total_elems(avg);
            }
            // Feed avg back to all encoders so v_l updates (standard PILT loop).
            for (auto& e : enc) e.update_importance(avg);
        }
        const int score_rounds = std::max(1, rounds - warmup);
        r.delivery_corr /= score_rounds;
        r.coverage_pct  /= score_rounds;
        r.rounds_simulated = score_rounds;
        // Every transmitted PILT byte enters the aggregate.
        r.goodput_ratio = 1.0;
        results.push_back(std::move(r));
    };

    run_dctcp();
    run_ltp();
    run_plot();
    run_pilt();

    if (out_path.empty()) {
        emit_json(results, std::cout, K, layers, seed);
    } else {
        std::ofstream f(out_path);
        emit_json(results, f, K, layers, seed);
        std::fprintf(stderr, "wrote %s\n", out_path.c_str());
    }
    return 0;
}
