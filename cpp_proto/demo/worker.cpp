// pilt_worker -- minimal TCP worker that encodes one round of synthetic
// gradients with libpilt and ships them to a pilt_parameter_server.
//
// Usage:
//
//   pilt_worker  <ps_host>  <port>  <worker_id>  <seed>  <S_1> [S_2 ...]
//
// Example (4 workers, 3 layers):
//
//   pilt_parameter_server 5005 4   1024 4096 256 &
//   pilt_worker 127.0.0.1 5005 0 100  1024 4096 256 &
//   pilt_worker 127.0.0.1 5005 1 101  1024 4096 256 &
//   pilt_worker 127.0.0.1 5005 2 102  1024 4096 256 &
//   pilt_worker 127.0.0.1 5005 3 103  1024 4096 256 &
//   wait

#include "pilt/encoder.hpp"
#include "pilt/wire.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <chrono>
#include <vector>

using pilt::PILTEncoder;
using pilt::EncoderConfig;
using pilt::PiltError;
using pilt::ByteBuf;

namespace {

bool write_exact(int fd, const void* buf, size_t n) {
    const auto* p = static_cast<const uint8_t*>(buf);
    while (n > 0) {
        ssize_t r = ::send(fd, p, n, MSG_NOSIGNAL);
        if (r < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        p += r;
        n -= static_cast<size_t>(r);
    }
    return true;
}

int connect_with_retry(const char* host, uint16_t port,
                       int max_retries = 30) {
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port   = htons(port);
    if (::inet_pton(AF_INET, host, &addr.sin_addr) <= 0) {
        std::fprintf(stderr, "worker: bad host %s\n", host);
        return -1;
    }
    for (int i = 0; i < max_retries; ++i) {
        int fd = ::socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0) { std::perror("socket"); return -1; }
        if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0) {
            int one = 1;
            ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
            return fd;
        }
        ::close(fd);
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    std::fprintf(stderr, "worker: could not connect to %s:%u\n", host, port);
    return -1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 6) {
        std::fprintf(stderr,
            "usage: %s <ps_host> <port> <worker_id> <seed> <S_1> [S_2 ...]\n",
            argv[0]);
        return 2;
    }
    const char*    host = argv[1];
    const uint16_t port = static_cast<uint16_t>(std::atoi(argv[2]));
    const uint16_t wid  = static_cast<uint16_t>(std::atoi(argv[3]));
    const uint64_t seed = static_cast<uint64_t>(std::atoll(argv[4]));

    std::vector<uint32_t> sizes;
    for (int i = 5; i < argc; ++i) {
        long s = std::atol(argv[i]);
        if (s < 0) { std::fprintf(stderr, "bad layer size: %s\n", argv[i]); return 2; }
        sizes.push_back(static_cast<uint32_t>(s));
    }

    EncoderConfig cfg;
    cfg.E_total = 0.5f;
    cfg.eps_min = 0.05f;
    cfg.beta    = 0.9f;
    cfg.d       = 0.05f;
    PILTEncoder enc(sizes, cfg);

    // Synthetic gradients: a fixed pattern per layer + small per-worker
    // noise so different workers hit different top-k positions.
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> noise(0.0f, 0.5f);
    pilt::LayerSet grads(sizes.size());
    for (size_t l = 0; l < sizes.size(); ++l) {
        grads[l].resize(sizes[l]);
        for (size_t i = 0; i < sizes[l]; ++i) {
            float base = (i % 7 == 0) ? 5.0f : 0.1f;     // structured signal
            grads[l][i] = base + noise(rng);
        }
    }

    pilt::EncodeStats stats;
    ByteBuf frame;
    try {
        frame = enc.encode(/*round=*/0, wid, grads, &stats);
    } catch (const PiltError& e) {
        std::fprintf(stderr, "worker %u: encode error: %s\n", wid, e.what());
        return 1;
    }
    std::printf("worker %u: encoded %zu bytes  ", wid, frame.size());
    for (size_t l = 0; l < sizes.size(); ++l) {
        std::printf("L%zu k=%u/%u  ", l, stats.k_sent[l], sizes[l]);
    }
    std::printf("\n");

    int fd = connect_with_retry(host, port);
    if (fd < 0) return 1;

    uint32_t length_le = static_cast<uint32_t>(frame.size());
    if (!write_exact(fd, &length_le, sizeof(length_le)) ||
        !write_exact(fd, frame.data(), frame.size())) {
        std::fprintf(stderr, "worker %u: send failed\n", wid);
        ::close(fd);
        return 1;
    }
    ::close(fd);
    std::printf("worker %u: sent OK\n", wid);
    return 0;
}
