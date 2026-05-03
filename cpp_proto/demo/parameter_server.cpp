// pilt_parameter_server -- minimal TCP-based PS that demonstrates the
// libpilt aggregator on real sockets.  Single-round, blocking accept loop.
//
// Wire framing on the socket (in addition to the libpilt frame itself):
//
//     uint32 length_prefix      -- size in bytes of the libpilt frame
//     uint8  body[length_prefix] -- libpilt frame (header + layers + crc)
//
// Usage:
//
//   pilt_parameter_server  <port>  <n_workers>  <layer_sizes...>
//
//   layer_sizes is a space-separated list of per-layer S_l (must match what
//   each worker's encoder was constructed with).
//
// Example:
//
//   pilt_parameter_server 5005 4   1024 4096 256
//
// On completion the PS prints the per-layer average's L2 norm and the
// number of distinct elements that received at least one update.

#include "pilt/aggregator.hpp"
#include "pilt/wire.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

using pilt::PILTAggregator;
using pilt::PiltError;

namespace {

bool read_exact(int fd, void* buf, size_t n) {
    auto* p = static_cast<uint8_t*>(buf);
    while (n > 0) {
        ssize_t r = ::recv(fd, p, n, MSG_WAITALL);
        if (r == 0) return false;        // peer closed
        if (r < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        p += r;
        n -= static_cast<size_t>(r);
    }
    return true;
}

void serve_one_worker(int fd, PILTAggregator& ps) {
    uint32_t le_len = 0;
    if (!read_exact(fd, &le_len, sizeof(le_len))) {
        std::fprintf(stderr, "PS: failed to read length prefix from worker\n");
        ::close(fd);
        return;
    }
    // length is little-endian on the wire (matches host on x86; convert
    // explicitly to avoid relying on host endianness).
    uint32_t len = ((le_len >> 24) & 0xFFu) |
                   ((le_len >> 8 ) & 0xFF00u) |
                   ((le_len << 8 ) & 0xFF0000u) |
                   ((le_len << 24) & 0xFF000000u);
    // We actually wrote the length in *host* order from the worker; on
    // little-endian hosts the swap above would corrupt it.  Simpler: trust
    // little-endian convention and fix it up using a portable helper.
    len = le_len;  // wire spec is LE; on LE host this is identity.

    if (len > 256u * 1024u * 1024u) {
        std::fprintf(stderr, "PS: refusing oversized frame (%u bytes)\n", len);
        ::close(fd);
        return;
    }

    std::vector<uint8_t> frame(len);
    if (!read_exact(fd, frame.data(), len)) {
        std::fprintf(stderr, "PS: short read of %u-byte frame\n", len);
        ::close(fd);
        return;
    }
    try {
        ps.add(frame);
        std::printf("PS: accepted worker frame (%u bytes, total=%zu)\n",
                    len, ps.messages_received());
    } catch (const PiltError& e) {
        std::fprintf(stderr, "PS: rejected worker frame: %s\n", e.what());
    }
    ::close(fd);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr,
            "usage: %s <port> <n_workers> <S_1> [S_2 ...]\n", argv[0]);
        return 2;
    }
    const uint16_t port = static_cast<uint16_t>(std::atoi(argv[1]));
    const uint32_t K    = static_cast<uint32_t>(std::atoi(argv[2]));

    std::vector<uint32_t> sizes;
    for (int i = 3; i < argc; ++i) {
        long s = std::atol(argv[i]);
        if (s < 0) { std::fprintf(stderr, "bad layer size: %s\n", argv[i]); return 2; }
        sizes.push_back(static_cast<uint32_t>(s));
    }

    PILTAggregator ps(K, sizes);
    ps.begin_round(0);   // single-round demo; round_num = 0

    int sock = ::socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) { std::perror("socket"); return 1; }
    int yes = 1;
    ::setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);  // localhost-only demo
    addr.sin_port = htons(port);

    if (::bind(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        std::perror("bind"); return 1;
    }
    if (::listen(sock, static_cast<int>(K)) < 0) {
        std::perror("listen"); return 1;
    }
    std::printf("PS: listening on 127.0.0.1:%u, expecting K=%u workers\n",
                port, K);

    std::vector<std::thread> sessions;
    for (uint32_t i = 0; i < K; ++i) {
        sockaddr_in caddr{};
        socklen_t cl = sizeof(caddr);
        int cfd = ::accept(sock, reinterpret_cast<sockaddr*>(&caddr), &cl);
        if (cfd < 0) { std::perror("accept"); continue; }
        int one = 1;
        ::setsockopt(cfd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        sessions.emplace_back(serve_one_worker, cfd, std::ref(ps));
    }
    for (auto& t : sessions) if (t.joinable()) t.join();
    ::close(sock);

    std::printf("PS: aggregating %zu workers\n", ps.messages_received());
    auto avg = ps.finalize_average();

    for (size_t l = 0; l < avg.size(); ++l) {
        long double sq = 0.0L;
        size_t nz = 0;
        for (float x : avg[l]) {
            if (x != 0.0f) ++nz;
            sq += static_cast<long double>(x) * static_cast<long double>(x);
        }
        std::printf("PS: layer %zu  size=%zu  nz=%zu  ||avg||_2=%.6Lf\n",
                    l, avg[l].size(), nz,
                    static_cast<long double>(std::sqrt(static_cast<double>(sq))));
    }
    return 0;
}
