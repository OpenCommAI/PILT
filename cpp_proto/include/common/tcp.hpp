// common/tcp.hpp -- POSIX TCP helpers shared across protocol demos.
//
// Pure header-only inline helpers so callers don't need an extra .cpp.
// Linux/macOS/BSD only (depends on <sys/socket.h>).

#pragma once

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <thread>

namespace proto::tcp {

inline bool read_exact(int fd, void* buf, size_t n) {
    auto* p = static_cast<uint8_t*>(buf);
    while (n > 0) {
        ssize_t r = ::recv(fd, p, n, MSG_WAITALL);
        if (r == 0) return false;
        if (r < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        p += r;
        n -= static_cast<size_t>(r);
    }
    return true;
}

inline bool write_exact(int fd, const void* buf, size_t n) {
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

// Send `len` bytes prefixed by a 4-byte little-endian length header.
inline bool write_framed(int fd, const void* buf, uint32_t len) {
    uint8_t hdr[4] = {
        static_cast<uint8_t>(len & 0xFFu),
        static_cast<uint8_t>((len >> 8) & 0xFFu),
        static_cast<uint8_t>((len >> 16) & 0xFFu),
        static_cast<uint8_t>((len >> 24) & 0xFFu),
    };
    if (!write_exact(fd, hdr, 4)) return false;
    return write_exact(fd, buf, len);
}

// Receive one length-prefixed frame.  out is resized to the frame length.
inline bool read_framed(int fd, std::vector<uint8_t>& out,
                        uint32_t max_bytes = 256u * 1024u * 1024u) {
    uint8_t hdr[4];
    if (!read_exact(fd, hdr, 4)) return false;
    uint32_t len = static_cast<uint32_t>(hdr[0]) |
                   (static_cast<uint32_t>(hdr[1]) << 8) |
                   (static_cast<uint32_t>(hdr[2]) << 16) |
                   (static_cast<uint32_t>(hdr[3]) << 24);
    if (len > max_bytes) return false;
    out.resize(len);
    return read_exact(fd, out.data(), len);
}

// Connect to host:port with retry/backoff (host as dotted-quad IPv4 only;
// loopback demo grade -- swap in getaddrinfo() for production).
inline int connect_with_retry(const char* host, uint16_t port,
                              int max_retries = 30,
                              int backoff_ms = 200) {
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port   = htons(port);
    if (::inet_pton(AF_INET, host, &addr.sin_addr) <= 0) return -1;
    for (int i = 0; i < max_retries; ++i) {
        int fd = ::socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0) return -1;
        if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0) {
            int one = 1;
            ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
            return fd;
        }
        ::close(fd);
        std::this_thread::sleep_for(std::chrono::milliseconds(backoff_ms));
    }
    return -1;
}

// Bind+listen on 127.0.0.1:port for loopback demos.
inline int listen_loopback(uint16_t port, int backlog = 16) {
    int sock = ::socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) return -1;
    int yes = 1;
    ::setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons(port);
    if (::bind(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        ::close(sock); return -1;
    }
    if (::listen(sock, backlog) < 0) { ::close(sock); return -1; }
    return sock;
}

// Best-effort: enable kernel DCTCP congestion control on a TCP socket.
// Returns true if accepted by the kernel, false otherwise.  On Linux this
// requires the `dctcp` module to be loaded and the cgroup / sysctl
// permission to be set; on other OSes this is a no-op returning false.
inline bool try_set_dctcp(int fd) {
#ifdef __linux__
    const char name[] = "dctcp";
    return ::setsockopt(fd, IPPROTO_TCP, TCP_CONGESTION,
                        name, sizeof(name) - 1) == 0;
#else
    (void)fd;
    return false;
#endif
}

}  // namespace proto::tcp
