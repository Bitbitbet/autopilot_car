#pragma once
#include <atomic>
#include <csignal>

namespace car_signal {
static_assert(std::atomic_bool::is_always_lock_free);
inline std::atomic_bool requested{false};
inline void handler(int) { requested.store(true, std::memory_order_relaxed); }
inline void install() {
    std::signal(SIGINT, handler);
    std::signal(SIGTERM, handler);
    std::signal(SIGHUP, handler);
}
}
