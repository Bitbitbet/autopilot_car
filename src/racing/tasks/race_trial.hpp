#pragma once
#include <atomic>
#include <chrono>
#include <cmath>
#include <functional>
#include <initializer_list>
#include <stdexcept>
#include <thread>

// 低速短测专用的一次性截止时间。计时从首次自动启动开始，界面和相机均不驱动此线程。
// 到期回调负责锁定零速；再次按 a 不会重置时间。默认禁用，不改变普通比赛入口。
class RaceTrialDeadline {
    using Clock=std::chrono::steady_clock;
    double seconds;
    std::atomic<long long> deadline{-1};
    std::atomic_bool fired{false};
    std::function<void()> stop;
    std::jthread worker;
    static long long now() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
    }
public:
    RaceTrialDeadline(double limit,std::function<void()> callback):seconds(limit),stop(std::move(callback)) {
        if(!std::isfinite(limit)||(limit!=0&&(limit<.5||limit>3)))
            throw std::invalid_argument("Trial duration must be 0 (disabled), or 0.5..3 seconds");
        if(enabled()) worker=std::jthread([this](std::stop_token token) {
            while(!token.stop_requested()) {
                const auto until=deadline.load();
                if(until>0&&now()>=until) {
                    if(!fired.exchange(true)) stop();
                    return;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
        });
    }
    bool enabled() const { return seconds>0; }
    bool arm() {
        if(!enabled()) return false;
        long long unused=-1;
        return deadline.compare_exchange_strong(unused,now()+static_cast<long long>(seconds*1e9));
    }
    bool expired() const { return fired.load(); }
    bool armed() const { return deadline.load()>0; }
    void finish() { if(worker.joinable()) { worker.request_stop(); worker.join(); } }
    static void requireLowSpeed(float maximum,float cruise,float curve,float degraded) {
        for(float value:{maximum,cruise,curve,degraded})
            if(!std::isfinite(value)||value<=0||value>.100001f)
                throw std::invalid_argument("Timed trial requires every configured speed <=0.10 m/s");
    }
};
