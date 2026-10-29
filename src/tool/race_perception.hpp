#pragma once
#include "race_config.hpp"
#include "race_direction.hpp"
#include "race_crosswalk.hpp"
#include "tools.hpp"
#include <chrono>
#include <condition_variable>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>

struct RaceSnapshot {
    uint64_t sequence=0;
    double captured=0;
    std::vector<PredictResult> detections;
    bool failed=false;
    std::string error;
    cv::Mat image; // 与结果对应的原始推理图像，只读共享，不能用新帧冒充。
    std::vector<ForkDirectionCheck> directionChecks{};
    CrosswalkGeometry crosswalk{};
};
inline double raceTime() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// 只保留最新待推理帧；结果携带原图时间，禁止同一旧结果被反复计数。
class RacePerception {
public:
    using Infer=std::function<std::vector<PredictResult>(const cv::Mat&)>;
private:
    Infer infer;
    RaceConfig config;
    std::mutex mutex;
    std::condition_variable wake;
    cv::Mat pending;
    double pendingTime=0;
    uint64_t pendingSequence=0;
    RaceSnapshot latest;
    bool stop=false;
    std::thread worker;
    void run() noexcept {
        try {
            for(;;) {
                cv::Mat frame; double captured; uint64_t sequence;
                {
                    std::unique_lock lock(mutex);
                    wake.wait(lock,[&] { return stop||!pending.empty(); });
                    if(stop) return;
                    frame=std::move(pending); captured=pendingTime; sequence=pendingSequence;
                }
                auto result=infer(frame);
                auto checks=config.verifyForkArrows?verifyForkDirections(frame,result):std::vector<ForkDirectionCheck>{};
                auto cross=config.verifyCrosswalkGeometry?detectCrosswalkGeometry(frame):CrosswalkGeometry{};
                std::lock_guard lock(mutex);
                latest={sequence,captured,std::move(result),false,{},std::move(frame),std::move(checks),std::move(cross)};
            }
        } catch(const std::exception &e) {
            std::lock_guard lock(mutex); latest.failed=true; latest.error=e.what();
        } catch(...) {
            std::lock_guard lock(mutex); latest.failed=true; latest.error="AI worker failed";
        }
    }
public:
    explicit RacePerception(Infer fn,const RaceConfig &cfg={}):infer(std::move(fn)),config(cfg),worker([this] { run(); }) {}
    ~RacePerception() { { std::lock_guard lock(mutex); stop=true; } wake.notify_all(); worker.join(); }
    void submit(const cv::Mat &image,uint64_t sequence,double captured) {
        auto copy=image.clone();
        { std::lock_guard lock(mutex); pending=std::move(copy); pendingSequence=sequence; pendingTime=captured; }
        wake.notify_one();
    }
    RaceSnapshot snapshot() { std::lock_guard lock(mutex); return latest; }
};

// 实车后端复用组员的 T710 NPU 检测器；离线回放不加载此后端。
std::unique_ptr<RacePerception> createRacePerception(const RaceConfig &config);
