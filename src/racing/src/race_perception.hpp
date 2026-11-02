#pragma once

#include "detection.hpp"
#include "race_config.hpp"
#include <chrono>
#include <condition_variable>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>

struct DetectionSnapshot {
    uint64_t sequence = 0;
    double captured = 0;
    std::vector<PredictResult> detections;
    bool failed = false;
    std::string error;
    cv::Mat image; // 与结果对应的原始推理图像
};
inline double raceTime() {
    return std::chrono::duration<double>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

class RacePerception {
  public:
    using Infer = std::function<std::vector<PredictResult>(const cv::Mat &)>;

  private:
    Infer infer;
    std::mutex mutex;
    std::condition_variable wake;
    cv::Mat pending;
    double pendingTime = 0;
    uint64_t pendingSequence = 0;
    DetectionSnapshot latest;
    bool stop = false;
    std::thread worker;

    void worker_run();

  public:
    explicit RacePerception(Infer fn)
        : infer(std::move(fn)), worker([this] { worker_run(); }) {}
    ~RacePerception() {
        {
            std::lock_guard lock(mutex);
            stop = true;
        }
        wake.notify_all();
        worker.join();
    }
    void submit(const cv::Mat &image, uint64_t sequence, double captured) {
        auto copy = image.clone();
        {
            std::lock_guard lock(mutex);
            pending = std::move(copy);
            pendingSequence = sequence;
            pendingTime = captured;
        }
        wake.notify_one();
    }
    DetectionSnapshot snapshot() {
        std::lock_guard lock(mutex);
        return latest;
    }
};

std::unique_ptr<RacePerception> createRacePerception(const RaceConfig &config);