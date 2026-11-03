#pragma once

#include "detection.hpp"
#include "race_config.hpp"
#include "timing.hpp"
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>

struct DetectionSnapshot {
    uint64_t captured = 0;
    std::vector<InferenceResult> detections;
};

class RacePerception {
  public:
    using Infer = std::function<std::vector<InferenceResult>(const cv::Mat &)>;

  private:
    Infer infer;
    std::mutex mutex;
    std::condition_variable wake;

    cv::Mat pending;
    uint64_t pendingTime = 0;

    DetectionSnapshot latest;
    bool worker_stop = false;
    std::thread worker;

    void worker_run();

  public:
    explicit RacePerception(Infer fn)
        : infer(std::move(fn)), worker([this] { worker_run(); }) {}

    ~RacePerception() {
        {
            std::lock_guard lock(mutex);
            worker_stop = true;
        }
        wake.notify_all();
        worker.join();
    }

    void submit(const cv::Mat &image) {
        auto copy = image.clone();
        {
            std::lock_guard lock(mutex);
            pending = std::move(copy);
            pendingTime = global_microsec();
        }
        wake.notify_one();
    }

    DetectionSnapshot snapshot() {
        std::lock_guard lock(mutex);
        return latest;
    }
};

std::unique_ptr<RacePerception> createRacePerception(const RaceConfig &config);