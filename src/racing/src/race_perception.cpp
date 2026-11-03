#include "race_perception.hpp"
#include "detection.hpp"
#include <filesystem>
#include <mutex>

using cv::Mat;
using std::lock_guard;
using std::make_shared;
using std::make_unique;
using std::unique_lock;
using std::unique_ptr;
namespace fs = std::filesystem;

unique_ptr<RacePerception> createRacePerception(const RaceConfig &config) {
    const fs::path path = config.modelDirectory;
    auto detector = make_shared<Detection>(path.string(), config.confidence);
    return make_unique<RacePerception>([detector](const cv::Mat &frame) {
        detector->inference(frame);
        return detector->results;
    });
}
void RacePerception::worker_run() {
    while (true) {
        Mat frame;
        uint64_t captured;
        {
            unique_lock lock(mutex);
            wake.wait(lock, [&] { return worker_stop || !pending.empty(); });
            if (worker_stop)
                return;
            frame = std::move(pending);
            captured = pendingTime;
        }
        auto result = infer(frame);

        lock_guard lock(mutex);
        latest = {captured, std::move(result)};
    }
}