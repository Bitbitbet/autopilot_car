#include "race_perception.hpp"
#include "detection.hpp"
#include <filesystem>

using cv::Mat;
using std::make_shared;
using std::make_unique;
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
    try {
        while (true) {
            Mat frame;
            double captured;
            uint64_t sequence;
            {
                std::unique_lock lock(mutex);
                wake.wait(lock, [&] { return stop || !pending.empty(); });
                if (stop)
                    return;
                frame = std::move(pending);
                captured = pendingTime;
                sequence = pendingSequence;
            }
            auto result = infer(frame);
            std::lock_guard lock(mutex);
            latest = {sequence, captured, std::move(result),
                      false,    {},       std::move(frame)};
        }
    } catch (const std::exception &e) {
        std::lock_guard lock(mutex);
        latest.failed = true;
        latest.error = e.what();
    } catch (...) {
        std::lock_guard lock(mutex);
        latest.failed = true;
        latest.error = "AI worker failed";
    }
}