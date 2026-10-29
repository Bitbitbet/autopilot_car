#include "race_perception.hpp"
#include "detection.hpp"
#include <filesystem>

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
