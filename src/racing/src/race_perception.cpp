#include "race_perception.hpp"
#include "detection.hpp"
#include <filesystem>
#include <fstream>

// 与 detection_tool 相同的接入方式：直接构造 Detection，在本机推理。
// 感知只负责给出当前帧的检测结果，复核与状态机都在上层。
std::unique_ptr<RacePerception> createRacePerception(const RaceConfig &config) {
    const std::filesystem::path path = config.modelDirectory;
    // 标签顺序决定 RaceManager 的类别索引，部署错模型要在开车前报错。
    const std::vector<std::string> expected = {
        "cone", "person", "busy", "limit", "unlimit", "stop",  "flow",  "park",
        "gate", "cross",  "fork", "left",  "choice",  "yleft", "yright"};
    std::ifstream labels(path / "label_list.txt");
    std::vector<std::string> actual;
    for (std::string line; std::getline(labels, line);) {
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        actual.push_back(line);
    }
    if (actual != expected)
        throw std::runtime_error("Model label order differs from race labels");
    // 停车牌与斑马线要留低分候选供复核，推理下限取三者中最低的。
    float floor =
        config.verifyStopCandidates
            ? std::min(config.confidence, config.stopCandidateConfidence)
            : config.confidence;
    if (config.verifyCrosswalkGeometry)
        floor = std::min(floor, config.crossCandidateConfidence);
    auto detector = std::make_shared<Detection>(path.string(), floor);
    return std::make_unique<RacePerception>(
        [detector, config](const cv::Mat &frame) {
            detector->inference(frame);
            std::vector<PredictResult> result;
            for (const auto &d : detector->results)
                if (d.score >= config.confidence ||
                    (config.verifyStopCandidates && d.type == LABEL_STOP &&
                     d.score >= config.stopCandidateConfidence) ||
                    (config.verifyCrosswalkGeometry && d.type == LABEL_CROSS &&
                     d.score >= config.crossCandidateConfidence))
                    result.push_back(d);
            return result;
        },
        config);
}
