#pragma once

#include "detection.hpp"
#include "race_perception.hpp"
#include "race_labels.hpp"
#include <algorithm>
#include <array>
#include <functional>
#include <map>
#include <utility>

// 单帧二值图无法区分画面顶边截断的红色圆弧与黑色道路分隔带。
// 仅检查同一帧彩色图中明确偏红、扁平、触顶且独立的暗色连通域。
// 只还原该连通域的像素，不扩张矩形、不沿用旧帧，也不修改巡线二值图。
// 黑色分隔带、与近处障碍连通的暗区、无颜色证据的情况仍保留原判断。
cv::Mat raceAppearanceGeometry(const cv::Mat &binary, const cv::Mat &image,
                               int &removed);

// 巡线边缘行集合：row -> (left_col, right_col)，与 track_identify_from_bin
// 一致。
using RaceRows = std::map<int, std::pair<int, int>, std::greater<int>>;

struct RacePath {
    RaceRows rows;
    int splitRows = 0;
    const char *geometryReason = "INVALID_ROAD_INPUT";
};

// 只做边界和分数检查：和 icar 一样信任模型标签，不再按图像内容复核。
inline bool raceDetectionAccepted(const PredictResult &d,
                                  const RaceConfig &cfg) {
    return d.type >= 0 && d.type < 15 && std::isfinite(d.score) &&
           d.score >= cfg.confidence && d.x >= 0 && d.y >= 0 && d.width > 0 &&
           d.height > 0 && d.x < RACE_COLS && d.y < RACE_ROWS &&
           d.width <= RACE_COLS - d.x && d.height <= RACE_ROWS - d.y;
}

// 交通牌属于场景中的物体，牌面上的深色图案不能充当道路分隔带。
// 只采用新鲜且达到阈值的有效检测框；行人、锥桶、道闸、斑马线不在此列。
inline std::vector<cv::Rect> raceSignRegions(const DetectionSnapshot &snap,
                                             const RaceConfig &cfg) {
    std::vector<cv::Rect> regions;
    for (const auto &d : snap.detections) {
        const bool sign = d.type == LABEL_BUSY || d.type == LABEL_LIMIT ||
                          d.type == LABEL_UNLIMIT || d.type == LABEL_STOP ||
                          d.type == LABEL_FLOW || d.type == LABEL_PARK ||
                          d.type == LABEL_FORK || d.type == LABEL_LEFT ||
                          d.type == LABEL_CHOICE || d.type == LABEL_YLEFT ||
                          d.type == LABEL_YRIGHT;
        if (!sign || !raceDetectionAccepted(d, cfg))
            continue;
        // 检测框在 640x480 原图上；留出抖动和异步帧位移的余量。
        constexpr int margin = 24;
        regions.push_back(cv::Rect(d.x - margin, d.y - margin,
                                   d.width + 2 * margin,
                                   d.height + 2 * margin) &
                          cv::Rect(0, 0, RACE_COLS, RACE_ROWS));
    }
    return regions;
}

// 仅保留与近处路面相接的白色区段，并把细小黑点和已检出的牌面合并回路面。
std::vector<std::pair<int, int>>
raceRoadBlocks(const cv::Mat &bin, int row, int low, int high,
               const RaceConfig &cfg, const std::vector<cv::Rect> &signRegions,
               int minimumBlockWidth = 0);

// 分叉必须是同一组通路持续延伸至搜索区远端，不能把不相邻的几行相加。
// 再用搜索区上方直到图像顶边的可见像素复核分隔带，避免裁掉圆环内部的白色。
// 顶部复核只参与判别，不把这些行送入 v6 的转向路径。
// 车头图案会在更远处重新合成一片白色；牌面的圆环会分成两个黑边，
// 都不满足道路分隔带连续的条件。判断使用本帧像素，AI 漏检时也不沿用旧框。
// 方向只在通过此检查的分叉上生效，不采用看到方向牌就删掉半边线的开环转向。
RacePath raceBranchPath(const cv::Mat &bin, const RaceRows &edges,
                        int direction, const RaceConfig &cfg,
                        const std::vector<cv::Rect> &signRegions = {});

struct RacePlan {
    float speedCap = 10.f;
    bool stop = false, buzzer = false, fresh = false;
    bool selectedBranch = false;
    int colorRejectedComponents = 0, geometrySplitRows = 0;
    const char *geometryReason = "INVALID_ROAD_INPUT";
    double detectionAge = -1;
    const char *state = "AI_WAIT", *reason = "AI_WAIT";
    RacePath path;
};

class RaceManager {
  private:
    RaceConfig cfg;
    std::array<int, 15> counts{};
    std::array<bool, 15> confirmed{};
    std::array<PredictResult, 15> boxes{};
    uint64_t lastSequence = 0;
    double lastCapture = -1, now = 0, stateSince = 0, cooldownUntil = 0;
    double gateClearSince = -1, hazardClearSince = -1;
    double directionUntil = 0, splitTime = 0, exitTime = 0, lastCrossSeen = 0;
    bool lastActive = false, limited = false, crossBottomSeen = false,
         gateWaiting = false, hazardWaiting = false;
    bool hazardPersonWaiting = false;
    bool hazardConeWaiting = false;
    int direction = 0;
    enum class State {
        Lane,
        ForkArmed,
        ForkSelect,
        CrossApproach,
        SignalWait,
        ReviewStop,
        Fault
    };
    State state = State::Lane;
    const char *reviewReason = "REVIEW_REQUIRED";
    PredictResult gateBox{};
    void enter(State s) {
        state = s;
        stateSince = now;
        splitTime = exitTime = 0;
    }
    bool seen(int label) const { return confirmed[label]; }
    bool near(int label, int row) const {
        return seen(label) && boxes[label].y + boxes[label].height >= row;
    }
    bool updateObservation(const DetectionSnapshot &snap, bool fresh) {
        if (!fresh || snap.sequence <= lastSequence ||
            snap.captured <= lastCapture)
            return false;
        if (lastCapture < 0 || snap.captured - lastCapture > cfg.confirmGap)
            counts.fill(0);
        lastSequence = snap.sequence;
        lastCapture = snap.captured;
        std::array<bool, 15> present{};
        for (const auto &d : snap.detections) {
            if (!raceDetectionAccepted(d, cfg))
                continue;
            if (!present[d.type] || d.score > boxes[d.type].score)
                boxes[d.type] = d;
            present[d.type] = true;
        }
        for (int i = 0; i < 15; ++i) {
            counts[i] =
                present[i] ? std::min(cfg.confirmFrames, counts[i] + 1) : 0;
            confirmed[i] = counts[i] >= cfg.confirmFrames;
        }
        return true;
    }

  public:
    explicit RaceManager(const RaceConfig &config) : cfg(config) {}
    bool aiReady(const DetectionSnapshot &s, double time) const {
        return !s.failed && s.sequence > 0 && std::isfinite(s.captured) &&
               time >= s.captured && time - s.captured <= cfg.maxDetectionAge;
    }
    RacePlan update(const DetectionSnapshot &snap, double time, float dt,
                    bool active, const cv::Mat &bin, const RaceRows &edges,
                    const cv::Mat &currentImage = {});
};
