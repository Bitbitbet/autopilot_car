#pragma once

#include "detection.hpp"
#include "race_config.hpp"
#include "race_perception.hpp"
#include "track.hpp"
#include <array>
#include <cmath>
#include <cstdint>

// 一次分叉几何扫描的结果。
// rows 只有 selectedBranch 时才会被主循环使用：direction==0 时是全宽，
// 否则是按方向裁掉另一侧之后的行集合。
struct RacePath {
    TrackRows rows;
    int splitRows = 0;                         // 连续判定为分叉的扫描行数
    const char *reason = "INVALID_ROAD_INPUT"; // 判定结论，可直接显示
};

// 本帧“该不该走 / 走多快 / 走哪条支路”，横向控制不在这里。
struct RacePlan {
    float speedCap = 10.f; // 速度上限，10 等于不限制
    bool stop = false, buzzer = false;
    bool selectedBranch = false; // path.rows 是否是可用的一条支路
    double detectionAge = -1;    // AI 结果的新鲜度（秒）
    const char *state = "AI_WAIT", *reason = "AI_WAIT";
    RacePath path;
};

// 比赛任务层：AI 标签 + 赛道几何 -> 停车/限速/选路。
class RaceManager {
  public:
    bool aiReady(const DetectionSnapshot &s, double time) const {
        return !s.failed && s.sequence > 0 && std::isfinite(s.captured) &&
               time >= s.captured && time - s.captured <= cfg.maxDetectionAge;
    }
    RacePlan update(const DetectionSnapshot &snap, double time, float dt,
                    bool active, const cv::Mat &bin, const TrackRows &edges);

  private:
    // 只有 Lane 是“正常巡线”，其余都是它前后的局部任务。
    // Review 是死锁状态：停车并等待人工重启。
    enum class State {
        Lane,
        ForkArmed,
        ForkSelect,
        CrossApproach,
        SignalWait,
        Review
    };

    RaceConfig cfg;
    State state = State::Lane;
    const char *reviewReason = "REVIEW_REQUIRED";

    // 标签确认：同一类别连续 confirmFrames 帧才算“看见”。
    std::array<int, LABEL_COUNT> counts{};
    std::array<bool, LABEL_COUNT> confirmed{};
    std::array<InferenceResult, LABEL_COUNT> boxes{};
    uint64_t lastSequence = 0;
    double lastCapture = -1;

    // 时间
    double now = 0, stateSince = 0, cooldownUntil = 0;
    double splitTime = 0, exitTime = 0; // 分叉确认 / 驶离计时
    double directionUntil = 0;          // 方向牌的记忆期限
    double lastCrossSeen = 0;           // 最近一次看到斑马线
    double gateClearSince = -1, hazardClearSince = -1;

    int direction = 0; // -1 左，+1 右，0 未知

    bool lastActive = false;
    bool limited = false;         // 限速区，跨人工接管保持
    bool crossBottomSeen = false; // 车身已压过斑马线
    bool gateWaiting = false;
    bool hazardWaiting = false;
    bool coneLatched = false; // 近处锥桶：没有绕行规划，看到就锁死
    InferenceResult gateBox{};

    void enter(State s) {
        state = s;
        stateSince = now;
        splitTime = exitTime = 0;
    }
    bool seen(int label) const { return confirmed[label]; }
    bool near(int label, int row) const {
        return seen(label) && boxes[label].y + boxes[label].height >= row;
    }
    bool updateObservation(const DetectionSnapshot &snap);
};
