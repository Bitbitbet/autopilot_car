#pragma once

#include "detection.hpp"
#include "race_config.hpp"
#include "race_perception.hpp"
#include <array>
#include <cstdint>

// 本帧“该不该走 / 走多快 / 往哪边打舵”，横向闭环在 AutoPilot。
struct RacePlan {
    float speedCap = 10.f; // 速度上限，10 等于不限制
    bool stop = false;
    double detectionAge = -1; // AI 结果的新鲜度（秒）
    const char *state = "AI_WAIT", *reason = "AI_WAIT";

    // 岔路：让 pilot 优先用哪一条边推中心线（-1 左，+1 右，0 自动）。
    // 地面上看到 fork 默认走左边；前方 yleft / yright 会覆盖它。
    int branchSide = 0;
};

// 比赛任务层：只看 AI 标签和时间，路面几何全部交给 track/AutoPilot。
class RaceManager {
  public:
    bool aiReady(const DetectionSnapshot &s, double time) const {
        return s.sequence > 0 && time >= s.captured &&
               time - s.captured <= cfg.maxDetectionAge;
    }
    RacePlan update(const DetectionSnapshot &snap, double time, float dt,
                    bool active);

  private:
    // 只有 Lane 是“正常巡线”，其余都是它前后的局部任务。
    // FatalStop 是死锁状态，进入后不能改变
    enum class State { Lane, CrossApproach, SignalWait, FatalStop };

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
    double now = 0, stateSince = 0;
    double lastCrossSeen = 0; // 最近一次看到斑马线
    double gateClearSince = -1, hazardClearSince = -1;

    // 岔路偏向：sign 消失后再保持一小段时间，避免单帧漏检来回跳。
    int side = 0; // -1 左，+1 右，0 自动
    double sideUntil = 0;

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
    }
    bool seen(int label) const { return confirmed[label]; }
    bool near(int label, int row) const {
        return seen(label) && boxes[label].y + boxes[label].height >= row;
    }
    bool updateObservation(const DetectionSnapshot &snap);
};
