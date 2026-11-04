#pragma once

#include "camera.hpp"
#include "detection.hpp"
#include "race_config.hpp"
#include "race_perception.hpp"
#include <array>
#include <vector>

// 本帧“该不该走 / 走多快 / 往哪边打舵”，横向闭环在 AutoPilot。
// 一个目标点：位置 + 期望到达方向（车体系，米）。
// 方向用来规划圆弧（沿圆弧切线到达），{0,0} 表示不约束、直接朝点开。
struct RaceTarget {
    cv::Point2d point{};
    cv::Point2d heading{};
};

struct RacePlan {
    float speedCap = 10.f; // 速度上限，10 等于不限制
    bool stop = false;
    double detectionAge = -1; // AI 结果的新鲜度（秒）
    const char *state = "AI_WAIT", *reason_for_stage = "AI_WAIT";

    // 岔路：让 pilot 优先用哪一条边推中心线（-1 左，+1 右，0 自动）。
    // 地面上看到 fork 默认走左边；前方 yleft / yright 会覆盖它。
    int branchSide = 0;

    // 目标点序列：车体系地面坐标（x 右正、y 前正，米），按顺序走。
    // 一次性给完；全部走完之前 targets 不变，走完后清空。
    // 换新序列 targetSeq 就 +1。
    int target_seq = 0;
    std::vector<RaceTarget> targets;
};

// 比赛任务层：只看 AI 标签和时间，路面几何全部交给 track/AutoPilot。
class RaceManager {
  public:
    // aiming：主循环正在执行这串目标；targets_done：这串目标全部走完了。
    RacePlan update(const DetectionSnapshot &snap, double dt, bool active,
                    const Camera &camera, bool aiming);

  private:
    // 只有 Lane 是“正常巡线”，其余都是它前后的局部任务。
    // FatalStop 是死锁状态，进入后不能改变
    enum class State { Lane, FatalStop };

    RaceConfig cfg;
    State state = State::Lane;
    const char *reviewReason = "REVIEW_REQUIRED";

    // 每个类别一份跟踪状态：最近一帧的框
    struct LabelTrack {
        bool confirmed = false;             // 够 confirmFrames 帧
        std::vector<InferenceResult> boxes; // 最近一帧这个类别的所有框
    };
    std::array<LabelTrack, LABEL_COUNT> tracks;

    // 时间
    double lastCrossSeen = 0; // 最近一次看到斑马线
    double gateClearSince = -1, hazardClearSince = -1;

    // 目标点序列号
    int target_seq = 0;

    bool lastActive = false;
    bool driving_on_park_lane = false;
    bool yfork_target_ready = false;
    RaceTarget yfork_left_target, yfork_right_target;

    const std::vector<InferenceResult> &boxes_of(int label) const {
        return tracks[label].boxes;
    }
    // 这一帧看到了这个类别（还没到确认帧数也算）
    bool present(int label) const { return !tracks[label].boxes.empty(); }
    bool seen(int label) const { return tracks[label].confirmed; }
    // 确认过的框里有没有已经开到 row 以下的（row 越小越远）
    bool near(int label, int row) const {
        if (!tracks[label].confirmed)
            return false;
        for (const auto &d : tracks[label].boxes)
            if (d.rect.y + d.rect.height >= row)
                return true;
        return false;
    }
    const InferenceResult &best_box(int label) const; // 分数最高的那个
    // 目标点用的 choice 框：多个 choice 时取离 park 最近的那个
    const InferenceResult *nearest_choice() const;
    void clear_observations();
    bool update_observation(const DetectionSnapshot &snap);
};
