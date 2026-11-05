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

// 主循环回传的车辆状态（岔路收尾判据用）
struct RaceFeedback {
    float speed = 0;    // 上一帧下发的速度（m/s）
    float distance = 0; // 上一帧走过的距离（米）
    bool tracking = false; // 现在跟踪住路面了
};

struct RacePlan {
    float speedCap = 10.f; // 速度上限，10 等于不限制
    bool stop = false;
    double detectionAge = -1; // AI 结果的新鲜度（秒）
    const char *state = "AI_WAIT", *reason_for_stage = "AI_WAIT";

    // 岔路：让 pilot 优先用哪一条边推中心线（-1 左，+1 右，0 自动）。
    // 地面上看到 fork 默认走左边；前方 yleft / yright 会覆盖它。
    int branchSide = 0;
    // 强制转向：-1 左 / +1 右 / 0 不强制。岔路时用它把车硬拐过去。
    int forceTurn = 0;
    // Y 岔路引导：方向（-1 左 / +1 右 / 0 无）和最小舵量（PWM）。
    // 舵量按脉冲给出：先强打，再线性松开。
    int guideDirection = 0;
    float guidePwm = 0;

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
    // aiming：主循环正在走目标点（停车场）。
    RacePlan update(const DetectionSnapshot &snap, double dt, bool active,
                    const Camera &camera, bool aiming,
                    const RaceFeedback &feedback);

  private:
    RaceConfig cfg;
    const char *reason_for_review = "REVIEW_REQUIRED";

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

    bool fatal_stop = false;
    bool lastActive = false;
    bool driving_on_park_lane = false;
    // Y 形岔路（引导脉冲）：方向、脉冲相位、行进距离、牌子消失与新路跟踪时长
    bool yfork = false;
    int yfork_dir = 0;         // -1 左 / +1 右
    double yfork_elapsed = 0;  // 进入岔路后过了多久
    double yfork_phase = 0;    // 脉冲相位（按速度推进，1.4 = 脉冲走完）
    double yfork_clear = 0;    // 岔路牌消失多久
    double yfork_track = 0;    // 新路被跟踪住多久
    float yfork_distance = 0;  // 这段岔路走了多远

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
