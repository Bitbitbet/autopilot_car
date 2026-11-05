#pragma once

#include "camera.hpp"
#include "control.hpp"
#include "track.hpp"
#include <algorithm>
#include <opencv2/core.hpp>
#include <vector>

namespace cfg {
constexpr float CRUISE_SPEED = 1.0f; // 默认巡航
constexpr float MAX_SPEED = 1.50f;   // 上限
constexpr float MIN_SPEED = 0.05f;
} // namespace cfg

enum class WeightMode { FarFull, NearDominant, Uniform };

inline const char *weight_mode_name(WeightMode mode) {
    switch (mode) {
    case WeightMode::FarFull:
        return "far-full";
    case WeightMode::NearDominant:
        return "near-dominant";
    case WeightMode::Uniform:
        return "uniform";
    }
    return "uniform";
}

// 与硬件无关的巡线控制：边缘行 -> 期望中心 -> PD -> 舵机 + 限速。
class AutoPilot {
    float curveLimit;
    float qualityLimit;
    int cols, rows;
    float filteredDerivative = 0, previousError = 0, commandedSpeed = 0;
    double missingTime = 0, qualityGoodTime = 0;
    bool qualityRestricted = false, degraded = false, hadPath = false;
    WeightMode weight_mode = WeightMode::FarFull;

    float weightOf(int row) const;
    float qualitySpeedLimit() const;

    // update 的几步：构中心线 -> 判可用 -> 平滑/定中心 -> 出控制量
    bool update_path_state(const std::vector<cv::Point2i> &raw,
                           size_t total_rows, int accepted, int rejected,
                           float dt);
    void build_center_edge(const std::vector<cv::Point2i> &raw);
    int weighted_center() const;
    float curve_heading() const;
    void update_steering(float dt, float previousSpeed, float curveHeading);

  public:
    explicit AutoPilot(int frameCols, int frameRows);

    // 每帧识别后调用。rows 为空、点数不足或帧间隔异常时判定不可信。
    // branchSide=-1/+1 时强制用左/右边缘推中心线（岔路），0 表示自动判断。
    void update(const TrackRows &rows, const Camera &camera, float dt,
                float previousSpeed, int branchSide = 0);

    // 岔路引导：在巡线算出的舵量上叠加“最小舵量”要求。
    // 方向同向时取较大者，绝不往反方向打；dir=0 时不动。
    void apply_guidance(int dir, float pwm);

    // 引导期间的速度：不受“丢线/置信度”影响（岔口本来就会丢线）。
    float guidedSpeed(float cruise, float dt);

    // 强制转向（岔路）：把期望中心压到图像一侧，用同一套 PD 打满舵，
    // 不依赖任何边线。dir<0 左转，dir>0 右转。
    void force_turn(int dir, float dt, float previous_speed);

    // 目标点模式：外部直接给定期望的图像列，用同一套 PD 出舵机。
    // 不依赖中心线，所以看不见路面时也不会被判丢线。
    void aim_at(int target_col, float dt, float previous_speed);

    // 按 severity / 置信度算出指令速度（含加、减速限幅）。
    float speedFor(float cruise, float dt);

    // 清掉 PD 与速度斜坡状态（切模式、停车、丢线时调用）。
    void resetControl() {
        previousError = filteredDerivative = commandedSpeed = 0;
        servo = PWMSERVOMID;
        hadPath = false;
    }

    // 状态（主循环与 HUD 读取）
    const char *path_issue = "NO_POINTS";
    std::vector<cv::Point2i> centerEdge; // 平滑后的中心线 (col, row)
    int center;
    float error = 0, confidence = 0, severity = 0;
    uint16_t servo = PWMSERVOMID;
    bool valid = false, lost = false;

    // 运行时可调的限速
    float curveSpeed() const { return curveLimit; }
    float degradedSpeed() const { return qualityLimit; }
    void setCurveSpeed(float value) {
        curveLimit = std::clamp(value, cfg::MIN_SPEED, cfg::MAX_SPEED);
    }
    void setDegradedSpeed(float value) {
        qualityLimit = std::clamp(value, cfg::MIN_SPEED, cfg::MAX_SPEED);
    }

    void cycleWeight() { weight_mode = WeightMode((int(weight_mode) + 1) % 3); }
    WeightMode get_weight_mode() const { return weight_mode; }
};
