#pragma once

#include "camera.hpp"
#include "control.hpp"
#include <algorithm>
#include <functional>
#include <map>
#include <opencv2/core.hpp>
#include <utility>
#include <vector>

// 主循环与 HUD 也用到的三个速度参数；其余调参都在 pilot.cpp 顶部。
namespace cfg {
constexpr float CRUISE_SPEED = 0.10f; // 固定入口初始巡航
constexpr float MAX_SPEED = 10.0f;   // 沿用可调上限
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

// 逐行双边坐标：row（0=远，越大越近，降序）-> {左边缘, 右边缘}
using TrackRows = std::map<int, std::pair<int, int>, std::greater<int>>;

// 与硬件无关的巡线控制：边缘行 -> 期望中心 -> PD -> 舵机 + 限速。
class AutoPilot {
    float curveLimit; // 运行时可调（[ ] , . 键）
    float qualityLimit;
    int imageCols, imageRows;
    float filteredDerivative = 0, previousError = 0, commandedSpeed = 0;
    double missingTime = 0, qualityGoodTime = 0;
    bool qualityRestricted = false, degraded = false, hadPath = false;
    WeightMode weightMode = WeightMode::FarFull;
    float runP1=1.10f,runP2=.00175f,turnD=1.75f,maxFrameGap=.25f;

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
    void configure(float curve,float quality,float p1,float p2,float derivative,float gap){curveLimit=curve;qualityLimit=quality;runP1=p1;runP2=p2;turnD=derivative;maxFrameGap=gap;}

    // 每帧识别后调用。rows 为空、点数不足或帧间隔异常时判定不可信。
    // 哪一边更平滑由自己判断：两边都好取中点，只有一边好就用它加减半个路宽。
    void update(const TrackRows &rows, const Camera &camera, float dt,
                float previousSpeed);

    // 按 severity / 置信度算出指令速度（含加、减速限幅）。
    float speedFor(float cruise, float dt);

    // 清掉 PD 与速度斜坡状态（切模式、停车、丢线时调用）。
    void resetControl() {
        previousError = filteredDerivative = commandedSpeed = 0;
        servo = PWMSERVOMID;
        hadPath = false;
    }

    // 状态（主循环与 HUD 读取）
    const char *pathIssue = "NO_POINTS";
    std::vector<cv::Point2i> centerEdge; // 平滑后的中心线 (col, row)
    int center = imageCols / 2;
    float error = 0, confidence = 0, severity = 0;
    uint16_t servo = PWMSERVOMID;
    bool valid = false, lost = false;
    bool is_singal_smooth = false; // 上一帧是用单边推的中心线

    // 运行时可调的限速
    float curveSpeed() const { return curveLimit; }
    float degradedSpeed() const { return qualityLimit; }
    void setCurveSpeed(float value) {
        curveLimit = std::clamp(value, cfg::MIN_SPEED, cfg::MAX_SPEED);
    }
    void setDegradedSpeed(float value) {
        qualityLimit = std::clamp(value, cfg::MIN_SPEED, cfg::MAX_SPEED);
    }

    void cycleWeight() { weightMode = WeightMode((int(weightMode) + 1) % 3); }
    const char *weightName() const { return weight_mode_name(weightMode); }
};
