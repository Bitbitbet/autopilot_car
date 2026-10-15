#pragma once
#include "control.hpp"
#include "tools.hpp"
#include "trackview_config.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <vector>

enum class WeightMode { FarFull, NearDominant, Uniform };
inline uint16_t clampServo(float pwm) {
    return static_cast<uint16_t>(std::lround(std::clamp(
        pwm, float(PWMSERVOMIN), float(PWMSERVOMAX))));
}

// 与硬件解耦的控制器，供实车运行和离线回放测试共用。
class AutoPilot {
    struct Width {
        float value = 0, pendingValue = 0;
        double time = -1;
        int observations = 0, pendingObservations = 0;
    };
    std::array<Width, IMAGE_ROW> widths{};
    TrackviewConfig config;
    float filteredDerivative = 0, previousError = 0, commandedSpeed = 0;
    double now = 0, missingTime = 0, stableTime = 0;
    double qualityGoodTime = 0;
    bool qualityRestricted = false;
    bool hadPath = false;

    bool trusted(const PointX &p) const {
        return p.x >= 0 && p.x < int(IMAGE_ROW) &&
               p.y > config.borderMargin && p.y < int(IMAGE_COL) - 1 - config.borderMargin;
    }
    float widthAt(int row) const {
        // 只在邻近图像行的观测之间插值，不把固定宽度外推到其他距离。
        int lower = -1, upper = -1;
        for (int r = std::max(0, row - 12); r <= std::min(int(IMAGE_ROW)-1, row+12); ++r) {
            const auto &w = widths[r];
            // 停车等待本身不会改变相机透视关系或赛道宽度。
            // 可信观测与记录冲突时，暂停使用该行宽度，直到重新验证。
            if (w.observations < 3 || w.pendingObservations > 0) continue;
            if (r <= row) lower = r;
            if (r >= row && upper < 0) upper = r;
        }
        if (lower < 0 && upper < 0) return 0;
        if (lower < 0) return widths[upper].value;
        if (upper < 0 || upper == lower) return widths[lower].value;
        const float t = float(row - lower) / (upper - lower);
        return widths[lower].value * (1-t) + widths[upper].value * t;
    }
    float weightOf(int row) const {
        if (weightMode == WeightMode::Uniform) return 1;
        if (weightMode == WeightMode::NearDominant)
            return std::min(float(row), IMAGE_ROW / 2.f);
        return row < int(IMAGE_ROW)/2 ? IMAGE_ROW/2.f : float(IMAGE_ROW-row);
    }
public:
    struct WidthReading {
        int row, observations, pendingObservations;
        float width;
        double age;
        bool refreshDue;
    };
    std::vector<WidthReading> widthProfile() const {
        std::vector<WidthReading> result;
        for (int row=0; row<int(IMAGE_ROW); ++row) {
            const auto &w=widths[row];
            if(w.observations) result.push_back({row,w.observations,w.pendingObservations,
                w.value,now-w.time,now-w.time>config.widthLifetime});
        }
        return result;
    }
    int learnedWidthRows() const {
        return int(std::count_if(widths.begin(),widths.end(),[](const Width &w) {
            return w.observations>=3 && w.pendingObservations==0;
        }));
    }
    const char *pathIssue = "NO_POINTS";
    int missingWidthPoints=0, untrustedPoints=0, widthRangePoints=0, jumpPoints=0;
    std::vector<PointX> centerEdge;
    int center = IMAGE_COL / 2, dualPoints = 0, singlePoints = 0, rejectedPoints = 0;
    float error = 0, heading = 0, confidence = 0, severity = 0;
    float rawSeverity = 0, curveHeading = 0, speedTarget = 0;
    bool curveWindowFallback = false;
    const char *curveCause = "NONE", *speedReason = "STOP";
    uint16_t servo = PWMSERVOMID;
    bool valid = false, ready = false, lost = false, degraded = false;
    WeightMode weightMode = WeightMode::FarFull;
    explicit AutoPilot(const TrackviewConfig &cfg = {}) : config(cfg) { config.validate(); }
    void cycleWeight() { weightMode = WeightMode((int(weightMode)+1)%3); }
    const char *weightName() const {
        if (weightMode == WeightMode::FarFull) return "far-full";
        if (weightMode == WeightMode::NearDominant) return "near-dominant";
        return "uniform";
    }
    float singleFraction() const {
        return float(singlePoints)/std::max(dualPoints+singlePoints,1);
    }
    float curveSpeed() const { return config.curveSpeed; }
    float degradedSpeed() const { return config.degradedSpeed; }
    void setCurveSpeed(float value) {
        if(!std::isfinite(value)) throw std::invalid_argument("Invalid curve speed");
        config.curveSpeed=std::clamp(value,config.minSpeed,config.maxSpeed);
    }
    void setDegradedSpeed(float value) {
        if(!std::isfinite(value)) throw std::invalid_argument("Invalid degraded speed");
        config.degradedSpeed=std::clamp(value,config.minSpeed,config.maxSpeed);
    }
    nlohmann::json runtimeConfig() const { return config.toJson(); }
    float qualitySpeedLimit() const {
        if(!valid || !ready || lost) return 0;
        if(qualityRestricted || degraded || confidence<0.65f) return config.degradedSpeed;
        // 正常的双边与单边混合路径共用巡航上限，不额外设置混合限速。
        return config.maxSpeed;
    }
    const char *qualityLimitReason() const {
        if(!valid || !ready || lost) return "PATH_WAIT";
        if(degraded) return singleFraction()>config.maxSingleFraction ? "SINGLE_SHARE" : "REJECT_SHARE";
        if(confidence<0.65f) return "LOW_QUALITY";
        if(qualityRestricted) return "QUALITY_RECOVERY";
        return "LIMIT";
    }
    void resetControl() {
        previousError = filteredDerivative = commandedSpeed = 0;
        speedTarget=0; speedReason="STOP";
        servo = PWMSERVOMID; hadPath = false;
    }
    void update(const std::vector<PointX> &left, const std::vector<PointX> &right,
                float dt, float previousSpeed = 0) {
        if (!std::isfinite(dt) || dt <= 0 || dt > config.maxFrameGap) {
            valid = ready = false; lost = true; centerEdge.clear();
            confidence = 0; stableTime = 0; degraded=false; pathIssue="FRAME_GAP";
            qualityGoodTime=0; rawSeverity=severity=curveHeading=0; curveCause="NONE";
            dualPoints=singlePoints=rejectedPoints=0;
            missingWidthPoints=untrustedPoints=widthRangePoints=jumpPoints=0;
            resetControl(); return;
        }
        now += dt;
        std::map<int, std::pair<int,int>, std::greater<int>> rows;
        for (const auto &p : left) {
            if (p.x < 0 || p.x >= int(IMAGE_ROW) || p.y < 0 || p.y >= int(IMAGE_COL)) continue;
            rows.try_emplace(p.x, -1, -1).first->second.first = p.y;
        }
        for (const auto &p : right) {
            if (p.x < 0 || p.x >= int(IMAGE_ROW) || p.y < 0 || p.y >= int(IMAGE_COL)) continue;
            rows.try_emplace(p.x, -1, -1).first->second.second = p.y;
        }
        std::vector<PointX> raw;
        std::vector<std::pair<int,float>> widthCandidates;
        dualPoints = singlePoints = rejectedPoints = 0;
        missingWidthPoints=untrustedPoints=widthRangePoints=jumpPoints=0;
        int previousRow = -1; float previousCenter = 0;
        for (const auto &[row, edges] : rows) {
            const auto [l, r] = edges;
            bool lOk = l >= 0 && trusted({row,l}), rOk = r >= 0 && trusted({row,r});
            float c = 0, w = r-l;
            bool dual = lOk && rOk;
            if (dual) {
                if (w < config.minWidth || w > config.maxWidth) { ++rejectedPoints; ++widthRangePoints; continue; }
                c = (l+r)/2.f;
            } else if (lOk != rOk) {
                w = widthAt(row);
                if (w == 0) { ++rejectedPoints; ++missingWidthPoints; continue; }
                c = lOk ? l+w/2.f : r-w/2.f;
            } else { ++rejectedPoints; ++untrustedPoints; continue; }
            if (c < 0 || c >= IMAGE_COL ||
                (previousRow >= 0 && std::fabs(c-previousCenter) >
                    10.f + 3.f * (previousRow-row))) { ++rejectedPoints; ++jumpPoints; continue; }
            raw.emplace_back(row, int(std::lround(c)));
            previousRow = row; previousCenter = c;
            if (dual) {
                ++dualPoints;
                widthCandidates.emplace_back(row,w);
            } else ++singlePoints;
        }
        if (raw.empty()) pathIssue=missingWidthPoints ? "WIDTH_UNAVAILABLE" : "NO_POINTS";
        else if(raw.size()<size_t(config.minPoints)) pathIssue="TOO_FEW_POINTS";
        else if(raw.front().x-raw.back().x<config.minSpan) pathIssue="SPAN_SHORT";
        else if(raw.front().x<int(IMAGE_ROW)*0.45f) pathIssue="NEAR_MISSING";
        else pathIssue="NONE";
        valid=std::string(pathIssue)=="NONE";
        // 路边离开视野时，少量图像行需要重建中心点属于正常情况。
        // 按单边重建点占有效中心点的比例判断降级，避免少量单边点就触发降级。
        degraded = singleFraction() > config.maxSingleFraction ||
                   rejectedPoints > int(rows.size()/3);
        centerEdge.clear();
        if (!valid) {
            missingTime += dt; stableTime = 0; ready = false; confidence = 0;
            qualityGoodTime=0; rawSeverity=severity=curveHeading=0; curveCause="NONE";
            lost = missingTime >= config.lostTimeout;
            // 首个无效帧立即将速度指令清零；短暂保持转向，持续丢线后回正。
            commandedSpeed=speedTarget=0; speedReason="PATH_WAIT";
            if (lost) resetControl();
            return;
        }
        missingTime = 0; lost = false; stableTime += dt;
        ready = stableTime >= config.recoverTime;
        confidence = float(dualPoints + 0.6f*singlePoints) /
                     std::max<size_t>(rows.size(), 1);
        if(degraded || confidence<0.65f) {
            qualityRestricted=true; qualityGoodTime=0;
        } else if(qualityRestricted) {
            qualityGoodTime+=dt;
            if(qualityGoodTime>=config.qualityRecoverTime) qualityRestricted=false;
        }
        for (size_t i=0; i<raw.size(); ++i) {
            const int a = raw[i ? i-1 : i].y, b = raw[i].y;
            const int c = raw[i+1 < raw.size() ? i+1 : i].y;
            centerEdge.emplace_back(raw[i].x,(a+b+c)/3);
        }
        float numerator = IMAGE_COL/2.f, denominator = 1;
        for (const auto &p : centerEdge) {
            float w = weightOf(p.x); numerator += w*p.y; denominator += w;
        }
        center = std::clamp(int(std::lround(numerator/denominator)),0,int(IMAGE_COL)-1);
        // 用远近中心线的横向位置差近似表示弯曲程度和道路朝向变化。
        const size_t n = std::max<size_t>(1, centerEdge.size()/4);
        float nearMean=0, farMean=0;
        for(size_t i=0;i<n;++i) {
            nearMean += centerEdge[i].y;
            farMean += centerEdge[centerEdge.size()-1-i].y;
        }
        heading = (farMean-nearMean)/n;
        // 速度预瞄可以改用较近的图像区域，而不改变转向所用的中心线
        // 或宽度学习条件所用的原始远近偏移量。
        const auto end=std::find_if(centerEdge.begin(),centerEdge.end(),[&](const PointX &p) {
            return p.x<config.curveLookaheadRow;
        });
        const size_t count=size_t(std::distance(centerEdge.begin(),end));
        curveWindowFallback=count<size_t(config.minPoints) ||
            centerEdge.front().x-centerEdge[count ? count-1 : 0].x<config.minSpan;
        curveHeading=heading;
        if(!curveWindowFallback) {
            const size_t section=std::max<size_t>(1,count/4);
            float near=0,far=0;
            for(size_t i=0;i<section;++i) {
                near+=centerEdge[i].y; far+=centerEdge[count-1-i].y;
            }
            curveHeading=(far-near)/section;
        }
        // 仅在车辆居中、道路近似直行且边界可信时学习各行的透视宽度。
        // 弯道和无效的局部片段不能覆盖参考宽度记录。
        if (dualPoints >= config.minPoints && std::fabs(heading)<15 && std::abs(center-160)<20) {
            for (const auto &[row,w]:widthCandidates) {
                auto &sample=widths[row];
                if (sample.observations==0) {
                    sample.value=w; sample.time=now; sample.observations=1;
                } else if (std::fabs(w-sample.value)<=std::max(20.f,sample.value*0.25f)) {
                    sample.value+=config.widthAlpha*(w-sample.value);
                    sample.time=now;
                    sample.observations=std::min(sample.observations+1,100);
                    sample.pendingObservations=0;
                } else {
                    // 需要多次一致的双边观测，才允许重新学习宽度。
                    // 重新验证期间，不使用该行旧宽度进行单边中心重建。
                    if(sample.pendingObservations==0 ||
                       std::fabs(w-sample.pendingValue)>std::max(10.f,sample.pendingValue*0.1f)) {
                        sample.pendingValue=w; sample.pendingObservations=1;
                    } else {
                        sample.pendingValue+=(w-sample.pendingValue)/++sample.pendingObservations;
                    }
                    if(sample.pendingObservations>=3) {
                        sample.value=sample.pendingValue; sample.time=now;
                        sample.observations=1; sample.pendingObservations=0;
                    }
                }
            }
        }
        error = float(center)-IMAGE_COL/2.f;
        error = std::clamp(error,previousError-config.maxErrorRate*dt,
                                previousError+config.maxErrorRate*dt);
        const float derivative = hadPath ? (error-previousError)*config.referenceDt/dt : 0;
        const float alpha = dt/(config.derivativeFilter+dt);
        filteredDerivative += alpha*(derivative-filteredDerivative);
        const float speedRatio = std::clamp(std::fabs(previousSpeed)/config.maxSpeed,0.f,1.f);
        const float gain = 1 + (config.highSpeedGain-1)*speedRatio;
        const float p = config.runP1+std::fabs(error)*config.runP2;
        servo = clampServo(PWMSERVOMID+gain*(error*p+filteredDerivative*config.turnD));
        const float e=std::fabs(error)/65.f, h=std::fabs(curveHeading)/65.f;
        const float s=std::fabs(float(servo)-PWMSERVOMID)/320.f;
        rawSeverity=std::clamp(std::max({e,h,s}),0.f,1.f);
        severity=std::clamp((rawSeverity-config.curveStart)/(1-config.curveStart),0.f,1.f);
        curveCause=severity<=0 ? "NONE" : h>=e && h>=s ? "CURVE_HEADING" :
                   s>=e ? "CURVE_STEERING" : "CURVE_ERROR";
        previousError = error; hadPath = true;
    }
    float speedFor(float cruise, float dt) {
        if (!std::isfinite(cruise) || !std::isfinite(dt) || dt <= 0 ||
            dt > config.maxFrameGap || !valid || !ready || lost) {
            commandedSpeed=speedTarget=0; speedReason="PATH_WAIT"; return 0;
        }
        const float limit=std::clamp(cruise,config.minSpeed,config.maxSpeed);
        const float curve=std::min(limit,limit+(config.curveSpeed-limit)*severity);
        const float quality=qualitySpeedLimit();
        speedTarget=std::min(curve,quality);
        speedReason=quality<limit && quality<=curve+1e-5f ? qualityLimitReason() :
                    curve<limit ? curveCause : "LIMIT";
        // 用户降低巡航上限时立即生效；上面的丢线停车同样
        // 跳过缓降过程。只有路径有效时的减速才按设定减速度逐步执行。
        commandedSpeed=std::min(commandedSpeed,limit);
        const bool qualityBraking=quality<limit && quality<=curve+1e-5f;
        const float braking=qualityBraking ? std::max(config.deceleration,config.qualityDeceleration) : config.deceleration;
        if(speedTarget<commandedSpeed)
            commandedSpeed=std::max(speedTarget,commandedSpeed-braking*dt);
        else commandedSpeed=std::min(speedTarget,commandedSpeed+config.acceleration*dt);
        return commandedSpeed;
    }
};
