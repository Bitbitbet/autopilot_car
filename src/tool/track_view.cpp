#include "camera.hpp"
#include "control.hpp"
#include "joystick.hpp"
#include "tools.hpp"
#include "track.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <deque>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using cv::Mat;
using std::atomic;
using std::atomic_bool;
using std::cerr;
using std::cout;
using std::deque;
using std::jthread;
using std::make_unique;
using std::mutex;
using std::ref;
using std::shared_ptr;
using std::string;
using std::unique_ptr;
using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::milliseconds;

using Clock = std::chrono::steady_clock;

// 像素量是 640x480 口径；识别门限是地面坐标（米）。
namespace cfg {
// 速度 (m/s)
constexpr float kCruiseSpeed = 0.25f; // 默认巡航
constexpr float kMaxSpeed = 1.50f;    // 上限
constexpr float kMinSpeed = 0.05f;
constexpr float kCurveSpeed = 0.15f;    // 弯道限速
constexpr float kDegradedSpeed = 0.12f; // 单边/低置信度限速
constexpr float kAcceleration = 0.30f;
constexpr float kDeceleration = 0.60f;
constexpr float kQualityDeceleration = 1.20f;
constexpr float kReferenceDt = 1.f / 30.f; // 无帧率信息时的 dt
constexpr float kMaxFrameGap = 0.25f;      // 超过按丢帧处理
constexpr float kLostTimeout = 0.20f;      // 丢线多久判定 LOST
constexpr float kRecoverTime = 0.25f;      // 稳定多久才允许进 AUTO
constexpr float kQualityRecoverTime = 0.30f;
// 转向：servo = mid + gain*(err*kRunP1 + err^2*kRunP2 + d*kTurnD)，err
// 是原生像素
constexpr float kRunP1 = 1.10f;
constexpr float kRunP2 = 0.00175f;
constexpr float kTurnD = 1.75f;
constexpr float kDerivativeFilter = 0.08f;
constexpr float kMaxErrorRate = 1920.f; // 偏差变化率上限 (px/s)
constexpr float kHighSpeedGain = 1.f;
constexpr float kCurveStart = 0.20f;        // severity 超过它开始减速
constexpr float kMaxSingleFraction = 0.35f; // 单边点占比上限
// 路径判定（原生像素）
constexpr int kBorderMargin = 6; // 图像左右边界余量
constexpr int kMinPoints = 16;   // 至少多少行才算可信路径
constexpr int kMinSpan = 60;     // 中心线最小纵向跨度
constexpr int kMinWidth = 60;    // 双边路宽的合理范围
constexpr int kMaxWidth = 600;
constexpr int kCurveLookaheadRow = 20; // 速度预判用的近处行
constexpr float kWidthAlpha = 0.1f;    // 宽度 profile 学习率
constexpr int kWidthWindow = 24;       // 单边补线时宽度插值的行窗口(±)
constexpr float kJumpLimitBase = 20.f; // 相邻行中心跳变上限
constexpr float kJumpLimitPerRow = 6.f;
constexpr float kStraightHeading = 30.f; // 学宽度要求的接近直道
constexpr float kCenterGate = 40.f;      // 学宽度要求的居中范围
constexpr float kWidthLearnStep = 40.f;  // profile 允许的偏差下限
constexpr float kWidthPendingStep = 20.f;
constexpr float kSeverityScale = 130.f; // severity 归一化尺度
// 其它
constexpr int kCommandTimeoutMs = 250;
} // namespace cfg
// Main thread alone owns HighGUI. Controller never waits for network rendering.
class TrackviewDisplay {
    mutex imageMutex, keyMutex;
    Mat latestImage, latestBinary;
    deque<int> keys;
    atomic_bool emergency{false};
    atomic<long long> progressMs{0};
    static long long clockMs() {
        return duration_cast<milliseconds>(
                   std::chrono::steady_clock::now().time_since_epoch())
            .count();
    }

  public:
    std::atomic_bool finished{false}, failed{false};
    TrackviewDisplay() { progress(); }
    void progress() { progressMs.store(clockMs()); }
    long long ageMs() const { return clockMs() - progressMs.load(); }
    bool stalled(int timeoutMs = 2000) const { return ageMs() >= timeoutMs; }
    void publish(const cv::Mat &image, const cv::Mat &binary) {
        // Clone outside the lock; no growing queue and no rendering under this
        // lock.
        cv::Mat a = image.clone(), b = binary.clone();
        std::unique_lock lock(imageMutex, std::try_to_lock);
        if (!lock.owns_lock())
            return;
        latestImage = std::move(a);
        latestBinary = std::move(b);
    }
    bool snapshot(cv::Mat &image, cv::Mat &binary) {
        std::lock_guard lock(imageMutex);
        if (latestImage.empty())
            return false;
        // Immutable reference-counted snapshots remain valid across
        // publications.
        image = latestImage;
        binary = latestBinary;
        return true;
    }
    void post(int key) {
        if (key < 0)
            return;
        if (key == ' ') {
            emergency = true;
            return;
        }
        std::lock_guard lock(keyMutex);
        if (keys.size() >= 32) {
            keys.clear();
            emergency = true;
        } else
            keys.push_back(key);
    }
    std::vector<int> takeKeys() {
        std::lock_guard lock(keyMutex);
        std::vector<int> result(keys.begin(), keys.end());
        keys.clear();
        if (emergency.exchange(false)) {
            result.clear();
            result.push_back(' ');
        }
        return result;
    }
};

enum class WeightMode { FarFull, NearDominant, Uniform };
inline uint16_t clampServo(float pwm) {
    return static_cast<uint16_t>(
        std::lround(std::clamp(pwm, float(PWMSERVOMIN), float(PWMSERVOMAX))));
}

// Hardware-independent controller, used by live driving and replay tests.
class AutoPilot {
    struct Width {
        float value = 0, pendingValue = 0;
        double time = -1;
        int observations = 0, pendingObservations = 0;
    };
    // 运行时可调的限速（键盘 [ ] , . 修改），初值来自 cfg
    float curveLimit = cfg::kCurveSpeed;
    float qualityLimit = cfg::kDegradedSpeed;
    int imageCols = IMAGE_COL, imageRows = IMAGE_ROW; // 实际帧尺寸
    std::vector<Width> widths;                        // 逐行宽度，原生像素
    float filteredDerivative = 0, previousError = 0, commandedSpeed = 0;
    double now = 0, missingTime = 0, stableTime = 0;
    double qualityGoodTime = 0;
    bool qualityRestricted = false;
    bool hadPath = false;

    bool trusted(const Point2i &p) const {
        return p.x >= 0 && p.x < imageRows && p.y > cfg::kBorderMargin &&
               p.y < imageCols - 1 - cfg::kBorderMargin;
    }
    float widthAt(int row) const {
        // Interpolate local observations only: no constant-width extrapolation.
        int lower = -1, upper = -1;
        for (int r = max(0, row - cfg::kWidthWindow);
             r <= min(imageRows - 1, row + cfg::kWidthWindow); ++r) {
            const auto &w = widths[r];
            // Stationary time does not change camera perspective or track
            // width. Contradictory trusted observations suspend a row until
            // revalidated.
            if (w.observations < 3 || w.pendingObservations > 0)
                continue;
            if (r <= row)
                lower = r;
            if (r >= row && upper < 0)
                upper = r;
        }
        if (lower < 0 && upper < 0)
            return 0;
        if (lower < 0)
            return widths[upper].value;
        if (upper < 0 || upper == lower)
            return widths[lower].value;
        const float t = float(row - lower) / (upper - lower);
        return widths[lower].value * (1 - t) + widths[upper].value * t;
    }
    float weightOf(int row) const {
        if (weightMode == WeightMode::Uniform)
            return 1;
        if (weightMode == WeightMode::NearDominant)
            return std::min(float(row), imageRows / 2.f);
        return row < imageRows / 2 ? imageRows / 2.f : float(imageRows - row);
    }

  public:
    int learnedWidthRows() const {
        return int(
            std::count_if(widths.begin(), widths.end(), [](const Width &w) {
                return w.observations >= 3 && w.pendingObservations == 0;
            }));
    }
    const char *pathIssue = "NO_POINTS";
    int missingWidthPoints = 0;
    std::vector<Point2i> centerEdge;
    int center = imageCols / 2, dualPoints = 0, singlePoints = 0,
        rejectedPoints = 0;
    float error = 0, heading = 0, confidence = 0, severity = 0;
    float rawSeverity = 0, curveHeading = 0, speedTarget = 0;
    bool curveWindowFallback = false;
    const char *curveCause = "NONE", *speedReason = "STOP";
    uint16_t servo = PWMSERVOMID;
    bool valid = false, ready = false, lost = false, degraded = false;
    WeightMode weightMode = WeightMode::FarFull;

    explicit AutoPilot(int frameCols = IMAGE_COL, int frameRows = IMAGE_ROW)
        : imageCols(frameCols), imageRows(frameRows) {
        widths.assign(imageRows, Width{});
    }
    void cycleWeight() { weightMode = WeightMode((int(weightMode) + 1) % 3); }
    const char *weightName() const {
        if (weightMode == WeightMode::FarFull)
            return "far-full";
        if (weightMode == WeightMode::NearDominant)
            return "near-dominant";
        return "uniform";
    }
    float singleFraction() const {
        return float(singlePoints) / std::max(dualPoints + singlePoints, 1);
    }
    float curveSpeed() const { return curveLimit; }
    float degradedSpeed() const { return qualityLimit; }
    void setCurveSpeed(float value) {
        if (!std::isfinite(value))
            throw std::invalid_argument("Invalid curve speed");
        curveLimit = std::clamp(value, cfg::kMinSpeed, cfg::kMaxSpeed);
    }
    void setDegradedSpeed(float value) {
        if (!std::isfinite(value))
            throw std::invalid_argument("Invalid degraded speed");
        qualityLimit = std::clamp(value, cfg::kMinSpeed, cfg::kMaxSpeed);
    }
    float qualitySpeedLimit() const {
        if (!valid || !ready || lost)
            return 0;
        if (qualityRestricted || degraded || confidence < 0.65f)
            return qualityLimit;
        // Healthy mixed paths share the cruise limit; no separate mixed-speed
        // cap.
        return cfg::kMaxSpeed;
    }
    const char *qualityLimitReason() const {
        if (!valid || !ready || lost)
            return "PATH_WAIT";
        if (degraded)
            return singleFraction() > cfg::kMaxSingleFraction ? "SINGLE_SHARE"
                                                              : "REJECT_SHARE";
        if (confidence < 0.65f)
            return "LOW_QUALITY";
        if (qualityRestricted)
            return "QUALITY_RECOVERY";
        return "LIMIT";
    }
    void resetControl() {
        previousError = filteredDerivative = commandedSpeed = 0;
        speedTarget = 0;
        speedReason = "STOP";
        servo = PWMSERVOMID;
        hadPath = false;
    }
    void update(const std::vector<Point2i> &left,
                const std::vector<Point2i> &right, float dt,
                float previousSpeed = 0) {
        if (!std::isfinite(dt) || dt <= 0 || dt > cfg::kMaxFrameGap) {
            valid = ready = false;
            lost = true;
            centerEdge.clear();
            confidence = 0;
            stableTime = 0;
            degraded = false;
            pathIssue = "FRAME_GAP";
            qualityGoodTime = 0;
            rawSeverity = severity = curveHeading = 0;
            curveCause = "NONE";
            dualPoints = singlePoints = rejectedPoints = 0;
            missingWidthPoints = 0;
            resetControl();
            return;
        }
        now += dt;
        std::map<int, std::pair<int, int>, std::greater<int>> rows;
        for (const auto &p : left) {
            if (p.x < 0 || p.x >= imageRows || p.y < 0 || p.y >= imageCols)
                continue;
            rows.try_emplace(p.x, -1, -1).first->second.first = p.y;
        }
        for (const auto &p : right) {
            if (p.x < 0 || p.x >= imageRows || p.y < 0 || p.y >= imageCols)
                continue;
            rows.try_emplace(p.x, -1, -1).first->second.second = p.y;
        }
        std::vector<Point2i> raw;
        std::vector<std::pair<int, float>> widthCandidates;
        dualPoints = singlePoints = rejectedPoints = 0;
        missingWidthPoints = 0;
        int previousRow = -1;
        float previousCenter = 0;
        for (const auto &[row, edges] : rows) {
            const auto [l, r] = edges;
            bool lOk = l >= 0 && trusted({row, l}),
                 rOk = r >= 0 && trusted({row, r});
            float c = 0, w = r - l;
            bool dual = lOk && rOk;
            if (dual) {
                if (w < cfg::kMinWidth || w > cfg::kMaxWidth) {
                    ++rejectedPoints;
                    continue;
                }
                c = (l + r) / 2.f;
            } else if (lOk != rOk) {
                w = widthAt(row);
                if (w == 0) {
                    ++rejectedPoints;
                    ++missingWidthPoints;
                    continue;
                }
                c = lOk ? l + w / 2.f : r - w / 2.f;
            } else {
                ++rejectedPoints;
                continue;
            }
            if (c < 0 || c >= imageCols ||
                (previousRow >= 0 &&
                 std::fabs(c - previousCenter) >
                     cfg::kJumpLimitBase +
                         cfg::kJumpLimitPerRow * (previousRow - row))) {
                ++rejectedPoints;
                continue;
            }
            raw.emplace_back(row, int(std::lround(c)));
            previousRow = row;
            previousCenter = c;
            if (dual) {
                ++dualPoints;
                widthCandidates.emplace_back(row, w);
            } else
                ++singlePoints;
        }
        if (raw.empty())
            pathIssue = missingWidthPoints ? "WIDTH_UNAVAILABLE" : "NO_POINTS";
        else if (raw.size() < size_t(cfg::kMinPoints))
            pathIssue = "TOO_FEW_POINTS";
        else if (raw.front().x - raw.back().x < cfg::kMinSpan)
            pathIssue = "SPAN_SHORT";
        else if (raw.front().x < imageRows * 0.45f)
            pathIssue = "NEAR_MISSING";
        else
            pathIssue = "NONE";
        valid = string(pathIssue) == "NONE";
        // A few reconstructed rows are normal when a road edge leaves the
        // image. Degrade by their share of accepted centers, not their mere
        // presence.
        degraded = singleFraction() > cfg::kMaxSingleFraction ||
                   rejectedPoints > int(rows.size() / 3);
        centerEdge.clear();
        if (!valid) {
            missingTime += dt;
            stableTime = 0;
            ready = false;
            confidence = 0;
            qualityGoodTime = 0;
            rawSeverity = severity = curveHeading = 0;
            curveCause = "NONE";
            lost = missingTime >= cfg::kLostTimeout;
            // Stop speed on the first bad frame; hold steering briefly, then
            // center.
            commandedSpeed = speedTarget = 0;
            speedReason = "PATH_WAIT";
            if (lost)
                resetControl();
            return;
        }
        missingTime = 0;
        lost = false;
        stableTime += dt;
        ready = stableTime >= cfg::kRecoverTime;
        confidence = float(dualPoints + 0.6f * singlePoints) /
                     std::max<size_t>(rows.size(), 1);
        if (degraded || confidence < 0.65f) {
            qualityRestricted = true;
            qualityGoodTime = 0;
        } else if (qualityRestricted) {
            qualityGoodTime += dt;
            if (qualityGoodTime >= cfg::kQualityRecoverTime)
                qualityRestricted = false;
        }
        for (size_t i = 0; i < raw.size(); ++i) {
            const int a = raw[i ? i - 1 : i].y, b = raw[i].y;
            const int c = raw[i + 1 < raw.size() ? i + 1 : i].y;
            centerEdge.emplace_back(raw[i].x, (a + b + c) / 3);
        }
        float numerator = imageCols / 2.f, denominator = 1;
        for (const auto &p : centerEdge) {
            float w = weightOf(p.x);
            numerator += w * p.y;
            denominator += w;
        }
        center = std::clamp(int(std::lround(numerator / denominator)), 0,
                            imageCols - 1);
        // Difference between far and near segments is a curve/heading proxy.
        const size_t n = std::max<size_t>(1, centerEdge.size() / 4);
        float nearMean = 0, farMean = 0;
        for (size_t i = 0; i < n; ++i) {
            nearMean += centerEdge[i].y;
            farMean += centerEdge[centerEdge.size() - 1 - i].y;
        }
        heading = (farMean - nearMean) / n;
        // Speed preview may use a nearer image region without changing steering
        // or the original heading used to qualify width-learning observations.
        const auto end = std::find_if(
            centerEdge.begin(), centerEdge.end(),
            [&](const Point2i &p) { return p.x < cfg::kCurveLookaheadRow; });
        const size_t count = size_t(std::distance(centerEdge.begin(), end));
        curveWindowFallback =
            count < size_t(cfg::kMinPoints) ||
            centerEdge.front().x - centerEdge[count ? count - 1 : 0].x <
                cfg::kMinSpan;
        curveHeading = heading;
        if (!curveWindowFallback) {
            const size_t section = std::max<size_t>(1, count / 4);
            float near = 0, far = 0;
            for (size_t i = 0; i < section; ++i) {
                near += centerEdge[i].y;
                far += centerEdge[count - 1 - i].y;
            }
            curveHeading = (far - near) / section;
        }
        // Learn perspective width only from a centered, nearly straight trusted
        // path. Curves and unusable fragments must not overwrite the reference
        // profile.
        if (dualPoints >= cfg::kMinPoints &&
            fabs(heading) < cfg::kStraightHeading &&
            std::abs(center - imageCols / 2) < cfg::kCenterGate) {
            for (const auto &[row, w] : widthCandidates) {
                auto &sample = widths[row];
                if (sample.observations == 0) {
                    sample.value = w;
                    sample.time = now;
                    sample.observations = 1;
                } else if (fabs(w - sample.value) <=
                           max(cfg::kWidthLearnStep, sample.value * 0.25f)) {
                    sample.value += cfg::kWidthAlpha * (w - sample.value);
                    sample.time = now;
                    sample.observations =
                        std::min(sample.observations + 1, 100);
                    sample.pendingObservations = 0;
                } else {
                    // Require repeated, consistent double-edge measurements to
                    // relearn. Do not use the old row for single-edge
                    // reconstruction meanwhile.
                    if (sample.pendingObservations == 0 ||
                        std::fabs(w - sample.pendingValue) >
                            max(cfg::kWidthPendingStep,
                                sample.pendingValue * 0.1f)) {
                        sample.pendingValue = w;
                        sample.pendingObservations = 1;
                    } else {
                        sample.pendingValue += (w - sample.pendingValue) /
                                               ++sample.pendingObservations;
                    }
                    if (sample.pendingObservations >= 3) {
                        sample.value = sample.pendingValue;
                        sample.time = now;
                        sample.observations = 1;
                        sample.pendingObservations = 0;
                    }
                }
            }
        }
        error = float(center) - imageCols / 2.f;
        error = std::clamp(error, previousError - cfg::kMaxErrorRate * dt,
                           previousError + cfg::kMaxErrorRate * dt);
        const float derivative =
            hadPath ? (error - previousError) * cfg::kReferenceDt / dt : 0;
        const float alpha = dt / (cfg::kDerivativeFilter + dt);
        filteredDerivative += alpha * (derivative - filteredDerivative);
        const float speedRatio =
            std::clamp(std::fabs(previousSpeed) / cfg::kMaxSpeed, 0.f, 1.f);
        const float gain = 1 + (cfg::kHighSpeedGain - 1) * speedRatio;
        const float p = cfg::kRunP1 + std::fabs(error) * cfg::kRunP2;
        servo =
            clampServo(PWMSERVOMID +
                       gain * (error * p + filteredDerivative * cfg::kTurnD));
        const float e = std::fabs(error) / cfg::kSeverityScale,
                    h = std::fabs(curveHeading) / cfg::kSeverityScale;
        const float s = std::fabs(float(servo) - PWMSERVOMID) / 320.f;
        rawSeverity = std::clamp(std::max({e, h, s}), 0.f, 1.f);
        severity = std::clamp((rawSeverity - cfg::kCurveStart) /
                                  (1 - cfg::kCurveStart),
                              0.f, 1.f);
        curveCause = severity <= 0      ? "NONE"
                     : h >= e && h >= s ? "CURVE_HEADING"
                     : s >= e           ? "CURVE_STEERING"
                                        : "CURVE_ERROR";
        previousError = error;
        hadPath = true;
    }
    float speedFor(float cruise, float dt) {
        if (!std::isfinite(cruise) || !std::isfinite(dt) || dt <= 0 ||
            dt > cfg::kMaxFrameGap || !valid || !ready || lost) {
            commandedSpeed = speedTarget = 0;
            speedReason = "PATH_WAIT";
            return 0;
        }
        const float limit = std::clamp(cruise, cfg::kMinSpeed, cfg::kMaxSpeed);
        const float curve =
            std::min(limit, limit + (curveLimit - limit) * severity);
        const float quality = qualitySpeedLimit();
        speedTarget = std::min(curve, quality);
        speedReason = quality < limit && quality <= curve + 1e-5f
                          ? qualityLimitReason()
                      : curve < limit ? curveCause
                                      : "LIMIT";
        // The operator's cruise ceiling takes effect immediately. Path loss
        // also bypasses this ramp above. Valid-path slowdowns have bounded
        // deceleration.
        commandedSpeed = std::min(commandedSpeed, limit);
        const bool qualityBraking = quality < limit && quality <= curve + 1e-5f;
        const float braking =
            qualityBraking
                ? std::max(cfg::kDeceleration, cfg::kQualityDeceleration)
                : cfg::kDeceleration;
        if (speedTarget < commandedSpeed)
            commandedSpeed =
                std::max(speedTarget, commandedSpeed - braking * dt);
        else
            commandedSpeed =
                std::min(speedTarget, commandedSpeed + cfg::kAcceleration * dt);
        return commandedSpeed;
    }
};

namespace {
std::atomic_bool stopping{false};
static_assert(std::atomic_bool::is_always_lock_free);
void onSignal(int) { stopping.store(true); }
} // namespace
enum class Mode { Manual, Auto };

// Declared after the driving thread: inhibit motion before unwinding/joining
// it.
struct StopOnExit {
    std::shared_ptr<CarControl> car;
    ~StopOnExit() {
        stopping = true;
        if (car) {
            try {
                car->latch_stop();
            } catch (...) {
            }
        }
    }
};

static void drawHud(Mat &image, const AutoPilot &pilot, Mode mode, bool panic,
                    float cruise, float speed, uint16_t servo, double fps,
                    bool displaySlow) {
    for (const auto &p : pilot.centerEdge)
        cv::circle(image, cv::Point(p.y, p.x), 1, cv::Scalar(0, 0, 255), -1);
    int y = 18;
    auto line = [&](const string &s) {
        cv::putText(image, s, {5, y}, cv::FONT_HERSHEY_PLAIN, 0.9,
                    {0, 255, 255}, 1);
        y += 18;
    };
    auto fmt = [](float v) {
        std::ostringstream s;
        s << std::fixed << std::setprecision(2) << v;
        return s.str();
    };
    line(string(mode == Mode::Auto ? "AUTO" : "MANUAL") +
         (panic ? " STOP LATCHED" : "") + (pilot.lost ? " LOST" : ""));
    line("speed=" + fmt(speed) + " limit=" + fmt(cruise) +
         " pwm=" + std::to_string(servo));
    line("turn=" + fmt(pilot.curveSpeed()) +
         " qualityLimit=" + fmt(pilot.degradedSpeed()));
    line(string(pilot.valid ? "center=" : "last center=") +
         std::to_string(pilot.center) + " err=" + fmt(pilot.error));
    line("dual=" + std::to_string(pilot.dualPoints) +
         " single=" + std::to_string(pilot.singlePoints) +
         " reject=" + std::to_string(pilot.rejectedPoints));
    line("quality=" + fmt(pilot.confidence) + " curve=" + fmt(pilot.severity) +
         " h=" + fmt(pilot.curveHeading));
    line(string(pilot.valid
                    ? (pilot.degraded
                           ? "DEGRADED"
                           : (pilot.singlePoints > 0 ? "MIXED" : "DUAL"))
                    : "NO TRUSTED PATH") +
         (pilot.ready ? " READY" : " WAIT"));
    line("target=" + fmt(pilot.speedTarget) + " " +
         (mode == Mode::Auto ? pilot.speedReason : "MANUAL"));
    if (!pilot.valid)
        line(string(pilot.pathIssue) +
             " width=" + std::to_string(pilot.learnedWidthRows()) +
             " miss=" + std::to_string(pilot.missingWidthPoints));
    if (displaySlow)
        line("DISPLAY DELAY: control independent");
    line("fps=" + std::to_string(int(fps)) + " weight=" + pilot.weightName());
    line("a:auto  SPACE:stop  +/-:limit  ESC:quit");
    line("[/]:turn  ,/.:quality  w:weight");
}

void driving_thread(Camera &camera, CarControl *car, AutoPilot &pilot,
                    JoyStick *js, TrackviewDisplay &display,
                    atomic_bool &joyLost) {
    float cruise = cfg::kCruiseSpeed, jsSpeed = 0, speed = 0;
    auto start = Clock::now(), last = start;
    uint16_t jsServo = PWMSERVOMID;
    auto mode = Mode::Manual;
    bool panic = false;

    string stage = "startup";
    try {
        while (!stopping.load()) {
            stage = "before_capture";
            if (car && car->hasFault())
                throw std::runtime_error(
                    "Serial fault or command watchdog timeout");
            cv::Mat image;
            stage = "capture";
            if (!camera.read(image) || image.empty()) {
                if (car)
                    car->resetVelocity();
                throw std::runtime_error("Camera frame unavailable");
            }
            stage = "preprocess";
            cv::Mat bin = camera.binarize(image);
            stage = "track";
            std::vector<Point2i> left_edges, right_edges;
            track_identify_from_bin(bin, camera, left_edges, right_edges);
            const auto now = Clock::now();
            stage = "control";
            auto dt = duration<double>(now - last).count();
            last = now;
            if (dt <= 0)
                dt = cfg::kReferenceDt;
            pilot.update(left_edges, right_edges, dt, speed);
            track_draw_result(left_edges, right_edges, image);
            const uint64_t motionEpoch = car ? car->motionEpoch() : 0;
            auto toggle = [&]() {
                if (mode == Mode::Auto) {
                    mode = Mode::Manual;
                    jsSpeed = 0;
                    pilot.resetControl();
                    cout << "[Mode] MANUAL\n";
                } else if (pilot.ready && !joyLost &&
                           (!js || !js->isStopped()) &&
                           std::fabs(jsSpeed) <= 0.05f) {
                    if (car && !car->resumeMotion(motionEpoch))
                        return;
                    mode = Mode::Auto;
                    panic = false;
                    jsSpeed = 0;
                    pilot.resetControl();
                    cout << "[Mode] AUTO\n";
                } else {
                    if (!pilot.ready)
                        cout << "[Wait] Lane " << pilot.pathIssue
                             << "; width rows=" << pilot.learnedWidthRows()
                             << " missing width=" << pilot.missingWidthPoints
                             << "; wait for stable trusted path.\n";
                    else
                        cout << "[Wait] Release gamepad throttle "
                                "or reconnect gamepad.\n";
                }
            };
            string stopReason;
            for (const int key : display.takeKeys()) {
                if (key == ' ') {
                    panic = true;
                    mode = Mode::Manual;
                    jsSpeed = 0;
                    pilot.resetControl();
                    if (car)
                        car->pauseMotion();
                    stopReason = "keyboard stop";
                } else if (key == 'a' || key == 'A')
                    toggle();
                else if (key == '+' || key == '=') {
                    cruise = std::min(cfg::kMaxSpeed, cruise + 0.05f);
                    cout << "[Set] limit=" << cruise << " m/s\n";
                } else if (key == '-' || key == '_') {
                    cruise = std::max(cfg::kMinSpeed, cruise - 0.05f);
                    cout << "[Set] limit=" << cruise << " m/s\n";
                } else if (key == '[' || key == ']') {
                    pilot.setCurveSpeed(pilot.curveSpeed() +
                                        (key == ']' ? 0.05f : -0.05f));
                    cout << "[Set] curveSpeed=" << pilot.curveSpeed()
                         << " m/s\n";
                } else if (key == ',' || key == '.') {
                    pilot.setDegradedSpeed(pilot.degradedSpeed() +
                                           (key == '.' ? 0.05f : -0.05f));
                    cout << "[Set] degradedSpeed=" << pilot.degradedSpeed()
                         << " m/s\n";
                } else if (key == 'w' || key == 'W') {
                    pilot.cycleWeight();
                    cout << "[Set] weight=" << pilot.weightName() << '\n';
                }
            }
            // X11 rendering is telemetry, not the control-loop
            // heartbeat. Keep command watchdog/vision loss independent
            // of remote UI delay.
            const auto displayAge = display.ageMs();
            if (js) {
                float s = 0;
                uint16_t sv = PWMSERVOMID;
                const bool newCommand = js->takeCarControl(s, sv);
                if (newCommand) {
                    jsSpeed = s;
                    jsServo = sv;
                }
                if (js->takeSampleOnce())
                    toggle();
                js->takeBuzzer();
                if (js->isStopped() || joyLost) {
                    if (!panic)
                        stopReason = "gamepad disconnected; stop latched";
                    panic = true;
                    mode = Mode::Manual;
                    jsSpeed = 0;
                    pilot.resetControl();
                } else if (mode == Mode::Auto && newCommand &&
                           std::fabs(jsSpeed) > 0.05f) {
                    mode = Mode::Manual;
                    pilot.resetControl();
                    cout << "[Mode] MANUAL (gamepad takeover)\n";
                }
            }
            if (mode == Mode::Auto && pilot.lost) {
                mode = Mode::Manual;
                panic = true;
                jsSpeed = 0;
                pilot.resetControl();
                stopReason = string("trusted lane lost; stop latched, "
                                    "press a after recovery; path=") +
                             pilot.pathIssue;
            }
            speed = 0;
            uint16_t servo = PWMSERVOMID;
            if (!panic && !stopping) {
                if (mode == Mode::Auto) {
                    speed = pilot.speedFor(cruise, dt);
                    servo = pilot.servo;
                } else {
                    speed =
                        std::clamp(jsSpeed, -cfg::kMaxSpeed, cfg::kMaxSpeed);
                    servo = jsServo;
                }
            }
            if (car) {
                if (car->hasFault())
                    throw std::runtime_error(
                        "Serial command timeout during processing");
                car->carControl(speed, servo);
            }
            stage = "display";
            drawHud(image, pilot, mode, panic, cruise, speed, servo, 1 / dt,
                    displayAge >= 2000);
            if (!stopReason.empty())
                cerr << "[Stop] " << stopReason << '\n';
            display.publish(image, bin);
        }
        if (car)
            car->resetVelocity();
        cout << "[Stop] "
             << (stopping.load() ? "signal or UI stop" : "normal exit") << '\n';
    } catch (const std::exception &e) {
        if (car) {
            try {
                car->latch_stop();
            } catch (...) {
            }
        }
        if (stopping.load() && !display.failed.load()) {
            cout << "[Stop] operator requested exit\n";
        } else {
            cerr << "[Error] " << e.what() << " (stage=" << stage << ")\n";
            display.failed = true;
        }
    } catch (...) {
        if (car) {
            try {
                car->latch_stop();
            } catch (...) {
            }
        }
        cerr << "[Error] unexpected driving thread failure (stage=" << stage
             << ")\n";
        display.failed = true;
    }
    display.finished = true;
}
int main() {
    signal(SIGINT, onSignal);
    signal(SIGTERM, onSignal);
    signal(SIGHUP, onSignal);
    auto car = CarControl::create();
    if (!car) {
        cerr << "无法加载串口控制。" << endl;
        return 1;
    }
    AutoPilot pilot;
    try {
        string source = "/dev/video0";
        // Load calibration before opening any vehicle control device.
        Camera camera;

        pilot = AutoPilot(camera.cols(), camera.rows());

        atomic_bool joyLost{false}; // Must outlive gamepad callback.
        car->setCommandTimeout(cfg::kCommandTimeoutMs);
        auto js = JoyStick::create([&joyLost] { joyLost = true; });
        if (!js) {
            cerr << "[Warn] No gamepad; use image window keyboard.\n";
        }

        TrackviewDisplay display;
        cout << "trackview | source=" << source << " | frame=" << camera.cols()
             << "x" << camera.rows() << " | limit=" << cfg::kCruiseSpeed
             << " m/s (max " << cfg::kMaxSpeed << ") turn=" << pilot.curveSpeed()
             << " quality=" << pilot.degradedSpeed() << "\n";
        jthread driving(driving_thread, ref(camera), car.get(), ref(pilot),
                        js.get(), ref(display), ref(joyLost));

        StopOnExit stopBeforeJoin{car};
        auto lastRender = Clock::now() - std::chrono::milliseconds(100);
        while (!stopping.load() && !display.finished.load()) {
            // GUI stays on the main thread; only immutable latest snapshots
            // cross threads.
            if (car && car->hasFault()) {
                display.failed = true;
                break;
            }
            const auto rendered = Clock::now();
            cv::Mat image, bin;
            if (rendered - lastRender >= std::chrono::milliseconds(100)) {
                if (!display.snapshot(image, bin)) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
                    display.progress();
                    continue;
                }
                cv::imshow("track_view", image);
                cv::imshow("binarized", bin);
                lastRender = Clock::now();
            }
            const int key = cv::waitKey(10);
            display.progress();
            if (key == 27)
                break;
            display.post(key);
            if (key == ' ' && car)
                car->pauseMotion(); // Veto racing worker writes immediately.
        }
        stopping = true;
        if (car) {
            try {
                car->latch_stop();
            } catch (...) {
            }
        }
        driving.join();
        cout << "[Stop] trackview finished.\n";
        return display.failed.load() ? 1 : 0;
    } catch (const std::exception &e) {
        if (car) {
            try {
                car->resetVelocity();
            } catch (...) {
            }
        }
        cerr << "[Error] " << e.what() << '\n';
        return 1;
    } catch (...) {
        if (car) {
            try {
                car->resetVelocity();
            } catch (...) {
            }
        }
        cerr << "[Error] unexpected exception\n";
        return 1;
    }
}
