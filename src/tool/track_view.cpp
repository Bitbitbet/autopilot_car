/** Track following with validated boundaries, adaptive speed and latched stops.
 */
#include "camera.hpp"
#include "control.hpp"
#include "joystick.hpp"
#include "tools.hpp"
#include "track.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <csignal>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using cv::Mat;
using std::cerr;
using std::cout;
using std::jthread;
using std::make_unique;
using std::shared_ptr;
using std::string;
using std::unique_ptr;
using std::chrono::duration;

using Clock = std::chrono::steady_clock;

struct TrackviewConfig {
    float cruiseSpeed = 0.25f, maxSpeed = 1.50f, minSpeed = 0.05f;
    float curveSpeed = 0.15f, degradedSpeed = 0.12f, acceleration = 0.30f;
    float deceleration = 0.60f, qualityDeceleration = 1.20f;
    float curveStart = 0.20f, qualityRecoverTime = 0.30f;
    int curveLookaheadRow = 10;
    float maxSingleFraction = 0.35f;
    float runP1 = 2.2f, runP2 = 0.007f, turnD = 3.5f;
    float referenceDt = 1.f / 30.f, derivativeFilter = 0.08f;
    float maxErrorRate = 960.f, highSpeedGain = 1.f;
    float lostTimeout = 0.20f, recoverTime = 0.25f, maxFrameGap = 0.25f;
    float widthAlpha = 0.1f;
    int borderMargin = 3, minPoints = 8, minSpan = 30;
    int minWidth = 30, maxWidth = 300;
    // 识别门限用地面坐标（米），见 Track::handle；上面的像素量（borderMargin/
    // minWidth/maxWidth/minSpan/minPoints/curveLookaheadRow/maxErrorRate）
    // 按 320x240 参考分辨率标定，AutoPilot 会按实际帧尺寸换算成原生像素运行
    float minDistance = 0.f, maxDistance = 1.20f;
    float minRoadWidth = 0.20f, maxGapDistance = 0.05f;
    int commandTimeoutMs = 250, binary = -1;

    void validate() const {
        auto positive = [](float x) { return std::isfinite(x) && x > 0; };
        if (!positive(cruiseSpeed) || !positive(maxSpeed) || maxSpeed > 1.5f ||
            !positive(minSpeed) || minSpeed > cruiseSpeed ||
            cruiseSpeed > maxSpeed || !positive(curveSpeed) ||
            curveSpeed > maxSpeed || !positive(degradedSpeed) ||
            degradedSpeed > maxSpeed || !positive(maxSingleFraction) ||
            maxSingleFraction >= 1 || !positive(acceleration) ||
            acceleration > 2 || !positive(runP1) || runP1 > 20 ||
            !positive(deceleration) || deceleration > 5 ||
            !positive(qualityDeceleration) || qualityDeceleration > 5 ||
            !std::isfinite(curveStart) || curveStart < 0 ||
            curveStart >= 0.95f || !std::isfinite(qualityRecoverTime) ||
            qualityRecoverTime < 0 || qualityRecoverTime > 5 ||
            curveLookaheadRow < 0 || curveLookaheadRow > 200 ||
            !std::isfinite(runP2) || runP2 < 0 || runP2 > 1 ||
            !std::isfinite(turnD) || turnD < 0 || turnD > 30 ||
            !positive(referenceDt) || referenceDt > 0.1f ||
            !positive(derivativeFilter) || derivativeFilter > 1 ||
            !positive(maxErrorRate) || maxErrorRate > 5000 ||
            !positive(highSpeedGain) || highSpeedGain > 2 ||
            !positive(lostTimeout) || lostTimeout > 1 ||
            !positive(recoverTime) || recoverTime > 5 ||
            !positive(maxFrameGap) || maxFrameGap > 1 ||
            !positive(widthAlpha) || widthAlpha > 1 || borderMargin < 1 ||
            borderMargin > 20 || minPoints < 5 || minPoints > 100 ||
            minSpan < 10 || minSpan > 200 || minWidth < 10 ||
            maxWidth <= minWidth || maxWidth >= 320 || minDistance < 0 ||
            maxDistance <= minDistance || maxDistance > 2 ||
            !positive(minRoadWidth) || minRoadWidth > 1 ||
            !positive(maxGapDistance) || maxGapDistance > 0.5f ||
            commandTimeoutMs < 100 || commandTimeoutMs > 1000 || binary < -1 ||
            binary > 255)
            throw std::invalid_argument(
                "Invalid trackview configuration values");
    }
    void load(const string &path, bool required = false) {
        std::ifstream file(path);
        if (!file) {
            if (required)
                throw std::runtime_error("Cannot read config: " + path);
            validate();
            return;
        }
        nlohmann::json j;
        file >> j;
        if (!j.is_object())
            throw std::invalid_argument("Config must be a JSON object");
        // Reject spelling mistakes rather than silently ignoring a speed
        // setting.
        const auto known = toJson();
        for (auto it = j.begin(); it != j.end(); ++it)
            if (!known.contains(it.key()) && it.key() != "mixedSpeed")
                throw std::invalid_argument("Unknown config key: " + it.key());
#define TV_READ(name)                                                          \
    if (j.contains(#name))                                                     \
    j.at(#name).get_to(name)
        TV_READ(cruiseSpeed);
        TV_READ(maxSpeed);
        TV_READ(minSpeed);
        TV_READ(curveSpeed);
        TV_READ(degradedSpeed);
        TV_READ(acceleration);
        TV_READ(deceleration);
        TV_READ(qualityDeceleration);
        TV_READ(curveStart);
        TV_READ(qualityRecoverTime);
        TV_READ(curveLookaheadRow);
        TV_READ(maxSingleFraction);
        // Accept v4 configuration files without retaining their extra speed
        // cap.
        if (j.contains("mixedSpeed"))
            std::cerr
                << "[Config] mixedSpeed ignored; healthy MIXED uses limit.\n";
        TV_READ(runP1);
        TV_READ(runP2);
        TV_READ(turnD);
        TV_READ(referenceDt);
        TV_READ(derivativeFilter);
        TV_READ(maxErrorRate);
        TV_READ(highSpeedGain);
        TV_READ(lostTimeout);
        TV_READ(recoverTime);
        TV_READ(maxFrameGap);
        TV_READ(widthAlpha);
        TV_READ(borderMargin);
        TV_READ(minPoints);
        TV_READ(minSpan);
        TV_READ(minWidth);
        TV_READ(maxWidth);
        TV_READ(minDistance);
        TV_READ(maxDistance);
        TV_READ(minRoadWidth);
        TV_READ(maxGapDistance);
        TV_READ(commandTimeoutMs);
        TV_READ(binary);
#undef TV_READ
        validate();
    }
    nlohmann::json toJson() const {
        return {{"cruiseSpeed", cruiseSpeed},
                {"maxSpeed", maxSpeed},
                {"minSpeed", minSpeed},
                {"curveSpeed", curveSpeed},
                {"degradedSpeed", degradedSpeed},
                {"acceleration", acceleration},
                {"deceleration", deceleration},
                {"qualityDeceleration", qualityDeceleration},
                {"curveStart", curveStart},
                {"qualityRecoverTime", qualityRecoverTime},
                {"curveLookaheadRow", curveLookaheadRow},
                {"maxSingleFraction", maxSingleFraction},
                {"runP1", runP1},
                {"runP2", runP2},
                {"turnD", turnD},
                {"referenceDt", referenceDt},
                {"derivativeFilter", derivativeFilter},
                {"maxErrorRate", maxErrorRate},
                {"highSpeedGain", highSpeedGain},
                {"lostTimeout", lostTimeout},
                {"recoverTime", recoverTime},
                {"maxFrameGap", maxFrameGap},
                {"widthAlpha", widthAlpha},
                {"borderMargin", borderMargin},
                {"minPoints", minPoints},
                {"minSpan", minSpan},
                {"minWidth", minWidth},
                {"maxWidth", maxWidth},
                {"minDistance", minDistance},
                {"maxDistance", maxDistance},
                {"minRoadWidth", minRoadWidth},
                {"maxGapDistance", maxGapDistance},
                {"commandTimeoutMs", commandTimeoutMs},
                {"binary", binary}};
    }
};
// Main thread alone owns HighGUI. Controller never waits for network rendering.
class TrackviewDisplay {
    std::mutex imageMutex, keyMutex;
    cv::Mat latestImage, latestBinary;
    std::deque<int> keys;
    std::atomic_bool emergency{false};
    std::atomic<long long> progressMs{0};
    static long long clockMs() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
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
    TrackviewConfig config;
    int imageCols = IMAGE_COL, imageRows = IMAGE_ROW; // 实际帧尺寸
    float refScale = 1.f;      // 实际分辨率 / 320x240 参考分辨率
    std::vector<Width> widths; // 逐行宽度，原生像素
    float filteredDerivative = 0, previousError = 0, commandedSpeed = 0;
    double now = 0, missingTime = 0, stableTime = 0;
    double qualityGoodTime = 0;
    bool qualityRestricted = false;
    bool hadPath = false;

    // 配置里的像素量按 320x240（IMAGE_COL/IMAGE_ROW）标定，这里按实际帧宽换算成
    // 原生像素，并把增益折算成原生像素口径，保证同样的横向偏差给出同样的舵机输出。
    void scaleToFrame() {
        refScale = float(imageCols) / IMAGE_COL;
        if (refScale == 1.f)
            return;
        config.borderMargin = int(std::lround(config.borderMargin * refScale));
        config.minWidth = int(std::lround(config.minWidth * refScale));
        config.maxWidth = int(std::lround(config.maxWidth * refScale));
        config.minSpan = int(std::lround(config.minSpan * refScale));
        config.minPoints = int(std::lround(config.minPoints * refScale));
        config.curveLookaheadRow =
            int(std::lround(config.curveLookaheadRow * refScale));
        config.maxErrorRate *= refScale;
        // error 变成 refScale 倍：一次项除 refScale，二次项除平方，微分项也除
        config.runP1 /= refScale;
        config.runP2 /= refScale * refScale;
        config.turnD /= refScale;
    }
    // 按 320x240 标定的像素常量换算到原生像素
    float px(float reference) const { return reference * refScale; }

    bool trusted(const Point2i &p) const {
        return p.x >= 0 && p.x < imageRows && p.y > config.borderMargin &&
               p.y < imageCols - 1 - config.borderMargin;
    }
    float widthAt(int row) const {
        // Interpolate local observations only: no constant-width extrapolation.
        int lower = -1, upper = -1;
        for (int r = std::max(0, row - int(px(12.f)));
             r <= std::min(imageRows - 1, row + int(px(12.f))); ++r) {
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

    explicit AutoPilot(const TrackviewConfig &cfg = {},
                       int frameCols = IMAGE_COL, int frameRows = IMAGE_ROW)
        : config(cfg), imageCols(frameCols), imageRows(frameRows) {
        config.validate();
        scaleToFrame();
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
    float curveSpeed() const { return config.curveSpeed; }
    float degradedSpeed() const { return config.degradedSpeed; }
    void setCurveSpeed(float value) {
        if (!std::isfinite(value))
            throw std::invalid_argument("Invalid curve speed");
        config.curveSpeed = std::clamp(value, config.minSpeed, config.maxSpeed);
    }
    void setDegradedSpeed(float value) {
        if (!std::isfinite(value))
            throw std::invalid_argument("Invalid degraded speed");
        config.degradedSpeed =
            std::clamp(value, config.minSpeed, config.maxSpeed);
    }
    float qualitySpeedLimit() const {
        if (!valid || !ready || lost)
            return 0;
        if (qualityRestricted || degraded || confidence < 0.65f)
            return config.degradedSpeed;
        // Healthy mixed paths share the cruise limit; no separate mixed-speed
        // cap.
        return config.maxSpeed;
    }
    const char *qualityLimitReason() const {
        if (!valid || !ready || lost)
            return "PATH_WAIT";
        if (degraded)
            return singleFraction() > config.maxSingleFraction ? "SINGLE_SHARE"
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
        if (!std::isfinite(dt) || dt <= 0 || dt > config.maxFrameGap) {
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
                if (w < config.minWidth || w > config.maxWidth) {
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
                     px(10.f) + px(3.f) * (previousRow - row))) {
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
        else if (raw.size() < size_t(config.minPoints))
            pathIssue = "TOO_FEW_POINTS";
        else if (raw.front().x - raw.back().x < config.minSpan)
            pathIssue = "SPAN_SHORT";
        else if (raw.front().x < imageRows * 0.45f)
            pathIssue = "NEAR_MISSING";
        else
            pathIssue = "NONE";
        valid = string(pathIssue) == "NONE";
        // A few reconstructed rows are normal when a road edge leaves the
        // image. Degrade by their share of accepted centers, not their mere
        // presence.
        degraded = singleFraction() > config.maxSingleFraction ||
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
            lost = missingTime >= config.lostTimeout;
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
        ready = stableTime >= config.recoverTime;
        confidence = float(dualPoints + 0.6f * singlePoints) /
                     std::max<size_t>(rows.size(), 1);
        if (degraded || confidence < 0.65f) {
            qualityRestricted = true;
            qualityGoodTime = 0;
        } else if (qualityRestricted) {
            qualityGoodTime += dt;
            if (qualityGoodTime >= config.qualityRecoverTime)
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
            [&](const Point2i &p) { return p.x < config.curveLookaheadRow; });
        const size_t count = size_t(std::distance(centerEdge.begin(), end));
        curveWindowFallback =
            count < size_t(config.minPoints) ||
            centerEdge.front().x - centerEdge[count ? count - 1 : 0].x <
                config.minSpan;
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
        if (dualPoints >= config.minPoints && std::fabs(heading) < px(15.f) &&
            std::abs(center - imageCols / 2) < px(20.f)) {
            for (const auto &[row, w] : widthCandidates) {
                auto &sample = widths[row];
                if (sample.observations == 0) {
                    sample.value = w;
                    sample.time = now;
                    sample.observations = 1;
                } else if (std::fabs(w - sample.value) <=
                           std::max(px(20.f), sample.value * 0.25f)) {
                    sample.value += config.widthAlpha * (w - sample.value);
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
                            std::max(px(10.f), sample.pendingValue * 0.1f)) {
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
        error = std::clamp(error, previousError - config.maxErrorRate * dt,
                           previousError + config.maxErrorRate * dt);
        const float derivative =
            hadPath ? (error - previousError) * config.referenceDt / dt : 0;
        const float alpha = dt / (config.derivativeFilter + dt);
        filteredDerivative += alpha * (derivative - filteredDerivative);
        const float speedRatio =
            std::clamp(std::fabs(previousSpeed) / config.maxSpeed, 0.f, 1.f);
        const float gain = 1 + (config.highSpeedGain - 1) * speedRatio;
        const float p = config.runP1 + std::fabs(error) * config.runP2;
        servo =
            clampServo(PWMSERVOMID +
                       gain * (error * p + filteredDerivative * config.turnD));
        const float e = std::fabs(error) / px(65.f),
                    h = std::fabs(curveHeading) / px(65.f);
        const float s = std::fabs(float(servo) - PWMSERVOMID) / 320.f;
        rawSeverity = std::clamp(std::max({e, h, s}), 0.f, 1.f);
        severity = std::clamp((rawSeverity - config.curveStart) /
                                  (1 - config.curveStart),
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
            dt > config.maxFrameGap || !valid || !ready || lost) {
            commandedSpeed = speedTarget = 0;
            speedReason = "PATH_WAIT";
            return 0;
        }
        const float limit =
            std::clamp(cruise, config.minSpeed, config.maxSpeed);
        const float curve =
            std::min(limit, limit + (config.curveSpeed - limit) * severity);
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
                ? std::max(config.deceleration, config.qualityDeceleration)
                : config.deceleration;
        if (speedTarget < commandedSpeed)
            commandedSpeed =
                std::max(speedTarget, commandedSpeed - braking * dt);
        else
            commandedSpeed = std::min(
                speedTarget, commandedSpeed + config.acceleration * dt);
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

static void drawHud(cv::Mat &image, const AutoPilot &pilot, Mode mode,
                    bool panic, float cruise, float speed, uint16_t servo,
                    double fps, bool displaySlow) {
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

void parse_arguments(int argc, char **argv, string &source, string &configPath,
                     bool &noControl, bool &headless, bool &replay,
                     bool &configRequired) {
    for (int i = 1; i < argc; ++i) {
        const string arg = argv[i];
        auto value = [&]() {
            if (++i >= argc)
                throw std::invalid_argument("Missing argument for " + arg);
            return string(argv[i]);
        };
        if (arg == "--config") {
            configPath = value();
            configRequired = true;
        } else if (arg == "--replay") {
            source = value();
            replay = noControl = true;
        } else if (arg == "--no-control")
            noControl = true;
        else if (arg == "--headless")
            headless = true;
        else if (arg == "--help") {
            std::cout << "trackview [--config trackview.json] [--no-control] "
                         "[--headless] [--replay video]\n"
                      << "keys: a=auto/manual  space=stop  +/-=limit  [/]=turn "
                         ",/.=quality  w=weight  ESC=quit\n"
                      << "Default MANUAL; replay never opens serial/gamepad.\n";
            exit(0);
        } else if (arg.rfind("--", 0) == 0)
            throw std::invalid_argument("Unknown option: " + arg);
        else
            source = arg;
    }
}
int main(int argc, char **argv) {
    signal(SIGINT, onSignal);
    signal(SIGTERM, onSignal);
    signal(SIGHUP, onSignal);
    shared_ptr<CarControl> car;
    AutoPilot pilot;
    try {
        string source = "/dev/video0", configPath = "trackview.json";
        bool noControl = false, headless = false, replay = false,
             configRequired = false;
        parse_arguments(argc, argv, source, configPath, noControl, headless,
                        replay, configRequired);

        if (!noControl && source.rfind("/dev/video", 0) != 0)
            throw std::invalid_argument(
                "Video files require --replay (no vehicle commands)");
        TrackviewConfig config;
        config.load(configPath, configRequired);
        // Load calibration before opening any vehicle control device.
        unique_ptr<Camera> camera =
            config.binary < 0
                ? make_unique<Camera>(source.c_str())
                : make_unique<Camera>(source.c_str(), uint8_t(config.binary));
        // pilot 的控制参数按 320x240 标定，这里把实际帧尺寸交给它换算
        pilot = AutoPilot(config, camera->cols(), camera->rows());
        std::atomic_bool joyLost{false}; // Must outlive gamepad callback.
        std::shared_ptr<JoyStick> js;
        if (!noControl) {
            car = CarControl::create();
            if (!car)
                throw std::runtime_error(
                    "Cannot initialize vehicle serial port");
            car->setCommandTimeout(config.commandTimeoutMs);
            js = JoyStick::create([&joyLost] { joyLost = true; });
            if (!js)
                cerr << "[Warn] No gamepad; use image window keyboard.\n";
        }
        Track track;
        track.minDistance = config.minDistance;
        track.maxDistance = config.maxDistance;
        track.minRoadWidth = config.minRoadWidth;
        track.maxGapDistance = config.maxGapDistance;
        Mode mode = replay ? Mode::Auto : Mode::Manual;
        bool panic = false;
        float cruise = config.cruiseSpeed, jsSpeed = 0, speed = 0;
        uint16_t jsServo = PWMSERVOMID;
        auto start = Clock::now(), last = start;
        const double videoFps = camera->get_capture().get(cv::CAP_PROP_FPS);
        const float replayDt = (videoFps >= 5 && videoFps <= 120)
                                   ? float(1 / videoFps)
                                   : config.referenceDt;
        TrackviewDisplay display;
        cout << "trackview | " << (noControl ? "NO CONTROL" : "MANUAL START")
             << " | source=" << source << " | frame=" << camera->cols() << "x"
             << camera->rows() << " | limit=" << cruise << " m/s (max "
             << config.maxSpeed << ") turn=" << pilot.curveSpeed()
             << " quality=" << pilot.degradedSpeed() << "\n";
        cout << "  track " << config.minDistance << "-" << config.maxDistance
             << " m, road>=" << config.minRoadWidth
             << " m, gap<=" << config.maxGapDistance << " m\n";
        jthread driving([&] {
            string stage = "startup";
            auto lastStatus = Clock::now();
            try {
                while (!stopping.load()) {
                    stage = "before_capture";
                    if (car && car->hasFault())
                        throw std::runtime_error(
                            "Serial fault or command watchdog timeout");
                    cv::Mat image;
                    stage = "capture";
                    if (!camera->read(image) || image.empty()) {
                        if (car)
                            car->resetVelocity();
                        if (replay) {
                            std::cout << "[Stop] replay finished\n";
                            break;
                        }
                        throw std::runtime_error("Camera frame unavailable");
                    }
                    stage = "preprocess";
                    cv::Mat bin = camera->binarize(image);
                    stage = "track";
                    track.handle(*camera, bin);
                    const auto now = Clock::now();
                    stage = "control";
                    float dt =
                        replay ? replayDt
                               : float(duration<double>(now - last).count());
                    last = now;
                    if (dt <= 0)
                        dt = config.referenceDt;
                    pilot.update(track.pointsEdgeLeft, track.pointsEdgeRight,
                                 dt, speed);
                    track.drawImage(image);
                    const uint64_t motionEpoch = car ? car->motionEpoch() : 0;
                    auto toggle = [&]() {
                        if (mode == Mode::Auto) {
                            mode = Mode::Manual;
                            jsSpeed = 0;
                            pilot.resetControl();
                            std::cout << "[Mode] MANUAL\n";
                        } else if (pilot.ready && !joyLost &&
                                   (!js || !js->isStopped()) &&
                                   std::fabs(jsSpeed) <= 0.05f) {
                            if (car && !car->resumeMotion(motionEpoch))
                                return;
                            mode = Mode::Auto;
                            panic = false;
                            jsSpeed = 0;
                            pilot.resetControl();
                            std::cout << "[Mode] AUTO\n";
                        } else {
                            if (!pilot.ready)
                                std::cout
                                    << "[Wait] Lane " << pilot.pathIssue
                                    << "; width rows="
                                    << pilot.learnedWidthRows()
                                    << " missing width="
                                    << pilot.missingWidthPoints
                                    << "; wait for stable trusted path.\n";
                            else
                                std::cout << "[Wait] Release gamepad throttle "
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
                            cruise = std::min(config.maxSpeed, cruise + 0.05f);
                            std::cout << "[Set] limit=" << cruise << " m/s\n";
                        } else if (key == '-' || key == '_') {
                            cruise = std::max(config.minSpeed, cruise - 0.05f);
                            std::cout << "[Set] limit=" << cruise << " m/s\n";
                        } else if (key == '[' || key == ']') {
                            pilot.setCurveSpeed(pilot.curveSpeed() +
                                                (key == ']' ? 0.05f : -0.05f));
                            std::cout
                                << "[Set] curveSpeed=" << pilot.curveSpeed()
                                << " m/s\n";
                        } else if (key == ',' || key == '.') {
                            pilot.setDegradedSpeed(
                                pilot.degradedSpeed() +
                                (key == '.' ? 0.05f : -0.05f));
                            std::cout << "[Set] degradedSpeed="
                                      << pilot.degradedSpeed() << " m/s\n";
                        } else if (key == 'w' || key == 'W') {
                            pilot.cycleWeight();
                            std::cout << "[Set] weight=" << pilot.weightName()
                                      << '\n';
                        }
                    }
                    // X11 rendering is telemetry, not the control-loop
                    // heartbeat. Keep command watchdog/vision loss independent
                    // of remote UI delay.
                    const auto displayAge = headless ? 0LL : display.ageMs();
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
                                stopReason =
                                    "gamepad disconnected; stop latched";
                            panic = true;
                            mode = Mode::Manual;
                            jsSpeed = 0;
                            pilot.resetControl();
                        } else if (mode == Mode::Auto && newCommand &&
                                   std::fabs(jsSpeed) > 0.05f) {
                            mode = Mode::Manual;
                            pilot.resetControl();
                            std::cout << "[Mode] MANUAL (gamepad takeover)\n";
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
                            speed = std::clamp(jsSpeed, -config.maxSpeed,
                                               config.maxSpeed);
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
                    drawHud(image, pilot, mode, panic, cruise, speed, servo,
                            1 / dt, displayAge >= 2000);
                    if (!stopReason.empty())
                        std::cerr << "[Stop] " << stopReason << '\n';
                    if (!headless)
                        display.publish(image, bin);
                    // headless 没有画面，用 1 Hz 的状态行代替
                    if (headless &&
                        now - lastStatus >= std::chrono::seconds(1)) {
                        lastStatus = now;
                        std::cout
                            << "[Status] "
                            << (mode == Mode::Auto ? "AUTO" : "MANUAL")
                            << " speed=" << speed << " center=" << pilot.center
                            << " err=" << pilot.error << ' ' << pilot.pathIssue
                            << '\n';
                    }
                }
                if (car)
                    car->resetVelocity();
                std::cout << "[Stop] "
                          << (stopping.load() ? "signal or UI stop"
                                              : "normal exit")
                          << '\n';
            } catch (const std::exception &e) {
                if (car) {
                    try {
                        car->latch_stop();
                    } catch (...) {
                    }
                }
                if (stopping.load() && !display.failed.load()) {
                    std::cout << "[Stop] operator requested exit\n";
                } else {
                    std::cerr << "[Error] " << e.what() << " (stage=" << stage
                              << ")\n";
                    display.failed = true;
                }
            } catch (...) {
                if (car) {
                    try {
                        car->latch_stop();
                    } catch (...) {
                    }
                }
                std::cerr << "[Error] unexpected driving thread failure (stage="
                          << stage << ")\n";
                display.failed = true;
            }
            display.finished = true;
        });
        StopOnExit stopBeforeJoin{car};
        auto lastRender = Clock::now() - std::chrono::milliseconds(100);
        while (!stopping.load() && !display.finished.load()) {
            // GUI stays on the main thread; only immutable latest snapshots
            // cross threads.
            if (car && car->hasFault()) {
                display.failed = true;
                break;
            }
            if (headless) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
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
        std::cout << "[Stop] trackview finished.\n";
        return display.failed.load() ? 1 : 0;
    } catch (const std::exception &e) {
        if (car) {
            try {
                car->resetVelocity();
            } catch (...) {
            }
        }
        std::cerr << "[Error] " << e.what() << '\n';
        return 1;
    } catch (...) {
        if (car) {
            try {
                car->resetVelocity();
            } catch (...) {
            }
        }
        std::cerr << "[Error] unexpected exception\n";
        return 1;
    }
}
