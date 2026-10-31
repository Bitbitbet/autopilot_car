#include "pilot.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

using cv::Point2d;
using cv::Point2i;
using std::vector;

namespace {
// ---- 调参（速度 m/s，加速度 m/s²；地面量是米） ----
constexpr float CURVE_SPEED = 0.8f;    // 弯道限速初值（运行时可调）
constexpr float DEGRADED_SPEED = 0.9f; // 单边/低置信度限速初值
constexpr float ACCELERATION = 0.30f;
constexpr float DECELERATION = 0.60f;
constexpr float QUALITY_DECELERATION = 1.20f;
constexpr float REFERENCE_DT = 1.f / 30.f; // 无帧率信息时的 dt
constexpr float MAX_FRAME_GAP = 0.6f; // 超过才算卡死（4~5fps 的环路也留余量）
constexpr float LOST_TIMEOUT = 0.20f; // 丢线多久判定 LOST
constexpr float QUALITY_RECOVER_TIME = 0.30f;
// 转向：servo = mid + gain*(err*RUN_P1 + err^2*RUN_P2 + d*TURN_D)，err 是像素
constexpr float RUN_P1 = 1.10f;
constexpr float RUN_P2 = 0.00175f;
constexpr float TURN_D = 1.75f;
constexpr float DERIVATIVE_FILTER = 0.08f;
constexpr float MAX_ERROR_RATE = 1920.f; // 偏差变化率上限 (px/s)
constexpr float HIGH_SPEED_GAIN = 1.f;
constexpr float CURVE_START = 0.20f; // severity 超过它开始减速
// 路径判定（原生像素）
constexpr size_t MIN_POINTS = 16;       // 至少多少行才算可信路径
constexpr int MIN_SPAN = 60;            // 中心线最小纵向跨度
constexpr int MIN_WIDTH = 30;           // 双边路宽的最小范围
constexpr float ROAD_WIDTH = 0.45f;     // 赛道实际宽度（米），单边推中心线用
constexpr int CURVE_LOOKAHEAD_ROW = 20; // 速度预判用的近处行
constexpr float JUMP_LIMIT_BASE = 20.f; // 相邻行中心跳变上限
constexpr float JUMP_LIMIT_PER_ROW = 6.f;
constexpr float SEVERITY_SCALE = 130.f; // severity 归一化尺度

uint16_t clampServo(float pwm) {
    return static_cast<uint16_t>(
        std::lround(std::clamp(pwm, float(PWMSERVOMIN), float(PWMSERVOMAX))));
}
// ---- 单边平滑度判定（地面坐标，米） ----
// 只看这一边是不是一条光滑曲线：直线、圆弧、S 弯都算光滑，
// 逐行抖动、跳变、贴图像边都算不好。
constexpr size_t EDGE_MIN_POINTS = 12;        // 够不够判断
constexpr double EDGE_SMOOTH_RESIDUAL = 0.02; // 残差上限，够小就算光滑
constexpr int EDGE_BORDER_MARGIN = 2;         // 距图像左右边几列算贴边
constexpr double EDGE_BORDER_SHARE = 0.4;     // 贴边点占比超过它算被裁掉

using GroundPoints = vector<Point2d>;

/**
 * @brief 一边的平滑程度
 *
 * 每个点到左右相邻点中点的距离就是“锯齿”的量；curve 是平滑后的点，
 * 用它算切向，避免把锯齿带到中心线上。
 */
struct EdgeQuality {
    GroundPoints curve;   // 平滑后的可用点（按行排列，地面坐标）
    vector<int> rows;     // curve 各点的图像行
    vector<size_t> index; // curve 各点在原始点集里的下标
    double residual = std::numeric_limits<double>::infinity(); // 残差（米）
    bool border = false; // 贴图像边的点太多，这边被裁掉了
    bool enough = false; // 没贴边的点够不够判断

    bool smooth() const {
        return enough && !border && residual <= EDGE_SMOOTH_RESIDUAL;
    }
};

EdgeQuality assess_edge(const GroundPoints &points, const vector<int> &cols,
                        const vector<int> &rows, int imageCols) {
    EdgeQuality quality;
    vector<size_t> idx; // 没贴图像边的点
    int border = 0;
    for (size_t i = 0; i < points.size(); ++i) {
        const int col = cols[i];
        if (col <= EDGE_BORDER_MARGIN ||
            col >= imageCols - 1 - EDGE_BORDER_MARGIN) {
            ++border;
            continue;
        }
        idx.push_back(i);
    }
    quality.border = double(border) > EDGE_BORDER_SHARE * double(points.size());
    if (idx.size() < EDGE_MIN_POINTS)
        return quality;
    quality.enough = true;
    quality.curve.resize(idx.size());
    quality.rows.resize(idx.size());
    quality.index = idx;
    for (size_t k = 0; k < idx.size(); ++k)
        quality.rows[k] = rows[idx[k]];
    double sum = 0;
    size_t measured = 0;
    for (size_t k = 0; k < idx.size(); ++k) {
        const Point2d &p = points[idx[k]];
        if (k == 0 || k + 1 == idx.size()) {
            quality.curve[k] = p; // 端点没有两侧邻居，不平滑
            continue;
        }
        const Point2d &a = points[idx[k - 1]], &b = points[idx[k + 1]];
        quality.curve[k] = (a + b + p) * (1.0 / 3.0);
        const Point2d d = p - quality.curve[k];
        sum += d.x * d.x + d.y * d.y;
        ++measured;
    }
    quality.residual = std::sqrt(sum / double(std::max<size_t>(measured, 1)));
    return quality;
}

/**
 * @brief 平滑曲线在第 k 点的单位法向（切向旋转 90°，方向未定）
 *
 * 只用行距在 8 行内的邻点估切向：白色标记会把可用点切成两段，
 * 跨着缺口取邻点算出来的切向是错的。
 */
Point2d curve_normal(const GroundPoints &curve, const vector<int> &rows,
                     size_t k) {
    const size_t n = curve.size();
    size_t a = k, b = k;
    for (size_t i = k; i-- > 0;)
        if (rows[k] - rows[i] <= 8) {
            a = i;
            break;
        }
    for (size_t i = k + 1; i < n; ++i)
        if (rows[k] - rows[i] <= 8) {
            b = i;
            break;
        }
    const Point2d tangent = curve[b] - curve[a];
    const double len = std::hypot(tangent.x, tangent.y);
    return len > 0 ? Point2d(-tangent.y / len, tangent.x / len) : Point2d(0, 0);
}

// 一帧的检测结果：图像列 + 对应的地面坐标
struct Edges {
    vector<int> rows, left_cols, right_cols;
    GroundPoints left, right;
};

/**
 * @brief 把 track 的行坐标整批换到地面坐标
 */
Edges gather_edges(const TrackRows &rows, const Camera &camera) {
    Edges edges;
    edges.rows.reserve(rows.size());
    edges.left_cols.reserve(rows.size());
    edges.right_cols.reserve(rows.size());
    edges.left.reserve(rows.size());
    edges.right.reserve(rows.size());
    for (const auto &[row, cols] : rows) {
        edges.rows.push_back(row);
        edges.left_cols.push_back(cols.first);
        edges.right_cols.push_back(cols.second);
        edges.left.push_back(
            camera.image_to_ground({double(cols.first), double(row)}));
        edges.right.push_back(
            camera.image_to_ground({double(cols.second), double(row)}));
    }
    return edges;
}

/**
 * @brief 只有一边可信：沿法向偏半个路宽推中心线
 *
 * 朝向用所有行累加判定（中心一定夹在好边和另一边之间）。单点判断朝向
 * 在斜视角下会翻，翻一次整条中心线就跑到画面外，只剩几个点。
 */
vector<Point2i> centerline_from_side(const EdgeQuality &good,
                                     const Edges &edges, bool use_left,
                                     const Camera &camera, int imageCols,
                                     int imageRows) {
    const GroundPoints &curve = good.curve;
    const GroundPoints &other = use_left ? edges.right : edges.left;
    double inward = 0;
    for (size_t k = 0; k < curve.size(); ++k) {
        const Point2d n = curve_normal(curve, good.rows, k);
        const Point2d d = other[good.index[k]] - curve[k];
        inward += n.x * d.x + n.y * d.y;
    }
    const double sign = inward >= 0 ? 1.0 : -1.0;

    vector<Point2i> out;
    for (size_t k = 0; k < curve.size(); ++k) {
        const Point2d center = curve[k] + curve_normal(curve, good.rows, k) *
                                              (sign * ROAD_WIDTH / 2.f);
        const Point2d img = camera.ground_to_image(center);
        if (!std::isfinite(img.x) || !std::isfinite(img.y))
            continue;
        if (img.x < 0 || img.x > imageCols - 1 || img.y < 0 ||
            img.y > imageRows - 1)
            continue; // 推出来的中心线不在画面里
        out.emplace_back(int(std::lround(img.x)), int(std::lround(img.y)));
    }
    return out;
}

/**
 * @brief 两边都用（或者都不够光滑，退回中点），按宽度和跳变挑可信的行
 */
vector<Point2i> centerline_from_mid(const TrackRows &rows, int &accepted,
                                    int &rejected) {
    vector<Point2i> out;
    int previousRow = -1;
    float previousCenter = 0;
    for (const auto &[row, cols] : rows) {
        const int l = cols.first, r = cols.second;
        if (r - l < MIN_WIDTH) {
            ++rejected;
            continue;
        }
        const float c = (l + r) / 2.f;
        if (previousRow >= 0 &&
            fabs(c - previousCenter) >
                JUMP_LIMIT_BASE + JUMP_LIMIT_PER_ROW * (previousRow - row)) {
            ++rejected;
            continue;
        }
        out.emplace_back(int(std::lround(c)), row);
        previousRow = row;
        previousCenter = c;
        ++accepted;
    }
    return out;
}
} // namespace

AutoPilot::AutoPilot(int frameCols, int frameRows)
    : curveLimit(CURVE_SPEED), qualityLimit(DEGRADED_SPEED),
      imageCols(frameCols), imageRows(frameRows) {}

float AutoPilot::weightOf(int row) const {
    switch (weightMode) {
    case WeightMode::Uniform:
        return 1;
    case WeightMode::NearDominant:
        return std::min(float(row), imageRows / 2.f);
    case WeightMode::FarFull:
        return row < imageRows / 2 ? imageRows / 2.f : float(imageRows - row);
    }
    return 1;
}

float AutoPilot::qualitySpeedLimit() const {
    if (!valid || lost)
        return 0;
    if (qualityRestricted || degraded || confidence < 0.65f)
        return qualityLimit;
    return cfg::MAX_SPEED; // 健康的混合路径不额外限速
}

void AutoPilot::update(const TrackRows &rows, const Camera &camera, float dt,
                       float previousSpeed) {
    // 帧间隔异常（真的卡死）：这一帧的积分/微分都没意义，按丢线处理
    if (dt <= 0 || dt > MAX_FRAME_GAP) {
        valid = false;
        lost = true;
        degraded = false;
        confidence = 0;
        severity = 0;
        pathIssue = "FRAME_GAP";
        qualityGoodTime = 0;
        centerEdge.clear();
        resetControl();
        return;
    }

    // 1) 两边换到地面坐标，判断哪一边更可信，构造中心线
    const Edges edges = gather_edges(rows, camera);
    const EdgeQuality left =
        assess_edge(edges.left, edges.left_cols, edges.rows, imageCols);
    const EdgeQuality right =
        assess_edge(edges.right, edges.right_cols, edges.rows, imageCols);
    // 两边都光滑就取中点；只有一边光滑就用它加减半个路宽推中心线
    const bool is_dual_smooth = left.smooth() && right.smooth();
    is_singal_smooth = !is_dual_smooth && (left.smooth() || right.smooth());

    int accepted = 0, rejected = 0;
    vector<Point2i> raw; // 中心线点 (col, row)
    if (is_singal_smooth) {
        const bool use_left = left.smooth();
        raw = centerline_from_side(use_left ? left : right, edges, use_left,
                                   camera, imageCols, imageRows);
        accepted = int(raw.size());
    } else {
        raw = centerline_from_mid(rows, accepted, rejected);
    }
    std::sort(raw.begin(), raw.end(),
              [](const Point2i &a, const Point2i &b) { return a.y > b.y; });
    std::cerr << raw.size() << " " << dt << std::endl; // DEBUG

    // 2) 这一帧的中心线够不够可信
    if (!update_path_state(raw, rows.size(), accepted, rejected, dt))
        return;

    // 3) 平滑 -> 控制中心 -> 航向 -> 舵机与 severity
    build_center_edge(raw);
    center = weighted_center();
    const float curveHeading = curve_heading();
    update_steering(dt, previousSpeed, curveHeading);
}

bool AutoPilot::update_path_state(const vector<Point2i> &raw, size_t total_rows,
                                  int accepted, int rejected, float dt) {
    valid = false;
    if (raw.empty())
        pathIssue = "NO_POINTS";
    else if (raw.size() < MIN_POINTS)
        pathIssue = "TOO_FEW_POINTS";
    else if (raw.front().y - raw.back().y < MIN_SPAN)
        pathIssue = "SPAN_SHORT";
    else if (raw.front().y < imageRows * 0.2f)
        pathIssue = "NEAR_MISSING";
    else {
        valid = true;
        pathIssue = "NONE";
    }
    degraded = rejected > int(total_rows / 3);

    centerEdge.clear();
    if (!valid) {
        missingTime += dt;
        confidence = 0;
        qualityGoodTime = 0;
        severity = 0;
        lost = missingTime >= LOST_TIMEOUT;
        // 坏帧立刻停速；转向保持几帧再回中。
        commandedSpeed = 0;
        if (lost)
            resetControl();
        return false;
    }
    missingTime = 0;
    lost = false;
    confidence = float(accepted) / std::max<size_t>(total_rows, 1);
    if (degraded || confidence < 0.65f) {
        qualityRestricted = true;
        qualityGoodTime = 0;
    } else if (qualityRestricted) {
        qualityGoodTime += dt;
        if (qualityGoodTime >= QUALITY_RECOVER_TIME)
            qualityRestricted = false;
    }
    return true;
}

void AutoPilot::build_center_edge(const vector<Point2i> &raw) {
    // 三点滑动平均抹平边缘锯齿。
    centerEdge.clear();
    centerEdge.reserve(raw.size());
    for (size_t i = 0; i < raw.size(); ++i) {
        const int a = raw[i ? i - 1 : i].x, b = raw[i].x;
        const int c = raw[i + 1 < raw.size() ? i + 1 : i].x;
        centerEdge.emplace_back((a + b + c) / 3, raw[i].y);
    }
}

int AutoPilot::weighted_center() const {
    // 越近的行权重越大（权重模式可切换），再夹到画面内。
    float numerator = imageCols / 2.f, denominator = 1;
    for (const auto &p : centerEdge) {
        const float w = weightOf(p.y);
        numerator += w * p.x;
        denominator += w;
    }
    return std::clamp(int(std::lround(numerator / denominator)), 0,
                      imageCols - 1);
}

float AutoPilot::curve_heading() const {
    // 远端与近端中心线的偏移是弯道/航向的代理量。
    const size_t n = std::max<size_t>(1, centerEdge.size() / 4);
    float nearMean = 0, farMean = 0;
    for (size_t i = 0; i < n; ++i) {
        nearMean += centerEdge[i].x;
        farMean += centerEdge[centerEdge.size() - 1 - i].x;
    }
    const float heading = (farMean - nearMean) / n;
    // 速度预判可以只看近处一段；点数不够时退回整条中心线。
    const auto end = std::find_if(
        centerEdge.begin(), centerEdge.end(),
        [](const Point2i &p) { return p.y < CURVE_LOOKAHEAD_ROW; });
    const size_t count = size_t(std::distance(centerEdge.begin(), end));
    if (count < MIN_POINTS ||
        centerEdge.front().y - centerEdge[count - 1].y < MIN_SPAN)
        return heading;
    const size_t section = std::max<size_t>(1, count / 4);
    float near = 0, far = 0;
    for (size_t i = 0; i < section; ++i) {
        near += centerEdge[i].x;
        far += centerEdge[count - 1 - i].x;
    }
    return (far - near) / section;
}

void AutoPilot::update_steering(float dt, float previousSpeed,
                                float curveHeading) {
    error = float(center) - imageCols / 2.f;
    error = std::clamp(error, previousError - MAX_ERROR_RATE * dt,
                       previousError + MAX_ERROR_RATE * dt);
    const float derivative =
        hadPath ? (error - previousError) * REFERENCE_DT / dt : 0;
    const float alpha = dt / (DERIVATIVE_FILTER + dt);
    filteredDerivative += alpha * (derivative - filteredDerivative);
    const float speedRatio =
        std::clamp(std::fabs(previousSpeed) / cfg::MAX_SPEED, 0.f, 1.f);
    const float gain = 1 + (HIGH_SPEED_GAIN - 1) * speedRatio;
    const float p = RUN_P1 + std::fabs(error) * RUN_P2;
    servo = clampServo(PWMSERVOMID +
                       gain * (error * p + filteredDerivative * TURN_D));

    // 弯道程度取偏差、航向、舵机量三者最大者。
    const float e = std::fabs(error) / SEVERITY_SCALE;
    const float h = std::fabs(curveHeading) / SEVERITY_SCALE;
    const float s = std::fabs(float(servo) - PWMSERVOMID) / 320.f;
    const float rawSeverity = std::clamp(std::max({e, h, s}), 0.f, 1.f);
    severity =
        std::clamp((rawSeverity - CURVE_START) / (1 - CURVE_START), 0.f, 1.f);
    previousError = error;
    hadPath = true;
}

float AutoPilot::speedFor(float cruise, float dt) {
    if (!std::isfinite(cruise) || !std::isfinite(dt) || dt <= 0 ||
        dt > MAX_FRAME_GAP || !valid || lost) {
        commandedSpeed = 0;
        return 0;
    }
    const float limit = std::clamp(cruise, cfg::MIN_SPEED, cfg::MAX_SPEED);
    const float curve =
        std::min(limit, limit + (curveLimit - limit) * severity);
    const float quality = qualitySpeedLimit();
    const float target = std::min(curve, quality);
    // 操作者的巡航上限立即生效；只有可信路径上的减速受减速度限制。
    commandedSpeed = std::min(commandedSpeed, limit);
    const bool qualityBraking = quality < limit && quality <= curve + 1e-5f;
    const float braking = qualityBraking
                              ? std::max(DECELERATION, QUALITY_DECELERATION)
                              : DECELERATION;
    if (target < commandedSpeed)
        commandedSpeed = std::max(target, commandedSpeed - braking * dt);
    else
        commandedSpeed = std::min(target, commandedSpeed + ACCELERATION * dt);
    return commandedSpeed;
}
