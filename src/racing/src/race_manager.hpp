#pragma once

#include "detection.hpp"
#include "race_perception.hpp"
#include "tools.hpp"
#include <algorithm>
#include <array>
#include <functional>
#include <map>
#include <utility>

// 单帧二值图无法区分画面顶边截断的红色圆弧与黑色道路分隔带。
// 仅检查同一帧彩色图中明确偏红、扁平、触顶且独立的暗色连通域。
// 只还原该连通域的像素，不扩张矩形、不沿用旧帧，也不修改巡线二值图。
// 黑色分隔带、与近处障碍连通的暗区、无颜色证据的情况仍保留原判断。
inline cv::Mat raceAppearanceGeometry(const cv::Mat &binary,
                                      const cv::Mat &image, int &removed) {
    removed = 0;
    if (binary.type() != CV_8UC1 || image.type() != CV_8UC3 ||
        image.size() != binary.size())
        return binary;
    const auto red = [](const cv::Vec3b &p) {
        return p[2] >= 70 && int(p[2]) - p[1] >= 25 && int(p[2]) - p[0] >= 25 &&
               int(p[2]) * 10 >= int(p[1]) * 14 &&
               int(p[2]) * 10 >= int(p[0]) * 14;
    };
    int redAtTop = 0;
    for (int col = 0; col < binary.cols; ++col)
        if (binary.at<uint8_t>(0, col) <= 127 &&
            red(image.at<cv::Vec3b>(0, col)))
            ++redAtTop;
    if (redAtTop < 60)
        return binary;
    cv::Mat dark, labels, stats, centroids;
    cv::compare(binary, 127, dark, cv::CMP_LE);
    const int components = cv::connectedComponentsWithStats(
        dark, labels, stats, centroids, 8, CV_32S);
    cv::Mat geometry;
    for (int label = 1; label < components; ++label) {
        const int x = stats.at<int>(label, cv::CC_STAT_LEFT),
                  y = stats.at<int>(label, cv::CC_STAT_TOP);
        const int width = stats.at<int>(label, cv::CC_STAT_WIDTH),
                  height = stats.at<int>(label, cv::CC_STAT_HEIGHT);
        const int area = stats.at<int>(label, cv::CC_STAT_AREA);
        if (y != 0 || x < 16 || x + width > binary.cols - 16 || width < 80 ||
            width > 360 || height < 12 || height > 80 || width < height * 3 ||
            area < 400)
            continue;
        int colored = 0;
        for (int row = y; row < y + height; ++row)
            for (int col = x; col < x + width; ++col)
                if (labels.at<int>(row, col) == label &&
                    red(image.at<cv::Vec3b>(row, col)))
                    ++colored;
        // 少量抗锯齿像素可以没有颜色，主体必须由本帧的红色像素证明。
        if (colored * 100 < area * 90)
            continue;
        if (geometry.empty())
            geometry = binary.clone();
        for (int row = y; row < y + height; ++row)
            for (int col = x; col < x + width; ++col)
                if (labels.at<int>(row, col) == label)
                    geometry.at<uint8_t>(row, col) = 255;
        ++removed;
    }
    return geometry.empty() ? binary : geometry;
}

// 巡线边缘行集合：row -> (left_col, right_col)，与 track_identify_from_bin
// 一致。
using RaceRows = std::map<int, std::pair<int, int>, std::greater<int>>;

struct RacePath {
    RaceRows rows;
    int splitRows = 0;
    const char *geometryReason = "INVALID_ROAD_INPUT";
};

// 只做边界和分数检查：和 icar 一样信任模型标签，不再按图像内容复核。
inline bool raceDetectionAccepted(const PredictResult &d,
                                  const RaceConfig &cfg) {
    return d.type >= 0 && d.type < 15 && std::isfinite(d.score) &&
           d.score >= cfg.confidence && d.x >= 0 && d.y >= 0 && d.width > 0 &&
           d.height > 0 && d.x < RACE_COLS && d.y < RACE_ROWS &&
           d.width <= RACE_COLS - d.x && d.height <= RACE_ROWS - d.y;
}

// 交通牌属于场景中的物体，牌面上的深色图案不能充当道路分隔带。
// 只采用新鲜且达到阈值的有效检测框；行人、锥桶、道闸、斑马线不在此列。
inline std::vector<cv::Rect> raceSignRegions(const RaceSnapshot &snap,
                                             const RaceConfig &cfg) {
    std::vector<cv::Rect> regions;
    for (const auto &d : snap.detections) {
        const bool sign = d.type == LABEL_BUSY || d.type == LABEL_LIMIT ||
                          d.type == LABEL_UNLIMIT || d.type == LABEL_STOP ||
                          d.type == LABEL_FLOW || d.type == LABEL_PARK ||
                          d.type == LABEL_FORK || d.type == LABEL_LEFT ||
                          d.type == LABEL_CHOICE || d.type == LABEL_YLEFT ||
                          d.type == LABEL_YRIGHT;
        if (!sign || !raceDetectionAccepted(d, cfg))
            continue;
        // 检测框在 640x480 原图上；留出抖动和异步帧位移的余量。
        constexpr int margin = 24;
        regions.push_back(cv::Rect(d.x - margin, d.y - margin,
                                   d.width + 2 * margin,
                                   d.height + 2 * margin) &
                          cv::Rect(0, 0, RACE_COLS, RACE_ROWS));
    }
    return regions;
}

// 仅保留与近处路面相接的白色区段，并把细小黑点和已检出的牌面合并回路面。
inline std::vector<std::pair<int, int>>
raceRoadBlocks(const cv::Mat &bin, int row, int low, int high,
               const RaceConfig &cfg, const std::vector<cv::Rect> &signRegions,
               int minimumBlockWidth = 0) {
    std::vector<std::pair<int, int>> blocks;
    for (int col = 0; col < bin.cols;) {
        if (bin.at<uint8_t>(row, col) <= 127) {
            ++col;
            continue;
        }
        const int first = col;
        while (col < bin.cols && bin.at<uint8_t>(row, col) > 127)
            ++col;
        if (col - first <
                (minimumBlockWidth ? minimumBlockWidth : cfg.minBranchWidth) ||
            std::min(col - 1, high) < std::max(first, low))
            continue;
        if (!blocks.empty()) {
            const int gapFirst = blocks.back().second + 1, gapLast = first - 1;
            const int middle = (gapFirst + gapLast) / 2;
            const bool covered = std::any_of(
                signRegions.begin(), signRegions.end(), [&](const cv::Rect &r) {
                    const int overlap = std::min(gapLast, r.x + r.width - 1) -
                                        std::max(gapFirst, r.x) + 1;
                    return r.contains({middle, row}) &&
                           overlap >= (gapLast - gapFirst + 2) / 2;
                });
            if (gapLast - gapFirst + 1 < cfg.minBranchGap || covered) {
                blocks.back().second = col - 1;
                continue;
            }
        }
        blocks.emplace_back(first, col - 1);
    }
    return blocks;
}

// 分叉必须是同一组通路持续延伸至搜索区远端，不能把不相邻的几行相加。
// 再用搜索区上方直到图像顶边的可见像素复核分隔带，避免裁掉圆环内部的白色。
// 顶部复核只参与判别，不把这些行送入 v6 的转向路径。
// 车头图案会在更远处重新合成一片白色；牌面的圆环会分成两个黑边，
// 都不满足道路分隔带连续的条件。判断使用本帧像素，AI 漏检时也不沿用旧框。
// 方向只在通过此检查的分叉上生效，不采用看到方向牌就删掉半边线的开环转向。
inline RacePath raceBranchPath(const cv::Mat &bin, const RaceRows &edges,
                               int direction, const RaceConfig &cfg,
                               const std::vector<cv::Rect> &signRegions = {}) {
    RacePath out;
    if (bin.empty() || bin.type() != CV_8UC1 || edges.empty())
        return out;
    int low = edges.begin()->second.first, high = edges.begin()->second.second;
    const int start = edges.begin()->first;
    if (start < 0 || start >= bin.rows || low < 0 || high >= bin.cols ||
        low >= high)
        return out;
    struct Row {
        int number;
        std::vector<std::pair<int, int>> blocks;
    };
    std::vector<Row> rows;
    std::vector<std::pair<int, int>> previousGaps;
    int missing = 0, continuous = 0, forkStart = -1;
    bool rejected = false, reachedFarEnd = false;
    const char *rejection = "NO_CONTINUOUS_SPLIT";

    for (int row = start; row >= 0;
         row = (row == 0 ? -1 : std::max(0, row - 4))) {
        auto blocks = raceRoadBlocks(bin, row, low, high, cfg, signRegions);
        if (blocks.empty()) {
            // 图像断层不能拼成连续分叉，也不能宣称已看清远端。
            rejected = true;
            continuous = 0;
            rejection = "ROAD_ROWS_MISSING";
            if (++missing > 3)
                break;
            continue;
        }
        missing = 0;
        std::vector<std::pair<int, int>> gaps;
        for (size_t i = 1; i < blocks.size(); ++i) {
            gaps.emplace_back(blocks[i - 1].second + 1, blocks[i].first - 1);
        }
        if (gaps.empty()) {
            continuous = 0;
            forkStart = -1;
            previousGaps.clear();
            rejected = false;
            rejection = "NO_CONTINUOUS_SPLIT";
        } else {
            if (!previousGaps.empty()) {
                // 同一个黑色分隔带不能突然变成圆环的两条边，也不能跳到别处的黑点。
                if (gaps.size() != previousGaps.size()) {
                    rejected = true;
                    rejection = "DIVIDER_COUNT_CHANGED";
                } else
                    for (size_t i = 0; i < gaps.size(); ++i) {
                        const int middle =
                            (previousGaps[i].first + previousGaps[i].second) /
                            2;
                        if (middle < gaps[i].first || middle > gaps[i].second) {
                            rejected = true;
                            rejection = "DIVIDER_DISCONTINUOUS";
                        }
                    }
            }
            if (!rejected && row > cfg.nearCutRow) {
                if (continuous == 0)
                    forkStart = row;
                ++continuous;
            }
            previousGaps = std::move(gaps);
        }
        // 几何检查保留所有连通支路，避免先选了牌面内部的白色而丢失两侧路面。
        low = blocks.front().first;
        high = blocks.back().second;
        if (row > cfg.nearCutRow)
            rows.push_back({row, std::move(blocks)});
        reachedFarEnd = row == 0;
    }
    if (reachedFarEnd && !rejected)
        out.splitRows = continuous;
    out.geometryReason = !reachedFarEnd ? "FAR_END_NOT_VISIBLE"
                         : rejected     ? rejection
                         : out.splitRows >= cfg.minBranchRows
                             ? "CONNECTED_SPLIT"
                             : "NO_CONTINUOUS_SPLIT";

    low = edges.begin()->second.first;
    high = edges.begin()->second.second;
    missing = 0;
    for (const auto &r : rows) {
        std::vector<std::pair<int, int>> blocks;
        for (const auto &b : r.blocks)
            if (std::min(b.second, high) >= std::max(b.first, low))
                blocks.push_back(b);
        if (blocks.empty()) {
            if (++missing > 3)
                break;
            continue;
        }
        missing = 0;
        auto best =
            std::min_element(blocks.begin(), blocks.end(), [&](auto a, auto b) {
                return std::abs(a.first + a.second - low - high) <
                       std::abs(b.first + b.second - low - high);
            });
        const bool fork =
            out.splitRows >= cfg.minBranchRows && r.number <= forkStart;
        if (fork && direction)
            best = direction < 0 ? blocks.begin() : blocks.end() - 1;
        low = (!direction) ? blocks.front().first : best->first;
        high = (!direction) ? blocks.back().second : best->second;
        out.rows.emplace(r.number, std::pair<int, int>{low, high});
    }
    return out;
}

struct RacePlan {
    float speedCap = 10.f;
    bool stop = false, buzzer = false, fresh = false;
    bool selectedBranch = false;
    int colorRejectedComponents = 0, geometrySplitRows = 0;
    const char *geometryReason = "INVALID_ROAD_INPUT";
    double detectionAge = -1;
    const char *state = "AI_WAIT", *reason = "AI_WAIT";
    RacePath path;
};

class RaceManager {
    RaceConfig cfg;
    std::array<int, 15> counts{};
    std::array<bool, 15> confirmed{};
    std::array<PredictResult, 15> boxes{};
    uint64_t lastSequence = 0;
    double lastCapture = -1, now = 0, stateSince = 0, cooldownUntil = 0;
    double gateClearSince = -1, hazardClearSince = -1;
    double directionUntil = 0, splitTime = 0, exitTime = 0, lastCrossSeen = 0;
    bool lastActive = false, limited = false, crossBottomSeen = false,
         gateWaiting = false, hazardWaiting = false;
    bool hazardPersonWaiting = false;
    bool hazardConeWaiting = false;
    int direction = 0;
    enum class State {
        Lane,
        ForkArmed,
        ForkSelect,
        CrossApproach,
        SignalWait,
        ReviewStop,
        Fault
    };
    State state = State::Lane;
    const char *reviewReason = "REVIEW_REQUIRED";
    PredictResult gateBox{};
    void enter(State s) {
        state = s;
        stateSince = now;
        splitTime = exitTime = 0;
    }
    bool seen(int label) const { return confirmed[label]; }
    bool near(int label, int row) const {
        return seen(label) && boxes[label].y + boxes[label].height >= row;
    }
    bool updateObservation(const RaceSnapshot &snap, bool fresh) {
        if (!fresh || snap.sequence <= lastSequence ||
            snap.captured <= lastCapture)
            return false;
        if (lastCapture < 0 || snap.captured - lastCapture > cfg.confirmGap)
            counts.fill(0);
        lastSequence = snap.sequence;
        lastCapture = snap.captured;
        std::array<bool, 15> present{};
        for (const auto &d : snap.detections) {
            if (!raceDetectionAccepted(d, cfg))
                continue;
            if (!present[d.type] || d.score > boxes[d.type].score)
                boxes[d.type] = d;
            present[d.type] = true;
        }
        for (int i = 0; i < 15; ++i) {
            counts[i] =
                present[i] ? std::min(cfg.confirmFrames, counts[i] + 1) : 0;
            confirmed[i] = counts[i] >= cfg.confirmFrames;
        }
        return true;
    }

  public:
    explicit RaceManager(const RaceConfig &config) : cfg(config) {}
    bool aiReady(const RaceSnapshot &s, double time) const {
        return !s.failed && s.sequence > 0 && std::isfinite(s.captured) &&
               time >= s.captured && time - s.captured <= cfg.maxDetectionAge;
    }
    RacePlan update(const RaceSnapshot &snap, double time, float dt,
                    bool active, const cv::Mat &bin, const RaceRows &edges,
                    const cv::Mat &currentImage = {}) {
        now = time;
        RacePlan p;
        p.detectionAge = snap.sequence ? time - snap.captured : -1;
        p.fresh = aiReady(snap, time);
        const bool newFrame = updateObservation(snap, p.fresh);
        const auto signRegions =
            p.fresh ? raceSignRegions(snap, cfg) : std::vector<cv::Rect>{};
        auto geometry = raceBranchPath(bin, edges, 0, cfg, signRegions);
        // 只在原几何规则将要判为分叉时检查颜色，避免正常巡线增加连通域分析。
        const auto geometryBin =
            geometry.splitRows >= cfg.minBranchRows
                ? raceAppearanceGeometry(bin, currentImage,
                                         p.colorRejectedComponents)
                : bin;
        if (p.colorRejectedComponents)
            geometry = raceBranchPath(geometryBin, edges, 0, cfg, signRegions);
        p.geometrySplitRows = geometry.splitRows;
        p.geometryReason = geometry.geometryReason;
        if (active != lastActive) {
            // 人工介入后不继续先前的定时动作；限速区域标志仍保留，不能靠按 a
            // 取消限速。
            if (state != State::ReviewStop && state != State::Fault) {
                enter(State::Lane);
                direction = 0;
            }
            counts.fill(0);
            confirmed.fill(false);
            lastActive = active;
        }
        if (!active) {
            p.state = "MANUAL";
            p.reason = "MANUAL";
            return p;
        }
        if (!p.fresh) {
            p.stop = true;
            p.state = "AI_WAIT";
            p.reason = snap.failed ? "AI_FAILED" : "AI_STALE";
            counts.fill(0);
            confirmed.fill(false);
            splitTime = exitTime = 0;
            // 无新观测时不累计恢复或穿越时间，也不能沿用旧方向继续完成动作。
            if (state == State::ForkSelect) {
                enter(State::Fault);
                reviewReason = "FORK_AI_STALE";
            }
            gateClearSince = hazardClearSince = -1;
            return p;
        }
        if (!std::isfinite(dt) || dt <= 0 || dt > 0.25f) {
            p.stop = true;
            p.state = "FRAME_WAIT";
            p.reason = "FRAME_GAP";
            return p;
        }
        if (newFrame) {
            if (seen(LABEL_LIMIT))
                limited = true;
            else if (seen(LABEL_UNLIMIT))
                limited = false;
        }
        if (limited)
            p.speedCap = cfg.limitedSpeed;

        // 施工区和泊车不能仅凭标签就宣称已完成；基础版明确停车等待专项规划验证。
        if (state != State::ReviewStop && state != State::Fault && newFrame &&
            (near(LABEL_PARK, cfg.triggerRow) ||
             near(LABEL_BUSY, cfg.triggerRow))) {
            reviewReason = seen(LABEL_PARK) ? "PARK_PLANNER_PENDING"
                                            : "CONSTRUCTION_PLANNER_PENDING";
            enter(State::ReviewStop);
        }
        if (state == State::ReviewStop || state == State::Fault) {
            p.stop = true;
            p.state = "REVIEW_STOP";
            p.reason = reviewReason;
            return p;
        }
        if (state == State::Lane && newFrame &&
            near(LABEL_STOP, cfg.triggerRow)) {
            crossBottomSeen = false;
            lastCrossSeen = now;
            enter(cfg.stopAfterCross ? State::CrossApproach
                                     : State::SignalWait);
        }
        // 道闸优先于路线选择；绿色通行牌不能放行尚未抬起的杆。
        if (newFrame && near(LABEL_GATE, cfg.triggerRow) &&
            boxes[LABEL_GATE].width >= boxes[LABEL_GATE].height &&
            !gateWaiting) {
            gateBox = boxes[LABEL_GATE];
            gateWaiting = true;
            gateClearSince = -1;
        }
        if (gateWaiting) {
            if (newFrame) {
                const auto &g = boxes[LABEL_GATE];
                const bool sameArea =
                    std::abs((g.x + g.width / 2) -
                             (gateBox.x + gateBox.width / 2)) < 80;
                const bool open =
                    seen(LABEL_GATE) && sameArea && g.height >= 60 &&
                    g.width < float(g.height) * cfg.gateOpenAspect;
                if (open) {
                    if (gateClearSince < 0)
                        gateClearSince = now;
                } else
                    gateClearSince = -1;
                if (gateClearSince >= 0 &&
                    now - gateClearSince >= cfg.clearTime)
                    gateWaiting = false;
            }
        }

        // 近桶停车不等三次确认：真实记录曾在接近时漏检，等重新凑满三帧已太迟。
        // 已看见的近桶在没有已验证绕行路线时保持停车，不能靠0.6秒漏检消除义务。
        const bool nearCone =
            newFrame &&
            std::any_of(snap.detections.begin(), snap.detections.end(),
                        [&](const PredictResult &d) {
                            return d.type == LABEL_CONE &&
                                   d.y + d.height >=
                                       std::min(cfg.obstacleRow, 290) &&
                                   raceDetectionAccepted(d, cfg);
                        });
        if (nearCone) {
            hazardConeWaiting = true;
            hazardWaiting = true;
            hazardClearSince = -1;
        }
        if (newFrame && (near(LABEL_PERSON, cfg.obstacleRow) ||
                         (near(LABEL_CONE, cfg.obstacleRow)))) {
            hazardWaiting = true;
            hazardClearSince = -1;
            if (near(LABEL_PERSON, cfg.obstacleRow))
                hazardPersonWaiting = true;
        }
        if (hazardWaiting) {
            if (newFrame) {
                // 单次漏检不能放行，连续的新鲜无障碍观测才能恢复。
                const bool any = counts[LABEL_PERSON] > 0 ||
                                 hazardConeWaiting || (counts[LABEL_CONE] > 0);
                if (any)
                    hazardClearSince = -1;
                else if (hazardClearSince < 0)
                    hazardClearSince = now;
                if (hazardClearSince >= 0 &&
                    now - hazardClearSince >= cfg.clearTime) {
                    hazardWaiting = false;
                    hazardPersonWaiting = false;
                }
            }
        }
        if (gateWaiting || hazardWaiting) {
            // 高优先级停车叠加在当前任务之上，不丢失信号灯等待等低优先级义务。
            // 停车期间不累计穿越/选路动作时间。
            stateSince += dt;
            splitTime = exitTime = 0;
            if (state == State::CrossApproach)
                lastCrossSeen = now;
            p.stop = true;
            p.state = gateWaiting ? "GATE_WAIT" : "OBSTACLE_WAIT";
            p.reason = gateWaiting         ? "GATE_NOT_VERIFIED_OPEN"
                       : hazardConeWaiting ? "CONE_ROUTE_OR_RESTART_REQUIRED"
                                           : "OBSTACLE_PLANNER_PENDING";
            return p;
        }
        if (state == State::CrossApproach) {
            p.speedCap = std::min(p.speedCap, cfg.approachSpeed);
            p.state = "CROSS_APPROACH";
            p.reason = "STOP_AFTER_CROSS";
            if (newFrame) {
                if (counts[LABEL_CROSS] > 0) {
                    lastCrossSeen = now;
                    if (seen(LABEL_CROSS) &&
                        near(LABEL_CROSS, cfg.crossPassedRow))
                        crossBottomSeen = true;
                } else if (crossBottomSeen &&
                           now - lastCrossSeen >= cfg.crossClearTime)
                    enter(State::SignalWait);
            }
            if (now - stateSince > 6 && state == State::CrossApproach) {
                enter(State::Fault);
                reviewReason = "CROSS_PASS_NOT_VERIFIED";
            }
        }
        if (state == State::SignalWait) {
            p.stop = true;
            p.state = "SIGNAL_WAIT";
            p.reason = "WAIT_FRESH_FLOW";
            if (newFrame && seen(LABEL_FLOW) && counts[LABEL_STOP] == 0 &&
                now - stateSince >= cfg.stopHoldTime) {
                enter(State::Lane);
                cooldownUntil = now + cfg.cooldown;
                p.stop = false;
                // 放行当帧同步更新显示，避免日志仍写等待、速度却已开始恢复。
                p.state = limited ? "LIMIT_ZONE" : "LANE";
                p.reason = limited ? "LIMIT_SIGN" : "CRUISE";
            }
            return p;
        }
        if (state == State::Lane && now >= cooldownUntil && newFrame) {
            if (seen(LABEL_YLEFT) && seen(LABEL_YRIGHT)) {
                direction = 0;
            } else if (seen(LABEL_YLEFT) || seen(LABEL_YRIGHT)) {
                direction = seen(LABEL_YLEFT) ? -1 : 1;
                directionUntil = now + cfg.directionMemory;
                enter(State::ForkArmed);
            }
        }
        if (state == State::ForkArmed) {
            p.speedCap = std::min(p.speedCap, cfg.forkSpeed);
            p.state = "FORK_ARMED";
            p.reason = direction < 0 ? "WAIT_LEFT_SPLIT" : "WAIT_RIGHT_SPLIT";
            if (newFrame && seen(LABEL_YLEFT) && seen(LABEL_YRIGHT)) {
                enter(State::Fault);
                reviewReason = "CONFLICTING_DIRECTIONS";
            } else if (newFrame && ((direction < 0 && seen(LABEL_YRIGHT)) ||
                                    (direction > 0 && seen(LABEL_YLEFT)))) {
                enter(State::Fault);
                reviewReason = "DIRECTION_CHANGED_BEFORE_SPLIT";
            } else {
                // Fresh same-direction observations renew memory, not the total
                // approach budget.
                if (newFrame &&
                    seen(direction < 0 ? LABEL_YLEFT : LABEL_YRIGHT))
                    directionUntil = now + cfg.directionMemory;
                if (now - stateSince > cfg.directionMemory + cfg.forkMaxTime) {
                    enter(State::Fault);
                    reviewReason = "FORK_NOT_CONFIRMED_TIMEOUT";
                } else if (now > directionUntil) {
                    enter(State::Fault);
                    reviewReason = "DIRECTION_MEMORY_EXPIRED";
                } else if (geometry.splitRows >= cfg.minBranchRows) {
                    splitTime += dt;
                    if (splitTime >= cfg.forkConfirmTime) {
                        enter(State::ForkSelect);
                        p.buzzer = true;
                    }
                } else
                    splitTime = 0;
            }
        }
        if (state == State::ForkSelect) {
            p.speedCap = std::min(p.speedCap, cfg.forkSpeed);
            p.state = direction < 0 ? "FORK_LEFT" : "FORK_RIGHT";
            p.reason = "CONNECTED_BRANCH";
            p.path =
                raceBranchPath(geometryBin, edges, direction, cfg, signRegions);
            p.selectedBranch = true;
            if (p.path.rows.size() < 8) {
                enter(State::Fault);
                reviewReason = "BRANCH_PATH_LOST";
            } else if (now - stateSince > cfg.forkMaxTime) {
                enter(State::Fault);
                reviewReason = "FORK_TIMEOUT";
            } else if (now - stateSince >= cfg.forkMinTime &&
                       geometry.splitRows == 0) {
                exitTime += dt;
                if (exitTime >= cfg.forkExitTime) {
                    enter(State::Lane);
                    cooldownUntil = now + cfg.cooldown;
                    p.selectedBranch = false;
                    p.path = {};
                }
            } else
                exitTime = 0;
        }
        if (state == State::Lane) {
            p.state = limited ? "LIMIT_ZONE" : "LANE";
            p.reason = limited ? "LIMIT_SIGN" : "CRUISE";
            if (geometry.splitRows >= cfg.minBranchRows &&
                now >= cooldownUntil) {
                p.stop = true;
                p.state = "FORK_WAIT";
                p.reason = "NO_CONFIRMED_DIRECTION";
            }
        }
        if (state == State::Fault) {
            p.stop = true;
            p.state = "REVIEW_STOP";
            p.reason = reviewReason;
        }
        return p;
    }
};
