#include "race_manager.hpp"

#include <algorithm>
#include <vector>

using cv::Mat;
using cv::Rect;
using std::array;
using std::pair;
using std::vector;

namespace {

// 近处锥桶比行人更早触发：实测接近时会漏检，等满三帧确认已经太迟。
constexpr int CONE_TRIGGER_ROW = 290;
// 抬起的道闸在图像里至少这么高，才算真的抬起来了。
constexpr int GATE_OPEN_HEIGHT = 60;
// 斑马线的兜底超时：这么久还没确认开过去就交给人。
constexpr double CROSS_TIMEOUT = 6.0;
// 几何扫描的行步长。
constexpr int SCAN_STEP = 4;
// 牌面框外扩：留出检测抖动和异步帧位移的余量。
constexpr int SIGN_MARGIN = 24;

// 只有这些牌子的画面可能盖住路面：牌面上的深色图案不能当分隔带。
bool isSignLabel(int label) {
    switch (label) {
    case LABEL_BUSY:
    case LABEL_LIMIT:
    case LABEL_UNLIMIT:
    case LABEL_STOP:
    case LABEL_FLOW:
    case LABEL_PARK:
    case LABEL_FORK:
    case LABEL_LEFT:
    case LABEL_CHOICE:
    case LABEL_YLEFT:
    case LABEL_YRIGHT:
        return true;
    default:
        return false;
    }
}

// 交通牌属于场景中的物体，牌面上的深色图案不能充当道路分隔带。
// 只采用新鲜且达到阈值的有效检测框；行人、锥桶、道闸、斑马线不在此列。
vector<Rect> signRegions(const DetectionSnapshot &snap) {
    vector<Rect> regions;
    for (const auto &d : snap.detections) {
        if (!isSignLabel(d.type))
            continue;
        regions.push_back(cv::Rect(d.x - SIGN_MARGIN, d.y - SIGN_MARGIN,
                                   d.width + 2 * SIGN_MARGIN,
                                   d.height + 2 * SIGN_MARGIN) &
                          cv::Rect(0, 0, RACE_COLS, RACE_ROWS));
    }
    return regions;
}

// 仅保留与近处路面相接的白色区段，并把细小黑点和已检出的牌面合并回路面。
vector<pair<int, int>> roadBlocks(const Mat &bin, int row, int low, int high,
                                  const RaceConfig &cfg,
                                  const vector<Rect> &signs) {
    vector<pair<int, int>> blocks;
    for (int col = 0; col < bin.cols;) {
        if (bin.at<uint8_t>(row, col) <= 127) {
            ++col;
            continue;
        }
        const int first = col;
        while (col < bin.cols && bin.at<uint8_t>(row, col) > 127)
            ++col;
        if (col - first < cfg.minBranchWidth ||
            std::min(col - 1, high) < std::max(first, low))
            continue;
        if (!blocks.empty()) {
            const int gapFirst = blocks.back().second + 1, gapLast = first - 1;
            const int middle = (gapFirst + gapLast) / 2;
            const bool covered =
                std::any_of(signs.begin(), signs.end(), [&](const Rect &r) {
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

// 判断“黑色分隔带”是否真的从车头一路延伸到图像顶端：
//   * 每 SCAN_STEP 行取一次白色区段，分隔带必须一直存在、而且同一块；
//   * 分隔带数量变化 / 断开 / 图像断层都算不连续；
//   * 只在通过检查的那一段上按 direction 裁掉另一侧。
// 圆环、牌面里的白块、车头图案都会在某个行上断开或跳变，因而被排除。
RacePath branchPath(const Mat &bin, const TrackRows &edges, int direction,
                    const RaceConfig &cfg, const vector<Rect> &signs) {
    RacePath out;
    if (bin.empty() || bin.type() != CV_8UC1 || edges.empty())
        return out;
    const int start = edges.begin()->first;
    const int edgeLow = edges.begin()->second.first;
    const int edgeHigh = edges.begin()->second.second;
    if (start < 0 || start >= bin.rows || edgeLow < 0 || edgeHigh >= bin.cols ||
        edgeLow >= edgeHigh)
        return out;

    struct Scanned {
        int row;
        vector<pair<int, int>> blocks;
    };
    vector<Scanned> scanned;
    vector<pair<int, int>> previousGaps;
    int low = edgeLow, high = edgeHigh;
    int missing = 0, continuous = 0, forkStart = -1;
    bool rejected = false, reachedFarEnd = false;
    const char *rejection = "NO_CONTINUOUS_SPLIT";

    // 第一遍：逐行扫描，只判断“是不是一个连通到远处的分叉”。
    for (int row = start; row >= 0;) {
        auto blocks = roadBlocks(bin, row, low, high, cfg, signs);
        if (blocks.empty()) {
            // 图像断层不能拼成连续分叉，也不能宣称已看清远端。
            rejected = true;
            continuous = 0;
            rejection = "ROAD_ROWS_MISSING";
            if (++missing > 3)
                break;
        } else {
            missing = 0;
            vector<pair<int, int>> gaps;
            for (size_t i = 1; i < blocks.size(); ++i)
                gaps.emplace_back(blocks[i - 1].second + 1,
                                  blocks[i].first - 1);
            if (gaps.empty()) {
                // 本行是完整路面：之前的分叉结论作废，往上重新累计。
                continuous = 0;
                forkStart = -1;
                previousGaps.clear();
                rejected = false;
                rejection = "NO_CONTINUOUS_SPLIT";
            } else {
                if (!previousGaps.empty()) {
                    // 同一个分隔带不能突然变成圆环的两条边，也不能跳到别处的黑点。
                    if (gaps.size() != previousGaps.size()) {
                        rejected = true;
                        rejection = "DIVIDER_COUNT_CHANGED";
                    } else {
                        for (size_t i = 0; i < gaps.size(); ++i) {
                            const int middle = (previousGaps[i].first +
                                                previousGaps[i].second) /
                                               2;
                            if (middle < gaps[i].first ||
                                middle > gaps[i].second) {
                                rejected = true;
                                rejection = "DIVIDER_DISCONTINUOUS";
                            }
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
            // 搜索区逐行收窄到本行路面，避免远处的牌面把范围拉宽。
            low = blocks.front().first;
            high = blocks.back().second;
            if (row > cfg.nearCutRow)
                scanned.push_back({row, std::move(blocks)});
            reachedFarEnd = row == 0;
        }
        row = row == 0 ? -1 : std::max(0, row - SCAN_STEP);
    }
    if (reachedFarEnd && !rejected)
        out.splitRows = continuous;
    out.reason = !reachedFarEnd                       ? "FAR_END_NOT_VISIBLE"
                 : rejected                           ? rejection
                 : out.splitRows >= cfg.minBranchRows ? "CONNECTED_SPLIT"
                                                      : "NO_CONTINUOUS_SPLIT";

    // 第二遍：在扫过的行里挑一条通路。direction==0 时保持全宽，
    // 否则从分叉起点往上只保留选中那一侧（分叉点以前仍贴合近处路面）。
    const bool cut =
        direction != 0 && out.splitRows >= cfg.minBranchRows && forkStart >= 0;
    low = edgeLow;
    high = edgeHigh;
    missing = 0;
    for (const auto &s : scanned) {
        vector<pair<int, int>> blocks;
        for (const auto &b : s.blocks)
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
        if (cut && s.row <= forkStart)
            best = direction < 0 ? blocks.begin() : blocks.end() - 1;
        low = cut ? best->first : blocks.front().first;
        high = cut ? best->second : blocks.back().second;
        out.rows.emplace(s.row, pair<int, int>{low, high});
    }
    return out;
}

} // namespace

bool RaceManager::updateObservation(const DetectionSnapshot &snap) {
    if (snap.sequence <= lastSequence || snap.captured <= lastCapture)
        return false;
    // 太久没有新观测：旧的连续帧数不再有效。
    if (lastCapture < 0 || snap.captured - lastCapture > cfg.confirmGap)
        counts.fill(0);
    lastSequence = snap.sequence;
    lastCapture = snap.captured;

    array<bool, LABEL_COUNT> present{};
    for (const auto &d : snap.detections) {
        if (!present[d.type] || d.score > boxes[d.type].score)
            boxes[d.type] = d;
        present[d.type] = true;
    }
    for (int i = 0; i < LABEL_COUNT; ++i) {
        counts[i] = present[i] ? std::min(cfg.confirmFrames, counts[i] + 1) : 0;
        confirmed[i] = counts[i] >= cfg.confirmFrames;
    }
    return true;
}

RacePlan RaceManager::update(const DetectionSnapshot &snap, double time,
                             float dt, bool active, const Mat &bin,
                             const TrackRows &edges) {
    now = time;
    RacePlan p;
    p.detectionAge = snap.sequence ? time - snap.captured : -1;
    const bool fresh = aiReady(snap, time);

    if (active != lastActive) {
        if (state != State::Review) {
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
    if (!fresh) {
        p.stop = true;
        p.state = "AI_WAIT";
        p.reason = snap.failed ? "AI_FAILED" : "AI_STALE";
        counts.fill(0);
        confirmed.fill(false);
        splitTime = exitTime = 0;
        gateClearSince = hazardClearSince = -1;
        // 选路途中没有新观测：方向已经没有依据，直接交给人。
        if (state == State::ForkSelect) {
            reviewReason = "FORK_AI_STALE";
            enter(State::Review);
        }
        return p;
    }
    if (dt > 0.25f) {
        p.stop = true;
        p.state = "FRAME_WAIT";
        p.reason = "FRAME_GAP";
        return p;
    }

    const bool newFrame = updateObservation(snap);
    if (newFrame) {
        if (seen(LABEL_LIMIT))
            limited = true;
        else if (seen(LABEL_UNLIMIT))
            limited = false;
    }
    if (limited)
        p.speedCap = cfg.limitedSpeed;

    // 施工区和泊车没有规划器，只能停下来等人；不能只凭标签就说已完成。
    if (state != State::Review && newFrame &&
        (near(LABEL_PARK, cfg.triggerRow) ||
         near(LABEL_BUSY, cfg.triggerRow))) {
        reviewReason = seen(LABEL_PARK) ? "PARK_PLANNER_PENDING"
                                        : "CONSTRUCTION_PLANNER_PENDING";
        enter(State::Review);
    }
    if (state == State::Review) {
        p.stop = true;
        p.state = "REVIEW_STOP";
        p.reason = reviewReason;
        return p;
    }

    // 禁止通行牌：先开过斑马线（stopAfterCross），再去等绿灯。
    if (state == State::Lane && newFrame && near(LABEL_STOP, cfg.triggerRow)) {
        crossBottomSeen = false;
        lastCrossSeen = now;
        enter(cfg.stopAfterCross ? State::CrossApproach : State::SignalWait);
    }

    // 道闸优先于选路：绿色通行牌不能放行尚未抬起的杆。
    if (newFrame && near(LABEL_GATE, cfg.triggerRow) &&
        boxes[LABEL_GATE].width >= boxes[LABEL_GATE].height && !gateWaiting) {
        gateBox = boxes[LABEL_GATE];
        gateWaiting = true;
        gateClearSince = -1;
    }
    if (gateWaiting && newFrame) {
        const auto &g = boxes[LABEL_GATE];
        const bool sameGate = std::abs((g.x + g.width / 2) -
                                       (gateBox.x + gateBox.width / 2)) < 80;
        const bool open = seen(LABEL_GATE) && sameGate &&
                          g.height >= GATE_OPEN_HEIGHT &&
                          g.width < float(g.height) * cfg.gateOpenAspect;
        if (open) {
            if (gateClearSince < 0)
                gateClearSince = now;
        } else {
            gateClearSince = -1;
        }
        if (gateClearSince >= 0 && now - gateClearSince >= cfg.clearTime)
            gateWaiting = false;
    }

    // 近处锥桶不等三帧确认，而且没有绕行路线：看到就锁死停车。
    bool hazardInFrame = false;
    if (newFrame) {
        for (const auto &d : snap.detections) {
            if (d.type == LABEL_CONE && d.y + d.height >= CONE_TRIGGER_ROW)
                coneLatched = true;
            if (d.type == LABEL_PERSON || d.type == LABEL_CONE)
                hazardInFrame = true;
        }
        if (!hazardWaiting && (near(LABEL_PERSON, cfg.obstacleRow) ||
                               near(LABEL_CONE, cfg.obstacleRow))) {
            hazardWaiting = true;
            hazardClearSince = -1;
        }
    }
    // 障碍：单次漏检不能放行，连续 clearTime 的新鲜无障碍观测才恢复。
    if (hazardWaiting && newFrame) {
        if (hazardInFrame)
            hazardClearSince = -1;
        else if (hazardClearSince < 0)
            hazardClearSince = now;
        if (hazardClearSince >= 0 && now - hazardClearSince >= cfg.clearTime)
            hazardWaiting = false;
    }

    if (gateWaiting || hazardWaiting || coneLatched) {
        // 停车叠在当前任务之上：不丢低优先级义务，只冻结它们的计时。
        stateSince += dt;
        splitTime = exitTime = 0;
        if (state == State::CrossApproach)
            lastCrossSeen = now;
        p.stop = true;
        p.state = gateWaiting ? "GATE_WAIT" : "OBSTACLE_WAIT";
        p.reason = gateWaiting   ? "GATE_NOT_VERIFIED_OPEN"
                   : coneLatched ? "CONE_ROUTE_OR_RESTART_REQUIRED"
                                 : "OBSTACLE_PLANNER_PENDING";
        return p;
    }

    // 斑马线：车已压过斑马线、且它消失 clearTime 之后才算通过。
    if (state == State::CrossApproach) {
        p.speedCap = std::min(p.speedCap, cfg.approachSpeed);
        p.state = "CROSS_APPROACH";
        p.reason = "STOP_AFTER_CROSS";
        if (newFrame) {
            if (counts[LABEL_CROSS] > 0) {
                lastCrossSeen = now;
                if (seen(LABEL_CROSS) && near(LABEL_CROSS, cfg.crossPassedRow))
                    crossBottomSeen = true;
            } else if (crossBottomSeen &&
                       now - lastCrossSeen >= cfg.crossClearTime) {
                enter(State::SignalWait);
            }
        }
        if (state == State::CrossApproach && now - stateSince > CROSS_TIMEOUT) {
            reviewReason = "CROSS_PASS_NOT_VERIFIED";
            enter(State::Review);
        }
        return p;
    }
    if (state == State::SignalWait) {
        p.stop = true;
        p.state = "SIGNAL_WAIT";
        p.reason = "WAIT_FRESH_FLOW";
        // 绿灯（畅通牌）且禁止牌已消失，并至少等够 stopHoldTime。
        if (newFrame && seen(LABEL_FLOW) && counts[LABEL_STOP] == 0 &&
            now - stateSince >= cfg.stopHoldTime) {
            enter(State::Lane);
            cooldownUntil = now + cfg.cooldown;
            p.stop = false;
            p.state = limited ? "LIMIT_ZONE" : "LANE";
            p.reason = limited ? "LIMIT_SIGN" : "CRUISE";
        }
        return p;
    }

    // 一次扫描同时给出分叉判定和（方向已知时的）支路路径。
    const auto signs = signRegions(snap);
    RacePath scan = branchPath(bin, edges, direction, cfg, signs);
    p.path.splitRows = scan.splitRows;
    p.path.reason = scan.reason;

    // 方向牌：左右同时出现视为没有方向；只有单侧才进入岔路流程。
    if (state == State::Lane && now >= cooldownUntil && newFrame) {
        if (seen(LABEL_YLEFT) && seen(LABEL_YRIGHT)) {
            direction = 0;
        } else if (seen(LABEL_YLEFT) || seen(LABEL_YRIGHT)) {
            direction = seen(LABEL_YLEFT) ? -1 : 1;
            directionUntil = now + cfg.directionMemory;
            // enter(State::ForkArmed); DEBUG!!!
        }
    }
    if (state == State::ForkArmed) {
        p.speedCap = std::min(p.speedCap, cfg.forkSpeed);
        p.state = "FORK_ARMED";
        p.reason = direction < 0 ? "WAIT_LEFT_SPLIT" : "WAIT_RIGHT_SPLIT";
        if (newFrame && seen(LABEL_YLEFT) && seen(LABEL_YRIGHT)) {
            reviewReason = "CONFLICTING_DIRECTIONS";
            enter(State::Review);
        } else if (newFrame && ((direction < 0 && seen(LABEL_YRIGHT)) ||
                                (direction > 0 && seen(LABEL_YLEFT)))) {
            reviewReason = "DIRECTION_CHANGED_BEFORE_SPLIT";
            enter(State::Review);
        } else {
            // 同向的新观测只续记忆，不重置接近预算。
            if (newFrame && seen(direction < 0 ? LABEL_YLEFT : LABEL_YRIGHT))
                directionUntil = now + cfg.directionMemory;
            if (now - stateSince > cfg.directionMemory + cfg.forkMaxTime) {
                reviewReason = "FORK_NOT_CONFIRMED_TIMEOUT";
                enter(State::Review);
            } else if (now > directionUntil) {
                reviewReason = "DIRECTION_MEMORY_EXPIRED";
                enter(State::Review);
            } else if (scan.splitRows >= cfg.minBranchRows) {
                splitTime += dt;
                if (splitTime >= cfg.forkConfirmTime) {
                    enter(State::ForkSelect);
                    p.buzzer = true;
                }
            } else {
                splitTime = 0;
            }
        }
    }
    if (state == State::ForkSelect) {
        p.speedCap = std::min(p.speedCap, cfg.forkSpeed);
        p.state = direction < 0 ? "FORK_LEFT" : "FORK_RIGHT";
        p.reason = "CONNECTED_BRANCH";
        if (scan.rows.size() < size_t(cfg.minBranchPathRows)) {
            reviewReason = "BRANCH_PATH_LOST";
            enter(State::Review);
        } else if (now - stateSince > cfg.forkMaxTime) {
            reviewReason = "FORK_TIMEOUT";
            enter(State::Review);
        } else {
            p.selectedBranch = true;
            p.path.rows = std::move(scan.rows);
            // 分叉消失并持续 forkExitTime：说明已经开出去了。
            if (now - stateSince >= cfg.forkMinTime && scan.splitRows == 0) {
                exitTime += dt;
                if (exitTime >= cfg.forkExitTime) {
                    p.selectedBranch = false;
                    p.path.rows.clear();
                    enter(State::Lane);
                    cooldownUntil = now + cfg.cooldown;
                }
            } else {
                exitTime = 0;
            }
        }
    }
    if (state == State::Lane) {
        p.state = limited ? "LIMIT_ZONE" : "LANE";
        p.reason = limited ? "LIMIT_SIGN" : "CRUISE";
        // 看到岔路却没有确认方向：不许自己猜，停在岔口前等人。
        if (scan.splitRows >= cfg.minBranchRows && now >= cooldownUntil) {
            p.stop = true;
            p.state = "FORK_WAIT";
            p.reason = "NO_CONFIRMED_DIRECTION";
        }
    }
    if (state == State::Review) {
        p.stop = true;
        p.state = "REVIEW_STOP";
        p.reason = reviewReason;
    }
    return p;
}
