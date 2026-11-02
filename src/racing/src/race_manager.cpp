#include "race_manager.hpp"
using cv::Mat;
using cv::Rect;
using std::pair;
using std::vector;

vector<pair<int, int>> raceRoadBlocks(const Mat &bin, int row, int low,
                                      int high, const RaceConfig &cfg,
                                      const vector<Rect> &signRegions,
                                      int minimumBlockWidth) {
    vector<pair<int, int>> blocks;
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
RacePath raceBranchPath(const cv::Mat &bin, const RaceRows &edges,
                        int direction, const RaceConfig &cfg,
                        const std::vector<cv::Rect> &signRegions) {
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
        vector<std::pair<int, int>> blocks;
    };
    vector<Row> rows;
    vector<pair<int, int>> previousGaps;
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
        vector<pair<int, int>> gaps;
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
        vector<pair<int, int>> blocks;
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
                return abs(a.first + a.second - low - high) <
                       abs(b.first + b.second - low - high);
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

RacePlan RaceManager::update(const DetectionSnapshot &snap, double time,
                             float dt, bool active, const cv::Mat &bin,
                             const RaceRows &edges) {
    now = time;
    RacePlan p;
    p.detectionAge = snap.sequence ? time - snap.captured : -1;
    p.fresh = aiReady(snap, time);
    const bool newFrame = updateObservation(snap, p.fresh);
    const auto signRegions =
        p.fresh ? raceSignRegions(snap, cfg) : std::vector<cv::Rect>{};
    auto geometry = raceBranchPath(bin, edges, 0, cfg, signRegions);
    if (p.colorRejectedComponents)
        geometry = raceBranchPath(bin, edges, 0, cfg, signRegions);
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
    if (state == State::Lane && newFrame && near(LABEL_STOP, cfg.triggerRow)) {
        crossBottomSeen = false;
        lastCrossSeen = now;
        enter(cfg.stopAfterCross ? State::CrossApproach : State::SignalWait);
    }
    // 道闸优先于路线选择；绿色通行牌不能放行尚未抬起的杆。
    if (newFrame && near(LABEL_GATE, cfg.triggerRow) &&
        boxes[LABEL_GATE].width >= boxes[LABEL_GATE].height && !gateWaiting) {
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
            const bool open = seen(LABEL_GATE) && sameArea && g.height >= 60 &&
                              g.width < float(g.height) * cfg.gateOpenAspect;
            if (open) {
                if (gateClearSince < 0)
                    gateClearSince = now;
            } else
                gateClearSince = -1;
            if (gateClearSince >= 0 && now - gateClearSince >= cfg.clearTime)
                gateWaiting = false;
        }
    }

    // 近桶停车不等三次确认：真实记录曾在接近时漏检，等重新凑满三帧已太迟。
    // 已看见的近桶在没有已验证绕行路线时保持停车，不能靠0.6秒漏检消除义务。
    const bool nearCone =
        newFrame && std::any_of(snap.detections.begin(), snap.detections.end(),
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
            const bool any = counts[LABEL_PERSON] > 0 || hazardConeWaiting ||
                             (counts[LABEL_CONE] > 0);
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
                if (seen(LABEL_CROSS) && near(LABEL_CROSS, cfg.crossPassedRow))
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
            if (newFrame && seen(direction < 0 ? LABEL_YLEFT : LABEL_YRIGHT))
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
        p.path = raceBranchPath(bin, edges, direction, cfg, signRegions);
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
        if (geometry.splitRows >= cfg.minBranchRows && now >= cooldownUntil) {
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