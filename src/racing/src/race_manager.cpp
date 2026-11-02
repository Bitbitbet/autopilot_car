#include "race_manager.hpp"

#include <algorithm>

using std::array;

namespace {

// 近处锥桶比行人更早触发：实测接近时会漏检，等满三帧确认已经太迟。
constexpr int CONE_TRIGGER_ROW = 290;
// 抬起的道闸在图像里至少这么高，才算真的抬起来了。
constexpr int GATE_OPEN_HEIGHT = 60;
// 斑马线的兜底超时：这么久还没确认开过去就交给人。
constexpr double CROSS_TIMEOUT = 6.0;

// 岔路偏向在牌子消失后保持多久（抗单帧漏检）。
constexpr double SIDE_HOLD_TIME = 0.5;

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
                             float dt, bool active) {
    now = time;
    RacePlan p;
    p.detectionAge = snap.sequence ? time - snap.captured : -1;
    const bool fresh = aiReady(snap, time);

    // 人工接管：丢掉确认结果方便重新开始；限速区状态必须保留，
    // 不能靠按一下 a 就取消限速。
    if (active != lastActive) {
        if (state != State::FatalStop)
            enter(State::Lane);
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
        p.reason = "AI_STALE";
        counts.fill(0);
        confirmed.fill(false);
        gateClearSince = hazardClearSince = -1;
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

    // 岔路偏向：地面上的 fork / choice / park 都靠左；前方 yleft / yright
    // 指定一边，一直保持到那块牌子消失（再留 SIDE_HOLD_TIME 抗漏检）。
    // 左右 Y 牌不会同时出现，所以先看到哪个就按哪个。
    if (newFrame) {
        if (seen(LABEL_YRIGHT)) {
            side = 1;
            sideUntil = now + SIDE_HOLD_TIME;
        } else if (seen(LABEL_YLEFT)) {
            side = -1;
            sideUntil = now + SIDE_HOLD_TIME;
        } else if (seen(LABEL_FORK) || seen(LABEL_CHOICE) || seen(LABEL_PARK)) {
            side = -1;
            sideUntil = now + SIDE_HOLD_TIME;
        } else if (now > sideUntil) {
            side = 0;
        }
    }
    p.branchSide = side;

    if (state == State::FatalStop) {
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
            enter(State::FatalStop);
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
            p.stop = false;
            p.state = limited ? "LIMIT_ZONE" : "LANE";
            p.reason = limited ? "LIMIT_SIGN" : "CRUISE";
        }
        return p;
    }

    if (state == State::Lane) {
        p.state = limited ? "LIMIT_ZONE" : "LANE";
        p.reason = limited ? "LIMIT_SIGN" : "CRUISE";
    }
    if (state == State::FatalStop) {
        p.stop = true;
        p.state = "REVIEW_STOP";
        p.reason = reviewReason;
    }
    return p;
}
