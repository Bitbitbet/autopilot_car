#include "race_manager.hpp"
#include "timing.hpp"

#include <algorithm>

using std::array;
using std::min;

namespace {

// 近处锥桶比行人更早触发：实测接近时会漏检，等满三帧确认已经太迟。
constexpr int CONE_TRIGGER_ROW = 290;
// 抬起的道闸在图像里至少这么高，才算真的抬起来了。
constexpr int GATE_OPEN_HEIGHT = 60;

// 岔路偏向在牌子消失后保持多久（抗单帧漏检）。
constexpr double SIDE_HOLD_TIME = 0.5;

} // namespace

bool RaceManager::updateObservation(const DetectionSnapshot &snap) {
    if (snap.captured <= lastCapture)
        return false;
    // 太久没有新观测：旧的连续帧数不再有效。
    if (lastCapture < 0 || snap.captured - lastCapture > cfg.confirmGap)
        counts.fill(0);
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

RacePlan RaceManager::update(const DetectionSnapshot &snap, float dt,
                             bool active) {
    const uint64_t now = global_microsec();
    RacePlan p;
    p.detectionAge = now - snap.captured;
    const bool fresh = p.detectionAge <= cfg.maxDetectionAge;

    // 人工接管：丢掉确认结果方便重新开始；限速区状态必须保留，
    // 不能靠按一下 a 就取消限速。
    if (active != lastActive) {
        if (state != State::FatalStop)
            state = State::Lane;
        counts.fill(0);
        confirmed.fill(false);
        lastActive = active;
    }
    if (!active) {
        p.state = "MANUAL";
        p.reason_for_stage = "MANUAL";
        return p;
    }
    if (!fresh) {
        p.stop = true;
        p.state = "AI_WAIT";
        p.reason_for_stage = "AI_STALE";
        counts.fill(0);
        confirmed.fill(false);
        gateClearSince = hazardClearSince = -1;
        return p;
    }
    if (dt > 0.25f) {
        p.stop = true;
        p.state = "FRAME_WAIT";
        p.reason_for_stage = "FRAME_GAP";
        return p;
    }

    const bool is_new_observation = updateObservation(snap);
    if (is_new_observation) {
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
    if (is_new_observation) {
        if (seen(LABEL_YRIGHT)) {
            side = 1;
            sideUntil = now + SIDE_HOLD_TIME;
        } else if (seen(LABEL_YLEFT)) {
            side = -1;
            sideUntil = now + SIDE_HOLD_TIME;
        } else if (seen(LABEL_FORK)) {
            side = -1;
            sideUntil = now + SIDE_HOLD_TIME;
        } else if (now > sideUntil) {
            side = 0;
        }
    }
    p.branchSide = side;

    // 禁止通行牌：先开过斑马线（stopAfterCross），再去等绿灯。
    if (state == State::Lane && is_new_observation &&
        near(LABEL_STOP, cfg.triggerRow)) {
        crossBottomSeen = false;
        lastCrossSeen = now;
        state = cfg.stopAfterCross ? State::CrossApproach : State::SignalWait;
    }

    // 道闸优先于选路：绿色通行牌不能放行尚未抬起的杆。
    if (is_new_observation && near(LABEL_GATE, cfg.triggerRow) &&
        boxes[LABEL_GATE].width >= boxes[LABEL_GATE].height && !gateWaiting) {
        gateBox = boxes[LABEL_GATE];
        gateWaiting = true;
        gateClearSince = -1;
    }
    if (gateWaiting && is_new_observation) {
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
    if (is_new_observation) {
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
    if (hazardWaiting && is_new_observation) {
        if (hazardInFrame)
            hazardClearSince = -1;
        else if (hazardClearSince < 0)
            hazardClearSince = now;
        if (hazardClearSince >= 0 && now - hazardClearSince >= cfg.clearTime)
            hazardWaiting = false;
    }

    if (gateWaiting || hazardWaiting || coneLatched) {
        // 停车叠在当前任务之上：不丢低优先级义务，只冻结它们的计时。
        if (state == State::CrossApproach)
            lastCrossSeen = now;
        p.stop = true;
        p.state = gateWaiting ? "GATE_WAIT" : "OBSTACLE_WAIT";
        p.reason_for_stage = gateWaiting   ? "GATE_NOT_VERIFIED_OPEN"
                             : coneLatched ? "CONE_ROUTE_OR_RESTART_REQUIRED"
                                           : "OBSTACLE_PLANNER_PENDING";
        return p;
    }

    // 斑马线：车已压过斑马线、且它消失 clearTime 之后才算通过。
    if (state == State::CrossApproach) {
        p.speedCap = min(p.speedCap, cfg.approachSpeed);
        p.state = "CROSS_APPROACH";
        p.reason_for_stage = "STOP_AFTER_CROSS";
        if (is_new_observation) {
            if (counts[LABEL_CROSS] > 0) {
                lastCrossSeen = now;
                if (seen(LABEL_CROSS) && near(LABEL_CROSS, cfg.crossPassedRow))
                    crossBottomSeen = true;
            } else if (crossBottomSeen &&
                       now - lastCrossSeen >= cfg.crossClearTime) {
                state = State::SignalWait;
            }
        }
    }
    if (state == State::SignalWait) {
        p.stop = true;
        p.state = "SIGNAL_WAIT";
        p.reason_for_stage = "WAIT_FRESH_FLOW";
    } else if (state == State::Lane) {
        p.state = limited ? "LIMIT_ZONE" : "LANE";
        p.reason_for_stage = limited ? "LIMIT_SIGN" : "CRUISE";
    } else if (state == State::FatalStop) {
        p.stop = true;
        p.state = "REVIEW_STOP";
        p.reason_for_stage = reviewReason;
    }
    return p;
}
