#include "race_manager.hpp"
#include "detection.hpp"
#include "timing.hpp"

#include <algorithm>
#include <cmath>

using cv::Point2d;
using cv::Rect;
using std::min;
using std::vector;

namespace {

// 抬起的道闸在图像里至少这么高，才算真的抬起来了。
constexpr int GATE_OPEN_HEIGHT = 60;

inline Point2d center_coord(Rect r) {
    return {r.x + r.width / 2.0, r.y + r.height / 2.0};
}

inline double distance(Point2d a, Point2d b) {
    return hypot(a.x - b.x, a.y - b.y);
}

} // namespace

const InferenceResult &RaceManager::best_box(int label) const {
    // 这一帧没看到这个类别时返回一个全零框，调用方不用先判空
    static const InferenceResult none{};
    const auto &boxes = tracks[label].boxes;
    if (boxes.empty())
        return none;
    return *std::max_element(
        boxes.begin(), boxes.end(),
        [](const InferenceResult &a, const InferenceResult &b) {
            return a.score < b.score;
        });
}

void RaceManager::clear_observations() {
    for (auto &track : tracks)
        track = LabelTrack{};
}

bool RaceManager::update_observation(const DetectionSnapshot &snap) {
    for (auto &track : tracks)
        track.boxes.clear();
    for (const auto &d : snap.detections) {
        tracks[d.type].boxes.push_back(d);
    }
    for (auto &track : tracks) {
        track.confirmed = !track.boxes.empty();
    }
    return true;
}

RacePlan RaceManager::update(const DetectionSnapshot &snap, double dt,
                             bool active, const Camera &camera, bool aiming) {
    const double now = global_microsec() / 1000000.0;
    RacePlan p;
    p.detectionAge = now - double(snap.captured) / 1000000.0;
    const bool fresh = p.detectionAge <= cfg.maxDetectionAge;

    // 人工接管：丢掉确认结果方便重新开始；限速区状态必须保留，
    // 不能靠按一下 a 就取消限速。
    if (active != lastActive) {
        if (state != State::FatalStop)
            state = State::Lane;
        clear_observations();
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
        clear_observations();
        gateClearSince = hazardClearSince = -1;
        return p;
    }
    if (dt > 0.25f) {
        p.stop = true;
        p.state = "FRAME_WAIT";
        p.reason_for_stage = "FRAME_GAP";
        return p;
    }

    const bool is_new_observation = update_observation(snap);
    if (is_new_observation) {
        if (seen(LABEL_LIMIT))
            limited = true;
        else if (seen(LABEL_UNLIMIT))
            limited = false;
    }
    if (limited)
        p.speedCap = cfg.limitedSpeed;

    p.branchSide = 0;

    // -------- TARGET POINT SECTION

    // 停车场左转：choice（地面标线）和
    // park（立牌）同时看到时生成一串目标点。
    if (!aiming) {
        vector<Point2d> targets;
        if (seen(LABEL_CHOICE) && seen(LABEL_PARK)) {
            const auto &choice_boxes = boxes_of(LABEL_CHOICE);
            auto park_coord = center_coord(boxes_of(LABEL_PARK).front().rect);

            auto choice = *std::min_element(
                choice_boxes.begin(), choice_boxes.end(),
                [=](const InferenceResult &a, const InferenceResult &b) {
                    auto a_coord = center_coord(a.rect);
                    auto b_coord = center_coord(b.rect);
                    return distance(a_coord, park_coord) <
                           distance(b_coord, park_coord);
                });

            const cv::Point2d ground =
                camera.image_to_ground(center_coord(choice.rect));
            if (isfinite(ground.x) && isfinite(ground.y)) {
                targets = {{float(ground.x), float(ground.y - 0.2)},
                           {float(ground.x - 0.4), float(ground.y)}};
                ++targetSeq;
            }
        } else if (seen(LABEL_LEFT)) {
            auto coord = camera.image_to_ground(
                center_coord(boxes_of(LABEL_LEFT).front().rect));
            coord.x -= 0.2;
            if (isfinite(coord.x) && isfinite(coord.y)) {
                ++targetSeq;
                points = {coord};
            }
        }
        p.targets = points;
    }
    p.targetSeq = targetSeq;

    // 禁止通行牌：先开过斑马线（stopAfterCross），再去等绿灯。
    if (state == State::Lane && is_new_observation &&
        near(LABEL_STOP, cfg.triggerRow)) {
        crossBottomSeen = false;
        lastCrossSeen = now;
        state = cfg.stopAfterCross ? State::CrossApproach : State::SignalWait;
    }

    // 道闸优先于选路：绿色通行牌不能放行尚未抬起的杆。
    if (is_new_observation && near(LABEL_GATE, cfg.triggerRow) &&
        best_box(LABEL_GATE).rect.width >= best_box(LABEL_GATE).rect.height &&
        !gateWaiting) {
        gateBox = best_box(LABEL_GATE);
        gateWaiting = true;
        gateClearSince = -1;
    }
    if (gateWaiting && is_new_observation) {
        const auto &g = best_box(LABEL_GATE);
        const bool sameGate =
            std::abs((g.rect.x + g.rect.width / 2) -
                     (gateBox.rect.x + gateBox.rect.width / 2)) < 80;
        const bool open =
            seen(LABEL_GATE) && sameGate && g.rect.height >= GATE_OPEN_HEIGHT &&
            g.rect.width < float(g.rect.height) * cfg.gateOpenAspect;
        if (open) {
            if (gateClearSince < 0)
                gateClearSince = now;
        } else {
            gateClearSince = -1;
        }
        if (gateClearSince >= 0 && now - gateClearSince >= cfg.clearTime)
            gateWaiting = false;
    }

    if (gateWaiting) {
        // 停车叠在当前任务之上：不丢低优先级义务，只冻结它们的计时。
        if (state == State::CrossApproach)
            lastCrossSeen = now;
        p.stop = true;
        p.state = gateWaiting ? "GATE_WAIT" : "OBSTACLE_WAIT";
        p.reason_for_stage =
            gateWaiting ? "GATE_NOT_VERIFIED_OPEN" : "OBSTACLE_PLANNER_PENDING";
        return p;
    }

    // 斑马线：车已压过斑马线、且它消失 clearTime 之后才算通过。
    if (state == State::CrossApproach) {
        p.speedCap = min(p.speedCap, cfg.approachSpeed);
        p.state = "CROSS_APPROACH";
        p.reason_for_stage = "STOP_AFTER_CROSS";
        if (is_new_observation) {
            if (present(LABEL_CROSS)) {
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
