#include "race_manager.hpp"
#include "detection.hpp"
#include "timing.hpp"

#include <algorithm>
#include <cmath>

using cv::Point2d;
using cv::Rect;
using std::cout;
using std::endl;
using std::vector;

namespace {

// Y 形岔路：脉冲式引导（数值照着分支的实测来）
constexpr double YFORK_STRONG_TIME = 0.40; // 强打舵持续多久（秒）
constexpr double YFORK_RAMP_TIME = 1.00;   // 之后线性回中所用的时间（秒）
constexpr float YFORK_PWM = 400.f;         // 脉冲幅度（PWM 偏离中值）
constexpr float YFORK_REF_SPEED = 0.20f;   // 脉冲相位的参考速度
constexpr float YFORK_MIN_DISTANCE = 0.30f; // 至少走这么远（米）
constexpr double YFORK_CLEAR_TIME = 0.60;   // 牌子消失这么久
constexpr double YFORK_TRACK_TIME = 0.60;   // 新路跟住这么久

/**
 * @brief 引导脉冲幅度
 *
 * Y 岔路比停车场入口浅：先强打方向，再线性松开，一直满舵会绕圈转过头。
 * 0~0.4 s: 360→400；0.4~1.4 s: 400→0。
 */
float guidance_pwm(double elapsed) {
    if (elapsed < YFORK_STRONG_TIME)
        return 360.f +
               40.f * float(std::clamp(elapsed / 0.12, 0.0, 1.0));
    return YFORK_PWM * float(
                          1.0 - std::clamp((elapsed - YFORK_STRONG_TIME) /
                                               YFORK_RAMP_TIME,
                                           0.0, 1.0));
}

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
                             bool active, const Camera &camera, bool aiming,
                             const RaceFeedback &feedback) {
    const double now = global_microsec() / 1000000.0;
    RacePlan p;
    p.detectionAge = now - double(snap.captured) / 1000000.0;
    const bool fresh = p.detectionAge <= cfg.maxDetectionAge;

    if (active != lastActive) {
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

    update_observation(snap);

    p.branchSide = 0;

    // -------- Y 形岔路 SECTION
    // 照分支的做法：限速 + 脉冲式强转向；转完之后用
    // （脉冲相位 / 行进距离 / 牌子消失时长 / 新路跟踪时长）判断这段走完。
    if (!yfork && (seen(LABEL_YLEFT) || seen(LABEL_YRIGHT))) {
        yfork = true;
        yfork_dir = seen(LABEL_YLEFT) ? -1 : 1;
        yfork_elapsed = yfork_phase = yfork_clear = yfork_track = 0;
        yfork_distance = 0;
        cout << "[YFork] 看到 " << (yfork_dir < 0 ? "yleft" : "yright")
             << "，脉冲引导转向" << endl;
    }
    if (yfork) {
        yfork_elapsed += dt;
        // 相位按速度推进：慢下来时脉冲也会拖长，但最多 2.8 s
        const float speed =
            feedback.speed > 0 ? std::min(feedback.speed, 1.f) : 0.f;
        yfork_phase = std::max(
            yfork_phase + dt * std::min(double(speed / YFORK_REF_SPEED), 1.0),
            yfork_elapsed * 0.5);
        yfork_distance += std::fabs(feedback.distance);
        if (seen(LABEL_YLEFT) || seen(LABEL_YRIGHT)) {
            yfork_clear = yfork_track = 0; // 牌子还在：重新计时
        } else {
            yfork_clear += dt;
            yfork_track = (yfork_phase >= 1.4 && feedback.tracking)
                              ? yfork_track + dt
                              : 0;
        }
        const bool done = yfork_phase >= 1.4 &&
                          yfork_distance >= YFORK_MIN_DISTANCE &&
                          yfork_clear >= YFORK_CLEAR_TIME &&
                          yfork_track >= YFORK_TRACK_TIME;
        if (done) {
            yfork = false;
            cout << "[YFork] 岔路完成" << endl;
        } else {
            p.guideDirection = yfork_dir;
            p.guidePwm = guidance_pwm(yfork_elapsed);
            // 限速：和分支一致——在停车场区域里用停车场速度当上限，
            // 否则用岔路速度。
            const float fork_speed =
                driving_on_park_lane ? cfg.targetSpeed : cfg.yforkSpeed;
            p.speedCap = std::min(p.speedCap, fork_speed);
        }
    }

    // -------- TARGET POINT SECTION
    // 走目标点期间限速
    if (aiming)
        p.speedCap = std::min(p.speedCap, cfg.targetSpeed);

    // 停车场左转：choice（地面标线）和
    // park（立牌）同时看到时生成一串目标点。
    if (!aiming) {
        if (seen(LABEL_CHOICE) && seen(LABEL_PARK)) {
            driving_on_park_lane = true;
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
                cout << "[Target] 看到停车场，准备左转" << endl;
                // 第 1 个点直着走到库口；第 2 个点沿左侧圆弧进库（到达时朝左）
                p.targets = {{{ground.x, ground.y - 0.2}, {0, 1}},
                             {{ground.x - 0.5, ground.y}, {-1, 0}}};
                ++target_seq;
            }
        } else if (seen(LABEL_LEFT)) {
            auto ground = camera.image_to_ground(
                center_coord(boxes_of(LABEL_LEFT).front().rect));
            if (isfinite(ground.x) && isfinite(ground.y)) {
                if (ground.y < 0.4) {
                    driving_on_park_lane = false;
                    cout << "[Target] 看到左转标识，准备左转离开停车场道"
                         << endl;
                    p.targets = {{{ground.x - 0.5, ground.y}, {-1, 0}}};
                    ++target_seq;
                }
            }
        }
    }
    p.target_seq = target_seq;

    if (fatal_stop) {
        p.stop = true;
        p.state = "REVIEW_STOP";
        p.reason_for_stage = reason_for_review;
    }
    return p;
}
