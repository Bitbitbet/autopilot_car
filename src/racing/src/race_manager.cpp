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

    p.branchSide = 0;

    // -------- TARGET POINT SECTION

    // 停车场左转：choice（地面标线）和
    // park（立牌）同时看到时生成一串目标点。
    if (!aiming) {
        vector<Point2d> targets;
        if (seen(LABEL_CHOICE) && seen(LABEL_PARK)) {
            driving_on_park_lane = true;
            cout << "[Target] 看到停车场，准备左转" << endl;
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
                targets = {{ground.x, ground.y - 0.3},
                           {ground.x - 0.4, ground.y}};
                ++targetSeq;
            }
        } else if (seen(LABEL_LEFT) && driving_on_park_lane) {
            auto ground = camera.image_to_ground(
                center_coord(boxes_of(LABEL_LEFT).front().rect));
            if (ground.y < 0.4) {
                driving_on_park_lane = false;
                cout << "[Target] 看到左转标识，准备左转" << endl;
                if (isfinite(ground.x) && isfinite(ground.y)) {
                    targets = {{ground.x - 0.2, ground.y}};
                    ++targetSeq;
                }
            }
        }
        p.targets = targets;
    }
    p.targetSeq = targetSeq;

    if (state == State::FatalStop) {
        p.stop = true;
        p.state = "REVIEW_STOP";
        p.reason_for_stage = reviewReason;
    }
    return p;
}
