#include "odometry.hpp"

#include <numbers>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

using cv::Mat;
using cv::Point;
using cv::Point2f;
using cv::Rect;
using cv::Size;
using std::isfinite;
using std::vector;
using std::numbers::pi;

namespace {

const Size WIN{21, 21};

} // namespace

void GroundOdometry::reset() {
    pose_ = {};
    prev_pyr_.clear();
    prev_pts_.clear();
    valid_ = false;
    inliers_ = 0;
    speed_ = 0;
}

void GroundOdometry::detect(const Mat &roi) {
    prev_pts_.clear();
    // 半分辨率提点：goodFeaturesToTrack 的开销主要在整幅求特征值和排序，
    // 缩一半就是四分之一；坐标乘回去，光流照样从这几个像素开始。
    Mat small, small_mask;
    resize(roi, small, Size(0, 0), 0.5, 0.5, cv::INTER_AREA);
    resize(mask_, small_mask, small.size(), 0, 0, cv::INTER_NEAREST);
    vector<Point2f> points;
    // 门槛压低：黑赛道上对比度弱，能抓一点是一点
    cv::goodFeaturesToTrack(small, points, WANT_POINTS, 0.01, 3, small_mask);
    prev_pts_.reserve(points.size());
    for (const auto &p : points)
        prev_pts_.emplace_back(p.x * 2.f, p.y * 2.f);
}

void GroundOdometry::update(const Mat &grey, const Camera &camera, double dt,
                            const vector<Rect> &blockers) {
    valid_ = false;
    const Rect roi_rect(ROI_LEFT, ROI_TOP, ROI_W, ROI_H);
    const Mat roi = grey(roi_rect);

    // 掩码：先全开，再把“立在地面上”的物体挖掉（它们的像不跟地面一起动）。
    mask_.setTo(255);
    for (const auto &b : blockers) {
        const Rect box = (b + Point(-BLOCK_MARGIN, -BLOCK_MARGIN) +
                          Size(2 * BLOCK_MARGIN, 2 * BLOCK_MARGIN)) &
                         roi_rect;
        if (box.width > 0 && box.height > 0)
            mask_(box - roi_rect.tl()).setTo(0);
    }

    // 金字塔每帧只建一次：正向、反向光流共用同一对。
    vector<Mat> pyramid;
    cv::buildOpticalFlowPyramid(roi, pyramid, WIN, MAX_LEVEL);

    // 第一帧、点数不够、或者帧间隔不连续：只重新提点，不改位姿。
    if (prev_pyr_.empty() || prev_pts_.size() < MIN_TRACKED || dt > MAX_GAP) {
        detect(roi);
        prev_pyr_.swap(pyramid);
        return;
    }

    vector<Point2f> tracked;
    vector<uchar> status;
    vector<float> error;
    cv::calcOpticalFlowPyrLK(prev_pyr_, pyramid, prev_pts_, tracked, status,
                             error, WIN, MAX_LEVEL);
    // 反向再跟一次，来回对不上的是跟丢了。
    vector<Point2f> back;
    vector<uchar> back_status;
    cv::calcOpticalFlowPyrLK(pyramid, prev_pyr_, tracked, back, back_status,
                             error, WIN, MAX_LEVEL);

    vector<Point2f> from, to, kept;
    for (size_t i = 0; i < prev_pts_.size(); ++i) {
        if (!status[i] || !back_status[i])
            continue;
        const float fb = float(
            std::hypot(back[i].x - prev_pts_[i].x, back[i].y - prev_pts_[i].y));
        if (fb > FB_ERROR)
            continue;
        const Point2f origin(float(roi_rect.x), float(roi_rect.y));
        const Point2f a = camera.image_to_ground(prev_pts_[i] + origin);
        const Point2f b = camera.image_to_ground(tracked[i] + origin);
        if (!isfinite(a.x) || !isfinite(a.y) || !isfinite(b.x) ||
            !isfinite(b.y))
            continue;
        from.push_back(a);
        to.push_back(b);
        kept.push_back(tracked[i]);
    }
    prev_pts_.swap(kept);
    prev_pyr_.swap(pyramid);
    if (from.size() < MIN_INLIERS) {
        prev_pts_.clear();
        return;
    }

    Mat inliers;
    const Mat motion = cv::estimateAffinePartial2D(
        from, to, inliers, cv::RANSAC, RANSAC_TOLERANCE, 500, 0.99);
    if (motion.empty() || inliers.empty())
        return;
    inliers_ = cv::countNonZero(inliers);
    if (inliers_ < MIN_INLIERS) {
        prev_pts_.clear();
        return;
    }
    const auto at = [&](int row, int col) {
        return float(motion.at<double>(row, col));
    };
    const float scale = std::hypot(at(0, 0), at(1, 0));
    const float turn = std::atan2(at(1, 0), at(0, 0));
    const Point2f step(at(0, 2), at(1, 2));
    const float moved = std::hypot(step.x, step.y);
    // 尺度跑偏、单帧位移/转角离谱都当作拟合失败，位姿保持不动。
    if (fabs(scale - 1.f) > SCALE_LIMIT || moved > MAX_STEP ||
        fabs(turn) > MAX_TURN) {
        prev_pts_.clear();
        return;
    }

    // 拟合给的是 车体系新 = R(turn) * 车体系旧 + step，
    // 而位姿定义是 世界 = R(theta) * 车体系 + t，所以反过来解。
    pose_.theta = wrap(pose_.theta - turn);
    const float c = std::cos(pose_.theta), s = std::sin(pose_.theta);
    pose_.x -= c * step.x - s * step.y;
    pose_.y -= s * step.x + c * step.y;
    speed_ = float(moved / dt);
    valid_ = true;
}

float GroundOdometry::wrap(float angle) {
    while (angle > pi)
        angle -= 2 * pi;
    while (angle < -pi)
        angle += 2 * pi;
    return angle;
}
