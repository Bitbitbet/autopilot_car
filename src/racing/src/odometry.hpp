#pragma once

#include "camera.hpp"
#include <algorithm>
#include <cmath>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <vector>

// 车在地面系里的位姿：世界点 = R(theta) * 车体系点 + (x, y)。
struct GroundPose {
    float x = 0, y = 0, theta = 0;

    cv::Point2f world(const cv::Point2f &c) const {
        const float s = std::sin(theta), k = std::cos(theta);
        return {x + k * c.x - s * c.y, y + s * c.x + k * c.y};
    }
    cv::Point2f local(const cv::Point2f &w) const {
        const float s = std::sin(theta), k = std::cos(theta);
        const float dx = w.x - x, dy = w.y - y;
        return {k * dx + s * dy, -s * dx + k * dy};
    }
};

/**
 * @brief 地面视觉里程计：用地面纹理估相机自己走了多少、转了多少
 *
 * 连续两帧跟踪同一批地面角点，把两组点都换到地面坐标再拟合 2D 刚体变换
 * （旋转 + 平移 + 尺度）：转向本来就在模型里，所以急转也不怕；尺度必须
 * 接近 1，否则说明拟合被俯仰、假特征或标定漂移带歪了，这一帧作废。
 *
 * 性能：只在近场 ROI（约 608x270）上做灰度、金字塔和角点，金字塔每帧
 * 只建一次（正向反向光流共用同一对），一帧的量级是几毫秒。
 */
class GroundOdometry {
  public:
    void reset() {
        pose_ = {};
        prev_pyr_.clear();
        prev_pts_.clear();
        valid_ = false;
        inliers_ = 0;
        speed_ = 0;
    }

    /**
     * @brief 每帧调用一次
     * @param image    与二值图同源的那一帧彩色图
     * @param camera   提供 image_to_ground
     * @param dt       距上一帧的秒数；不连续时这一帧只重新提点
     * @param blockers 需要剔除的矩形（检测框等），可以为空
     */
    void update(const cv::Mat &image, const Camera &camera, float dt,
                const std::vector<cv::Rect> &blockers = {}) {
        step(image, camera, dt, blockers);
    }

    const GroundPose &pose() const { return pose_; }
    bool valid() const { return valid_; }
    int inliers() const { return inliers_; }
    int tracked() const { return int(prev_pts_.size()); }
    float speed() const { return speed_; } // 最近一次拟合出来的地面速度（m/s）

  private:
    void step(const cv::Mat &image, const Camera &camera, float dt,
              const std::vector<cv::Rect> &blockers) {
        valid_ = false;
        if (image.empty() || image.type() != CV_8UC3) {
            drop();
            return;
        }
        if (image.size() != frame_size_)
            setup(image.size());

        // 只对近场 ROI 做灰度：远景分辨率低、偏航流量大，还白费算力。
        cv::Mat gray;
        cv::cvtColor(image(roi_), gray, cv::COLOR_BGR2GRAY);
        mask_.setTo(255);
        for (const auto &b : blockers) {
            const cv::Rect box =
                (b + cv::Point(-BLOCK_MARGIN, -BLOCK_MARGIN) +
                 cv::Size(2 * BLOCK_MARGIN, 2 * BLOCK_MARGIN)) &
                roi_;
            if (box.width > 0 && box.height > 0)
                mask_(box - roi_.tl()).setTo(0);
        }

        // 金字塔每帧只建一次：正向、反向光流共用同一对。
        std::vector<cv::Mat> pyramid;
        cv::buildOpticalFlowPyramid(gray, pyramid, WIN, MAX_LEVEL);

        // 第一帧、点数不够、或者帧间隔不连续：只重新提点，不改位姿。
        if (prev_pyr_.empty() || prev_pts_.size() < MIN_TRACKED || !(dt > 0) ||
            dt > MAX_GAP) {
            detect(gray);
            prev_pyr_.swap(pyramid);
            return;
        }

        std::vector<cv::Point2f> tracked;
        std::vector<uchar> status;
        std::vector<float> error;
        cv::calcOpticalFlowPyrLK(prev_pyr_, pyramid, prev_pts_, tracked, status,
                                 error, WIN, MAX_LEVEL);
        // 反向再跟一次，来回对不上的是跟丢了。
        std::vector<cv::Point2f> back;
        std::vector<uchar> back_status;
        cv::calcOpticalFlowPyrLK(pyramid, prev_pyr_, tracked, back, back_status,
                                 error, WIN, MAX_LEVEL);

        std::vector<cv::Point2f> from, to, kept;
        for (size_t i = 0; i < prev_pts_.size(); ++i) {
            if (!status[i] || !back_status[i])
                continue;
            const float fb = float(std::hypot(back[i].x - prev_pts_[i].x,
                                              back[i].y - prev_pts_[i].y));
            if (fb > FB_ERROR)
                continue;
            const cv::Point2f origin(float(roi_.x), float(roi_.y));
            const cv::Point2f a = camera.image_to_ground(prev_pts_[i] + origin);
            const cv::Point2f b = camera.image_to_ground(tracked[i] + origin);
            if (!std::isfinite(a.x) || !std::isfinite(a.y) ||
                !std::isfinite(b.x) || !std::isfinite(b.y))
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

        cv::Mat inliers;
        const cv::Mat motion = cv::estimateAffinePartial2D(
            from, to, inliers, cv::RANSAC, RANSAC_TOLERANCE, 500, 0.99);
        if (motion.empty() || inliers.empty()) {
            return;
        }
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
        const cv::Point2f step(at(0, 2), at(1, 2));
        const float moved = std::hypot(step.x, step.y);
        // 尺度跑偏、单帧位移/转角离谱都当作拟合失败，位姿保持不动。
        if (std::fabs(scale - 1.f) > SCALE_LIMIT || moved > MAX_STEP ||
            std::fabs(turn) > MAX_TURN) {
            prev_pts_.clear();
            return;
        }

        // 拟合给的是 车体系新 = R(turn) * 车体系旧 + step，
        // 而位姿定义是 世界 = R(theta) * 车体系 + t，所以反过来解。
        pose_.theta = wrap(pose_.theta - turn);
        const float c = std::cos(pose_.theta), s = std::sin(pose_.theta);
        pose_.x -= c * step.x - s * step.y;
        pose_.y -= s * step.x + c * step.y;
        speed_ = moved / dt;
        valid_ = true;
    }

  private:
    // 取特征的图像范围：近处地面、避开远景和车头
    static constexpr int ROW_FAR = 100;  // 远（小行号）
    static constexpr int ROW_NEAR = 370; // 近（大行号）
    static constexpr int COL_MARGIN = 16;
    static constexpr int BLOCK_MARGIN = 8;
    static constexpr int WANT_POINTS = 80;            // 想维持的角点数
    static constexpr int MIN_TRACKED = 20;            // 低于它重新提点
    static constexpr int MIN_INLIERS = 15;            // RANSAC 内点下限
    static constexpr int MAX_LEVEL = 3;               // 光流金字塔层数
    static constexpr float FB_ERROR = 0.5f;           // 双向光流一致性（像素）
    static constexpr float RANSAC_TOLERANCE = 0.006f; // 内点阈值（米）
    static constexpr float SCALE_LIMIT = 0.02f;       // 刚体尺度允许偏差
    static constexpr float MAX_STEP = 0.20f;          // 单帧最大位移（米）
    static constexpr float MAX_TURN = 0.35f;          // 单帧最大转角（弧度）
    static constexpr float MAX_GAP = 0.25f;           // 帧间隔超时（秒）
    static constexpr float PI = 3.14159265f;
    static inline const cv::Size WIN{21, 21};

    cv::Size frame_size_;
    cv::Rect roi_;
    std::vector<cv::Mat> prev_pyr_;
    cv::Mat mask_;
    std::vector<cv::Point2f> prev_pts_;
    GroundPose pose_;
    bool valid_ = false;
    int inliers_ = 0;
    float speed_ = 0;

    static float wrap(float angle) {
        while (angle > PI)
            angle -= 2 * PI;
        while (angle < -PI)
            angle += 2 * PI;
        return angle;
    }
    static float seconds(int64_t ticks) {
        return float(double(ticks) * 1000.0 / cv::getTickFrequency());
    }
    void setup(cv::Size size) {
        frame_size_ = size;
        const int top = std::clamp(size.height - ROW_NEAR, 0, size.height);
        const int bottom = std::clamp(size.height - ROW_FAR, 0, size.height);
        roi_ =
            cv::Rect(COL_MARGIN, top, std::max(0, size.width - 2 * COL_MARGIN),
                     std::max(0, bottom - top));
        mask_ = cv::Mat(roi_.size(), CV_8UC1);
        prev_pyr_.clear();
        prev_pts_.clear();
    }
    void drop() {
        prev_pyr_.clear();
        prev_pts_.clear();
    }
    void detect(const cv::Mat &gray) {
        prev_pts_.clear();
        if (gray.empty() || mask_.empty())
            return;
        // 半分辨率提点：goodFeaturesToTrack 的开销主要花在整幅求特征值和
        // 排序上，缩一半就是四分之一；坐标乘回去，光流照样从这几像素开始。
        cv::Mat small, small_mask;
        cv::resize(gray, small, cv::Size(), 0.5, 0.5, cv::INTER_AREA);
        cv::resize(mask_, small_mask, small.size(), 0, 0, cv::INTER_NEAREST);
        std::vector<cv::Point2f> points;
        cv::goodFeaturesToTrack(small, points, WANT_POINTS, 0.05, 4,
                                small_mask);
        prev_pts_.reserve(points.size());
        for (const auto &p : points)
            prev_pts_.emplace_back(p.x * 2.f, p.y * 2.f);
    }
};
