#pragma once

#include "camera.hpp"
#include <opencv2/core.hpp>
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
 * 连续两帧跟踪同一批地面角点，两组点都换到地面坐标后拟合 2D 刚体变换
 * （旋转 + 平移 + 尺度）：转向本来就在模型里，所以急转也不怕；尺度必须
 * 接近 1，否则说明拟合被俯仰、假特征或标定漂移带歪了，这一帧作废。
 *
 * 输入是 Camera::grey() 的灰度图。图像固定 640x480，只在近场 ROI 上做
 * 灰度、金字塔和角点，金字塔每帧只建一次（正反向光流共用同一对）。
 */
class GroundOdometry {
  public:
    void reset();

    /**
     * @brief 每帧调用一次
     * @param grey     当前帧灰度图（Camera::grey()）
     * @param camera   提供 image_to_ground
     * @param dt       距上一帧的秒数；不连续时这一帧只重新提点
     * @param blockers 需要剔除的矩形（非地面物体），可以为空
     */
    void update(const cv::Mat &grey, const Camera &camera, double dt,
                const std::vector<cv::Rect> &blockers = {});

    const GroundPose &pose() const { return pose_; }
    bool valid() const { return valid_; }
    int inliers() const { return inliers_; }
    int tracked() const { return int(prev_pts_.size()); }
    float speed() const { return speed_; } // 最近一次拟合出来的地面速度（m/s）

  private:
    // ROI（图像行/列，默认640x480）限制
    static constexpr int ROI_TOP = 40;
    static constexpr int ROI_BOTTOM = 420;
    static constexpr int ROI_LEFT = 0;
    static constexpr int ROI_RIGHT = 640;
    static constexpr int ROI_W = ROI_RIGHT - ROI_LEFT;
    static constexpr int ROI_H = ROI_BOTTOM - ROI_TOP;

    static constexpr int BLOCK_MARGIN = 8;  // 检测框外扩，防止边缘漏进来
    static constexpr int WANT_POINTS = 80;  // 想维持的角点数
    static constexpr int MIN_TRACKED = 12;  // 低于它重新提点
    static constexpr int MIN_INLIERS = 6;   // RANSAC 内点下限
    static constexpr int MAX_LEVEL = 4;     // 光流金字塔层数
    static constexpr float FB_ERROR = 0.5f; // 双向光流一致性（像素）
    static constexpr float RANSAC_TOLERANCE = 0.006f; // 内点阈值（米）
    static constexpr float SCALE_LIMIT = 0.02f;       // 刚体尺度允许偏差
    static constexpr float MAX_STEP = 0.20f;          // 单帧最大位移（米）
    static constexpr float MAX_TURN = 0.35f;          // 单帧最大转角（弧度）
    static constexpr float MAX_GAP = 0.25f;           // 帧间隔超时（秒）

    // 注意必须用圆括号：花括号会构造出 1x3 的 CV_32S（initializer_list 重载）
    cv::Mat mask_ = cv::Mat(ROI_H, ROI_W, CV_8UC1); // 每帧先全开，再挖洞
    std::vector<cv::Mat> prev_pyr_;
    std::vector<cv::Point2f> prev_pts_; // ROI 坐标
    // 上一帧拟合出的运动（2x3，地面系）：给下一帧的光流当初值
    cv::Mat last_motion_;
    GroundPose pose_;
    bool valid_ = false;
    int inliers_ = 0;
    float speed_ = 0;

    void detect(const cv::Mat &roi);
    static float wrap(float angle);
};
