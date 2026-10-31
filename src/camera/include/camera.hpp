#pragma once

#include <opencv2/opencv.hpp>

// 黑色赛道：路面是否是暗色。
constexpr bool BLACK_ROAD = true;

class Camera {
  public:
    struct CalibrationOnly {};
    Camera(CalibrationOnly, int threshold=-1);
    explicit Camera(const cv::Mat &imageToGround);
    void correct(cv::Mat &frame) const;
    void setGroundHomography(const cv::Mat &h);
    bool blackRoad=BLACK_ROAD;
    Camera();
    Camera(uint8_t threshold_value);
    Camera(const char *capture_source);
    Camera(const char *capture_source, uint8_t threshold_value);
    ~Camera();
    cv::Mat binarize(const cv::Mat &img);
    bool read(cv::Mat &frame_out);

    /**
     * @brief 图像坐标 -> 地面坐标（米）
     *
     * 输入是 read() 返回的矫正后图像（640x480）上的像素点；
     * 只对地面上的点有意义。
     * @param img_pt 图像坐标(px)
     * @return 地面坐标 (X, Y)，单位米；X 左负右正，Y 前正后负
     */
    cv::Point2d image_to_ground(const cv::Point2d &img_pt) const;

    /**
     * @brief 地面坐标（米） -> 图像坐标
     *
     * @param gnd_pt 地面坐标 (X, Y)，单位米；X 左负右正，Y 前正后负
     * @return 矫正后图像(640x480)上的像素坐标
     */
    cv::Point2d ground_to_image(const cv::Point2d &gnd_pt) const;

    int cols() const { return capture_width; }
    int rows() const { return capture_height; }

    cv::VideoCapture &get_capture() { return capture; }

  private:
    // 矫正用
    cv::Mat mapx, mapy;

    // 地面单应（由 calibrate_homography 工具写进 calibration.xml）
    cv::Mat H_img2gnd; // 图像 -> 地面(米)
    cv::Mat H_gnd2img; // 地面(米) -> 图像

    uint8_t threshold_value;
    bool use_otsu;

    cv::VideoCapture capture;
    int capture_width, capture_height;

    void initialize(const char *capture_source);
    void initializeCalibration();
};
