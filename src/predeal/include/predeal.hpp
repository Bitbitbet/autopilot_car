#pragma once

#include <opencv2/core/mat.hpp>

class Predeal {
  public:
    Predeal();
    Predeal(uint8_t threshold_value);
    ~Predeal() {};
    cv::Mat binarize(const cv::Mat &img);
    void correct(cv::Mat &img);
    void cutImage(cv::Mat &img);
    uint8_t threshold_value = 127;
    bool use_otsu;

  private:
    cv::Mat cameraMatrix; // 摄像机内参矩阵
    cv::Mat distCoeffs;   // 相机的畸变矩阵
    cv::Mat mapx, mapy;

    void initialize();
};