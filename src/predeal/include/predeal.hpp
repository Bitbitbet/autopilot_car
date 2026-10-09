#pragma once

#include <opencv2/core/mat.hpp>

class Predeal {
  public:
    Predeal(int bin);
    ~Predeal() {};
    cv::Mat binarize(cv::Mat &img);
    void correct(cv::Mat &img);
    void cutImage(cv::Mat &img);
    int binary = -1; // 图像二值化阈值：<0 默认使用大津法

  private:
    bool enable = false;  // 图像矫正使能：初始化完成
    cv::Mat cameraMatrix; // 摄像机内参矩阵
    cv::Mat distCoeffs;   // 相机的畸变矩阵
};