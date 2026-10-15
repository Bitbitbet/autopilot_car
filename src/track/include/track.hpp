#pragma once
#include "camera.hpp"
#include <opencv2/opencv.hpp>
#include <vector>

class Track {
  public:
    std::vector<cv::Point2i> pointsEdgeLeft;  // 赛道左边缘点集
    std::vector<cv::Point2i> pointsEdgeRight; // 赛道右边缘点集
    std::vector<cv::Point2i> widthBlock;      // 色块宽度=终-起（每行）
    std::vector<cv::Point2i> spurroad;        // 保存岔路信息
    double stdevLeft = 1000;                  // 边缘斜率方差（左）
    double stdevRight = 1000;                 // 边缘斜率方差（右）
    int validRowsLeft = 0;                    // 边缘有效行数（左）
    int validRowsRight = 0;                   // 边缘有效行数（右）
    int rowCutUp = 1;                         // 图像顶部切行
    int rowCutBottom = 20;                    // 图像底部切行

    // camera 暂未参与识别；保留参数以统一调用方签名
    void handle([[maybe_unused]] Camera &camera, cv::Mat img);
    void drawImage(cv::Mat &img);
};

double stdevEdgeCal(const std::vector<cv::Point2i> &v_edge, int img_height);