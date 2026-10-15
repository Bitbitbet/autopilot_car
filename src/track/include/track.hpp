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

    // 单位：地面坐标（米）
    float minDistance = 0.f;      // 最近的搜索距离（车头前方）
    float maxDistance = 1.20f;    // 最远的搜索距离（图像前向可见约 1.5 m）
    float minRoadWidth = 0.20f;   // 路面色块的最小实际宽度
    float maxGapDistance = 0.05f; // 允许桥接的最大缺口

    void handle(Camera &camera, cv::Mat img);
    void drawImage(cv::Mat &img);
};

double stdevEdgeCal(const std::vector<cv::Point2i> &v_edge);