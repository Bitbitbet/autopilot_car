#pragma once
#include "camera.hpp"
#include <opencv2/opencv.hpp>
#include <vector>

/**
 * @brief 赛道边缘识别
 *
 * 二值图自下而上逐行搜索，行号 0 在图像顶部（远处），越大越靠近车头。
 *
 * @param bin_img           赛道识别基准图像（CV_8UC1 二值图）
 * @param camera            相机
 * @param left_edge_points  输出：左边缘点集（x=行, y=列），会被清空
 * @param right_edge_points 输出：右边缘点集（x=行, y=列），会被清空
 */
void track_identify_from_bin(const cv::Mat &bin_img, Camera &camera,
                             std::vector<cv::Point2i> &left_edge_points,
                             std::vector<cv::Point2i> &right_edge_points);

/**
 * @brief 把识别结果画到图上（绿=左边缘，黄=右边缘）
 *
 * @param left_edge_points  左边缘点集（x=行, y=列）
 * @param right_edge_points 右边缘点集（x=行, y=列）
 * @param result            被叠加绘制的图像
 */
void track_draw_result(const std::vector<cv::Point2i> &left_edge_points,
                       const std::vector<cv::Point2i> &right_edge_points,
                       cv::Mat &result);

double stdevEdgeCal(const std::vector<cv::Point2i> &v_edge);