#pragma once

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;

class Mapping {
  public:
    /**
     * @brief IPM初始化
     *
     * @param origSize 输入原始图像Size
     * @param dstSize 输出图像Size
     */
    Mapping(const cv::Size &origSize, const cv::Size &dstSize);

    /**
     * @brief 单应性反透视变换
     *
     * @param _inputImg 原始域图像
     * @param _dstImg 矫正域图像
     * @param _borderMode 矫正模式
     */
    void homographyInv(const Mat &_inputImg, Mat &_dstImg, int _borderMode);

    /**
     * @brief 单应性反透视变换
     *
     * @param _point 矫正域坐标
     * @return Point2d 原始域坐标
     */
    Point2d homographyInv(const Point2d &_point) {
        return homography(_point, m_H_inv);
    }

    /**
     * @brief 单应性反透视变换
     *
     * @param _point 矫正域坐标
     * @return Point3d 原始域坐标
     */
    Point3d homographyInv(const Point3d &_point) {
        return homography(_point, m_H_inv);
    }

    /**
     * @brief 单应性透视变换
     *
     * @param _point 原始域坐标
     * @return Point2d 矫正域坐标
     */
    Point2d homography(const Point2d &_point) {
        return homography(_point, m_H);
    }

    /**
     * @brief 单应性透视变换
     *
     * @param _point 原始域坐标
     * @param _H 转换矩阵
     * @return Point2d 矫正域坐标
     */
    Point2d homography(const Point2d &_point, const Mat &_H);

    /**
     * @brief 单应性透视变换
     *
     * @param _point 原始域坐标
     * @return Point3d 矫正域坐标
     */
    Point3d homography(const Point3d &_point) {
        return homography(_point, m_H);
    }

    /**
     * @brief 单应性透视变换
     *
     * @param _point 原始域坐标
     * @param _H 转换矩阵
     * @return Point3d
     */
    Point3d homography(const Point3d &_point, const cv::Mat &_H);

    /**
     * @brief 单应性透视变换
     *
     * @param _inputImg 原始域图像
     * @param _dstImg 矫正域图像
     */
    void homography(const Mat &_inputImg, Mat &_dstImg);

    cv::Mat getH() const { return m_H; }
    cv::Mat getHinv() const { return m_H_inv; }
    void getPoints(std::vector<Point2f> &_origPts,
                   std::vector<Point2f> &_ipmPts) {
        _origPts = m_origPoints;
        _ipmPts = m_dstPoints;
    }

    /**
     * @brief 绘制掩膜外框
     *
     * @param _points
     * @param _img
     */
    void drawBorder(const std::vector<cv::Point2f> &_points,
                    cv::Mat &_img) const;

  private:
    // Sizes
    cv::Size m_origSize;
    cv::Size m_dstSize;

    // Points
    std::vector<cv::Point2f> m_origPoints;
    std::vector<cv::Point2f> m_dstPoints;

    // Homography
    cv::Mat m_H;
    cv::Mat m_H_inv;

    // Maps
    cv::Mat m_mapX, m_mapY;
    cv::Mat m_invMapX, m_invMapY;

    void createMaps();
};
