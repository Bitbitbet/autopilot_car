#include "predeal.hpp"
#include "tools.hpp"
#include <iostream>
#include <stdexcept>
#include <opencv2/core/persistence.hpp>
#include <opencv2/imgproc.hpp>

using cv::FileStorage;
using cv::Mat;
using cv::Scalar;
using cv::Size;
using std::cout;
using std::endl;

Predeal::Predeal() : use_otsu(true) { initialize(); }
Predeal::Predeal(uint8_t threshold)
    : threshold_value(threshold), use_otsu(false) {
    initialize();
}

void Predeal::initialize() {
    // 读取xml中的相机标定参数
    cameraMatrix = Mat(3, 3, CV_32FC1, Scalar::all(0)); // 摄像机内参矩阵
    distCoeffs = Mat(1, 5, CV_32FC1, Scalar::all(0));   // 相机的畸变矩阵
    FileStorage file;
    if (file.open("../res/calibration/calibration.xml",
                  FileStorage::READ)) // 读取本地保存的标定文件
    {
        file["cameraMatrix"] >> cameraMatrix;
        file["distCoeffs"] >> distCoeffs;
        file.release();
        cout << "相机矫正参数初始化成功!" << endl;
    } else {
        cout << "打开相机矫正参数失败!!!" << endl;
        throw std::runtime_error("Cannot open ../res/calibration/calibration.xml");
    }

    if (cameraMatrix.rows != 3 || cameraMatrix.cols != 3 ||
        distCoeffs.empty() || !cv::checkRange(cameraMatrix) ||
        !cv::checkRange(distCoeffs))
        throw std::runtime_error("Invalid camera calibration matrices");

    Size camera_size{IMAGE_COL, IMAGE_ROW};
    mapx = Mat(camera_size,
               CV_32FC1); // 经过矫正后的X坐标重映射参数
    mapy = Mat(camera_size,
               CV_32FC1);                   // 经过矫正后的Y坐标重映射参数
    Mat rotMatrix = Mat::eye(3, 3, CV_32F); // 内参矩阵与畸变矩阵之间的旋转矩阵

    // 采用initUndistortRectifyMap+remap进行图像矫正
    initUndistortRectifyMap(cameraMatrix, distCoeffs, rotMatrix, cameraMatrix,
                            camera_size, CV_32FC1, mapx, mapy);
}

/**
 * @brief 图像二值化
 *
 * @param img  输入图像
 * @return cv::Mat 二值化图像
 */
Mat Predeal::binarize(const Mat &img) {
    Mat imgGray, imgBin;
    cvtColor(img, imgGray, cv::COLOR_BGR2GRAY); // RGB转灰度图

    if (use_otsu)
        threshold(imgGray, imgBin, 0, 255,
                  cv::THRESH_OTSU); //  采用大津法二值化
    else
        threshold(imgGray, imgBin, threshold_value, 255,
                  cv::THRESH_BINARY); // 固定阈值方法

    return imgBin;
}

/**
 * @brief 矫正图像
 *
 * @param img 输入图像
 * @return cv::Mat 输出图像
 */
void Predeal::correct(Mat &camera_frame) {
    if (camera_frame.empty() || camera_frame.type() != CV_8UC3 ||
        camera_frame.size() != Size(IMAGE_COL, IMAGE_ROW))
        throw std::runtime_error("Camera must supply 320x240 BGR frames");
    remap(camera_frame, camera_frame, mapx, mapy, cv::INTER_LINEAR);
}

/**
 * @brief 图像裁剪
 *
 * @param img
 */
void Predeal::cutImage(Mat &img) {
    // 图像裁剪
    // 提取 55~175 行，列方向为 30~310
    cv::Rect roi(40, 55, 260, 120); // (x, y, width, height)
    img = img(roi);                 // 抠图

    // 纵向拉伸到300x180（height=300, width=180）
    cv::resize(img, img, cv::Size(300, 180), 0, 0, cv::INTER_NEAREST);
}
