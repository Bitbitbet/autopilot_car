#include "predeal.hpp"
#include <iostream>
#include <opencv2/core/persistence.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using std::cout;
using std::endl;

/**
 * @brief 图像矫正参数初始化
 *
 * @param bin 二值化阈值
 */
Predeal::Predeal(int bin) : binary(bin) {
    // 读取xml中的相机标定参数
    cameraMatrix = cv::Mat(3, 3, CV_32FC1, Scalar::all(0)); // 摄像机内参矩阵
    distCoeffs = cv::Mat(1, 5, CV_32FC1, Scalar::all(0));   // 相机的畸变矩阵
    FileStorage file;
    if (file.open("../res/calibration/calibration.xml",
                  FileStorage::READ)) // 读取本地保存的标定文件
    {
        file["cameraMatrix"] >> cameraMatrix;
        file["distCoeffs"] >> distCoeffs;
        cout << "相机矫正参数初始化成功!" << endl;
        enable = true;
    } else {
        cout << "打开相机矫正参数失败!!!" << endl;
        enable = false;
    }
}

/**
 * @brief 图像二值化
 *
 * @param img  输入图像
 * @return cv::Mat 二值化图像
 */
cv::Mat Predeal::binarize(cv::Mat &img) {
    cv::Mat imgGray, imgBin;
    cvtColor(img, imgGray, COLOR_BGR2GRAY); // RGB转灰度图

    if (binary < 0)
        threshold(imgGray, imgBin, 0, 255, THRESH_OTSU); //  采用大津法二值化
    else {
        if (binary > 255)
            binary = 255;
        threshold(imgGray, imgBin, binary, 255, THRESH_BINARY); // 固定阈值方法
    }

    // 图像转换
    Mat imgInv = Mat::zeros(imgBin.size(), imgBin.type());
    bitwise_not(imgBin, imgInv);

    return imgInv;
}

/**
 * @brief 矫正图像
 *
 * @param img 输入图像
 * @return cv::Mat 输出图像
 */
void Predeal::correct(cv::Mat &img) {
    if (enable) {
        Size sizeImage; // 图像的尺寸
        sizeImage.width = img.cols;
        sizeImage.height = img.rows;

        cv::Mat mapx =
            cv::Mat(sizeImage, CV_32FC1); // 经过矫正后的X坐标重映射参数
        cv::Mat mapy =
            cv::Mat(sizeImage, CV_32FC1); // 经过矫正后的Y坐标重映射参数
        cv::Mat rotMatrix =
            cv::Mat::eye(3, 3, CV_32F); // 内参矩阵与畸变矩阵之间的旋转矩阵

        // 采用initUndistortRectifyMap+remap进行图像矫正
        initUndistortRectifyMap(cameraMatrix, distCoeffs, rotMatrix,
                                cameraMatrix, sizeImage, CV_32FC1, mapx, mapy);
        remap(img, img, mapx, mapy, INTER_LINEAR);
    }
}

/**
 * @brief 图像裁剪
 *
 * @param img
 */
void Predeal::cutImage(cv::Mat &img) {
    // 图像裁剪
    // 提取 55~175 行，列方向为 30~310
    cv::Rect roi(40, 55, 260, 120); // (x, y, width, height)
    img = img(roi);                 // 抠图

    // 纵向拉伸到300x180（height=300, width=180）
    cv::resize(img, img, cv::Size(300, 180), 0, 0, cv::INTER_NEAREST);
}
