#include "opencv2/calib3d.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "show.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/core/persistence.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <sys/stat.h>
#include <utility>

using namespace cv;
using std::cerr;
using std::cout;
using std::endl;
using std::ofstream;
using std::string;
using std::vector;

namespace fs = std::filesystem;
const Size BOARD_SIZE = Size(11, 8); // 标定板的角点数(行,列)
const Size SQUARE_SIZE =
    Size(20, 20); // 实际测量标定板上每个棋盘格的边长（长,宽）：单位mmd
/*
 * 第一步：提取图片中的角点信息
 */
vector<vector<Point2f>> extract_corners(const fs::path &temp_images_folder,
                                        vector<fs::path> &temp_image_paths,
                                        Size &image_size) {
    temp_image_paths.clear();
    //[1] 提取每张图片的角点信息
    cout << "[1] 提取每张图片的角点信息" << endl;
    vector<vector<Point2f>> pointsCorners; // 保存检测到的所有角点
    bool image_size_initialized = false;
    for (const auto &entry : fs::directory_iterator(temp_images_folder)) {
        auto path = entry.path();
        Mat image = imread(path.string()); // OpenCV读取图像
        if (image.empty()) {
            cerr << "无法读取" << path << "。" << endl;
            exit(2);
        }
        if (!image_size_initialized) { // 初始化图像Size信息
            image_size.width = image.cols;
            image_size.height = image.rows;
            image_size_initialized = true;
            cout << "图片宽度：" << image_size.width << "\n图片高度："
                 << image_size.height << endl;
        }

        vector<Point2f> pointCorners; // 每幅图像上检测到的角点坐标
        if (findChessboardCorners(image, BOARD_SIZE,
                                  pointCorners)) { // 寻找图像角点
            Mat imageGray;
            cvtColor(image, imageGray, CV_RGB2GRAY);
            //[2] 亚像素精确化
            find4QuadCornerSubpix(imageGray, pointCorners,
                                  Size(5, 5)); // 对粗提取的角点信息进一步优化
            // cornerSubPix(imageGray,pointCorners,Size(5,5),Size(-1,-1),TermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER,30,0.1));

            drawChessboardCorners(image, BOARD_SIZE, pointCorners,
                                  true); // 在标定板图像上绘制已识别的角点信息

            cv::namedWindow("Corners", WINDOW_NORMAL); // 图像名称
            imshow("Corners", image); //[3] 在原始图形中绘制角点信息并展示

            // 存储角点绘制图像
            auto filename = path.filename();
            auto rst_path =
                (path.parent_path() / "../corners").lexically_normal();
            fs::create_directories(rst_path);
            rst_path = rst_path / filename;

            if (!imwrite(rst_path.string(), image)) {
                cerr << "试图写入 " << rst_path << " 失败!" << endl;
                exit(2);
            }

            uint16_t countCorner =
                BOARD_SIZE.width *
                BOARD_SIZE.height; // 每张图像上该有的角点数量
            if (pointCorners.size() >= countCorner) {
                temp_image_paths.push_back(path);
                pointsCorners.push_back(
                    std::move(pointCorners)); // 保存亚像素角点
            }

            cout << path << "样片合格!" << endl;
            waitKey(500); // 停顿500ms
        } else {          // 图像不存在角点
            cerr << "样片" << path << "找不到角点，不合格，应该删除。" << endl;
            exit(1); // 直接退出标定：否者后续文件名无法对应
        }
    }
    if (pointsCorners.empty()) {
        cerr << "没有合格的图片。" << endl;
        exit(1);
    }
    cout << "已筛选 " << pointsCorners.size()
         << "张合格照片进行标定 | Loading...." << endl;
    return pointsCorners;
}

/**
 * @brief 相机标定程序
 * @note 输出标定参数文件.yml 和 评价结果.txt
 * @param imagesFolder 标定板图像路径（文件夹）
 */
int main(int argc, char *argv[]) {

    fs::path working_path("../res/calibration");
    if (!fs::is_directory(working_path)) {
        cerr << "找不到" << working_path << "文件夹，请在正确位置执行该程序"
             << endl;
        return 1;
    }

    auto temp_images_folder = working_path / "temp";
    if (!fs::is_directory(working_path)) {
        cerr << "找不到" << working_path << "文件夹" << endl;
    }

    Size image_size;                                        // 图像的尺寸
    Mat cameraMatrix = Mat(3, 3, CV_32FC1, Scalar::all(0)); // 摄像机内参矩阵
    Mat distCoeffs = Mat(1, 5, CV_32FC1, Scalar::all(0));   // 相机的畸变矩阵

    vector<fs::path> temp_image_paths;
    auto pointsCorners =
        extract_corners(temp_images_folder, temp_image_paths, image_size);

    //[4] 开始标定图像
    vector<vector<Point3f>> pointsObject; // 标定板角点的三维坐标
    for (int t = 0; t < pointsCorners.size(); t++) {
        vector<Point3f> tempPointSet;
        for (int i = 0; i < BOARD_SIZE.height; i++) {
            for (int j = 0; j < BOARD_SIZE.width; j++) {
                Point3f realPoint;
                // 假设标定板放在世界坐标系中z=0的平面上
                realPoint.x = i * SQUARE_SIZE.width;
                realPoint.y = j * SQUARE_SIZE.height;
                realPoint.z = 0;
                tempPointSet.push_back(realPoint);
            }
        }
        pointsObject.push_back(tempPointSet);
    }

    vector<int> countCorners; // 每幅图像中角点的数量
    // 初始化每幅图像中的角点数量，假定每幅图像中都可以看到完整的标定板
    for (int i = 0; i < pointsCorners.size(); i++) {
        countCorners.push_back(BOARD_SIZE.width * BOARD_SIZE.height);
    }

    vector<Mat> rvecsMat; // 图像的旋转向量
    vector<Mat> tvecsMat; // 图像的平移向量
    calibrateCamera(pointsObject, pointsCorners, image_size, cameraMatrix,
                    distCoeffs, rvecsMat, tvecsMat, 0); // 开始标定

    cout << "相机标定完成!" << endl;
    cout << "正在评价标定结果 | Loading...." << endl;

    // 输出标定误差的评价结果
    double errorTotal = 0.0;     // 所有图像的误差总和
    double error = 0.0;          // 每幅图像的平均误差
    vector<Point2f> pointsImage; // 重新计算得到的投影点
    ofstream assessment(working_path / "assessment.txt"); // 保存标定的评价结果

    cout << "相机标定分辨率: " << image_size.width << "x" << image_size.height
         << "\n";
    assessment << "相机标定分辨率: " << image_size.width << "x"
               << image_size.height << "\n";

    cout << "每幅图像的标定误差: \n";
    assessment << "每幅图像的标定误差: \n";
    for (int t = 0; t < pointsCorners.size(); t++) {
        vector<Point3f> tempPointSet = pointsObject[t];
        /* 通过得到的摄像机内外参数，对空间的三维点进行重新投影计算，得到新的投影点
         */
        projectPoints(tempPointSet, rvecsMat[t], tvecsMat[t], cameraMatrix,
                      distCoeffs, pointsImage);
        /* 计算新的投影点和旧的投影点之间的误差*/
        const vector<Point2f> &tempImagePoint = pointsCorners[t];
        Mat tempImagePointMat = Mat(1, tempImagePoint.size(), CV_32FC2);
        Mat image_points2Mat = Mat(1, pointsImage.size(), CV_32FC2);
        for (int j = 0; j < tempImagePoint.size(); j++) {
            image_points2Mat.at<Vec2f>(0, j) =
                Vec2f(pointsImage[j].x, pointsImage[j].y);
            tempImagePointMat.at<Vec2f>(0, j) =
                Vec2f(tempImagePoint[j].x, tempImagePoint[j].y);
        }
        error = norm(image_points2Mat, tempImagePointMat, NORM_L2);
        errorTotal += error /= countCorners[t];

        const auto &filename = temp_image_paths[t].filename();
        cout << "图像[" << filename << "]的平均误差: " << error << "像素"
             << endl;
        assessment << "图像[" << filename << "]的平均误差: " << error << "像素"
                   << endl;
    }

    cout << "总体平均误差: " << errorTotal / pointsCorners.size() << "像素"
         << endl;
    assessment << "总体平均误差: " << errorTotal / pointsCorners.size()
               << "像素" << endl
               << endl;

    Mat rotation_matrix =
        Mat(3, 3, CV_32FC1, Scalar::all(0)); /* 保存每幅图像的旋转矩阵 */
    assessment << "相机内参数矩阵: " << endl;
    assessment << cameraMatrix << endl << endl;
    assessment << "畸变系数: \n";
    assessment << distCoeffs << endl << endl << endl;
    for (int t = 0; t < pointsCorners.size(); t++) {
        const auto &filename = temp_image_paths[t].filename();
        assessment << "图像[" << filename << "]的旋转向量: " << endl;
        assessment << rvecsMat[t] << endl;
        /* 将旋转向量转换为相对应的旋转矩阵 */
        Rodrigues(rvecsMat[t], rotation_matrix);
        assessment << "图像[" << filename << "]的旋转矩阵: " << endl;
        assessment << rotation_matrix << endl;
        assessment << "图像[" << filename << "]的平移向量: " << endl;
        assessment << tvecsMat[t] << endl << endl;
    }

    assessment << endl;

    //[5] 将相机标定参数写入xml文件
    FileStorage calibration((working_path / "calibration.xml").string(),
                            FileStorage::WRITE);
    calibration << "cameraMatrix" << cameraMatrix;
    calibration << "distCoeffs" << distCoeffs;
    calibration.release();

    //[6] 开始矫正图像
    cout << "正在显示图像矫正结果..." << endl;
    Show show(2); // 初始化UI显示窗口
    for (int t = 0; t < pointsCorners.size(); t++) {
        const auto &path = temp_image_paths[t];
        Mat imageSource = imread(path.string()); // OpenCV读取图像
        Mat mapx = Mat(image_size, CV_32FC1);    // 经过矫正后的X坐标重映射参数
        Mat mapy = Mat(image_size, CV_32FC1);    // 经过矫正后的Y坐标重映射参数
        Mat R = Mat::eye(3, 3, CV_32F); // 内参矩阵与畸变矩阵之间的旋转矩阵

        // 采用initUndistortRectifyMap + remap进行图像矫正
        initUndistortRectifyMap(cameraMatrix, distCoeffs, R, cameraMatrix,
                                image_size, CV_32FC1, mapx, mapy);
        Mat imageCorrect = imageSource.clone();
        remap(imageSource, imageCorrect, mapx, mapy, INTER_LINEAR);

        show.setNewWindow(1, "imgSource", imageSource);   // 添加需要显示的图像
        show.setNewWindow(2, "imgCorrect", imageCorrect); // 添加需要显示的图像
        show.show();                                      // 图像窗口显示

        // 存储矫正图像
        auto rst_path = working_path / "correct";
        fs::create_directories(rst_path);
        rst_path = rst_path / path.filename();
        imwrite(rst_path.string(), imageCorrect);

        waitKey(500); // 延时2s
    }

    return 0;
}

// /**
//  * @brief 矫正图像
//  *
//  * @param imagesPath 图像路径
//  */
// void imageCorrecte(string imagesPath) {
//     Size sizeImage; // 图像的尺寸
//     Show show(2);   // 图像显示窗口重绘
//     // 读取xml中的相机标定参数
//     Mat matrix = Mat(3, 3, CV_32FC1, Scalar::all(0)); // 摄像机内参矩阵
//     Mat coeffs = Mat(1, 5, CV_32FC1, Scalar::all(0)); // 相机的畸变矩阵
//     FileStorage fs;
//     fs.open("../res/calibration/valid/calibration.xml", FileStorage::READ);
//     fs["cameraMatrix"] >> matrix;
//     fs["distCoeffs"] >> coeffs;

//     Mat imageSource = imread(imagesPath); // 读取矫正图像
//     sizeImage.width = imageSource.cols;
//     sizeImage.height = imageSource.rows;
//     Mat mapx = Mat(sizeImage, CV_32FC1); // 经过矫正后的X坐标重映射参数
//     Mat mapy = Mat(sizeImage, CV_32FC1); // 经过矫正后的Y坐标重映射参数
//     Mat R = Mat::eye(3, 3, CV_32F);      // 内参矩阵与畸变矩阵之间的旋转矩阵

//     Mat imageCorrect = imageSource.clone();
//     // 采用initUndistortRectifyMap+remap进行图像矫正
//     initUndistortRectifyMap(matrix, coeffs, R, matrix, sizeImage, CV_32FC1,
//                             mapx, mapy);
//     remap(imageSource, imageCorrect, mapx, mapy, INTER_LINEAR);

//     // 采用undistort进行图像矫正
//     //  undistort(imageSource, imageCorrect, matrix, coeffs);

//     show.setNewWindow(0, "imgSource", imageSource);   // 添加需要显示的图像
//     show.setNewWindow(1, "imgCorrect", imageCorrect); // 添加需要显示的图像
//     show.show();                                      // 图像窗口显示

//     while (1) {
//         waitKey(200);
//     }
// }