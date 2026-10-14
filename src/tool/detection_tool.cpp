#include "camera.hpp"
#include "detection.hpp"
#include <iostream>           // 输入输出类
#include <opencv2/opencv.hpp> // OpenCV终端部署

using namespace cv;
using std::cout;
using std::endl;
using std::make_shared;
using std::shared_ptr;
namespace chrono = std::chrono;

bool enable = false;  // 图像矫正使能：初始化完成
cv::Mat cameraMatrix; // 摄像机内参矩阵
cv::Mat distCoeffs;   // 相机的畸变矩阵
void correction(cv::Mat &img);

int main() {
    // 打开摄像头
    Camera camera;

    // 目标检测类(AI模型文件)
    shared_ptr<Detection> detection =
        make_shared<Detection>("../res/models/yolov3_mobilenet_v1", 0.3);

    // 读取xml中的相机标定参数
    cameraMatrix = cv::Mat(3, 3, CV_32FC1, Scalar::all(0)); // 摄像机内参矩阵
    distCoeffs = cv::Mat(1, 5, CV_32FC1, Scalar::all(0));   // 相机的畸变矩阵
    FileStorage file;
    if (file.open("../res/calibration/valid/calibration.xml",
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

    while (1) {
        Mat img;
        if (!camera.read(img)) {
            cout << "相机读取错误" << endl;
            break;
        }

        // 启动AI推理
        auto preTime = chrono::duration_cast<chrono::milliseconds>(
                           chrono::system_clock::now().time_since_epoch())
                           .count();
        detection->inference(img); // AI推理
        detection->drawBox(img);   // 图像绘制AI结果

        // 帧率计算
        auto startTime = chrono::duration_cast<chrono::milliseconds>(
                             chrono::system_clock::now().time_since_epoch())
                             .count();
        printf(">> FrameTime: %ldms | %.2ffps \n", startTime - preTime,
               1000.0 / (startTime - preTime));
        imshow("img", img);
        waitKey(1);
    }
}
