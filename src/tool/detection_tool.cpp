#include "detection.hpp"
#include "predeal.hpp"
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

int main(int argc, char const *argv[]) {
    // 打开摄像头
    // VideoCapture capture("../res/samples/sample.mp4");
    VideoCapture capture("/dev/video0", CAP_V4L2);
    if (!capture.isOpened()) {
        cout << "can not open video device " << endl;
        return 1;
    }

    capture.set(CAP_PROP_FRAME_WIDTH, 320);  // 设置图像的列数
    capture.set(CAP_PROP_FRAME_HEIGHT, 240); // 设置图像的行数

    double rate = capture.get(CAP_PROP_FPS);            // 读取图像的帧率
    double width = capture.get(CAP_PROP_FRAME_WIDTH);   // 读取图像的宽度
    double height = capture.get(CAP_PROP_FRAME_HEIGHT); // 读取图像的高度
    cout << "Camera Param: frame rate = " << rate << " width = " << width
         << " height = " << height << endl;

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

    // 读取xml中的相机标定参数
    shared_ptr<Predeal> predeal = make_shared<Predeal>(-1); // 图像预处理类

    while (1) {
        Mat img;
        if (!capture.read(img)) {
            cout << "no video frame" << endl;
            continue;
        }

        // 图像预处理
        predeal->correct(img); // 图像矫正

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
    capture.release();
}
