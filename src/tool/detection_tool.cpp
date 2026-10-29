#include "camera.hpp"
#include "detection.hpp"
#include <chrono>
#include <iostream>           // 输入输出类
#include <opencv2/opencv.hpp> // OpenCV终端部署

using namespace cv;
using std::cerr;
using std::endl;
using std::chrono::microseconds;
using std::chrono::steady_clock;

int main() {
    // 打开摄像头
    Camera camera;

    // 目标检测类(AI模型文件)
    Detection detection("../res/models/yolov3_mobilenet_v1", 0.3);

    while (1) {
        Mat img;
        if (!camera.read(img)) {
            cerr << "相机读取错误" << endl;
            break;
        }

        // 启动AI推理
        auto timepoint_bf = steady_clock::now();
        detection.inference(img); // AI推理
        auto timepoint_aft = steady_clock::now();
        detection.drawBox(img); // 图像绘制AI结果

        auto microsec =
            duration_cast<microseconds>(timepoint_aft - timepoint_bf).count();
        printf("推理用时: %.2lfms，即%.2lf ps\n", microsec / 1000.0,
               1000000.0 / microsec);
        imshow("img", img);
        waitKey(1);
    }
}
