#include "camera.hpp"
#include "detection.hpp"
#include "timing.hpp"
#include <iostream>           // 输入输出类
#include <opencv2/opencv.hpp> // OpenCV终端部署

using namespace cv;
using std::cerr;
using std::endl;
using std::chrono::microseconds;

int main() {
    // 打开摄像头
    Camera camera;

    // 目标检测类(AI模型文件)
    Detection detection("../res/models/yolov3_mobilenet_v1", 0.3);

    LoopTiming timing;
    while (1) {
        timing.begin();
        Mat img;
        if (!camera.read(img)) {
            cerr << "相机读取错误" << endl;
            break;
        }
        timing.lap("read");

        detection.inference(img); // AI推理
        timing.lap("infer");
        detection_draw_box(img, detection.results); // 图像绘制AI结果

        imshow("img", img);
        waitKey(1);

        timing.end();
    }
}
