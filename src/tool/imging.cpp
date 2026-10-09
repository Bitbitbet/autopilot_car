#include "predeal.hpp"
#include "tools.hpp"
#include <iostream>
#include <memory>

using namespace cv;

int main(int argc, char const *argv[]) {
    // 打开摄像头
    VideoCapture capture("../res/samples/sample.mp4");
    // VideoCapture capture("/dev/video0", CAP_V4L2);
    if (!capture.isOpened()) {
        std::cerr << "can not open video device " << std::endl;
        return 1;
    }

    capture.set(CAP_PROP_FRAME_WIDTH, COLSCAMERA);  // 设置图像的列数
    capture.set(CAP_PROP_FRAME_HEIGHT, ROWSCAMERA); // 设置图像的行数

    double rate = capture.get(CAP_PROP_FPS);            // 读取图像的帧率
    double width = capture.get(CAP_PROP_FRAME_WIDTH);   // 读取图像的宽度
    double height = capture.get(CAP_PROP_FRAME_HEIGHT); // 读取图像的高度
    std::cout << "Camera Param: frame rate = " << rate << " width = " << width
              << " height = " << height << std::endl;

    // 读取xml中的相机标定参数
    std::shared_ptr<Predeal> predeal =
        std::make_shared<Predeal>(-1); // 图像预处理类

    while (1) {
        //[01] 图像采集
        Mat img;
        if (!capture.read(img)) {
            std::cout << "no video frame" << std::endl;
            // 如果没有帧了，重置到视频开头
            capture.set(cv::CAP_PROP_POS_FRAMES, 0);
            continue;
        }

        //[02] 图像预处理
        predeal->correction(img); // 图像矫正
        imshow("imgCor", img);
        // predeal->imgCutting(img); // 图像裁剪
        // imshow("imgCut", img);
        cv::Mat imgBin = predeal->binaryzation(img); // 图像二值化
        //[03] 透视变换
        cv::Mat imgIpm;
        ipm.homography(img, imgIpm);
        imshow("imgIpm", imgIpm);
        int key = cv::waitKey(20); // 等待用户按键（无限时长）
        if (key == 32) {
            savePicture(imgBin);
            savePicture(imgIpm);
        }
    }
    capture.release();
}
