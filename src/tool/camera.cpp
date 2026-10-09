#include "predeal.hpp"
#include <chrono>
#include <iostream> // 输入输出类
#include <memory>
#include <opencv2/opencv.hpp> // OpenCV终端部署

using std::cout;
using std::endl;
using std::make_shared;
using std::shared_ptr;
namespace chrono = std::chrono;
using std::cerr;
using std::chrono::steady_clock;
using namespace cv;

#define COLS_IMG 640 // 摄像头：图像的列数
#define ROWS_IMG 480 // 摄像头：图像的行数

int main(int argc, char const *argv[]) {
    // 打开摄像头
    VideoCapture capture("/dev/video0", CAP_V4L2);
    if (!capture.isOpened()) {
        cerr << "无法打开视频设备 /dev/video0" << endl;
        return 1;
    }

    capture.set(CAP_PROP_FRAME_WIDTH, COLS_IMG);  // 设置图像的列数
    capture.set(CAP_PROP_FRAME_HEIGHT, ROWS_IMG); // 设置图像的行数

    double rate = capture.get(CAP_PROP_FPS);            // 读取图像的帧率
    double width = capture.get(CAP_PROP_FRAME_WIDTH);   // 读取图像的宽度
    double height = capture.get(CAP_PROP_FRAME_HEIGHT); // 读取图像的高度
    cout << "相机参数:\n  帧率 " << rate << "\n  宽度 " << width << "\n  高度 "
         << height << endl;

    // 读取xml中的相机标定参数
    shared_ptr<Predeal> predeal = make_shared<Predeal>(-1); // 图像预处理类

    while (1) {
        static auto timeLast = steady_clock::now();
        static double fps = 0.0;
        auto timeNow = steady_clock::now();
        double dt =
            chrono::duration_cast<chrono::microseconds>(timeNow - timeLast)
                .count() /
            1e6;
        timeLast = timeNow;
        if (dt > 0)
            fps = 1.0 / dt;

        Mat img;
        if (!capture.read(img)) {
            cerr << "no video frame" << endl;
            continue;
        }
        Mat imgCor = img.clone();

        // 绘制田字格：基准线
        uint16_t rows = ROWS_IMG / 30; // 8
        uint16_t cols = COLS_IMG / 32; // 10

        for (size_t i = 1; i < rows; i++) { // 使用for循环绘制行线
            line(img, Point(0, 30 * i), Point(img.cols - 1, 30 * i),
                 Scalar(211, 211, 211), 1);
        }
        for (size_t i = 1; i < cols; i++) { // 使用for循环绘制列线
            if (i == (int)(cols / 2))
                line(img, Point(32 * i, 0), Point(32 * i, img.rows - 1),
                     Scalar(0, 0, 255), 2);
            else
                line(img, Point(32 * i, 0), Point(32 * i, img.rows - 1),
                     Scalar(211, 211, 211), 1);
        }
        imshow("imgOri", img);

        //[02] 图像预处理
        cv::resize(imgCor, imgCor, cv::Size(320, 240), 0, 0, cv::INTER_NEAREST);
        predeal->correct(imgCor); // 图像矫正
        imshow("imgCor", imgCor);

        waitKey(10);
    }

    capture.release();
}