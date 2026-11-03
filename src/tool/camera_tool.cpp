#include "camera.hpp"
#include "timing.hpp"
#include <iostream>           // 输入输出类
#include <opencv2/opencv.hpp> // OpenCV终端部署

using cv::Mat;
using cv::Point;
using cv::Scalar;
using cv::waitKey;
using std::cerr;
using std::endl;

int main() {
    // 打开摄像头
    Camera camera;

    LoopTiming timing;
    while (1) {
        timing.begin();
        Mat img;
        if (!camera.read(img)) {
            cerr << "no video frame" << endl;
            continue;
        }
        Mat imgCor = img.clone();
        timing.lap("read_and_correct");

        // 绘制田字格：基准线
        uint16_t row_cnt = camera.rows() / 30;
        uint16_t col_cnt = camera.cols() / 32;

        for (size_t i = 1; i < row_cnt; i++) { // 使用for循环绘制行线
            line(img, Point(0, 30 * i), Point(img.cols - 1, 30 * i),
                 Scalar(211, 211, 211), 1);
        }
        for (size_t i = 1; i < col_cnt; i++) { // 使用for循环绘制列线
            if (i == (col_cnt / 2))
                line(img, Point(32 * i, 0), Point(32 * i, img.rows - 1),
                     Scalar(0, 0, 255), 2);
            else
                line(img, Point(32 * i, 0), Point(32 * i, img.rows - 1),
                     Scalar(211, 211, 211), 1);
        }
        imshow("imgCor", imgCor);

        waitKey(10);
        timing.end();
    }
}