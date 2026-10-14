#include "camera.hpp"
#include "tools.hpp"

int main() {
    // 打开摄像头
    Camera camera("../res/samples/sample.mp4");

    while (1) {
        //[01] 图像采集
        Mat img;
        if (!camera.read(img)) {
            // 如果没有帧了，重置到视频开头
            camera.get_capture().set(cv::CAP_PROP_POS_FRAMES, 0);
            continue;
        }

        //[02] 图像预处理
        imshow("imgCor", img);
        cv::Mat imgBin = camera.binarize(img); // 图像二值化
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
}
