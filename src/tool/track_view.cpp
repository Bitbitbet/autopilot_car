/**
 * trackview：只看赛道切割结果。
 * 打开相机 → 二值化 → 取左右边缘 → 画在原图上显示。不驱动车辆、不接手柄。
 */
#include "camera.hpp"
#include "track.hpp"
#include <iostream>
#include <map>
#include <opencv2/opencv.hpp>
#include <utility>

using std::greater;
using std::map;
using std::pair;

int main() {
    Camera camera;
    map<int, pair<int, int>, greater<int>> rows;
    std::cout << "trackview | " << camera.cols() << "x" << camera.rows()
              << " | ESC 退出\n";
    while (true) {
        cv::Mat image;
        if (!camera.read(image)) {
            std::cerr << "[Error] 取不到图像\n";
            break;
        }
        const cv::Mat bin = camera.binarize(image);
        track_identify_from_bin(bin, camera, rows);
        track_draw_result(rows, image);
        cv::imshow("track_view", image);
        cv::imshow("binarized", bin);
        if ((cv::waitKey(1) & 0xff) == 27)
            break;
    }
    cv::destroyAllWindows();
    return 0;
}
