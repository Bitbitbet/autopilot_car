#include "collection_session.hpp"
#include "control.hpp"
#include "predeal.hpp"
#include "stop_signal.hpp"
#include "tools.hpp"
#include <fcntl.h>
#include <filesystem>
#include <iostream> // 输入输出类
#include <libevdev/libevdev.h>
#include <libudev.h>
#include <linux/input.h>
#include <linux/joystick.h>
#include <memory>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp> // OpenCV终端部署
#include <poll.h>
#include <string>      // 字符串类
#include <sys/stat.h>  // 获取文件属性
#include <sys/types.h> // 基本系统数据类型
#include <unistd.h>

using std::cerr;
using std::cout;
using std::endl;
using std::make_shared;
using std::shared_ptr;
using std::string;
using std::to_string;

using namespace cv;
namespace fs = std::filesystem;

int main(int argc, char const *argv[]) {
    car_signal::install();
    try {
        // 通信
        auto car = CarControl::create();
        if (!car) {
            cerr << "Failed to initialize uart!\n";
            return 1;
        }
        car->resetVelocity();
        auto js = JoyStick::create([car] { car->resetVelocity(); });
        if (!js) {
            cerr << "Failed to initialize joystick." << endl;
            return 1;
        }
        VideoCapture capture;
        CollectionSession session(car, js, car_signal::requested);

        car->buzzerSound(Buzzer::start);

        // 摄像头初始化
        capture.open("/dev/video0");
        if (!capture.isOpened()) {
            cerr << "Failed to open video device!" << endl;
            return 2;
        }
        capture.set(CAP_PROP_FRAME_WIDTH, COLSIMAGE);  // 设置图像的列
        capture.set(CAP_PROP_FRAME_HEIGHT, ROWSIMAGE); // 设置图像的行

        // 读取xml中的相机标定参数
        shared_ptr<Predeal> predeal = make_shared<Predeal>(-1); // 图像预处理类

        uint32_t index = 0;
        while (session.running()) {
            // 读取图像
            Mat img;
            if (!capture.read(img) || img.empty())
                throw std::runtime_error("Camera frame unavailable; stopping.");
            if (!session.running())
                break;

            // 图像预处理
            predeal->correction(img); // 图像矫正

            // 图像采集
            if (js->takeSampleOnce()) {
                // 保存到本地
                index++;

                string imgPath = "../res/samples/train/";
                string imgName = to_string(index) + ".jpg";

                fs::create_directory(imgPath);

                imwrite(imgPath + imgName, img);
                cout << "Saved image: " << imgName << endl;
            }

            putText(img, to_string(index), Point(10, 30),
                    cv::FONT_HERSHEY_TRIPLEX, 1, cv::Scalar(0, 0, 254), 1,
                    CV_AA); // 显示图片保存序号
            imshow("img", img);
            int key = waitKey(10);
            if (key == 32) // 空格采图
                js->requestSampleOnce();
        }

        return session.hasFailed() ? 1 : 0;
    } catch (const std::exception &e) {
        cerr << "[Error]: " << e.what() << endl;
        return 1;
    } catch (...) {
        cerr << "[Error]: Unexpected collection failure." << endl;
        return 1;
    }
}
