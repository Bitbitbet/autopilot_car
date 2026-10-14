#include "camera.hpp"
#include "control.hpp"
#include "joystick.hpp"
#include "joystick_car_control.hpp"
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
using std::shared_ptr;
using std::string;
using std::to_string;

using namespace cv;
namespace fs = std::filesystem;

int main() {
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
        auto controlThread = startJoystickControlThread(car, js);

        car->buzzerSound(Buzzer::start);

        // 摄像头初始化
        VideoCapture camera("/dev/video0");

        uint32_t index = 0;
        while (true) {
            // 读取图像
            Mat frame;
            if (!camera.read(frame)) {
                cerr << "相机读取错误" << endl;
                break;
            }

            // 图像采集
            if (js->takeSampleOnce()) {
                // 保存到本地
                index++;

                string imgPath = "../res/samples/train/";
                string imgName = to_string(index) + ".jpg";

                fs::create_directory(imgPath);

                imwrite(imgPath + imgName, frame);
                cout << "Saved image: " << imgName << endl;
            }

            putText(frame, to_string(index), Point(10, 30),
                    cv::FONT_HERSHEY_TRIPLEX, 1, cv::Scalar(0, 0, 254), 1,
                    CV_AA); // 显示图片保存序号
            imshow("img", frame);
            int key = waitKey(10);
            if (key == 32) // 空格采图
                js->requestSampleOnce();
        }

        return 0;
    } catch (const std::exception &e) {
        cerr << "[Error]: " << e.what() << endl;
        return 1;
    } catch (...) {
        cerr << "[Error]: Unexpected collection failure." << endl;
        return 1;
    }
}
