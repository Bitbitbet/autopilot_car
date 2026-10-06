#include "predeal.hpp"
#include "tools.hpp"
#include "uart.hpp"
#include <fcntl.h>
#include <filesystem>
#include <fstream>  // 文件操作类
#include <iostream> // 输入输出类
#include <linux/input.h>
#include <linux/joystick.h>
#include <memory>
#include <mutex>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp> // OpenCV终端部署
#include <string>             // 字符串类
#include <sys/stat.h>         // 获取文件属性
#include <sys/types.h>        // 基本系统数据类型
#include <thread>             // 线程类
#include <unistd.h>

using std::cerr;
using std::cout;
using std::endl;
using std::ifstream;
using std::lock_guard;
using std::make_shared;
using std::make_unique;
using std::mutex;
using std::nullopt;
using std::optional;
using std::shared_ptr;
using std::string;
using std::thread;
using std::to_string;
using std::unique_ptr;

using namespace cv;
namespace fs = std::filesystem;

/**
 * @brief 读取sysfs单值
 */
optional<string> readToken(const string &path) {
    if (!fs::exists(path)) {
        return nullopt;
    }
    ifstream f(path);
    string v;
    f >> v;
    return v;
}
enum class JoyStickMode : uint8_t {
    BD2A, // 北通2
    BD4A, // 北通4
    SW02, // 星途2
    GR01, // 罗技
};

/**
 * @brief 按 vendor/product 识别手柄品牌
 */
optional<JoyStickMode> getJoyStickMode() {
    auto deviceName = readToken("/sys/class/input/js0/device/name");
    auto vendor = readToken("/sys/class/input/js0/device/id/vendor");
    auto product = readToken("/sys/class/input/js0/device/id/product");
    if (!deviceName || !vendor || !product)
        return nullopt;

    cout << "Joystick Information:\n"
         << "  Device Name: " << *deviceName << "\n  Vendor ID: " << *vendor
         << "\n  Product ID: " << *product << endl;

    if (vendor == "045e" && product == "028e")
        return JoyStickMode::BD2A; // 北通2
    else if (vendor == "20bc" && product == "5046")
        return JoyStickMode::BD4A; // 北通4
    else if (vendor == "3537" && product == "1007")
        return JoyStickMode::SW02; // 星途2
    else if (vendor == "046d" && product == "c219")
        return JoyStickMode::GR01; // 罗技
    else
        return JoyStickMode::SW02; // 未识别,默认星途2
}
class JoyStick {
  private:
    JoyStickMode mode;
    int jsDesc;
    unique_ptr<thread> jsEventThread; // 遥控手柄子线程

    bool sampleMore = false;   // 连续图像采样使能
    bool sampleOnce = false;   // 单次图像采样使能
    float speed = 0;           // 车速：m/s
    float servo = PWMSERVOMID; // 打舵：PWM
    bool forward = true;       // 车辆速度方向:默认向前
    bool uartSend = false;     // 串口发送使能
    bool buzzer = false;       // 提示音效
    mutable mutex mtx;         // 保护所有共享状态

    JoyStick() {}

    // —— 北通2(BD2A) 映射 ——
    void bd2a(const js_event &event) {
        lock_guard<mutex> lk(mtx);

        if (event.type == JS_EVENT_AXIS) // 摇杆
        {
            // cout << "AXIS: " << to_string(joy.number) << " | " <<
            // to_string(joy.value) << endl;
            switch (event.number) {
            case 0: // 方向控制
                servo = PWMSERVOMID +
                        event.value * (PWMSERVOMID - PWMSERVOMIN) / 32767.0;
                uartSend = true;
                break;
            case 5: // 两档速度选择:慢速档
                if (event.value >= 1) {
                    if (forward)
                        speed = 0.3;
                    else
                        speed = -0.3;
                    uartSend = true;
                } else {
                    speed = 0.0;
                    uartSend = true;
                }
                break;
            }
        } else if (event.type == JS_EVENT_BUTTON) // 按键
        {
            // cout << "BUTTON: " << to_string(joy.number) << " | " <<
            // to_string(joy.value) << endl;
            switch (event.number) {
            case 5: // 两档速度选择: 高速档
                if (event.value >= 1) {
                    if (forward)
                        speed = 0.5;
                    else
                        speed = -0.5;
                    uartSend = true;
                } else {
                    speed = 0.0;
                    uartSend = true;
                }
                break;
            case 2: // 开始单次采图
                if (event.value == 1) {
                    buzzer = true;     // 蜂鸣器音效
                    sampleOnce = true; // 开启单张采图使能
                }
                break;
            case 3: // 开始连续采图
                if (event.value == 1) {
                    buzzer = true;     // 蜂鸣器音效
                    sampleMore = true; // 开启连续采图使能
                }
                break;
            case 0:                   // 停止采图
                if (event.value == 1) // 关闭采图使能
                {
                    sampleMore = false;
                    sampleOnce = false;
                    buzzer = true; // 蜂鸣器音效
                }
                break;
            case 7:
                if (event.value == 1) // 向前
                {
                    forward = true;
                    if (speed < 0) {
                        speed = -speed;
                        uartSend = true;
                    }
                    buzzer = true; // 蜂鸣器音效
                }
                break;
            case 6:
                if (event.value == 1) // 向后
                {
                    forward = false;
                    if (speed < 0) {
                        speed = -speed;
                        uartSend = true;
                    }
                    buzzer = true; // 蜂鸣器音效
                }
                break;
            default: // 任意键停止运动
                speed = 0;
                uartSend = true;
                break;
            }
        }
    }

    // —— 星途2(SW02) 映射,按键号按参考 joystick.hpp 派生 ——
    void sw02(const js_event &event) {
        lock_guard<mutex> lk(mtx);

        if (event.type == JS_EVENT_AXIS) { // 摇杆
            switch (event.number) {
            case 0: // 方向控制(LX):罗技轴取负(参考LX=-axes[0])
                servo = PWMSERVOMID -
                        event.value * (PWMSERVOMID - PWMSERVOMIN) / 32767.0;
                uartSend = true;
                break;
            }
        } else if (event.type == JS_EVENT_BUTTON) { // 按键
            switch (event.number) {
            case 9: // 低速档(ZR/RT)
                if (event.value >= 1) {
                    if (forward)
                        speed = 0.3;
                    else
                        speed = -0.3;
                    uartSend = true;
                } else {
                    speed = 0.0;
                    uartSend = true;
                }
                break;
            case 7: // 高速档(R/RB)
                if (event.value >= 1) {
                    if (forward)
                        speed = 0.5;
                    else
                        speed = -0.5;
                    uartSend = true;
                } else {
                    speed = 0.0;
                    uartSend = true;
                }
                break;
            case 3: // 开始单次采图(X)
                if (event.value == 1) {
                    buzzer = true;
                    sampleOnce = true;
                }
                break;
            case 4: // 开始连续采图(Y)
                if (event.value == 1) {
                    buzzer = true;
                    sampleMore = true;
                }
                break;
            case 0: // 停止采图(A)
                if (event.value == 1) {
                    sampleMore = false;
                    sampleOnce = false;
                    buzzer = true;
                }
                break;
            case 11: // 向前(Start)
                if (event.value == 1) {
                    forward = true;
                    if (speed < 0) {
                        speed = -speed;
                        uartSend = true;
                    }
                    buzzer = true;
                }
                break;
            case 10: // 向后(Back)
                if (event.value == 1) {
                    forward = false;
                    if (speed < 0) {
                        speed = -speed;
                        uartSend = true;
                    }
                    buzzer = true;
                }
                break;
            default: // 任意键停止运动
                speed = 0;
                uartSend = true;
                break;
            }
        }
    }

    // —— 罗技(GR01) 映射,按键号按参考 joystick.hpp 派生 ——
    void gr01(const js_event &event) {
        lock_guard<mutex> lk(mtx);

        if (event.type == JS_EVENT_AXIS) // 摇杆
        {
            switch (event.number) {
            case 0: // 方向控制(LX)
                servo = PWMSERVOMID +
                        event.value * (PWMSERVOMID - PWMSERVOMIN) / 32767.0;
                uartSend = true;
                break;
            }
        } else if (event.type == JS_EVENT_BUTTON) // 按键
        {
            switch (event.number) {
            case 7: // 低速档(RT)
                if (event.value >= 1) {
                    if (forward)
                        speed = 0.3;
                    else
                        speed = -0.3;
                    uartSend = true;
                } else {
                    speed = 0.0;
                    uartSend = true;
                }
                break;
            case 5: // 高速档(RB)
                if (event.value >= 1) {
                    if (forward)
                        speed = 0.5;
                    else
                        speed = -0.5;
                    uartSend = true;
                } else {
                    speed = 0.0;
                    uartSend = true;
                }
                break;
            case 0: // 开始单次采图(X)
                if (event.value == 1) {
                    buzzer = true;
                    sampleOnce = true;
                }
                break;
            case 3: // 开始连续采图(Y)
                if (event.value == 1) {
                    buzzer = true;
                    sampleMore = true;
                }
                break;
            case 1: // 停止采图(A)
                if (event.value == 1) {
                    sampleMore = false;
                    sampleOnce = false;
                    buzzer = true;
                }
                break;
            case 9: // 向前(Start)
                if (event.value == 1) {
                    forward = true;
                    if (speed < 0) {
                        speed = -speed;
                        uartSend = true;
                    }
                    buzzer = true;
                }
                break;
            case 8: // 向后(Back)
                if (event.value == 1) {
                    forward = false;
                    if (speed < 0) {
                        speed = -speed;
                        uartSend = true;
                    }
                    buzzer = true;
                }
                break;
            default: // 任意键停止运动
                speed = 0;
                uartSend = true;
                break;
            }
        }
    }

  public:
    ~JoyStick() {
        jsEventThread->join();
        close(jsDesc);
    }

    bool takeUartControl(float &outSpeed, float &outServo) {
        lock_guard<mutex> lk(mtx);
        if (!uartSend)
            return false;
        outSpeed = speed;
        outServo = servo;
        uartSend = false;
        return true;
    }

    bool takeBuzzer() {
        lock_guard<mutex> lk(mtx);
        if (!buzzer)
            return false;
        buzzer = false;
        return true;
    }

    bool takeSampleOnce() {
        lock_guard<mutex> lk(mtx);
        if (!sampleOnce)
            return false;
        sampleOnce = false;
        return true;
    }

    bool isSampleMore() {
        lock_guard<mutex> lk(mtx);
        return sampleMore;
    }

    void requestSampleOnce() {
        lock_guard<mutex> lk(mtx);
        sampleOnce = true;
    }
    static shared_ptr<JoyStick> create() {
        auto joystick = shared_ptr<JoyStick>(new JoyStick);

        joystick->jsDesc = open("/dev/input/js0", O_RDONLY);
        if (joystick->jsDesc < 0) {
            cerr << "Failed to access /dev/input/js0." << endl;
            return nullptr;
        }
        auto mode = getJoyStickMode();
        if (!mode) {
            cerr << "Failed to obtain information about the joystick device."
                 << endl;
            close(joystick->jsDesc);
            return nullptr;
        }
        joystick->mode = *mode;

        joystick->jsEventThread = make_unique<thread>([joystick]() {
            while (1) {
                js_event event;
                if (read(joystick->jsDesc, &event, sizeof(event)) !=
                    sizeof(event))
                    continue;

                switch (joystick->mode) {
                case JoyStickMode::BD4A:
                case JoyStickMode::SW02:
                    joystick->sw02(event);
                    break;
                case JoyStickMode::GR01:
                    joystick->gr01(event);
                    break;
                case JoyStickMode::BD2A:
                    joystick->bd2a(event);
                    break;
                }
            }
        });

        return joystick;
    }
};

int main(int argc, char const *argv[]) {
    auto js = JoyStick::create(); // 遥控手柄类
    if (!js)
        return 1;

    // 通信
    auto client = Uart::create("/dev/ttyUSB0");
    if (!client) {
        cerr << "Failed to initialize uart!\n";
        return 1;
    }

    client->buzzerSound(Buzzer::start);

    // 摄像头初始化
    VideoCapture capture("/dev/video0");
    if (!capture.isOpened()) {
        cerr << "Failed to open video device!" << endl;
        return 2;
    }
    capture.set(CAP_PROP_FRAME_WIDTH, COLSIMAGE);  // 设置图像的列
    capture.set(CAP_PROP_FRAME_HEIGHT, ROWSIMAGE); // 设置图像的行

    // 读取xml中的相机标定参数
    shared_ptr<Predeal> predeal = make_shared<Predeal>(-1); // 图像预处理类

    uint32_t index = 0;
    while (1) {
        float speed, servo;
        if (js->takeUartControl(speed, servo)) {
            client->carControl(speed, servo); // 运动
        }

        // 读取图像
        Mat img;
        if (!capture.read(img))
            continue;

        // 图像预处理
        predeal->correction(img); // 图像矫正

        // 图像采集
        if (js->isSampleMore() || js->takeSampleOnce()) {
            // 保存到本地
            index++;

            string imgPath = "../res/samples/train/";
            string imgName = to_string(index) + ".jpg";

            fs::create_directory(imgPath);

            imwrite(imgPath + imgName, img);
            cout << "Saved image: " << imgName << endl;
        }

        if (js->takeBuzzer()) {
            client->buzzerSound(Buzzer::ding);
        }

        putText(img, to_string(index), Point(10, 30), cv::FONT_HERSHEY_TRIPLEX,
                1, cv::Scalar(0, 0, 254), 1, CV_AA); // 显示图片保存序号
        imshow("img", img);
        int key = waitKey(10);
        if (key == 32) // 空格采图
            js->requestSampleOnce();
    }

    return 0;
}