#include "control.hpp"
#include "predeal.hpp"
#include "tools.hpp"
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <fcntl.h>
#include <filesystem>
#include <fstream>  // 文件操作类
#include <iostream> // 输入输出类
#include <libevdev/libevdev.h>
#include <libudev.h>
#include <linux/input.h>
#include <linux/joystick.h>
#include <memory>
#include <mutex>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp> // OpenCV终端部署
#include <optional>
#include <poll.h>
#include <string>      // 字符串类
#include <sys/stat.h>  // 获取文件属性
#include <sys/types.h> // 基本系统数据类型
#include <thread>      // 线程类
#include <unistd.h>

using std::atomic;
using std::atomic_bool;
using std::atomic_uint8_t;
using std::cerr;
using std::condition_variable;
using std::cout;
using std::endl;
using std::ifstream;
using std::lock_guard;
using std::make_shared;
using std::make_unique;
using std::memory_order_relaxed;
using std::mutex;
using std::nullopt;
using std::optional;
using std::shared_ptr;
using std::string;
using std::thread;
using std::to_string;
using std::unique_lock;
using std::unique_ptr;

using namespace cv;
namespace fs = std::filesystem;

string get_joystick_device_path() {
    struct udev *udev_ctx = udev_new();
    if (!udev_ctx) {
        return {};
    }

    struct udev_enumerate *enumerate = udev_enumerate_new(udev_ctx);
    if (!enumerate) {
        udev_unref(udev_ctx);
        return {};
    }

    udev_enumerate_add_match_subsystem(enumerate, "input");
    udev_enumerate_add_match_property(enumerate, "ID_INPUT_JOYSTICK", "1");

    if (udev_enumerate_scan_devices(enumerate) < 0) {
        udev_enumerate_unref(enumerate);
        udev_unref(udev_ctx);
        return {};
    }

    struct udev_list_entry *devices = udev_enumerate_get_list_entry(enumerate);
    struct udev_list_entry *entry;
    std::string result;

    udev_list_entry_foreach(entry, devices) {
        const char *syspath = udev_list_entry_get_name(entry);
        struct udev_device *dev =
            udev_device_new_from_syspath(udev_ctx, syspath);
        if (!dev) {
            continue;
        }

        const char *devnode = udev_device_get_devnode(dev);
        const char *sysname = udev_device_get_sysname(dev);

        if (devnode && sysname && std::strncmp(sysname, "event", 5) == 0) {
            result = devnode;
            udev_device_unref(dev);
            break;
        }

        udev_device_unref(dev);
    }

    udev_enumerate_unref(enumerate);
    udev_unref(udev_ctx);

    return result;
}
class JoyStick {
  public:
    static shared_ptr<JoyStick> create() {
        auto joystick_path = get_joystick_device_path();
        if (joystick_path.empty()) {
            cerr << "Failed to find joystick device." << endl;
            return nullptr;
        }

        int joystick_fd = open(joystick_path.c_str(), O_RDONLY | O_NONBLOCK);
        if (joystick_fd < 0) {
            cerr << "Failed to open joystick device " << joystick_path << "."
                 << endl;
            return nullptr;
        }
        cout << "Joystick device " << joystick_path << " opened." << endl;

        struct libevdev *dev = nullptr;
        if (libevdev_new_from_fd(joystick_fd, &dev) < 0) {
            close(joystick_fd);
            return nullptr;
        }

        auto joystick = shared_ptr<JoyStick>(new JoyStick);
        joystick->joystick_fd = joystick_fd;
        joystick->dev = dev;

        JoyStick *raw = joystick.get();
        joystick->recv_thread =
            make_unique<thread>([raw]() { raw->recv_thread_main(); });

        return joystick;
    }

    ~JoyStick() {
        thread_stop.store(true, memory_order_relaxed);
        recv_thread->join();
        libevdev_free(dev);
        close(joystick_fd);
    }

    // ---------------- 对外接口 ----------------

    bool takeCarControl(float &outSpeed, uint16_t &outServo) {
        lock_guard lk(mtx);
        if (!do_car_control)
            return false;
        outSpeed = speed;
        outServo = servo;
        do_car_control = false;
        return true;
    }

    bool takeBuzzer() {
        lock_guard lk(mtx);
        auto rst = do_buzzer;
        do_buzzer = false;
        return rst;
    }

    bool takeSampleOnce() {
        lock_guard lk(mtx);
        auto rst = do_sample_once;
        do_sample_once = false;
        return rst;
    }
    void waitEvent() {
        unique_lock lk(mtx);
        cv.wait(lk, [this]() {
            return do_sample_once || do_car_control || do_buzzer;
        });
    }

    void requestSampleOnce() {
        bool old_do_sample_once;
        {
            lock_guard lk(mtx);
            old_do_sample_once = do_sample_once;
            do_sample_once = true;
        }
        if (!old_do_sample_once)
            cv.notify_one();
    }

  private:
    int joystick_fd = -1;
    struct libevdev *dev = nullptr;
    unique_ptr<thread> recv_thread;
    atomic<bool> thread_stop{false};

    // ---------------- 共享状态 ----------------
    bool do_sample_once = false;
    bool do_car_control = false;
    bool do_buzzer = false;
    float speed = 0;              // 车速：m/s
    uint16_t servo = PWMSERVOMID; // 打舵：PWM
    mutex mtx;
    condition_variable cv;

    bool forward = true; // 车辆速度方向：默认向前

    JoyStick() {}

    void requestCarControl(float newSpeed, uint16_t newServo) {
        bool old_do_car_control;
        {
            lock_guard lk(mtx);
            old_do_car_control = do_car_control;
            speed = newSpeed;
            servo = newServo;
            do_car_control = true;
        }
        if (!old_do_car_control)
            cv.notify_one();
    }
    void requestBuzzer() {
        bool old_do_buzzer;
        {
            lock_guard lk(mtx);
            old_do_buzzer = do_buzzer;
            do_buzzer = true;
        }
        if (!old_do_buzzer)
            cv.notify_one();
    }
    void clearBuzzer() {
        lock_guard lk(mtx);
        do_buzzer = false;
    }
    void clearSampleOnce() {
        lock_guard lk(mtx);
        do_sample_once = false;
    }

    // ---------------- 事件处理 ----------------
    void processEvent(const input_event &ev) {
        optional<bool> force_speed_forward = nullopt;
        optional<float> new_speed = nullopt;
        optional<uint16_t> new_servo = nullopt;
        optional<bool> new_do_sample_once = nullopt;
        optional<bool> new_do_buzzer = nullopt;

        if (ev.type == EV_ABS) {
            switch (ev.code) {
            case ABS_X: { // 方向控制
                const struct input_absinfo *abs =
                    libevdev_get_abs_info(dev, ev.code);
                if (!abs)
                    break;
                // v [-1, 1]
                auto v = (double)(ev.value - abs->minimum) /
                             (abs->maximum - abs->minimum) * 2 -
                         1;
                new_servo = PWMSERVOMID + v * (PWMSERVOMID - PWMSERVOMIN);
                break;
            }
            case ABS_RZ: { // 右扳机，油门控制
                const struct input_absinfo *abs =
                    libevdev_get_abs_info(dev, ev.code);
                if (!abs)
                    break;
                // v [0, 1]
                auto v = (double)(ev.value - abs->minimum) /
                         (abs->maximum - abs->minimum);
                new_speed = (forward ? 1 : -1) * v * 0.5;
                break;
            }
            default:
                break;
            }
        } else if (ev.type == EV_KEY) {
            switch (ev.code) {
            case BTN_WEST: // 单次采图 (X)
                if (ev.value == 1) {
                    new_do_buzzer = true;
                    new_do_sample_once = true;
                }
                break;

            case BTN_NORTH: // 连续采图 (Y)
                if (ev.value == 1) {
                    new_do_buzzer = true;
                    // sampleMore = true;
                }
                break;

            case BTN_SOUTH: // 停止采图 (A)
                if (ev.value == 1) {
                    // sampleMore = false;
                    new_do_sample_once = false;
                    new_do_buzzer = true;
                }
                break;

            case BTN_START: // 向前
                if (ev.value == 1) {
                    force_speed_forward = true;
                    new_do_buzzer = true;
                }
                break;

            case BTN_SELECT: // 向后
                if (ev.value == 1) {
                    force_speed_forward = false;
                    new_do_buzzer = true;
                }
                break;

            default:
                new_speed = 0;
                break;
            }
        }
        if (new_speed || new_servo) {
            float speed;
            uint16_t servo;
            if (new_speed)
                speed = *new_speed;
            else {
                lock_guard lk(mtx);
                speed = this->speed;
            }
            if (force_speed_forward) {
                forward = *force_speed_forward;
                speed = (forward ? 1 : -1) * fabs(speed);
            }
            if (new_servo)
                servo = *new_servo;
            else {
                lock_guard lk(mtx);
                servo = this->servo;
            }
            requestCarControl(speed, servo);
        } else if (force_speed_forward) {
            forward = *force_speed_forward;
            float speed;
            uint16_t servo;
            {
                lock_guard lk(mtx);
                speed = this->speed;
                servo = this->servo;
            }
            speed = (forward ? 1 : -1) * fabs(speed);
            requestCarControl(speed, servo);
        }
        if (new_do_buzzer) {
            if (*new_do_buzzer)
                requestBuzzer();
            else
                clearBuzzer();
        }
        if (new_do_sample_once) {
            if (*new_do_sample_once)
                requestSampleOnce();
            else
                clearSampleOnce();
        }
    }

    void recv_thread_main() {
        struct input_event ev;
        struct pollfd pfd{joystick_fd, POLLIN, 0};

        while (!thread_stop.load(memory_order_relaxed)) {
            int pr = poll(&pfd, 1, 100); // 100ms 超时
            if (pr < 0) {
                if (errno == EINTR)
                    continue;
                break;
            }
            if (pr == 0)
                continue; // 超时，回到循环顶检查 thread_stop

            while (true) {
                auto rc =
                    libevdev_next_event(dev, LIBEVDEV_READ_FLAG_NORMAL, &ev);
                // 成功
                if (rc == LIBEVDEV_READ_STATUS_SUCCESS) {
                    processEvent(ev);
                    continue;
                }

                // ---- 失败 ----
                if (rc == LIBEVDEV_READ_STATUS_SYNC) {
                    // 出现 SYN_DROPPED
                    continue;
                }

                if (rc == -EAGAIN)
                    break;

                // 其他负值：设备错误/被拔掉
                return;
            }
        }
    }
};

int main(int argc, char const *argv[]) {
    auto js = JoyStick::create(); // 遥控手柄类
    if (!js) {
        cerr << "Failed to initialize joystick." << endl;
        return 1;
    }

    // 通信
    auto car = CarControl::create();
    if (!car) {
        cerr << "Failed to initialize uart!\n";
        return 1;
    }

    thread carControlThread([car, js]() {
        while (true) {
            js->waitEvent();
            float speed;
            uint16_t servo;
            if (js->takeCarControl(speed, servo)) {
                // cout << "Speed: " << speed << ", Servo: " << servo << endl;
                car->carControl(speed, servo); // 运动
            }

            if (js->takeBuzzer()) {
                car->buzzerSound(Buzzer::ding);
            }
        }
    });

    car->buzzerSound(Buzzer::start);

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
        // 读取图像
        Mat img;
        if (!capture.read(img))
            continue;

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

        putText(img, to_string(index), Point(10, 30), cv::FONT_HERSHEY_TRIPLEX,
                1, cv::Scalar(0, 0, 254), 1, CV_AA); // 显示图片保存序号
        imshow("img", img);
        int key = waitKey(10);
        if (key == 32) // 空格采图
            js->requestSampleOnce();
    }

    return 0;
}