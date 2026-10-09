#pragma once
#include "control.hpp"
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <fcntl.h>
#include <functional>
#include <iostream>
#include <libevdev/libevdev.h>
#include <libudev.h>
#include <linux/input.h>
#include <memory>
#include <mutex>
#include <optional>
#include <poll.h>
#include <string>
#include <thread>
#include <unistd.h>
using std::atomic;
using std::cerr;
using std::condition_variable;
using std::cout;
using std::endl;
using std::lock_guard;
using std::make_unique;
using std::memory_order_relaxed;
using std::mutex;
using std::nullopt;
using std::optional;
using std::shared_ptr;
using std::string;
using std::thread;
using std::unique_lock;
using std::unique_ptr;
inline string get_joystick_device_path() {
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
    static shared_ptr<JoyStick> create(std::function<void()> onDisconnect,
                                      const string &devicePath = "") {
        auto joystick_path = devicePath.empty() ? get_joystick_device_path() : devicePath;
        if (joystick_path.empty()) {
            cerr << "Failed to find joystick device." << endl;
            return nullptr;
        }

        auto joystick = shared_ptr<JoyStick>(new JoyStick);
        joystick->onDisconnect = std::move(onDisconnect);
        int joystick_fd = open(joystick_path.c_str(), O_RDONLY | O_NONBLOCK);
        joystick->joystick_fd = joystick_fd;
        if (joystick_fd < 0) {
            cerr << "Failed to open joystick device " << joystick_path << "."
                 << endl;
            return nullptr;
        }
        cout << "Joystick device " << joystick_path << " opened." << endl;

        struct libevdev *dev = nullptr;
        if (libevdev_new_from_fd(joystick_fd, &dev) < 0) {
            return nullptr;
        }

        joystick->dev = dev;

        JoyStick *raw = joystick.get();
        joystick->recv_thread =
            make_unique<thread>([raw]() { raw->recv_thread_main(); });

        return joystick;
    }

    ~JoyStick() {
        requestShutdown();
        if (recv_thread && recv_thread->joinable())
            recv_thread->join();
        if (dev)
            libevdev_free(dev);
        if (joystick_fd >= 0)
            close(joystick_fd);
    }

    bool isStopped() const { return thread_stop.load() || disconnected.load(); }

    void requestShutdown() {
        {
            lock_guard lk(mtx);
            thread_stop = true;
            do_car_control = false;
            speed = 0;
            servo = PWMSERVOMID;
        }
        cv.notify_all();
    }

    // ---------------- 对外接口 ----------------

    bool takeCarControl(float &outSpeed, uint16_t &outServo) {
        lock_guard lk(mtx);
        if (isStopped() || !do_car_control)
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
    bool waitEvent() {
        unique_lock lk(mtx);
        cv.wait_for(lk, std::chrono::milliseconds(50), [this]() {
            return isStopped() || do_car_control || do_buzzer;
        });
        return !isStopped() && (do_car_control || do_buzzer);
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
    atomic<bool> disconnected{false};
    std::function<void()> onDisconnect;

    void disconnect() noexcept {
        disconnected = true;
        requestShutdown();
        try {
            if (onDisconnect)
                onDisconnect();
        } catch (...) {
            cerr << "[Error]: Joystick disconnect callback failed." << endl;
        }
    }

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
            if (isStopped())
                return;
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
                disconnect();
                return;
            }
            if (pr == 0)
                continue; // 超时，回到循环顶检查 thread_stop

            if (pfd.revents & (POLLHUP | POLLERR | POLLNVAL)) {
                disconnect();
                return;
            }

            while (!thread_stop.load()) {
                auto rc =
                    libevdev_next_event(dev, LIBEVDEV_READ_FLAG_NORMAL, &ev);
                // 成功
                if (rc == LIBEVDEV_READ_STATUS_SUCCESS) {
                    processEvent(ev);
                    continue;
                }

                // ---- 失败 ----
                if (rc == LIBEVDEV_READ_STATUS_SYNC) {
                    disconnect(); // 输入事件丢失时不继续沿用旧油门值
                    return;
                }

                if (rc == -EAGAIN)
                    break;

                // 其他负值：设备错误/被拔掉
                disconnect();
                return;
            }
        }
    }
};
