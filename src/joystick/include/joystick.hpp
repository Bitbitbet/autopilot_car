#pragma once
#include "control.hpp"
#include <atomic>
#include <condition_variable>
#include <fcntl.h>
#include <functional>
#include <libudev.h>
#include <linux/input.h>
#include <memory>
#include <mutex>
#include <poll.h>
#include <string>
#include <thread>
#include <unistd.h>

class JoyStick {
  public:
    static std::shared_ptr<JoyStick> create(std::function<void()> onDisconnect,
                                            const std::string &devicePath = "");

    ~JoyStick();

    bool isStopped() const { return thread_stop.load() || disconnected.load(); }

    void requestShutdown();

    // ---------------- 对外接口 ----------------

    bool takeCarControl(float &outSpeed, uint16_t &outServo);

    bool takeBuzzer();

    bool takeSampleOnce();
    bool waitEvent();
    void requestSampleOnce();

  private:
    int joystick_fd;
    struct libevdev *dev;
    std::thread recv_thread;
    std::atomic<bool> thread_stop{false};
    std::atomic<bool> disconnected{false};
    std::function<void()> onDisconnect;

    void disconnect() noexcept;

    // ---------------- 共享状态 ----------------
    bool do_sample_once = false;
    bool do_car_control = false;
    bool do_buzzer = false;
    float speed = 0;              // 车速：m/s
    uint16_t servo = PWMSERVOMID; // 打舵：PWM
    std::mutex mtx;
    std::condition_variable cv;

    bool forward = true; // 车辆速度方向：默认向前

    JoyStick();

    void requestCarControl(float newSpeed, uint16_t newServo);
    void requestBuzzer();
    void clearBuzzer();
    void clearSampleOnce();

    // 事件处理
    void processEvent(const input_event &ev);

    void recv_thread_main();
};
