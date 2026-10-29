#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <libserial/SerialPort.h>
#include <memory>
#include <mutex>
#include <thread>

// USB通信帧
#define USB_FRAME_HEAD 0x42 // USB通信帧头
#define USB_FRAME_LENMIN 4  // USB通信帧最短字节长度
#define USB_FRAME_LENMAX 12 // USB通信帧最长字节长度

// USB通信地址
#define USB_ADDR_HEART 0   // 心跳信号，特指Boot
#define USB_ADDR_CARCTRL 1 // 智能车速度+方向控制
#define USB_ADDR_BUZZER 4  // 蜂鸣器音效控制
#define USB_ADDR_LED 5     // LED灯效控制
#define USB_ADDR_KEY 0x10  // 按键信息

constexpr size_t PWMSERVOMAX = 1900; // 舵机PWM最大值（左）
constexpr size_t PWMSERVOMID = 1500; // 舵机PWM中值
constexpr size_t PWMSERVOMIN = 1100; // 舵机PWM最小值（右）

/**
 * @brief 蜂鸣器音效
 */
enum class Buzzer : uint8_t {
    ok = 0,  // 确认
    warning, // 报警
    finish,  // 完成
    ding,    // 提示
    start,   // 开机
};

class CarControl {
  private:
    std::unique_ptr<std::thread> recvThread; // 串口接收子线程
    LibSerial::SerialPort serialPort;
    std::atomic_bool recvThreadStop{true};
    std::atomic_bool serialFault{false};
    std::atomic_bool commandExpired{false};
    std::mutex writeMutex;
    int serialFd = -1;
    bool moving = false;
    std::chrono::steady_clock::time_point lastCommand;
    void writeBufferLocked(const void *buffer, size_t len);
    void carControlLocked(float speed, uint16_t servo);

    CarControl();

    /**
     * @brief 串口通信协议数据转换
     */
    void translateBuffer(const std::array<uint8_t, USB_FRAME_LENMAX> &buffer);

    void recvThreadMain();

  public:
    ~CarControl() noexcept;
    bool hasFault() const {
        return serialFault.load() || commandExpired.load();
    }

    /* Exit/UI failure: prevent any racing worker from restarting. */
    void latch_stop();

    bool exitBoot = false;
    bool keypress = false;

    static std::shared_ptr<CarControl> create();

    /**
     * @brief 速度+方向控制
     *
     * @param speed 速度：m/s
     * @param servo 方向：PWM（500~2500）
     */
    void carControl(float speed, uint16_t servo);

    void resetVelocity() { carControl(0, PWMSERVOMID); }

    /**
     * @brief 蜂鸣器音效控制
     *
     * @param sound
     */
    void buzzerSound(Buzzer sound);

    void writeBuffer(void *buffer, size_t len);

    /**
     * @brief 发送心跳信号
     *
     */
    void sendHeart();
};
