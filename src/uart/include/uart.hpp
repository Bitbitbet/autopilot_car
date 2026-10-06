#pragma once

#include <array>
#include <cstdint>
#include <cstdio>
#include <libserial/SerialPort.h> // 串口通信
#include <libserial/SerialPortConstants.h>
#include <math.h> // 数学函数类
#include <memory>
#include <stdint.h> // 整型数据类
#include <string.h>
#include <thread>

using namespace LibSerial;
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

#define PWMSERVOMAX 1900 // 舵机PWM最大值（左）1840
#define PWMSERVOMID 1500 // 舵机PWM中值 1520
#define PWMSERVOMIN 1100 // 舵机PWM最小值（右）1200

/**
 * @brief 32位数据内存对齐/联合体
 */
typedef union {
    uint8_t buff[4];
    float float32;
    int int32;
} Bit32Union;

/**
 * @brief 16位数据内存对齐/联合体
 */
typedef union {
    uint8_t buff[2];
    int int16;
    uint16_t uint16;
} Bit16Union;

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

class Uart {
  private:
    std::unique_ptr<std::thread> threadRecv; // 串口接收子线程
    std::unique_ptr<SerialPort> serialPort;
    Uart();

    /**
     * @brief 串口通信协议数据转换
     */
    void translateBuffer(const std::array<uint8_t, USB_FRAME_LENMAX> &buffer);

  public:
    Uart(Uart &&);
    ~Uart();

    static std::shared_ptr<Uart> create(std::string portName);

    bool keypress = false; // 按键
    bool killAll = false;  // 杀进程
    bool exitBoot = false; // 退出boot

    /**
     * @brief 速度+方向控制
     *
     * @param speed 速度：m/s
     * @param servo 方向：PWM（500~2500）
     */
    void carControl(float speed, uint16_t servo);

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
