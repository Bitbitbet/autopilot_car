#include "control.hpp"
#include <array>
#include <atomic>
#include <iostream>
#include <libserial/SerialPort.h>
#include <memory>
#include <string>

using namespace LibSerial;
using std::cerr;
using std::endl;
using std::make_unique;
using std::memory_order_relaxed;
using std::shared_ptr;
using std::string;
using std::thread;

CarControl::CarControl() = default;

shared_ptr<CarControl> CarControl::create() {
    auto carControl = shared_ptr<CarControl>(new CarControl);

    const char *portName = "/dev/ttyUSB0";
    try {
        carControl->serialPort.Open(portName);                     // 打开串口
        carControl->serialPort.SetBaudRate(BaudRate::BAUD_115200); // 设置波特率
        carControl->serialPort.SetCharacterSize(
            CharacterSize::CHAR_SIZE_8); // 8位数据位
        carControl->serialPort.SetFlowControl(
            FlowControl::FLOW_CONTROL_NONE);                       // 设置流控
        carControl->serialPort.SetParity(Parity::PARITY_NONE);     // 无校验
        carControl->serialPort.SetStopBits(StopBits::STOP_BITS_1); // 1个停止位
    } catch (const OpenFailed &) {
        cerr << "Serial port: " << portName << "open failed ..." << endl;
        // error = -2;
        return nullptr;
    } catch (const AlreadyOpen &) {
        cerr << "Serial port: " << portName << "open failed ..." << endl;
        // error = -3;
        return nullptr;
    } catch (...) {
        cerr << "Serial port: " << portName << " received exception ..."
             << endl;
        // error = -4;
        return nullptr;
    }

    carControl->recvThreadStop.store(false, memory_order_relaxed);
    auto raw = carControl.get();

    carControl->threadRecv =
        make_unique<thread>([raw]() { raw->recvThreadMain(); });

    // error = 0;
    return carControl;
}

void CarControl::recvThreadMain() {
    size_t index = 0;
    std::array<uint8_t, USB_FRAME_LENMAX> buffer; // 临时缓冲数据
    while (!recvThreadStop.load(memory_order_relaxed)) {
        uint8_t byte;
        try {
            serialPort.ReadByte(byte, 100); // 定期检查退出，避免永久阻塞
        } catch (const ReadTimeout &) {
            continue;
        } catch (const std::exception &e) {
            cerr << "[Error]: Serial receive failed: " << e.what() << endl;
            killAll = true;
            halt();
            break;
        }

        /* 起始帧不是USB_FRAME_HEAD就不开始接收 */
        if (index == 0 && byte != USB_FRAME_HEAD) {
            continue;
        }
        /* 长度过小或过大，重新接收 */
        if (index == 2 &&
            (byte > USB_FRAME_LENMAX || byte < USB_FRAME_LENMIN)) {
            index = 0;
            continue;
        }
        /* 接收完毕，计算校验、提交后重置 */
        if (index >= 3 && index == buffer[2] - 1) {

            uint8_t check = 0;
            for (size_t i = 0; i < index; ++i)
                check += buffer[i];

            if (check == byte) {
                buffer[index] = byte;
                translateBuffer(buffer);
            }
            index = 0;
            continue;
        }

        buffer[index] = byte;
        ++index;
    }
}

CarControl::~CarControl() {
    stop();
    recvThreadStop.store(true, memory_order_relaxed);
    if (threadRecv && threadRecv->joinable())
        threadRecv->join();
    try {
        if (serialPort.IsOpen())
            serialPort.Close();
    } catch (const std::exception &e) {
        cerr << "[Error]: Serial close failed: " << e.what() << endl;
    }
};

void CarControl::stop() noexcept {
    try {
        if (serialPort.IsOpen()) {
            carControl(0, PWMSERVOMID);
            cerr << "[STOP] Zero-speed command sent (no hardware acknowledgement)."
                 << endl;
        }
    } catch (const std::exception &e) {
        cerr << "[Error]: Stop transmission failed: " << e.what() << endl;
    } catch (...) {
        cerr << "[Error]: Stop transmission failed." << endl;
    }
}

void CarControl::halt() noexcept {
    halted = true;
    stop();
}
void CarControl::translateBuffer(
    const std::array<uint8_t, USB_FRAME_LENMAX> &buffer) {
    /* DEBUG 打印接收的帧 */
    printf("USB Frame Received: [");
    bool first = true;
    for (size_t i = 0; i < buffer[2]; ++i) {
        if (first) {
            first = false;
        } else {
            printf(", ");
        }
        printf("%d", buffer[i]);
    }
    printf("]\n");

    switch (buffer[1]) {
    case USB_ADDR_KEY: // 接收按键信息
        if (buffer[3] == 1) {
            keypress = true;
            killAll = false;
            exitBoot = false;
        } else if (buffer[3] == 2) {
            keypress = false;
            killAll = true;
            exitBoot = false;
        } else if (buffer[3] == 3) {
            keypress = false;
            killAll = false;
            exitBoot = true;
        }

        break;
    case 6: // 接收按键信息
        if (buffer[3] == 1 || (buffer[3] > 100 && buffer[3] < 2000)) // 发车
        {
            keypress = true;
            killAll = false;
            exitBoot = false;
        } else if (buffer[3] == 2 || (buffer[3] > 2000 && buffer[3] < 5000)) {
            keypress = false;
            killAll = true;
            exitBoot = false;
        } else if (buffer[3] == 3 || buffer[3] > 5000) {
            keypress = false;
            killAll = false;
            exitBoot = true;
        }
        break;

    default:
        break;
    }
}
void CarControl::carControl(float speed, uint16_t servo) {
    std::lock_guard<std::mutex> lock(writeMutex);
    if (halted || killAll || exitBoot) { // 停车后不得发送新的行驶指令
        speed = 0;
        servo = PWMSERVOMID;
    }
    uint8_t buff[11]{}; // 保留额外字节，固定为零
    uint8_t check = 0; // 校验位
    Bit32Union bit32U;
    Bit16Union bit16U;

    buff[0] = USB_FRAME_HEAD;   // 通信帧头
    buff[1] = USB_ADDR_CARCTRL; // 地址
    buff[2] = 10;               // 帧长

    bit32U.float32 = speed; // X轴线速度
    for (int i = 0; i < 4; i++)
        buff[i + 3] = bit32U.buff[i];

    bit16U.uint16 = servo; // Y轴线速度
    buff[7] = bit16U.buff[0];
    buff[8] = bit16U.buff[1];

    for (int i = 0; i < 9; i++)
        check += buff[i];
    buff[9] = check; // 校验位

    writeBufferUnlocked(buff, 11);
}

/**
 * @brief 蜂鸣器音效控制
 *
 * @param sound
 */
void CarControl::buzzerSound(Buzzer sound) {
    uint8_t buff[6]{}; // 保留额外字节，固定为零
    uint8_t check = 0; // 校验位

    buff[0] = USB_FRAME_HEAD;  // 帧头
    buff[1] = USB_ADDR_BUZZER; // 地址
    buff[2] = 5;               // 帧长
    switch (sound) {
    case Buzzer::ok: // 确认
        buff[3] = 1;
        break;
    case Buzzer::warning: // 报警
        buff[3] = 2;
        break;
    case Buzzer::finish: // 完成
        buff[3] = 3;
        break;
    case Buzzer::ding: // 提示
        buff[3] = 4;
        break;
    case Buzzer::start: // 开机
        buff[3] = 5;
        break;
    }

    for (size_t i = 0; i < 4; i++)
        check += buff[i];
    buff[4] = check;

    writeBuffer(buff, 6);
}
void CarControl::writeBuffer(void *buffer, size_t len) {
    std::lock_guard<std::mutex> lock(writeMutex); // 防止完整帧交叉发送
    writeBufferUnlocked(buffer, len);
}

void CarControl::writeBufferUnlocked(void *buffer, size_t len) {
    for (size_t i = 0; i < len; ++i)
        serialPort.WriteByte(((uint8_t *)buffer)[i]);
    serialPort.DrainWriteBuffer();
}

/**
 * @brief 发送心跳信号
 *
 */
void CarControl::sendHeart() {
    uint8_t buff[5]{}; // 保留额外字节，固定为零
    uint8_t check = 0; // 校验位

    buff[0] = USB_FRAME_HEAD; // 通信帧头
    buff[1] = USB_ADDR_HEART; // 地址
    buff[2] = 4;              // 帧长

    for (int i = 0; i < 3; i++)
        check += buff[i];
    buff[3] = check; // 校验位

    writeBuffer(buff, 5);
}
