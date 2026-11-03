#include "control.hpp"
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <poll.h>
#include <stdexcept>
#include <unistd.h>

using std::array;
using std::cerr;
using std::clamp;
using std::endl;
using std::lock_guard;
using std::memory_order_relaxed;
using std::thread;
using std::chrono::milliseconds;

using namespace LibSerial;
using Clock = std::chrono::steady_clock;
CarControl::CarControl() = default;

std::shared_ptr<CarControl> CarControl::create() {
    auto car = std::shared_ptr<CarControl>(new CarControl);
    try {
        car->serialPort.Open("/dev/ttyUSB0");
        car->serialPort.SetBaudRate(BaudRate::BAUD_115200);
        car->serialPort.SetCharacterSize(CharacterSize::CHAR_SIZE_8);
        car->serialPort.SetFlowControl(FlowControl::FLOW_CONTROL_NONE);
        car->serialPort.SetParity(Parity::PARITY_NONE);
        car->serialPort.SetStopBits(StopBits::STOP_BITS_1);
        auto serialFd = car->serialPort.GetFileDescriptor();
        const int flags = fcntl(serialFd, F_GETFL);
        if (flags < 0 || fcntl(serialFd, F_SETFL, flags | O_NONBLOCK) < 0)
            throw std::runtime_error("Cannot enable nonblocking serial I/O");
        car->resetVelocity();
        car->recvThreadStop = false;
        auto raw = car.get();
        car->recvThread = thread([=] { raw->recvThreadMain(); });

        return car;
    } catch (const std::exception &e) {
        cerr << "Serial initialization failed: " << e.what() << endl;
        return nullptr;
    }
}

void CarControl::latch_stop() {
    commandExpired = true;
    resetVelocity();
}

void CarControl::recvThreadMain() {
    size_t index = 0;
    array<uint8_t, USB_FRAME_LENMAX> buffer{};
    auto serialFd = serialPort.GetFileDescriptor();
    try {
        while (!recvThreadStop.load()) {
            pollfd fd{serialFd, POLLIN, 0};
            const int status = poll(&fd, 1, 20);
            if (status < 0) {
                if (errno == EINTR)
                    continue;
                throw std::runtime_error("Serial receive poll failed");
            }
            if (fd.revents & (POLLERR | POLLHUP | POLLNVAL))
                throw std::runtime_error("Serial disconnected");
            if (!(fd.revents & POLLIN))
                continue;
            uint8_t byte = 0;
            const auto n = read(serialFd, &byte, 1);
            if (n < 0 &&
                (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR))
                continue;
            if (n <= 0)
                throw std::runtime_error("Serial receive failed");
            if (index == 0 && byte != USB_FRAME_HEAD)
                continue;
            if (index == 2 &&
                (byte < USB_FRAME_LENMIN || byte > USB_FRAME_LENMAX)) {
                index = 0;
                continue;
            }
            buffer[index++] = byte;
            if (index >= 3 && index == buffer[2]) {
                uint8_t checksum = 0;
                for (size_t i = 0; i + 1 < index; ++i)
                    checksum += buffer[i];
                if (checksum == buffer[index - 1])
                    translateBuffer(buffer);
                index = 0;
            }
        }
    } catch (const std::exception &e) {
        serialFault = true; // All later motion is vetoed.
        cerr << "[Error] Serial receive: " << e.what() << endl;
        try {
            resetVelocity();
        } catch (...) {
        }
    } catch (...) {
        serialFault = true;
        try {
            resetVelocity();
        } catch (...) {
        }
    }
}

CarControl::~CarControl() noexcept {
    try {
        resetVelocity();
    } catch (...) {
    }
    recvThreadStop.store(true, memory_order_relaxed);
    recvThread.join();
    try {
        if (serialPort.IsOpen())
            serialPort.Close();
    } catch (...) {
    }
}

void CarControl::translateBuffer(const array<uint8_t, USB_FRAME_LENMAX> &) {
    // Currently do nothing here
}

void CarControl::carControl(float speed, uint16_t servo) {
    lock_guard lock(writeMutex);
    if (!std::isfinite(speed)) {
        serialFault = true;
        carControlLocked(0, PWMSERVOMID);
        throw std::invalid_argument("Non-finite vehicle speed");
    }
    if (hasFault()) {
        speed = 0;
        servo = PWMSERVOMID;
    }
    carControlLocked(speed, servo);
}

void CarControl::carControlLocked(float speed, uint16_t servo) {
    uint8_t buff[11]{}; // Preserve the existing 10-byte frame + trailing zero.
    buff[0] = USB_FRAME_HEAD;
    buff[1] = USB_ADDR_CARCTRL;
    buff[2] = 10;
    servo = clamp<uint16_t>(servo, PWMSERVOMIN, PWMSERVOMAX);
    memcpy(buff + 3, &speed, 4);
    memcpy(buff + 7, &servo, 2);
    for (int i = 0; i < 9; ++i)
        buff[9] += buff[i];
    writeBufferLocked(buff, sizeof(buff));
    moving = speed != 0;
}

void CarControl::buzzerSound(Buzzer sound) {
    uint8_t buff[6]{USB_FRAME_HEAD,
                    USB_ADDR_BUZZER,
                    5,
                    static_cast<uint8_t>(static_cast<int>(sound) + 1),
                    0,
                    0};
    for (int i = 0; i < 4; ++i)
        buff[4] += buff[i];
    writeBuffer(buff, sizeof(buff));
}

void CarControl::writeBuffer(void *buffer, size_t len) {
    std::lock_guard lock(writeMutex);
    writeBufferLocked(buffer, len);
}

void CarControl::writeBufferLocked(const void *buffer, size_t len) {
    for (size_t i = 0; i < len; ++i)
        serialPort.WriteByte(((uint8_t *)buffer)[i]);
    serialPort.DrainWriteBuffer();
}

void CarControl::sendHeart() {
    uint8_t buff[5]{USB_FRAME_HEAD, USB_ADDR_HEART, 4, 0, 0};
    buff[3] = buff[0] + buff[1] + buff[2];
    writeBuffer(buff, sizeof(buff));
}
