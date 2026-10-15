#include "control.hpp"
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <poll.h>
#include <stdexcept>
#include <unistd.h>

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
        car->serialFd = car->serialPort.GetFileDescriptor();
        const int flags = fcntl(car->serialFd, F_GETFL);
        if (flags < 0 || fcntl(car->serialFd, F_SETFL, flags | O_NONBLOCK) < 0)
            throw std::runtime_error("Cannot enable nonblocking serial I/O");
        car->resetVelocity();
        car->recvThreadStop = false;
        auto raw = car.get();
        car->threadRecv = std::make_unique<std::thread>([raw] { raw->recvThreadMain(); });
        return car;
    } catch (const std::exception &e) {
        std::cerr << "Serial initialization failed: " << e.what() << '\n';
        return nullptr;
    }
}

void CarControl::setCommandTimeout(int milliseconds) {
    if (milliseconds < 0 || (milliseconds > 0 && milliseconds < 100))
        throw std::invalid_argument("Serial command timeout must be 0 or >=100ms");
    std::lock_guard lock(writeMutex);
    commandTimeoutMs = milliseconds;
}

void CarControl::latchStop() {
    commandExpired = true;
    resetVelocity();
}

void CarControl::pauseMotion() {
    std::lock_guard lock(writeMutex);
    ++stopEpoch;
    motionAllowed = false;
    carControlLocked(0, PWMSERVOMID);
}

bool CarControl::resumeMotion(uint64_t expectedEpoch) {
    std::lock_guard lock(writeMutex);
    if (hasFault() || expectedEpoch != stopEpoch.load()) return false;
    motionAllowed = true;
    return true;
}

void CarControl::checkCommandTimeout() {
    std::lock_guard lock(writeMutex);
    if (moving && commandTimeoutMs > 0 &&
        Clock::now() - lastCommand >= std::chrono::milliseconds(commandTimeoutMs)) {
        commandExpired = true; // Latched: old frames cannot restart the car.
        carControlLocked(0, PWMSERVOMID);
        std::cerr << "[Stop] Control command expired; restart program to rearm.\n";
    }
}

void CarControl::recvThreadMain() {
    size_t index = 0;
    std::array<uint8_t, USB_FRAME_LENMAX> buffer{};
    try {
        while (!recvThreadStop.load()) {
            checkCommandTimeout();
            pollfd fd{serialFd, POLLIN, 0};
            const int status = poll(&fd, 1, 20);
            if (status < 0) {
                if (errno == EINTR) continue;
                throw std::runtime_error("Serial receive poll failed");
            }
            if (fd.revents & (POLLERR | POLLHUP | POLLNVAL))
                throw std::runtime_error("Serial disconnected");
            if (!(fd.revents & POLLIN)) continue;
            uint8_t byte = 0;
            const auto n = ::read(serialFd, &byte, 1);
            if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR))
                continue;
            if (n <= 0) throw std::runtime_error("Serial receive failed");
            if (index == 0 && byte != USB_FRAME_HEAD) continue;
            if (index == 2 && (byte < USB_FRAME_LENMIN || byte > USB_FRAME_LENMAX)) {
                index = 0; continue;
            }
            buffer[index++] = byte;
            if (index >= 3 && index == buffer[2]) {
                uint8_t checksum = 0;
                for (size_t i = 0; i + 1 < index; ++i) checksum += buffer[i];
                if (checksum == buffer[index - 1]) translateBuffer(buffer);
                index = 0;
            }
        }
    } catch (const std::exception &e) {
        serialFault = true; // All later motion is vetoed.
        std::cerr << "[Error] Serial receive: " << e.what() << '\n';
        try { resetVelocity(); } catch (...) {}
    } catch (...) {
        serialFault = true;
        try { resetVelocity(); } catch (...) {}
    }
}

CarControl::~CarControl() noexcept {
    recvThreadStop = true;
    try { resetVelocity(); } catch (...) {}
    if (threadRecv && threadRecv->joinable()) threadRecv->join();
    try { if (serialPort.IsOpen()) serialPort.Close(); } catch (...) {}
}

void CarControl::translateBuffer(const std::array<uint8_t, USB_FRAME_LENMAX> &) {
    // Preserve current protocol behavior, without per-byte console output.
}

void CarControl::carControl(float speed, uint16_t servo) {
    std::lock_guard lock(writeMutex);
    if (!std::isfinite(speed)) {
        serialFault = true;
        carControlLocked(0, PWMSERVOMID);
        throw std::invalid_argument("Non-finite vehicle speed");
    }
    if (hasFault() || !motionAllowed.load()) { speed = 0; servo = PWMSERVOMID; }
    carControlLocked(speed, servo);
}

void CarControl::carControlLocked(float speed, uint16_t servo) {
    uint8_t buff[11]{}; // Preserve the existing 10-byte frame + trailing zero.
    buff[0] = USB_FRAME_HEAD; buff[1] = USB_ADDR_CARCTRL; buff[2] = 10;
    servo = std::clamp<uint16_t>(servo, PWMSERVOMIN, PWMSERVOMAX);
    std::memcpy(buff + 3, &speed, 4);
    std::memcpy(buff + 7, &servo, 2);
    for (int i = 0; i < 9; ++i) buff[9] += buff[i];
    writeBufferLocked(buff, sizeof(buff));
    moving = speed != 0;
    lastCommand = Clock::now();
}

void CarControl::buzzerSound(Buzzer sound) {
    uint8_t buff[6]{USB_FRAME_HEAD, USB_ADDR_BUZZER, 5,
                    static_cast<uint8_t>(static_cast<int>(sound) + 1), 0, 0};
    for (int i = 0; i < 4; ++i) buff[4] += buff[i];
    writeBuffer(buff, sizeof(buff));
}

void CarControl::writeBuffer(void *buffer, size_t len) {
    std::lock_guard lock(writeMutex);
    writeBufferLocked(buffer, len);
}

void CarControl::writeBufferLocked(const void *buffer, size_t len) {
    if (serialFd < 0) return; // Clean destruction after failed open.
    const auto deadline = Clock::now() + std::chrono::milliseconds(100);
    size_t sent = 0;
    try {
        while (sent < len) {
            const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
                deadline - Clock::now()).count();
            if (remaining <= 0) throw std::runtime_error("Serial write timed out");
            pollfd fd{serialFd, POLLOUT, 0};
            const int status = poll(&fd, 1, static_cast<int>(remaining));
            if (status < 0 && errno == EINTR) continue;
            if (status <= 0 || (fd.revents & (POLLERR | POLLHUP | POLLNVAL)))
                throw std::runtime_error("Serial write unavailable");
            const auto n = ::write(serialFd,
                static_cast<const uint8_t *>(buffer) + sent, len - sent);
            if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR))
                continue;
            if (n <= 0) throw std::runtime_error("Serial write failed");
            sent += static_cast<size_t>(n);
        }
    } catch (...) { serialFault = true; throw; }
}

void CarControl::sendHeart() {
    uint8_t buff[5]{USB_FRAME_HEAD, USB_ADDR_HEART, 4, 0, 0};
    buff[3] = buff[0] + buff[1] + buff[2];
    writeBuffer(buff, sizeof(buff));
}
