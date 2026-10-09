#include "control.hpp"
#include <cassert>
#include <cstring>
#include <iostream>

static void checkFrame(const std::vector<uint8_t> &frame, size_t size,
                       uint8_t address) {
    assert(frame.size() == size);
    assert(frame[0] == USB_FRAME_HEAD && frame[1] == address);
    assert(frame[2] == size - 1 && frame.back() == 0);
    uint8_t checksum = 0;
    for (size_t i = 0; i < size - 2; ++i) checksum += frame[i];
    assert(frame[size - 2] == checksum);
}

int main() {
    LibSerial::failOpen = true;
    assert(!CarControl::create()); // failed initialization must not dereference a thread
    LibSerial::failOpen = false;
    auto control = CarControl::create();
    assert(control);
    control->carControl(0.05f, 1600);
    control->buzzerSound(Buzzer::ding);
    control->sendHeart();
    {
        std::lock_guard<std::mutex> lock(LibSerial::outputMutex);
        checkFrame(LibSerial::frames[0], 11, USB_ADDR_CARCTRL);
        float speed;
        std::memcpy(&speed, LibSerial::frames[0].data() + 3, sizeof speed);
        assert(speed == 0.05f);
        assert(LibSerial::frames[0][7] == 0x40 && LibSerial::frames[0][8] == 0x06);
        checkFrame(LibSerial::frames[1], 6, USB_ADDR_BUZZER);
        assert(LibSerial::frames[1][3] == 4);
        checkFrame(LibSerial::frames[2], 5, USB_ADDR_HEART);
    }
    LibSerial::failWrite = true;
    control->stop(); // transport failure must not escape noexcept cleanup
    LibSerial::failWrite = false;
    LibSerial::failRead = true;
    for (int i = 0; i < 50 && !control->killAll; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    assert(control->killAll);
    control->carControl(1.0f, 1800); // a racing drive command must remain stopped
    {
        std::lock_guard<std::mutex> lock(LibSerial::outputMutex);
        float speed;
        std::memcpy(&speed, LibSerial::frames.back().data() + 3, sizeof speed);
        assert(speed == 0.0f);
    }
    LibSerial::failRead = false;
    auto start = std::chrono::steady_clock::now();
    control.reset();
    assert(std::chrono::steady_clock::now() - start < std::chrono::seconds(1));
    {
        std::lock_guard<std::mutex> lock(LibSerial::outputMutex);
        const auto &frame = LibSerial::frames.back();
        checkFrame(frame, 11, USB_ADDR_CARCTRL);
        float speed;
        std::memcpy(&speed, frame.data() + 3, sizeof speed);
        assert(speed == 0.0f);
        assert(frame[7] == (PWMSERVOMID & 0xff) && frame[8] == (PWMSERVOMID >> 8));
    }
    auto waiting = CarControl::create();
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    start = std::chrono::steady_clock::now();
    waiting.reset(); // the worker is waiting for input, rather than disconnected
    assert(std::chrono::steady_clock::now() - start < std::chrono::seconds(1));
    std::cout << "PASS: serial payload/checksum/padding, failed open, disconnect stop, bounded cleanup\n";
}
