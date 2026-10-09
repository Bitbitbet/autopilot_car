#pragma once
#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

// Test double only: no physical serial device is opened.
namespace LibSerial {
struct OpenFailed : std::runtime_error { using runtime_error::runtime_error; };
struct AlreadyOpen : std::runtime_error { using runtime_error::runtime_error; };
struct ReadTimeout : std::runtime_error { using runtime_error::runtime_error; };
enum class BaudRate { BAUD_115200 };
enum class CharacterSize { CHAR_SIZE_8 };
enum class FlowControl { FLOW_CONTROL_NONE };
enum class Parity { PARITY_NONE };
enum class StopBits { STOP_BITS_1 };
inline std::atomic_bool failOpen{false}, failWrite{false}, failRead{false};
inline std::mutex outputMutex;
inline std::vector<std::vector<uint8_t>> frames;
class SerialPort {
    bool opened = false;
    std::vector<uint8_t> pending;
public:
    void Open(const char *) {
        if (failOpen) throw OpenFailed("test open failure");
        opened = true;
    }
    bool IsOpen() const { return opened; }
    void Close() { opened = false; }
    void SetBaudRate(BaudRate) {}
    void SetCharacterSize(CharacterSize) {}
    void SetFlowControl(FlowControl) {}
    void SetParity(Parity) {}
    void SetStopBits(StopBits) {}
    void ReadByte(uint8_t &, size_t timeout) {
        if (!timeout) throw std::runtime_error("unbounded receive");
        if (failRead) throw std::runtime_error("test disconnect");
        std::this_thread::sleep_for(std::chrono::milliseconds(timeout));
        throw ReadTimeout("test timeout");
    }
    void WriteByte(uint8_t byte) {
        if (failWrite) throw std::runtime_error("test write failure");
        pending.push_back(byte);
    }
    void DrainWriteBuffer() {
        std::lock_guard<std::mutex> lock(outputMutex);
        frames.push_back(pending);
        pending.clear();
    }
};
}
