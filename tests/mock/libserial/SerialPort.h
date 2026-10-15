#pragma once
#include <stdexcept>
#include <sys/socket.h>
#include <unistd.h>
namespace LibSerial {
enum class BaudRate { BAUD_115200 };
enum class CharacterSize { CHAR_SIZE_8 };
enum class FlowControl { FLOW_CONTROL_NONE };
enum class Parity { PARITY_NONE };
enum class StopBits { STOP_BITS_1 };
inline int peerFd=-1, deviceFd=-1;
class SerialPort {
    int fd=-1;
public:
    void Open(const char *) {
        int pair[2];
        if(socketpair(AF_UNIX,SOCK_STREAM,0,pair)) throw std::runtime_error("socketpair");
        fd=pair[0]; peerFd=pair[1]; deviceFd=fd;
    }
    void Close() { if(fd>=0) close(fd); fd=-1; deviceFd=-1; }
    bool IsOpen() const { return fd>=0; }
    int GetFileDescriptor() const { return fd; }
    void SetBaudRate(BaudRate) {}
    void SetCharacterSize(CharacterSize) {}
    void SetFlowControl(FlowControl) {}
    void SetParity(Parity) {}
    void SetStopBits(StopBits) {}
};
}
