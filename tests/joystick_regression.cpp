#include "collection_session.hpp"
#include "stop_signal.hpp"
#include <cassert>
#include <sys/stat.h>

using namespace std::chrono_literals;

template <typename Predicate> void eventually(Predicate predicate) {
    auto deadline = std::chrono::steady_clock::now() + 2s;
    while (!predicate() && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(5ms);
    assert(predicate());
}

float lastSpeed() {
    std::lock_guard lock(LibSerial::outputMutex);
    for (auto it = LibSerial::frames.rbegin(); it != LibSerial::frames.rend(); ++it) {
        if ((*it)[1] != USB_ADDR_CARCTRL) continue;
        float speed;
        std::memcpy(&speed, it->data() + 3, sizeof speed);
        return speed;
    }
    return -1;
}

struct Device {
    std::string path = "/tmp/icar-joystick-test-" + std::to_string(getpid());
    int writer;
    Device() {
        unlink(path.c_str());
        assert(mkfifo(path.c_str(), 0600) == 0);
        // Keeper avoids a HUP before the receiver has opened its end.
        writer = open(path.c_str(), O_RDWR | O_NONBLOCK);
        assert(writer >= 0);
    }
    void throttle() {
        input_event event{};
        event.type = EV_ABS;
        event.code = ABS_RZ;
        event.value = 128;
        assert(write(writer, &event, sizeof event) == sizeof event);
    }
    void disconnect() { close(writer); writer = -1; }
    ~Device() { if (writer >= 0) close(writer); unlink(path.c_str()); }
};

int main() {
    assert(!JoyStick::create([] {}, "/tmp/no-such-car-joystick"));
    {
        Device device;
        auto car = CarControl::create();
        auto joystick = JoyStick::create([car] { car->halt(); }, device.path);
        assert(joystick);
        std::atomic_bool signal{false};
        CollectionSession session(car, joystick, signal);
        device.throttle();
        eventually([] { return lastSpeed() > 0; });
        device.disconnect();
        eventually([&] { return joystick->isStopped() && lastSpeed() == 0; });
        assert(!session.running());
        car->carControl(0.5f, 1800); // A stale command cannot restart the motor.
        assert(lastSpeed() == 0);
        auto start = std::chrono::steady_clock::now();
        session.shutdown();
        joystick.reset();
        assert(std::chrono::steady_clock::now() - start < 1s);
    }
    for (int stopSignal : {SIGINT, SIGTERM, SIGHUP}) {
        Device device;
        auto car = CarControl::create();
        auto joystick = JoyStick::create([car] { car->halt(); }, device.path);
        CollectionSession session(car, joystick, car_signal::requested);
        car_signal::install();
        std::raise(stopSignal); // Worker must stop even if camera/main is busy.
        eventually([&] { return joystick->isStopped() && lastSpeed() == 0; });
        car_signal::requested = false;
    }
    {
        Device device;
        auto car = CarControl::create();
        auto joystick = JoyStick::create([car] { car->halt(); }, device.path);
        std::atomic_bool signal{false};
        auto start = std::chrono::steady_clock::now();
        try {
            CollectionSession session(car, joystick, signal);
            device.throttle();
            eventually([] { return lastSpeed() > 0; });
            throw std::runtime_error("camera failure");
        } catch (const std::runtime_error &) {}
        assert(lastSpeed() == 0);
        assert(joystick->isStopped());
        assert(std::chrono::steady_clock::now() - start < 1s);
    }
    {
        Device device;
        auto car = CarControl::create();
        auto joystick = JoyStick::create([car] { car->halt(); }, device.path);
        std::atomic_bool signal{false};
        CollectionSession session(car, joystick, signal);
        LibSerial::failWrite = true;
        device.throttle();
        eventually([&] { return joystick->isStopped(); });
        assert(session.hasFailed());
        LibSerial::failWrite = false;
        session.shutdown();
        assert(lastSpeed() == 0);
    }
    std::cout << "PASS: joystick disconnect, stale drive rejection, SIGINT/SIGTERM/SIGHUP, camera failure, write failure, thread cleanup\n";
}
