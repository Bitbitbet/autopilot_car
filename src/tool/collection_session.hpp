#pragma once
#include "joystick.hpp"
#include <stdexcept>

// 控制线程先停车、再退出；即使相机初始化或采图抛异常，也会完成回收。
class CollectionSession {
    shared_ptr<CarControl> car;
    shared_ptr<JoyStick> joystick;
    const std::atomic_bool &signal;
    std::atomic_bool failed{false};
    std::jthread worker;

  public:
    CollectionSession(shared_ptr<CarControl> car, shared_ptr<JoyStick> joystick,
                      const std::atomic_bool &signal)
        : car(std::move(car)), joystick(std::move(joystick)), signal(signal),
          worker([this](std::stop_token token) {
              try {
                  while (!token.stop_requested() && running()) {
                      if (!this->joystick->waitEvent())
                          continue;
                      if (token.stop_requested() || !running())
                          break;
                      float speed;
                      uint16_t servo;
                      if (this->joystick->takeCarControl(speed, servo))
                          this->car->carControl(speed, servo);
                      if (this->joystick->takeBuzzer())
                          this->car->buzzerSound(Buzzer::ding);
                  }
              } catch (const std::exception &e) {
                  cerr << "[Error]: Gamepad control failed: " << e.what() << endl;
                  failed = true;
              } catch (...) {
                  cerr << "[Error]: Unexpected gamepad control failure." << endl;
                  failed = true;
              }
              this->car->halt();
              this->joystick->requestShutdown();
          }) {}

    bool running() const {
        return !signal.load() && !failed.load() && !joystick->isStopped() &&
               !car->killAll.load() && !car->exitBoot.load();
    }
    bool hasFailed() const { return failed.load(); }
    void shutdown() noexcept {
        worker.request_stop();
        car->halt();
        joystick->requestShutdown();
        if (worker.joinable())
            worker.join();
    }
    ~CollectionSession() { shutdown(); }
};
