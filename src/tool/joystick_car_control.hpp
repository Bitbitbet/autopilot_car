#pragma once

#include "control.hpp"
#include "joystick.hpp"
#include <stop_token>
#include <thread>
using std::jthread;
using std::stop_token;
inline jthread startJoystickControlThread(shared_ptr<CarControl> car,
                                          shared_ptr<JoyStick> js) {
    jthread worker(
        [car = std::move(car), js = std::move(js)](stop_token token) {
            try {
                while (!token.stop_requested() && !js->isStopped()) {
                    if (!js->waitEvent())
                        continue;
                    if (token.stop_requested() || js->isStopped())
                        break;
                    float speed;
                    uint16_t servo;
                    if (js->takeCarControl(speed, servo))
                        car->carControl(speed, servo);
                    if (js->takeBuzzer())
                        car->buzzerSound(Buzzer::ding);
                }
            } catch (const std::exception &e) {
                cerr << "[Error]: Gamepad control failed: " << e.what() << endl;
            } catch (...) {
                cerr << "[Error]: Unexpected gamepad control failure." << endl;
            }
            car->resetVelocity();
            js->requestShutdown();
        });

    return worker;
}