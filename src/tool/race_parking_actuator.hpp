#pragma once
#include "control.hpp"
// Camera-only mission preview shares decisions but never constructs CarControl.
class RaceParkingActuator {
    std::shared_ptr<CarControl> actual;std::atomic<uint64_t> epoch{0};
public:
    explicit RaceParkingActuator(bool control){if(control){actual=CarControl::create();if(!actual)throw std::runtime_error("Cannot open vehicle serial");}}
    bool hasFault()const{return actual&&actual->hasFault();}
    void setCommandTimeout(int ms){if(actual)actual->setCommandTimeout(ms);}
    void pauseMotion(){if(actual)actual->pauseMotion();else ++epoch;}
    void latchStop(){if(actual)actual->latchStop();else ++epoch;}
    uint64_t motionEpoch()const{return actual?actual->motionEpoch():epoch.load();}
    bool resumeMotion(uint64_t n){return actual?actual->resumeMotion(n):n==epoch;}
    void carControl(float speed,uint16_t servo){if(actual)actual->carControl(speed,servo);}
};
