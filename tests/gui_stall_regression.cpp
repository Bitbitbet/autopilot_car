#include "control.hpp"
#include "trackview_display.hpp"
#include "trackview_pilot.hpp"
#include <atomic>
#include <chrono>
#include <cstring>
#include <iostream>
#include <poll.h>
#include <stdexcept>
#include <thread>
#include <unistd.h>

using namespace std::chrono_literals;
void check(bool ok,const char *message) { if(!ok) throw std::runtime_error(message); }

int main() {
    // Same controller/display mailbox and real serial implementation as the program.
    // Socketpair replaces the hardware, and sleeping renderer replaces network X11.
    auto car=CarControl::create();
    if(!car) return 1;
    car->setCommandTimeout(250);
    TrackviewDisplay display;
    std::atomic_bool stop{false}, cameraStall{false}, loopFault{false};
    std::atomic_int cycles{0};
    std::atomic<float> lastSpeed{0};
    std::jthread sink([&] {
        while(!stop) {
            pollfd p{LibSerial::peerFd,POLLIN,0};
            if(poll(&p,1,20)>0 && (p.revents&POLLIN)) {
                uint8_t bytes[1024];
                if(read(p.fd,bytes,sizeof(bytes))<=0) break;
            }
        }
    });
    std::jthread control([&] {
        TrackviewConfig cfg;
        AutoPilot pilot(cfg);
        std::vector<PointX> left,right;
        for(int row=219;row>=30;row-=2) {
            left.emplace_back(row,70); right.emplace_back(row,250);
        }
        cv::Mat image(240,320,CV_8UC3,cv::Scalar(0)), bin(240,320,CV_8UC1,cv::Scalar(0));
        try {
            while(!stop) {
                if(cameraStall.exchange(false)) std::this_thread::sleep_for(400ms);
                if(car->hasFault()) { loopFault=true; break; }
                pilot.update(left,right,1.f/30,lastSpeed.load());
                float speed=pilot.speedFor(0.25f,1.f/30);
                car->carControl(speed,pilot.servo); lastSpeed=speed;
                display.publish(image,bin); ++cycles;
                std::this_thread::sleep_for(30ms);
            }
        } catch(...) { loopFault=true; }
    });
    bool ok=true;
    std::string error;
    try {
        for(int i=0;i<3;++i) {
            cv::Mat image,bin; display.snapshot(image,bin);
            std::this_thread::sleep_for(400ms); // Formerly expires the 250ms command lease.
            display.progress();
            check(!car->hasFault() && !loopFault,"slow GUI still caused control timeout");
        }
        check(cycles>=20 && lastSpeed>0,"control did not continue during slow display");
        std::cout<<"PASS 400ms renderer delays: control continued, 250ms watchdog unchanged\n";
        std::this_thread::sleep_for(2200ms);
        check(display.stalled() && lastSpeed>0 && !car->hasFault() && !loopFault,
            "2s rendering delay incorrectly stopped a healthy controller");
        std::cout<<"PASS 2.2s renderer delay: commands continue; display lag is diagnostic only\n";
        display.progress();
        // Genuine vision/controller stall must still trip the original lease.
        std::this_thread::sleep_for(150ms);
        cameraStall=true;
        std::this_thread::sleep_for(500ms);
        check(car->hasFault() && loopFault,"genuine controller stall failed to stop");
        std::cout<<"PASS 400ms camera/control stall still triggers latched stop\n";
        display.post('a'); display.post(' ');
        auto keys=display.takeKeys(); check(keys.size()==1 && keys[0]==' ',"stop key lost behind AUTO key");
        std::cout<<"PASS emergency key priority\n";
    } catch(const std::exception &e) { ok=false; error=e.what(); }
    stop=true;
    try { car->latchStop(); } catch(...) {}
    control.join(); sink.join(); car.reset(); close(LibSerial::peerFd);
    if(!ok) { std::cerr<<"FAIL "<<error<<'\n'; return 1; }
    return 0;
}
