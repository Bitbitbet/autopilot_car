#include "race_trial.hpp"
#include "control.hpp"
#include <future>
#include <iostream>
#include <limits>
#include <poll.h>
#include <unistd.h>
using namespace std::chrono_literals;
static void check(bool ok,const char *why) { if(!ok) throw std::runtime_error(why); }
int main() {
    try {
        RaceTrialDeadline::requireLowSpeed(.1f,.1f,.1f,.1f);
        for(float invalid:{.15f,10.f,std::numeric_limits<float>::quiet_NaN()}) {
            bool rejected=false;
            try { RaceTrialDeadline::requireLowSpeed(invalid,.1f,.1f,.1f); } catch(...) { rejected=true; }
            check(rejected,"high or invalid trial speed was accepted");
        }
        std::atomic_int calls{0};
        { RaceTrialDeadline disabled(0,[&] { ++calls; });
          check(!disabled.enabled()&&!disabled.arm(),"disabled timer changed ordinary mode"); }
        std::promise<void> stopped; auto completion=stopped.get_future();
        {
            RaceTrialDeadline trial(.5,[&] { ++calls; stopped.set_value(); });
            std::this_thread::sleep_for(600ms);
            check(calls==0,"timer fired before AUTO arm");
            const auto armed=std::chrono::steady_clock::now();
            check(trial.arm()&&!trial.arm(),"repeated AUTO reset timer");
            check(completion.wait_for(2s)==std::future_status::ready,"timer failed while caller was waiting");
            check(std::chrono::steady_clock::now()-armed>=490ms,"timer fired early");
            check(trial.expired()&&!trial.arm()&&calls==1,"one shot can rearm");
        }
        // 使用真实串口控制类及成对套接字模拟设备；调用方卡住时，截止线程仍锁定零速。
        auto car=CarControl::create(); check(bool(car),"mock serial unavailable");
        car->setCommandTimeout(250);
        std::atomic_bool done{false};
        std::jthread drain([&] { while(!done) { pollfd p{LibSerial::peerFd,POLLIN,0};
            if(poll(&p,1,20)>0) { unsigned char data[4096]; if(read(p.fd,data,sizeof(data))<=0) break; } } });
        std::jthread driving([&] { while(!done) { car->carControl(.1f,1500); std::this_thread::sleep_for(20ms); } });
        std::promise<void> wireStop; auto wireReady=wireStop.get_future();
        { RaceTrialDeadline trial(.5,[&] { car->latchStop(); wireStop.set_value(); });
          trial.arm(); std::this_thread::sleep_for(900ms);
          const bool good=wireReady.wait_for(0ms)==std::future_status::ready&&car->hasFault();
          done=true; driving.join(); drain.join();
          check(good,"blocked renderer prevented independent latched stop"); }
        car.reset(); close(LibSerial::peerFd);
        std::cout<<"PASS timed trial: wait for arm, single deadline, low-speed validation, independent serial latch during blocked caller\n";
    } catch(const std::exception &e) { std::cerr<<"FAIL "<<e.what()<<'\n'; return 1; }
}
