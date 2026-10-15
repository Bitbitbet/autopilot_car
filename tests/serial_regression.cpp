#include "control.hpp"
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <poll.h>
#include <stdexcept>
#include <sys/socket.h>
#include <thread>
#include <vector>
using Clock=std::chrono::steady_clock;
void check(bool ok,const char *message) { if(!ok) throw std::runtime_error(message); }
std::vector<float> frames(int timeout=30) {
    std::vector<uint8_t> bytes;
    for(;;) {
        pollfd p{LibSerial::peerFd,POLLIN,0};
        if(poll(&p,1,timeout)<=0 || !(p.revents&POLLIN)) break;
        uint8_t b[512]; const auto n=read(p.fd,b,sizeof(b));
        if(n<=0) break;
        bytes.insert(bytes.end(),b,b+n); timeout=5;
    }
    std::vector<float> speeds;
    for(size_t i=0;i+10<bytes.size();i+=11) {
        check(bytes[i]==0x42 && bytes[i+1]==1 && bytes[i+2]==10,"control protocol changed");
        uint8_t sum=0; for(size_t j=0;j<9;++j) sum+=bytes[i+j];
        check(sum==bytes[i+9] && bytes[i+10]==0,"checksum or trailing byte changed");
        float s; memcpy(&s,bytes.data()+i+3,4); speeds.push_back(s);
    }
    return speeds;
}
int main() {
    try {
        {
            auto car=CarControl::create(); check(bool(car),"create");
            car->carControl(0.25f,1600); auto s=frames();
            check(s.size()==2 && s.front()==0 && s.back()==0.25f,"startup/motion frames");
            auto start=Clock::now(); car.reset();
            check(Clock::now()-start<std::chrono::milliseconds(300),"silent serial destructor blocked");
            s=frames(); check(!s.empty() && s.back()==0,"destructor omitted stop");
            close(LibSerial::peerFd);
        }
        std::cout<<"PASS silent serial shutdown and exact protocol\n";
        {
            auto car=CarControl::create(); frames();
            car->setCommandTimeout(100); car->carControl(0.5f,1700); frames();
            std::this_thread::sleep_for(std::chrono::milliseconds(160));
            check(car->hasFault(),"watchdog did not latch");
            auto s=frames(); check(!s.empty() && s.back()==0,"watchdog omitted stop");
            car->carControl(0.5f,1700); s=frames(); check(s.back()==0,"expired command restarted car");
            car.reset(); close(LibSerial::peerFd);
        }
        std::cout<<"PASS stalled main loop watchdog and restart veto\n";
        {
            auto car=CarControl::create(); frames(); car->carControl(0.12f,1500); frames();
            shutdown(LibSerial::deviceFd,SHUT_RD); // Receive fails, write remains usable.
            std::this_thread::sleep_for(std::chrono::milliseconds(60));
            check(car->hasFault(),"receive failure not published");
            auto s=frames(); check(!s.empty() && s.back()==0,"receive failure omitted stop");
            car->carControl(0.25f,1500); s=frames(); check(s.back()==0,"receive fault allowed motion");
            car.reset(); close(LibSerial::peerFd);
        }
        std::cout<<"PASS receive fault cleanup and motion veto\n";
        {
            auto car=CarControl::create(); frames();
            int size=1024; setsockopt(LibSerial::deviceFd,SOL_SOCKET,SO_SNDBUF,&size,sizeof(size));
            bool threw=false; const auto start=Clock::now();
            for(int i=0;i<500 && !threw;++i) {
                try { car->carControl(0.1f,1500); } catch(...) { threw=true; }
            }
            check(threw && car->hasFault(),"blocked write not bounded");
            check(Clock::now()-start<std::chrono::seconds(2),"write blocked indefinitely");
            frames(); car.reset(); close(LibSerial::peerFd);
        }
        std::cout<<"PASS write backpressure timeout\n";
        {
            auto car=CarControl::create(); frames();
            const auto stale=car->motionEpoch();
            car->pauseMotion();
            check(!car->resumeMotion(stale),"stale AUTO key cleared newer stop");
            car->carControl(0.2f,1600);
            auto s=frames(); check(s.back()==0,"paused motion restarted");
            check(car->resumeMotion(car->motionEpoch()),"explicit new rearm rejected");
            car->carControl(0.2f,1600); s=frames(); check(s.back()==0.2f,"rearm did not restore control");
            car->latchStop(); check(!car->resumeMotion(car->motionEpoch()),"exit latch cleared");
            car.reset(); close(LibSerial::peerFd);
        }
        std::cout<<"PASS stop/rearm epochs and exit race veto\n";
        return 0;
    } catch(const std::exception &e) { std::cerr<<"FAIL "<<e.what()<<'\n'; return 1; }
}
