#include "race_runtime.hpp"
#include "trackview_display.hpp"
#include "race_parking_control.hpp"
#include <iostream>
static void check(bool b,const char *s){if(!b)throw std::runtime_error(s);}
int main(){try{
    raceConfigureRuntime();check(cv::getNumThreads()==1,"Unexpected OpenCV oversubscription");
    cv::Mat original(240,320,CV_8UC3);cv::randu(original,0,255);cv::Mat capture;
    cv::resize(original,capture,{640,480},0,0,cv::INTER_NEAREST);raceNormalizeFrame(capture);
    check(cv::norm(original,capture,cv::NORM_INF)==0,"640->320 camera geometry changed");
    auto invalid=cv::Mat(360,640,CV_8UC3);bool rejected=false;
    try{raceNormalizeFrame(invalid);}catch(const std::invalid_argument&){rejected=true;}
    check(rejected,"Stretched 16:9 calibration accepted");
    TrackviewDisplay display;display.publish(original,{});cv::Mat image,binary;
    check(display.snapshot(image,binary)&&!display.snapshot(image,binary),"GUI redraws old frames");
    check(!display.frameDue(),"GUI publish throttle not working");
    std::vector<RaceParkingPose> a,b;RaceParkingDimensions d;
    for(int i=0;i<=100;++i){auto p=raceParkingAdvance({0,-.4f,0},i*.005f,1.f,d);a.push_back(p);b.push_back(p);b.push_back(p);}
    const auto x=raceParkingReverseSamples(a,d),y=raceParkingReverseSamples(b,d);
    check(x.size()==y.size()&&x.size()<a.size(),"Duplicate video frames change reverse distance");
    for(size_t i=0;i<x.size();++i)check(std::hypot(x[i].x-y[i].x,x[i].z-y[i].z)<1e-6,"Reverse path depends on camera FPS");
    RaceParkingSpace space;const auto route=racePlanTrackableParking(space,{0,-.4f,0});check(route.valid,"No baseline parking route");
    const auto goal=route.entry.back();bool noiseCase=false;
    for(int i=-10;i<=10;++i)for(int j=-10;j<=10;++j){auto p=goal;p.x+=i*.001f;p.z+=j*.001f;
        if(!space.parked(p)&&space.parkedObserved(p))noiseCase=true;}
    check(noiseCase,"Dwell still confuses planned clearance with physical containment");
    auto outside=goal;outside.x-=.15f;check(!space.parkedObserved(outside),"Large actual bay violation accepted");
    RaceParkingControl control(space);RaceParkingObservation observation{route.entry.front(),1,1,1,true};
    check(control.arm(route,observation,1,true),"Cannot arm settle test");double time=1;RaceParkingCommand command;
    for(int i=0;i<1500;++i){time+=.08;observation.captured=time;++observation.sequence;command=control.update(observation,time,.08f,true);
        if(std::string(command.state)=="PARK_DWELL")break;
        observation.pose=raceParkingAdvance(observation.pose,command.speed*.08f,command.curvature,d);}
    check(std::string(command.state)=="PARK_DWELL","Settle test did not park");
    auto lastInside=observation.pose;bool found=false;
    // Smooth visual drift up to the edge, then one transient outside sample;
    // all inter-frame displacements remain below continuity limits.
    for(int i=0;i<40&&!found;++i){lastInside=observation.pose;observation.pose.x-=.003f;
        time+=.08;observation.captured=time;++observation.sequence;command=control.update(observation,time,.08f,true);
        found=!space.parkedObserved(observation.pose);}
    check(found,"No near-boundary drift case");
    check(!control.failed()&&command.speed==0&&std::string(command.state)=="PARK_DWELL"&&control.terminalRecoveryUsed(),
          "Dwell jitter must latch recovery, keep stopped and never restart forward entry");
    observation.pose=lastInside;
    for(int i=0;i<25;++i){time+=.08;observation.captured=time;++observation.sequence;command=control.update(observation,time,.08f,true);
        check(!control.failed()&&command.speed==0&&std::string(command.state)=="PARK_DWELL","Jitter counted toward completed dwell");}
    bool exit=false;
    for(int i=0;i<25;++i){time+=.08;observation.captured=time;++observation.sequence;command=control.update(observation,time,.08f,true);
        check(!control.failed()&&command.speed<=0,"Recovery after dwell drives forward or faults");
        if(std::string(command.state)=="PARK_EXIT"){exit=true;break;}}
    check(exit&&control.terminalRecoveryUsed(),"Three stationary seconds must lead to reverse with recovery still reported");
    std::cout<<"PASS 640 capture normalization, aspect rejection, GUI latest-only/throttle, OpenCV worker limit, FPS-independent reverse samples and dwell observation margin\n";
}catch(const std::exception &e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}}
