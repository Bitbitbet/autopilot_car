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
    std::cout<<"PASS 640 capture normalization, aspect rejection, GUI latest-only/throttle, OpenCV worker limit, FPS-independent reverse samples and dwell observation margin\n";
}catch(const std::exception &e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}}
