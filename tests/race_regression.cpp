#include "race_manager.hpp"
#include "race_replay.hpp"
#include "trackview_pilot.hpp"
#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>
#include <filesystem>

static void check(bool ok,const char *what) { if(!ok) throw std::runtime_error(what); }
static PredictResult box(int type,int y=120,int w=30,int h=40,int x=100) {
    PredictResult d{}; d.type=type; d.score=0.95f; d.x=x; d.y=y; d.width=w; d.height=h; return d;
}
static cv::Mat road(bool fork=false) {
    cv::Mat bin=cv::Mat::zeros(240,320,CV_8UC1);
    cv::rectangle(bin,{40,0,241,240},cv::Scalar(255),-1);
    if(fork) cv::rectangle(bin,{131,0,59,165},cv::Scalar(0),-1);
    return bin;
}
struct Fixture {
    RaceConfig config;
    RaceManager manager{config};
    Track track;
    double time=0;
    uint64_t sequence=0;
    cv::Mat image=road();
    RaceSnapshot snapshot;
    RacePlan step(std::vector<PredictResult> ds={},bool active=true,bool newFrame=true,float dt=.07f) {
        time+=dt; track.handle(image);
        if(newFrame) snapshot={++sequence,time,std::move(ds),false,{}};
        return manager.update(snapshot,time,dt,active,image,track);
    }
    RacePlan repeat(std::vector<PredictResult> ds,int n=4) { RacePlan p; while(n--) p=step(ds); return p; }
};
int main(int argc,char **argv) {
    try {
        Fixture f;
        check(!f.repeat({}).stop,"fresh empty AI permits clear straight lane");
        auto p=f.repeat({box(LABEL_LIMIT)});
        check(p.speedCap==.4f,"limit sign latch");
        check(f.repeat({}).speedCap==.4f,"sign disappearing must not cancel limit");
        p=f.repeat({box(LABEL_UNLIMIT)});
        check(p.speedCap==10.f,"confirmed unlimit clears latch");
        f.repeat({box(LABEL_LIMIT)});
        f.step({},false); f.step({},true);
        check(f.step({}).speedCap==.4f,"manual toggle cannot cancel speed zone");

        Fixture repeated;
        repeated.step({}); repeated.step({box(LABEL_LIMIT)});
        for(int i=0;i<4;++i) p=repeated.step({},true,false);
        check(p.speedCap==10.f,"one AI result cannot count as repeated evidence");
        p=repeated.step({},true,false,.2f);
        check(p.stop&&!p.fresh,"old capture timestamp stops autonomous motion");
        auto failed=repeated.snapshot; failed.failed=true;
        check(repeated.manager.update(failed,repeated.time,.07f,true,repeated.image,repeated.track).stop,"failed AI stops");

        Fixture fork;
        p=fork.repeat({box(LABEL_YLEFT)});
        check(!p.selectedBranch&&std::string(p.state)=="FORK_ARMED","sign alone must not turn");
        fork.image=road(true);
        p=fork.repeat({box(LABEL_YLEFT)},4);
        check(p.selectedBranch&&std::string(p.state)=="FORK_LEFT","confirmed left branch");
        check(p.path.left.back().y<100&&p.path.right.back().y<150,"path follows left connected road");
        AutoPilot pilot;
        for(int i=0;i<8;++i) pilot.update(p.path.left,p.path.right,.07f);
        check(pilot.valid&&pilot.center<160,"selected branch supplies valid v6 steering path");
        fork.image=road(); p=fork.repeat({},14);
        check(!p.selectedBranch,"branch exits after sustained single corridor");
        Fixture right;
        right.repeat({box(LABEL_YRIGHT)}); right.image=road(true);
        p=right.repeat({box(LABEL_YRIGHT)},4);
        check(p.selectedBranch&&p.path.left.back().y>180,"right route geometry");
        Fixture unknown; unknown.image=road(true);
        check(unknown.repeat({}).stop,"unmarked fork must wait");
        Fixture conflict;
        conflict.repeat({box(LABEL_YLEFT),box(LABEL_YRIGHT)}); conflict.image=road(true);
        check(conflict.repeat({box(LABEL_YLEFT),box(LABEL_YRIGHT)}).stop,"conflicting signs cannot choose route");
        Fixture changed;
        changed.repeat({box(LABEL_YLEFT)});
        check(changed.repeat({box(LABEL_YRIGHT)}).stop,"changed direction before split cannot retain old route");
        Fixture interrupted;
        interrupted.repeat({box(LABEL_YLEFT)}); interrupted.image=road(true);
        interrupted.repeat({box(LABEL_YLEFT)},4);
        interrupted.step({},true,false,.5f);
        check(interrupted.repeat({}).stop,"AI loss during branch latches review stop");

        Fixture gate;
        p=gate.repeat({box(LABEL_GATE,130,90,20)});
        check(p.stop&&std::string(p.state)=="GATE_WAIT","horizontal gate stops");
        check(gate.repeat({},15).stop,"missing gate detections cannot prove raised bar");
        p=gate.repeat({box(LABEL_GATE,80,20,100,130)},14);
        check(!p.stop,"sustained upright same-area gate permits through");
        Fixture hazard;
        check(hazard.repeat({box(LABEL_PERSON,150)}).stop,"near person stops");
        check(hazard.repeat({},2).stop,"single missed person cannot release");
        check(!hazard.repeat({},13).stop,"fresh sustained clear scene releases wait");
        Fixture park;
        p=park.repeat({box(LABEL_PARK)});
        check(p.stop&&std::string(p.reason)=="PARK_PLANNER_PENDING","parking is explicitly pending real planner validation");
        check(park.repeat({box(LABEL_FLOW)},15).stop,"flow cannot release unvalidated parking task");

        Fixture signal;
        p=signal.repeat({box(LABEL_STOP)});
        check(std::string(p.state)=="CROSS_APPROACH"&&!p.stop,"2026 stop sign waits until cross passed");
        signal.repeat({box(LABEL_STOP),box(LABEL_CROSS,190,60,30)});
        p=signal.repeat({},6);
        check(p.stop&&std::string(p.state)=="SIGNAL_WAIT","cross disappearing after bottom approach triggers stop");
        check(signal.repeat({box(LABEL_GATE,130,90,20)}).stop,"gate overlays existing red-signal wait");
        p=signal.repeat({box(LABEL_GATE,80,20,100,130)},14);
        check(p.stop&&std::string(p.state)=="SIGNAL_WAIT","gate clearing must not discard signal-stop obligation");
        check(signal.repeat({box(LABEL_FLOW),box(LABEL_STOP)},18).stop,"stop beats simultaneous flow");
        check(!signal.repeat({box(LABEL_FLOW)},5).stop,"fresh confirmed flow resumes after minimum stop");

        Fixture invalid;
        auto d=box(LABEL_LIMIT); d.x=-1;
        check(invalid.repeat({d}).speedCap==10.f,"malformed boxes ignored");
        d=box(LABEL_LIMIT); d.score=std::numeric_limits<float>::quiet_NaN();
        check(invalid.repeat({d}).speedCap==10.f,"NaN detections ignored");
        std::atomic_int calls{0};
        {
            RacePerception worker([&](const cv::Mat&) {
                ++calls; std::this_thread::sleep_for(std::chrono::milliseconds(20));
                return std::vector<PredictResult>{};
            });
            const double captured=raceTime();
            for(int i=1;i<=30;++i) worker.submit(road(),i,captured);
            bool complete=false;
            for(int i=0;i<100;++i) {
                auto snap=worker.snapshot();
                if(snap.sequence==30) { complete=true; check(snap.captured==captured,"capture timestamp preserved"); break; }
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
            check(complete&&calls<30,"inference queue keeps latest frame without backlog");
        }
        if(argc==2) {
            const std::filesystem::path dir=argv[1];
            std::filesystem::create_directories(dir);
            cv::VideoWriter video((dir/"race-scenario.avi").string(),
                cv::VideoWriter::fourcc('M','J','P','G'),30,{320,240});
            check(video.isOpened(),"create full entry replay fixture");
            std::ofstream labels(dir/"race-scenario.jsonl");
            for(int frame=1;frame<=120;++frame) {
                cv::Mat rgb; cv::cvtColor(road(frame>=36&&frame<=65),rgb,cv::COLOR_GRAY2BGR); video.write(rgb);
                nlohmann::json ds=nlohmann::json::array();
                if(frame>=21&&frame<=45) ds.push_back({{"type",LABEL_YLEFT},{"score",.95f},
                    {"x",200},{"y",80},{"width",30},{"height",40}});
                if(frame>=96) ds.push_back({{"type",LABEL_LIMIT},{"score",.95f},
                    {"x",200},{"y",80},{"width",30},{"height",40}});
                labels<<nlohmann::json({{"frame",frame},{"detections",ds}}).dump()<<'\n';
            }
        }
        std::cout<<"PASS race limits/fork geometry/AI freshness/signal/gate/task boundaries/asynchronous queue\n";
    } catch(const std::exception &e) { std::cerr<<"FAIL: "<<e.what()<<'\n'; return 1; }
}
