#include "race_manager.hpp"
#include "race_replay.hpp"
#include "race_preview.hpp"
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
        if(newFrame) snapshot={++sequence,time,std::move(ds),false,{},{}};
        return manager.update(snapshot,time,dt,active,image,track);
    }
    RacePlan repeat(std::vector<PredictResult> ds,int n=4) { RacePlan p; while(n--) p=step(ds); return p; }
};
int main(int argc,char **argv) {
    try {
        if(argc==3&&std::string(argv[1])=="--elevated-sign-field") {
            std::ifstream input(std::filesystem::path(argv[2])/"detections.jsonl");std::string line;
            Fixture field;field.image=cv::Mat::zeros(240,320,CV_8UC1);
            cv::fillConvexPoly(field.image,std::vector<cv::Point>{{240,0},{319,0},{280,239},{40,239}},{255});
            bool started=false;int count=0,guided=0,maxBottom=0;
            while(std::getline(input,line)) {
                auto row=nlohmann::json::parse(line);std::vector<PredictResult> ds;
                for(const auto &d:row.at("detections"))if(d.at("type")==LABEL_YLEFT&&d.at("accepted").get<bool>()) {
                    ds.push_back(box(LABEL_YLEFT,d.at("y"),d.at("width"),d.at("height"),d.at("x")));
                    maxBottom=std::max(maxBottom,d.at("y").get<int>()+d.at("height").get<int>());
                }
                if(!started&&ds.empty())continue;started=true;
                auto p=field.step(ds);++count;
                if(p.guideDirection==-1) {
                    check(!p.stop&&!p.selectedBranch&&p.guidePwm>=180&&raceDirectionServo(p,1700)<=1320,
                          "Recorded elevated sign waits for left road or row145");++guided;
                }
                if(count>=30)break;
            }
            check(guided>=15&&maxBottom<145,"Elevated sign field sequence not exercised");
            cv::flip(field.image,field.image,1);auto p=field.repeat({},4);
            check(p.selectedBranch&&!p.stop&&std::string(p.state)=="FORK_LEFT","Sign-led turn cannot hand off to newly visible left road");
            std::cout<<"PASS actual elevated sign detections with synthetic hidden-left road: guided_frames="<<guided
                     <<" maximum_box_bottom="<<maxBottom<<"; immediate steering then left-road handoff\n";return 0;
        }
        for(int sign:{-1,1}) {
            cv::Mat picture(240,320,CV_8UC3,cv::Scalar(30,30,30));
            cv::circle(picture,{160,120},80,{220,40,20},-1);
            std::vector<cv::Point> arrow{{100,120},{150,80},{150,105},{210,105},{210,135},{150,135},{150,160}};
            if(sign>0)for(auto &v:arrow)v.x=320-v.x;
            cv::fillPoly(picture,std::vector<std::vector<cv::Point>>{arrow},{255,255,255});
            std::vector<PredictResult> ds{box(LABEL_LEFT,40,160,160,80)};
            auto checks=verifyForkDirections(picture,ds);
            check(checks.size()==1&&ds.size()==1&&ds[0].type==(sign<0?LABEL_YLEFT:LABEL_YRIGHT),"horizontal blue arrow direction not verified");
        }
        if(argc==3&&std::string(argv[1])=="--sign-photo") {
            auto raw=cv::imread(argv[2]);check(!raw.empty(),"Uploaded left-sign fixture missing");
            cv::Mat picture;cv::resize(raw(cv::Rect(100,485,890,780)),picture,{320,240});
            std::vector<PredictResult> ds{box(LABEL_LEFT,0,320,240,0)};
            verifyForkDirections(picture,ds);
            check(ds.size()==1&&ds[0].type==LABEL_YLEFT,"User's photographed blue arrow not recognized as left");
            std::cout<<"PASS user photographed left arrow shape (phone image; vehicle recognition still needs field check)\n";
        }
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
        check(!p.selectedBranch&&p.guideDirection==-1&&std::string(p.state)=="FORK_SEEK_LEFT","near confirmed sign must seek the signed road without claiming a branch");
        check(raceDirectionServo(p,1650)<PWMSERVOMID,"visible right road overrides left sign");
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

        Fixture expired;
        expired.repeat({box(LABEL_YLEFT)});
        for(int i=0;i<65;++i)p=expired.step({});
        check(p.stop&&(std::string(p.reason)=="DIRECTION_MEMORY_EXPIRED"||std::string(p.reason)=="SIGNED_ROAD_REACQUIRE_TIMEOUT")&&!p.selectedBranch,
              "unresolved direction silently resumed ordinary lane following");
        expired.step({},false);p=expired.repeat({},4);
        check(p.stop,"manual toggle cleared unresolved direction fault");

        Fixture renewed;
        renewed.repeat({box(LABEL_YLEFT,20)});
        for(int i=0;i<30;++i)p=renewed.step({box(LABEL_YLEFT,20)});
        check(!p.stop&&std::string(p.state)=="FORK_SEEK_LEFT"&&p.guidePwm>=280,"confirmed high sign did not actively seek left");
        renewed.image=road(true);p=renewed.repeat({box(LABEL_YLEFT)},4);
        check(p.selectedBranch&&std::string(p.state)=="FORK_LEFT","renewed direction cannot select a later real split");

        Fixture neverSplit;
        neverSplit.repeat({box(LABEL_YRIGHT,20)});
        for(int i=0;i<125;++i)p=neverSplit.step({box(LABEL_YRIGHT,20)});
        check(p.stop&&std::string(p.reason)=="SIGNED_ROAD_REACQUIRE_TIMEOUT",
              "continuous direction sign permits indefinite unconfirmed forward approach");
        Fixture horizontalLeft;
        p=horizontalLeft.repeat({box(LABEL_LEFT)});
        check(p.guideDirection==-1&&p.guidePwm>=90,"ordinary LEFT label never arms direction guidance");
        horizontalLeft.image=road(true);p=horizontalLeft.repeat({},4);
        check(p.selectedBranch&&std::string(p.state)=="FORK_LEFT","guided left cannot acquire visible branch");
        Fixture distant;
        p=distant.repeat({box(LABEL_YLEFT,20)});
        check(p.guidePwm>=180&&raceDirectionServo(p,1650)<=1320,"confirmed elevated sign still waits for row145 or permits opposite steering");
        AutoPilot missingRoad(TrackviewConfig{});
        missingRoad.update({}, {},.07f,0);
        check(!missingRoad.ready&&raceDirectionSeeking(p)&&raceDirectionSpeed(p,missingRoad,.1f,.07f,0)>0,
              "confirmed sign cannot move without visible lane");
        check(raceDirectionSpeed(p,missingRoad,.1f,1.f,0)==0,"sign seeking ignores frame gap");
        auto stoppedGuide=p;stoppedGuide.stop=true;
        check(raceDirectionSpeed(stoppedGuide,missingRoad,.1f,.07f,.1f)==0,"sign seeking ignores priority stop");
        stoppedGuide=p;stoppedGuide.fresh=false;
        check(raceDirectionSpeed(stoppedGuide,missingRoad,.1f,.07f,.1f)==0,"sign seeking ignores stale AI");
        Fixture hidden;
        hidden.image=cv::Mat::zeros(240,320,CV_8UC1);
        cv::fillConvexPoly(hidden.image,std::vector<cv::Point>{{240,0},{319,0},{280,239},{40,239}},{255});
        p=hidden.repeat({box(LABEL_YLEFT)},8);
        check(!p.stop&&!p.selectedBranch&&p.guideDirection==-1&&raceDirectionServo(p,1700)<1500,
              "only-right-visible scene defeats confirmed left guidance");
        cv::flip(hidden.image,hidden.image,1);p=hidden.repeat({},4);
        check(!p.stop&&p.selectedBranch&&std::string(p.state)=="FORK_LEFT",
              "single connected left road entering view cannot replace guidance");
        Fixture seekGate;
        seekGate.repeat({box(LABEL_LEFT)});
        p=seekGate.repeat({box(LABEL_LEFT),box(LABEL_GATE,130,90,20)});
        check(p.stop&&std::string(p.state)=="GATE_WAIT","sign guidance overrides gate");
        Fixture seekStale;seekStale.repeat({box(LABEL_LEFT)});p=seekStale.step({},true,false,.5f);
        check(p.stop&&!p.fresh&&p.guideDirection==0,"stale image keeps sign steering active");

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
        Fixture taught;
        auto taughtStep=[&](std::vector<PredictResult> detections) {
            taught.time+=.07;taught.track.handle(taught.image);
            taught.snapshot={++taught.sequence,taught.time,std::move(detections),false,{},{}};
            return taught.manager.update(taught.snapshot,taught.time,.07f,true,taught.image,taught.track,{},false,true);
        };
        for(int i=0;i<5;++i)p=taughtStep({box(LABEL_PARK)});
        check(!p.stop,"explicit taught-bay mode still blocks on its own parking sign");
        for(int i=0;i<5;++i)p=taughtStep({box(LABEL_GATE,130,90,20)});
        check(p.stop&&std::string(p.state)=="GATE_WAIT","taught-bay authorization bypasses gate stop");
        for(int i=0;i<15;++i)p=taughtStep({});
        check(p.stop,"taught parking treats missing gate as open");

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
        bool released=false;
        for(int i=0;i<5;++i) {
            p=signal.step({box(LABEL_FLOW)});
            if(!p.stop) {
                check(std::string(p.state)=="LANE"&&std::string(p.reason)=="CRUISE",
                      "released flow must update state text in the same frame");
                released=true;
            }
        }
        check(released,"fresh confirmed flow resumes after minimum stop");
        // 等待期间的限速义务不能随绿牌一起解除，放行文字也必须与之相符。
        Fixture limitedSignal;
        limitedSignal.repeat({box(LABEL_LIMIT)});
        limitedSignal.repeat({box(LABEL_STOP)});
        limitedSignal.repeat({box(LABEL_CROSS,190,60,30)});
        limitedSignal.repeat({},20);
        bool limitedReleased=false;
        for(int i=0;i<5;++i) {
            p=limitedSignal.step({box(LABEL_FLOW)});
            if(!p.stop) {
                check(std::string(p.state)=="LIMIT_ZONE"&&std::string(p.reason)=="LIMIT_SIGN"&&p.speedCap==.4f,
                      "flow release must preserve speed cap and matching state text");
                limitedReleased=true;
            }
        }
        check(limitedReleased,"limited signal release fixture did not release");

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
            for(int i=1;i<=30;++i) {
                auto frame=road(); frame.at<uint8_t>(0,0)=i;
                worker.submit(frame,i,captured);
            }
            bool complete=false;
            for(int i=0;i<100;++i) {
                auto snap=worker.snapshot();
                if(snap.sequence==30) {
                    complete=true;
                    check(snap.captured==captured,"capture timestamp preserved");
                    check(!snap.image.empty()&&snap.image.at<uint8_t>(0,0)==30,"preview image matches inferred frame");
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
            check(complete&&calls<30,"inference queue keeps latest frame without backlog");
        }
        if(argc==2) {
            const std::filesystem::path dir=argv[1];
            std::filesystem::create_directories(dir);
            RaceSnapshot sample{9,1.0,{box(LABEL_YLEFT)},false,{},{}};
            sample.image=cv::Mat(240,320,CV_8UC3,cv::Scalar(35,35,35));
            const auto before=sample.image.clone();
            const auto preview=racePreviewImage(sample,1.1,RaceConfig{});
            check(preview.size()==cv::Size(640,300),"AI preview layout");
            check(cv::norm(before,sample.image,cv::NORM_INF)==0,"drawing cannot alter raw AI frame");
            check(saveRaceSample(dir,sample,1.1,RaceConfig{},1),"save AI sample");
            const auto json=raceDetectionJson(sample,1.1,.65f);
            check(json.at("detections").at(0).at("label")=="yleft","sample uses actual class mapping");
            auto malformed=box(LABEL_YLEFT); malformed.x=-100;
            sample.detections.push_back(malformed);
            check(raceDetectionJson(sample,1.1,.65f).at("detections").size()==1,"invalid boxes excluded from preview metadata");
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
            // 长直道供一次性定时模式验证整程序退出，不连接真实车辆设备。
            cv::VideoWriter longVideo((dir/"trial-long.avi").string(),
                cv::VideoWriter::fourcc('M','J','P','G'),15,{320,240});
            check(longVideo.isOpened(),"create timed entry fixture");
            std::ofstream empty(dir/"trial-long.jsonl");
            cv::Mat straight; cv::cvtColor(road(),straight,cv::COLOR_GRAY2BGR);
            for(int frame=1;frame<=1000;++frame) {
                longVideo.write(straight);
                empty<<nlohmann::json({{"frame",frame},{"detections",nlohmann::json::array()}}).dump()<<'\n';
            }
        }
        std::cout<<"PASS race limits/fork geometry/AI freshness/signal/gate/task boundaries/asynchronous queue/matched preview and samples\n";
    } catch(const std::exception &e) { std::cerr<<"FAIL: "<<e.what()<<'\n'; return 1; }
}
