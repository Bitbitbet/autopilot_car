#include "race_manager.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>

static void check(bool ok,const char *what) { if(!ok) throw std::runtime_error(what); }
static cv::Mat forkRoad() {
    cv::Mat bin=cv::Mat::zeros(240,320,CV_8UC1);
    cv::rectangle(bin,{40,0,241,240},cv::Scalar(255),-1);
    cv::rectangle(bin,{131,0,59,165},cv::Scalar(0),-1);
    return bin;
}
int main(int argc,char **argv) {
    try {
        if(argc!=2) return 2;
        const std::filesystem::path base=argv[1];
        RaceConfig cfg; cfg.stopAfterCross=false;
        int cases=0;
        for(const auto &sample:{"1792309701567-10891/ai-sample-1",
                               "1792309155760-32754/ai-sample-1",
                               "1792309155760-32754/ai-sample-2"}) {
            auto image=cv::imread((base/sample/"camera.png").string());
            check(!image.empty(),"missing recorded sign image");
            cv::Mat gray,bin; cv::cvtColor(image,gray,cv::COLOR_BGR2GRAY);
            cv::threshold(gray,bin,0,255,cv::THRESH_OTSU);
            Track track; track.rowCutUp=10; track.rowCutBottom=20; track.handle(bin);
            std::ifstream file(base/sample/"detections.json"); nlohmann::json meta; file>>meta;
            RaceSnapshot snap; snap.sequence=1; snap.captured=1;
            for(const auto &v:meta["detections"]) {
                PredictResult d{}; d.type=v["type"]; d.score=v["score"];
                d.x=v["x"]; d.y=v["y"]; d.width=v["width"]; d.height=v["height"];
                snap.detections.push_back(d);
            }
            const int raw=raceBranchPath(bin,track,0,cfg).splitRows;
            const int verified=raceBranchPath(bin,track,0,cfg,raceSignRegions(snap,cfg)).splitRows;
            std::cout<<sample<<" raw_split_rows="<<raw<<" verified_split_rows="<<verified<<'\n';
            check(raw<cfg.minBranchRows,"sign topology must also work when AI misses the sign");
            check(verified<cfg.minBranchRows,"printed sign must not count as road fork");
            RaceManager manager(cfg); RacePlan plan;
            for(int i=0;i<6;++i) {
                snap.sequence++; snap.captured+=.07;
                plan=manager.update(snap,snap.captured,.07,true,bin,track);
                check(std::string(plan.state)!="FORK_WAIT","sign must not generate fork wait");
            }
            if(snap.detections.front().type==LABEL_STOP)
                check(plan.stop&&std::string(plan.state)=="SIGNAL_WAIT","stop obligation retained");
            else check(!plan.stop&&std::string(plan.state)=="LIMIT_ZONE","limit obligation retained");
            ++cases;
        }
        auto bin=forkRoad(); Track track; track.handle(bin);
        check(raceBranchPath(bin,track,0,cfg).splitRows>=cfg.minBranchRows,"real fork geometry retained");
        PredictResult side{}; side.type=LABEL_LIMIT; side.score=.99f;
        side.x=5; side.y=5; side.width=50; side.height=90;
        RaceSnapshot snap{1,1,{side},false,{},{}};
        check(raceBranchPath(bin,track,0,cfg,raceSignRegions(snap,cfg)).splitRows>=cfg.minBranchRows,
              "roadside sign must not hide real fork");
        RaceManager unknown(cfg); RacePlan p;
        for(int i=0;i<5;++i) { snap.sequence++; snap.captured+=.07;
            p=unknown.update(snap,snap.captured,.07,true,bin,track); }
        check(p.stop&&std::string(p.state)=="FORK_WAIT","unmarked fork still stops");
        cases+=3;
        auto bad=side; bad.score=.1f; snap.detections={bad};
        check(raceSignRegions(snap,cfg).empty(),"low score cannot mask geometry");
        bad=side; bad.x=-1; snap.detections={bad};
        check(raceSignRegions(snap,cfg).empty(),"invalid box cannot mask geometry");
        for(int type:{LABEL_PERSON,LABEL_CONE,LABEL_GATE,LABEL_CROSS}) {
            bad=side; bad.type=type; snap.detections={bad};
            check(raceSignRegions(snap,cfg).empty(),"road and hazard targets cannot mask geometry");
        }
        cases+=6;
        // 连续确认左右方向后，真实岔路依然能分别选择左右支路。
        for(int direction:{LABEL_YLEFT,LABEL_YRIGHT}) {
            RaceManager manager(cfg);
            auto sign=side; sign.type=direction; snap.detections={sign};
            for(int i=0;i<8;++i) { snap.sequence++; snap.captured+=.07;
                p=manager.update(snap,snap.captured,.07,true,bin,track); }
            check(p.selectedBranch&&!p.stop,"verified direction still selects real branch");
            check(std::string(p.state)==(direction==LABEL_YLEFT?"FORK_LEFT":"FORK_RIGHT"),"branch direction retained");
            ++cases;
        }
        std::cout<<"PASS "<<cases<<" sign/road separation cases\n";
    } catch(const std::exception &e) { std::cerr<<"FAIL "<<e.what()<<'\n'; return 1; }
}
