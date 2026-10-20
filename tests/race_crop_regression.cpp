#include "race_manager.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>

static void check(bool ok,const char *why) { if(!ok) throw std::runtime_error(why); }
int main(int argc,char **argv) {
    try {
        if(argc!=2) return 2;
        const std::filesystem::path base=argv[1];
        RaceConfig cfg; cfg.stopAfterCross=false;
        int frames=0,triggers=0;
        for(const char *run:{"1792475658486-13101","1792482459384-641"}) {
            auto dir=base/run/"fork-trace-1";
            std::ifstream file(dir/"trace.json"); nlohmann::json trace; file>>trace;
            check(trace["frames"].size()==7,"missing withdrawal/entry frames");
            for(bool mirror:{false,true}) {
                RaceManager manager(cfg);
                for(const auto &f:trace["frames"]) {
                    cv::Mat bin=cv::imread((dir/f["binary"].get<std::string>()).string(),cv::IMREAD_GRAYSCALE);
                    check(!bin.empty(),"missing recorded binary");
                    if(mirror) cv::flip(bin,bin,1);
                    Track track; track.rowCutUp=f["row_cut_up"]; track.rowCutBottom=f["row_cut_bottom"];
                    for(const auto &v:f[mirror?"right":"left"])
                        track.pointsEdgeLeft.emplace_back(v[0].get<int>(),mirror?319-v[1].get<int>():v[1].get<int>());
                    for(const auto &v:f[mirror?"left":"right"])
                        track.pointsEdgeRight.emplace_back(v[0].get<int>(),mirror?319-v[1].get<int>():v[1].get<int>());
                    const auto path=raceBranchPath(bin,track,0,cfg);
                    check(path.splitRows<cfg.minBranchRows,"top-clipped sign still counted as fork");
                    for(const auto &point:path.left)
                        check(point.x>track.rowCutUp,"top validation pixels entered steering path");
                    if(f["frame"]==trace["trigger_frame"]&&!mirror) {
                        check(f["masked_split_rows"].get<int>()>=cfg.minBranchRows,"record no longer proves old failure");
                        ++triggers;
                    }
                    RaceSnapshot snap; snap.sequence=f["ai"]["frame"]; snap.captured=f["ai"]["captured"];
                    const auto plan=manager.update(snap,f["time"],.07f,true,bin,track);
                    check(!plan.stop&&std::string(plan.state)=="LANE","empty AI snapshot must not invent a fork");
                    ++frames;
                }
            }
        }
        // 真实持续分隔带仍在首次新鲜观测时停车，不用额外等几帧掩盖误判。
        cv::Mat fork=cv::Mat::zeros(240,320,CV_8UC1);
        cv::rectangle(fork,{40,0,240,240},cv::Scalar(255),-1);
        cv::rectangle(fork,{135,0,50,160},cv::Scalar(0),-1);
        Track track; track.pointsEdgeLeft={{219,40}}; track.pointsEdgeRight={{219,279}};
        RaceManager manager(cfg); RaceSnapshot snap; snap.sequence=1; snap.captured=1;
        const auto p=manager.update(snap,1,.07f,true,fork,track);
        check(p.stop&&std::string(p.state)=="FORK_WAIT","unknown fork must still stop immediately");
        std::cout<<"PASS crop frames="<<frames<<" recorded_triggers="<<triggers
                 <<" steering_ROI_preserved immediate_unknown_fork_stop\n";
    } catch(const std::exception &e) { std::cerr<<"FAIL "<<e.what()<<'\n'; return 1; }
}
