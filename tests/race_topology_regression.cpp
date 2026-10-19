#include "race_manager.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <chrono>

static void check(bool ok,const std::string &what) { if(!ok) throw std::runtime_error(what); }

// 保留取证时算法作对照，只在测试中使用，确认测试确实覆盖曾经触发的画面。
static int legacyRows(const cv::Mat &bin,const Track &track,const RaceConfig &cfg,
                      const std::vector<cv::Rect> &regions) {
    int low=track.pointsEdgeLeft.front().y,high=track.pointsEdgeRight.front().y;
    int count=0,missing=0;
    for(int row=track.pointsEdgeLeft.front().x;row>track.rowCutUp;row-=2) {
        std::vector<std::pair<int,int>> blocks;
        for(int col=0;col<320;) {
            if(bin.at<uint8_t>(row,col)<=127) { ++col; continue; }
            int first=col;
            while(col<320&&bin.at<uint8_t>(row,col)>127) ++col;
            if(col-first>=cfg.minBranchWidth&&std::min(col-1,high)>=std::max(first,low))
                blocks.emplace_back(first,col-1);
        }
        if(blocks.empty()) { if(++missing>3) break; continue; }
        missing=0; bool split=false;
        for(size_t i=1;i<blocks.size();++i) {
            int first=blocks[i-1].second+1,last=blocks[i].first-1;
            if(last-first+1<cfg.minBranchGap) continue;
            const bool covered=std::any_of(regions.begin(),regions.end(),[&](const cv::Rect &r) {
                return r.contains({(first+last)/2,row})&&
                    std::min(last,r.x+r.width-1)-std::max(first,r.x)+1>=(last-first+2)/2;
            });
            if(!covered) { split=true; break; }
        }
        if(split) ++count;
        auto best=std::min_element(blocks.begin(),blocks.end(),[&](auto a,auto b) {
            return std::abs(a.first+a.second-low-high)<std::abs(b.first+b.second-low-high);
        });
        low=split?blocks.front().first:best->first;
        high=split?blocks.back().second:best->second;
    }
    return count;
}

static Track recordedTrack(const nlohmann::json &f,bool mirror) {
    Track track; track.rowCutUp=f["row_cut_up"]; track.rowCutBottom=f["row_cut_bottom"];
    for(const auto &p:f[mirror?"right":"left"])
        track.pointsEdgeLeft.emplace_back(p[0].get<int>(),mirror?319-p[1].get<int>():p[1].get<int>());
    for(const auto &p:f[mirror?"left":"right"])
        track.pointsEdgeRight.emplace_back(p[0].get<int>(),mirror?319-p[1].get<int>():p[1].get<int>());
    return track;
}

static RaceSnapshot recordedSnapshot(const nlohmann::json &f,bool mirror) {
    const auto &a=f["ai"];
    RaceSnapshot s; s.sequence=a["frame"]; s.captured=a["captured"]; s.failed=a["failed"];
    for(const auto &v:a["detections"]) {
        PredictResult d{}; d.type=v["type"]; d.score=v["score"]; d.x=v["x"]; d.y=v["y"];
        d.width=v["width"]; d.height=v["height"];
        if(mirror) d.x=320-d.x-d.width;
        s.detections.push_back(d);
    }
    return s;
}

int main(int argc,char **argv) {
    try {
        if(argc!=2) return 2;
        const std::filesystem::path base=argv[1];
        RaceConfig cfg; cfg.stopAfterCross=false;
        int frames=0,episodes=0,oldWaits=0,states=0;
        const auto began=std::chrono::steady_clock::now();
        for(int episode=1;episode<=9;++episode) {
            const auto dir=base/("fork-trace-"+std::to_string(episode));
            std::ifstream file(dir/"trace.json"); nlohmann::json trace; file>>trace;
            check(trace["frames"].size()==7,"incomplete episode");
            ++episodes;
            for(bool mirror:{false,true}) {
                RaceManager paired(cfg),missed(cfg);
                for(const auto &f:trace["frames"]) {
                    cv::Mat bin=cv::imread((dir/f["binary"].get<std::string>()).string(),cv::IMREAD_GRAYSCALE);
                    check(!bin.empty(),"missing binary");
                    if(mirror) cv::flip(bin,bin,1);
                    auto track=recordedTrack(f,mirror); auto snap=recordedSnapshot(f,mirror);
                    auto regions=raceSignRegions(snap,cfg);
                    if(!mirror) {
                        const int old=legacyRows(bin,track,cfg,regions);
                        check(old==f["masked_split_rows"].get<int>(),"saved geometry no longer reproduced exactly");
                        if(old>=cfg.minBranchRows) ++oldWaits;
                    }
                    check(raceBranchPath(bin,track,0,cfg,regions).splitRows<cfg.minBranchRows,
                          "recorded straight road still treated as fork");
                    check(raceBranchPath(bin,track,0,cfg).splitRows<cfg.minBranchRows,
                          "missed sign must not create a fork");
                    auto p=paired.update(snap,f["time"],.07f,true,bin,track);
                    check(std::string(p.state)!="FORK_WAIT","paired decision still waits for false fork");
                    snap.detections.clear();
                    p=missed.update(snap,f["time"],.07f,true,bin,track);
                    check(std::string(p.state)=="LANE"&&!p.stop,"empty AI on recorded straight road must cruise");
                    states+=2; ++frames;
                }
            }
        }
        check(oldWaits==18,"expected all 18 previously triggering frames");
        int synthetic=0;
        for(int kind=0;kind<8;++kind) {
            cv::Mat bin=cv::Mat::zeros(240,320,CV_8UC1);
            cv::rectangle(bin,{20,0,280,240},cv::Scalar(255),-1);
            if(kind==0) cv::rectangle(bin,{135,0,50,160},cv::Scalar(0),-1); // 真正的持续分隔带。
            if(kind==1) { // 弯曲分隔带。
                std::vector<cv::Point> polygon{{100,0},{150,0},{200,160},{150,160}};
                cv::fillConvexPoly(bin,polygon,cv::Scalar(0));
            }
            if(kind==2) { // 两个分隔带，三条持续通路。
                cv::rectangle(bin,{90,0,25,160},cv::Scalar(0),-1);
                cv::rectangle(bin,{205,0,25,160},cv::Scalar(0),-1);
            }
            if(kind==3) cv::rectangle(bin,{130,70,60,70},cv::Scalar(0),-1); // 有限黑色孤岛。
            if(kind==4) cv::circle(bin,{160,55},60,cv::Scalar(0),12); // 顶部被裁掉的圆环。
            if(kind==5) cv::circle(bin,{160,90},55,cv::Scalar(0),12); // 完整圆环。
            if(kind==6) for(int row:{70,90,110}) cv::rectangle(bin,{130,row,60,2},cv::Scalar(0),-1);
            Track track; track.rowCutUp=10; track.pointsEdgeLeft={{219,20}}; track.pointsEdgeRight={{219,299}};
            const bool fork=kind<3;
            const auto path=raceBranchPath(bin,track,0,cfg);
            check((path.splitRows>=cfg.minBranchRows)==fork,"synthetic topology kind "+std::to_string(kind));
            RaceSnapshot snap; snap.sequence=1; snap.captured=1;
            RaceManager manager(cfg); RacePlan p;
            for(int n=0;n<6;++n) { snap.sequence++; snap.captured+=.07;
                p=manager.update(snap,snap.captured,.07,true,bin,track); }
            check(p.stop==fork,"unknown genuine fork must still stop");
            if(fork) {
                check(std::string(p.state)=="FORK_WAIT","unknown fork state");
                for(int label:{LABEL_YLEFT,LABEL_YRIGHT}) {
                    RaceManager directed(cfg); PredictResult d{};
                    d.type=label; d.score=.99f; d.x=0; d.y=0; d.width=12; d.height=15;
                    snap.detections={d};
                    for(int n=0;n<8;++n) { snap.sequence++; snap.captured+=.07;
                        p=directed.update(snap,snap.captured,.07,true,bin,track); }
                    check(p.selectedBranch&&!p.stop,"known direction must select genuine fork");
                    check(p.path.left.size()>20,"selected path must remain connected");
                    const int center=p.path.left.back().y+p.path.right.back().y;
                    check(label==LABEL_YLEFT?center<320:center>320,"selected path on wrong side");
                    // 过期结果仍须停车，不能借几何修复取消原有保护。
                    p=directed.update(snap,snap.captured+1,.07,true,bin,track);
                    check(p.stop&&std::string(p.state)=="AI_WAIT","stale inference must stop");
                }
            }
            ++synthetic;
        }
        std::cout<<"PASS episodes="<<episodes<<" recorded_frames="<<frames<<" old_wait_frames="<<oldWaits
                 <<" decision_checks="<<states<<" synthetic="<<synthetic<<" elapsed_ms="
                 <<std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-began).count()<<'\n';
    } catch(const std::exception &e) { std::cerr<<"FAIL "<<e.what()<<'\n'; return 1; }
}
