#include "race_manager.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>

static void check(bool ok,const char *why) { if(!ok) throw std::runtime_error(why); }
int main(int argc,char **argv) {
    try {
        if(argc!=2&&argc!=3) return 2;
        const std::filesystem::path directory=argv[1];
        std::ifstream file(directory/"trace.json"); nlohmann::json trace; file>>trace;
        check(trace["frames"].size()==7,"missing recorded frames");
        RaceConfig cfg; cfg.stopAfterCross=false;
        int cases=0,originalFailures=0;
        for(bool mirror:{false,true}) for(const auto &f:trace["frames"]) {
            cv::Mat bin=cv::imread((directory/f["binary"].get<std::string>()).string(),cv::IMREAD_GRAYSCALE);
            cv::Mat image=cv::imread((directory/f["camera"].get<std::string>()).string());
            check(!bin.empty()&&!image.empty(),"missing recorded image");
            if(mirror) { cv::flip(bin,bin,1); cv::flip(image,image,1); }
            Track track; track.rowCutUp=f["row_cut_up"]; track.rowCutBottom=f["row_cut_bottom"];
            for(const auto &v:f[mirror?"right":"left"])
                track.pointsEdgeLeft.emplace_back(v[0].get<int>(),mirror?319-v[1].get<int>():v[1].get<int>());
            for(const auto &v:f[mirror?"left":"right"])
                track.pointsEdgeRight.emplace_back(v[0].get<int>(),mirror?319-v[1].get<int>():v[1].get<int>());
            const auto before=bin.clone(),originalImage=image.clone();
            RaceSnapshot ai; ai.sequence=f["ai"]["frame"]; ai.captured=f["ai"]["captured"];
            RaceManager original(cfg);
            const auto old=original.update(ai,f["time"],.07f,true,bin,track);
            if(f["frame"]==trace["trigger_frame"]) {
                check(old.stop&&std::string(old.state)=="FORK_WAIT","saved case must reproduce old failure without color evidence");
                ++originalFailures;
            }
            RaceManager candidate(cfg);
            const auto plan=candidate.update(ai,f["time"],.07f,true,bin,track,image);
            check(!plan.stop&&std::string(plan.state)=="LANE","recorded partial red sign still creates false fork");
            check(cv::countNonZero(before!=bin)==0,"geometry filter changed steering binary");
            check(cv::norm(originalImage,image,cv::NORM_INF)==0,"geometry filter changed camera image");
            if(f["frame"]==trace["trigger_frame"]) {
                check(plan.colorRejectedComponents==1&&plan.geometrySplitRows==0,"missing appearance explanation");
                cv::Mat dim; image.convertTo(dim,CV_8UC3,.7);
                RaceManager darker(cfg);
                check(!darker.update(ai,f["time"],.07f,true,bin,track,dim).stop,"moderately dim red cap regression");
                cv::Mat gray,neutral; cv::cvtColor(image,gray,cv::COLOR_BGR2GRAY);
                cv::cvtColor(gray,neutral,cv::COLOR_GRAY2BGR);
                RaceManager cautious(cfg);
                check(cautious.update(ai,f["time"],.07f,true,bin,track,neutral).stop,"ambiguous grayscale evidence must retain stop");
            }
            ++cases;
        }
        check(originalFailures==2,"original failure and mirror both reproduced");
        // 无彩色图时不能凭空忽略黑色岔路；彩色图错尺寸或类型也必须回退。
        cv::Mat fork=cv::Mat::zeros(240,320,CV_8UC1);
        cv::rectangle(fork,{40,0,240,240},cv::Scalar(255),-1);
        cv::rectangle(fork,{135,0,50,160},cv::Scalar(0),-1);
        cv::Mat road; cv::cvtColor(fork,road,cv::COLOR_GRAY2BGR);
        Track track; track.pointsEdgeLeft={{219,40}}; track.pointsEdgeRight={{219,279}};
        RaceSnapshot ai; ai.sequence=1; ai.captured=1;
        for(const auto &image:std::vector<cv::Mat>{road,cv::Mat{},fork,road(cv::Rect(0,0,100,100))}) {
            RaceManager manager(cfg);
            const auto plan=manager.update(ai,1,.07f,true,fork,track,image);
            check(plan.stop&&std::string(plan.state)=="FORK_WAIT","black unknown fork must still stop immediately");
            ++cases;
        }
        // 红色小物体不能擦除与其连接的真实长分隔带，也不能覆盖矩形内的其他暗区。
        cv::Mat connected=road.clone(); cv::ellipse(connected,{160,0},{50,16},0,0,180,{0,0,180},-1);
        RaceManager manager(cfg);
        check(manager.update(ai,1,.07f,true,fork,track,connected).stop,"color cannot remove a connected black fork");
        ++cases;
        int removed=-1;
        const auto unchanged=raceAppearanceGeometry(fork,connected,removed);
        check(removed==0&&cv::countNonZero(unchanged!=fork)==0,"long dark components must remain intact");
        // 彩色物体只有符合触顶扁平连通域条件才能参与排除，其他形状不能套用。
        for(const cv::Rect &shape:std::vector<cv::Rect>{{100,12,80,12},{100,0,50,100},{0,0,80,12}}) {
            cv::Mat b(240,320,CV_8UC1,cv::Scalar(255)); cv::rectangle(b,shape,cv::Scalar(0),-1);
            cv::Mat rgb(240,320,CV_8UC3,cv::Scalar(255,255,255)); cv::rectangle(rgb,shape,{0,0,180},-1);
            const auto result=raceAppearanceGeometry(b,rgb,removed);
            check(removed==0&&cv::countNonZero(result!=b)==0,"unqualified colored object incorrectly removed");
            ++cases;
        }
        // 像素掩膜不能像外接矩形那样擦掉另一个真实黑色分隔带。
        cv::Mat b(240,320,CV_8UC1,cv::Scalar(255)),rgb(240,320,CV_8UC3,cv::Scalar(255,255,255));
        cv::ellipse(b,{160,0},{50,16},0,0,180,cv::Scalar(0),-1);
        cv::ellipse(rgb,{160,0},{50,16},0,0,180,{0,0,180},-1);
        cv::rectangle(b,{230,0,25,160},cv::Scalar(0),-1);
        cv::rectangle(rgb,{230,0,25,160},cv::Scalar(0,0,0),-1);
        const auto filtered=raceAppearanceGeometry(b,rgb,removed);
        check(removed==1,"synthetic clipped red cap missing");
        check(cv::countNonZero(filtered(cv::Rect(230,0,25,160)))==0,"independent black divider was erased");
        ++cases;
        // 即使同帧存在可排除的红色小区域，AI 确认的停止义务不能被几何修补解除。
        Track plain; plain.pointsEdgeLeft={{219,0}}; plain.pointsEdgeRight={{219,319}};
        RaceManager stopManager(cfg); RacePlan stopPlan;
        PredictResult stop{}; stop.type=LABEL_STOP; stop.score=.99f;
        stop.x=100; stop.y=120; stop.width=100; stop.height=100;
        ai.detections={stop};
        for(int i=0;i<5;++i) { ai.sequence++; ai.captured+=.07;
            stopPlan=stopManager.update(ai,ai.captured,.07f,true,b,plain,rgb); }
        check(stopPlan.stop&&std::string(stopPlan.state)=="SIGNAL_WAIT","appearance filter bypassed stop sign");
        ++cases;
        if(argc==3) {
            const std::filesystem::path out=argv[2]; std::filesystem::create_directories(out);
            cv::VideoWriter video((out/"recorded-entry.avi").string(),
                cv::VideoWriter::fourcc('F','F','V','1'),15,{320,240});
            check(video.isOpened(),"lossless full-entry replay fixture unavailable");
            std::ofstream labels(out/"recorded-entry.jsonl");
            for(int frame=1;frame<=100;++frame) {
                const int index=frame<=60?0:std::min(frame-61,6);
                auto image=cv::imread((directory/trace["frames"][index]["camera"].get<std::string>()).string());
                video.write(image);
                labels<<nlohmann::json({{"frame",frame},{"detections",nlohmann::json::array()}}).dump()<<'\n';
            }
        }
        std::cout<<"PASS appearance recorded_frames=14 cases="<<cases<<" original_failures="<<originalFailures
                 <<" dim_red_pass grayscale_fallback immutable_binary immediate_black_fork_stop\n";
    } catch(const std::exception &e) { std::cerr<<"FAIL "<<e.what()<<'\n'; return 1; }
}
