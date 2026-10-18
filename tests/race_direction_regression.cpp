#include "race_preview.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

int main(int argc,char **argv) {
    if(argc!=2) return 2;
    const std::filesystem::path base=argv[1];
    int failures=0,tested=0;
    for(const auto &run:std::filesystem::directory_iterator(base)) {
        if(!run.is_directory()) continue;
        const auto runName=run.path().filename().string();
        if(runName!="1792307034685-12162"&&runName!="1792307575938-18396") continue;
        for(const auto &sample:std::filesystem::directory_iterator(run)) {
            if(!sample.is_directory()||sample.path().filename().string().rfind("ai-sample-",0)!=0) continue;
            std::ifstream f(sample.path()/"detections.json");
            if(!f) continue;
            nlohmann::json meta; f>>meta;
            cv::Mat image=cv::imread((sample.path()/"camera.png").string());
            for(const auto &v:meta["detections"]) {
                if(v["type"]!=LABEL_YLEFT&&v["type"]!=LABEL_YRIGHT) continue;
                PredictResult d{};
                d.type=v["type"]; d.x=v["x"]; d.y=v["y"]; d.width=v["width"]; d.height=v["height"]; d.score=v["score"];
                const char *reason;
                int result=forkArrowDirection(image,d,reason);
                const bool green=sample.path().parent_path().filename()=="1792307034685-12162"&&
                    sample.path().filename()=="ai-sample-10";
                // 人工核对：现有蓝牌均为左下箭头；绿色右箭头不是本复核器的蓝色岔路牌。
                const bool clipped=sample.path().parent_path().filename()=="1792307575938-18396"&&
                    sample.path().filename()=="ai-sample-6";
                const int expected=(green||clipped)?0:-1;
                std::cout<<sample.path().parent_path().filename()<<'/'<<sample.path().filename()
                    <<" raw="<<d.type<<" shape="<<result<<" reason="<<reason<<'\n';
                ++tested;
                if(result!=expected) ++failures;
                std::vector<PredictResult> corrected{d};
                const auto checks=verifyForkDirections(image,corrected);
                if(checks.size()!=1||checks.front().rawType!=d.type) ++failures;
                if(!expected) { if(!corrected.empty()) ++failures; }
                else if(corrected.size()!=1||corrected.front().type!=LABEL_YLEFT||corrected.front().score!=d.score) ++failures;
                if(!green) {
                    cv::Mat mirrored; cv::flip(image,mirrored,1);
                    PredictResult right=d; right.x=image.cols-d.x-d.width;
                    const int mirrorResult=forkArrowDirection(mirrored,right,reason);
                    std::cout<<" mirrored shape="<<mirrorResult<<" reason="<<reason<<'\n';
                    if(mirrorResult!=(clipped?0:1)) ++failures;
                    ++tested;
                }
            }
        }
    }
    if(tested!=9) { std::cerr<<"Missing real fixtures\n"; ++failures; }
    cv::Mat blank(240,320,CV_8UC3,cv::Scalar(255,255,255));
    PredictResult bad{}; bad.type=LABEL_YLEFT; bad.score=.99f; bad.x=80; bad.y=40; bad.width=160; bad.height=160;
    std::vector<PredictResult> rejected{bad};
    verifyForkDirections(blank,rejected);
    if(!rejected.empty()) ++failures;
    bad.type=LABEL_BUSY; std::vector<PredictResult> other{bad};
    if(!verifyForkDirections(blank,other).empty()||other.size()!=1||other.front().type!=LABEL_BUSY) ++failures;
    cv::circle(blank,{160,120},65,{255,0,0},-1);
    cv::rectangle(blank,{140,105,40,30},{255,255,255},-1);
    const char *reason;
    if(forkArrowDirection(blank,bad,reason)!=0) ++failures;
    tested+=3;
    {
        const auto rawPath=base/"1792307575938-18396"/"ai-sample-7";
        const auto frame=cv::imread((rawPath/"camera.png").string());
        std::ifstream input(rawPath/"detections.json"); nlohmann::json meta; input>>meta;
        const auto v=meta["detections"][0];
        PredictResult wrong{}; wrong.type=LABEL_YRIGHT; wrong.score=v["score"];
        wrong.x=v["x"]; wrong.y=v["y"]; wrong.width=v["width"]; wrong.height=v["height"];
        RacePerception worker([&](const cv::Mat&) { return std::vector<PredictResult>{wrong}; });
        worker.submit(frame,123,raceTime());
        RaceSnapshot snapshot;
        for(int i=0;i<200;++i) {
            snapshot=worker.snapshot(); if(snapshot.sequence==123) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        if(snapshot.sequence!=123||snapshot.detections.size()!=1||snapshot.detections[0].type!=LABEL_YLEFT||
           snapshot.directionChecks.size()!=1||snapshot.directionChecks[0].rawType!=LABEL_YRIGHT) ++failures;
        else {
            const auto json=raceDetectionJson(snapshot,raceTime(),.65f);
            if(json["direction_checks"][0]["final_label"]!="yleft"||
               json["direction_checks"][0]["raw_label"]!="yright") ++failures;
            if(!saveRaceSample(base/"direction-preview",snapshot,raceTime(),RaceConfig{},1)) ++failures;
        }
        tested+=2;
    }
    std::cout<<"cases="<<tested<<" failures="<<failures<<'\n';
    return failures?1:0;
}
