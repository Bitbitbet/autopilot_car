#pragma once
#include "race_perception.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

inline const char *raceLabel(int type) {
    static const char *names[]={"cone","person","busy","limit","unlimit","stop","flow",
        "park","gate","cross","fork","left","choice","yleft","yright"};
    return type>=0&&type<15?names[type]:"unknown";
}
inline bool raceBoxValid(const PredictResult &d) {
    return d.type>=0&&d.type<15&&std::isfinite(d.score)&&d.score>=0&&d.score<=1&&
        d.x>=0&&d.y>=0&&d.width>0&&d.height>0&&d.x<320&&d.y<240&&
        d.width<=320-d.x&&d.height<=240-d.y;
}
inline nlohmann::json raceDetectionJson(const RaceSnapshot &s,double time,float threshold) {
    nlohmann::json boxes=nlohmann::json::array();
    for(const auto &d:s.detections) if(raceBoxValid(d)) boxes.push_back({
        {"type",d.type},{"label",raceLabel(d.type)},{"score",d.score},
        {"x",d.x},{"y",d.y},{"width",d.width},{"height",d.height},
        {"passes_threshold",d.score>=threshold}});
    return {{"frame",s.sequence},{"captured",s.captured},{"age_s",time-s.captured},
        {"failed",s.failed},{"error",s.error},{"confidence_threshold",threshold},{"detections",boxes}};
}
inline cv::Mat racePreviewImage(const RaceSnapshot &s,double time,const RaceConfig &config) {
    cv::Mat canvas=cv::Mat::zeros(300,640,CV_8UC3);
    const double age=s.sequence?time-s.captured:-1;
    const bool fresh=!s.failed&&s.sequence&&age>=0&&age<=config.maxDetectionAge;
    auto text=[&](std::string value,int y,cv::Scalar color) {
        cv::putText(canvas,value,{325,y},cv::FONT_HERSHEY_PLAIN,.85,color,1);
    };
    const cv::Scalar status=fresh?cv::Scalar(0,255,0):cv::Scalar(0,0,255);
    if(!s.image.empty()&&s.image.type()==CV_8UC3&&s.image.size()==cv::Size(320,240))
        s.image.copyTo(canvas(cv::Rect(0,0,320,240)));
    int count=0,line=65;
    for(const auto &d:s.detections) {
        if(!raceBoxValid(d)||d.score<config.confidence) continue;
        ++count;
        cv::rectangle(canvas,{d.x,d.y,d.width,d.height},status,1);
        std::ostringstream label; label<<raceLabel(d.type)<<' '<<std::fixed<<std::setprecision(2)<<d.score;
        cv::putText(canvas,label.str(),{d.x,std::max(12,d.y-3)},cv::FONT_HERSHEY_PLAIN,.8,status,1);
        if(line<280) { text(label.str(),line,status); line+=15; }
    }
    std::ostringstream header; header<<"frame="<<s.sequence<<" age="<<std::fixed<<std::setprecision(2)<<age<<"s";
    text(header.str(),15,status);
    text(s.failed?"AI FAILED":fresh?"FRESH":"STALE / WAIT",32,status);
    text("objects="+std::to_string(count),49,status);
    cv::putText(canvas,"s: save sample (NO CONTROL only)",{5,260},cv::FONT_HERSHEY_PLAIN,.9,{0,255,255},1);
    cv::putText(canvas,"Check box + label + stability, not score alone",{5,280},cv::FONT_HERSHEY_PLAIN,.85,{255,255,255},1);
    return canvas;
}

// GUI 主线程画框和保存；控制线程只发布引用计数快照，不等待远程绘图。
class RacePreview {
    std::mutex mutex;
    RaceSnapshot latest;
    double time=0;
public:
    void publish(RaceSnapshot s,double now) {
        std::unique_lock lock(mutex,std::try_to_lock);
        if(!lock.owns_lock()) return;
        if(s.image.empty()&&s.sequence==latest.sequence) s.image=latest.image;
        latest=std::move(s); time=now;
    }
    std::pair<RaceSnapshot,double> snapshot() {
        std::lock_guard lock(mutex); return {latest,time};
    }
};

inline bool saveRaceSample(const std::filesystem::path &directory,const RaceSnapshot &s,
                           double time,const RaceConfig &config,int index) {
    if(directory.empty()||s.image.empty()) return false;
    const auto path=directory/("ai-sample-"+std::to_string(index));
    std::filesystem::create_directories(path);
    if(!cv::imwrite((path/"camera.png").string(),s.image)||
       !cv::imwrite((path/"detections.png").string(),racePreviewImage(s,time,config)))
        throw std::runtime_error("Cannot save AI sample image");
    std::ofstream result(path/"detections.json");
    result<<raceDetectionJson(s,time,config.confidence).dump(2);
    if(!result) throw std::runtime_error("Cannot save AI sample metadata");
    return true;
}
