#pragma once
#include "race_perception.hpp"
#include "race_detection_review.hpp"
#include "race_obstacle_plan.hpp"
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
    return raceDetectionBoxValid(d);
}
inline nlohmann::json raceDirectionJson(const RaceSnapshot &s) {
    nlohmann::json checks=nlohmann::json::array();
    for(const auto &c:s.directionChecks) checks.push_back({{"raw_type",c.rawType},
        {"raw_label",raceLabel(c.rawType)},{"model_score",c.modelScore},
        {"final_type",c.finalType},{"final_label",c.finalType<0?"unverified":raceLabel(c.finalType)},
        {"reason",c.reason},{"x",c.box.x},{"y",c.box.y},{"width",c.box.width},{"height",c.box.height}});
    return checks;
}
inline bool raceCrosswalkFallback(const RaceSnapshot &s,const RaceConfig &config) {
    if(!config.verifyCrosswalkGeometry||!s.crosswalk.accepted||
       s.image.type()!=CV_8UC3||s.image.size()!=cv::Size(640,480)) return false;
    return std::none_of(s.detections.begin(),s.detections.end(),[&](const PredictResult &d) {
        return d.type==LABEL_CROSS&&reviewRaceDetection(d,s.image,config,&s.crosswalk).accepted;
    });
}
inline nlohmann::json raceCrosswalkJson(const RaceSnapshot &s,const RaceConfig &config) {
    const auto &c=s.crosswalk;
    return {{"enabled",config.verifyCrosswalkGeometry},{"accepted",config.verifyCrosswalkGeometry&&c.accepted},
        {"source","STRIPE_GEOMETRY"},{"model_score",nullptr},{"fallback_used",raceCrosswalkFallback(s,config)},
        {"stripes",c.stripes},{"contrast",c.contrast},{"reason",c.reason},
        {"x",c.box.x},{"y",c.box.y},{"width",c.box.width},{"height",c.box.height}};
}
inline nlohmann::json raceDetectionJson(const RaceSnapshot &s,double time,const RaceConfig &config) {
    nlohmann::json boxes=nlohmann::json::array();
    for(const auto &d:s.detections) if(raceBoxValid(d)) {
        const auto review=reviewRaceDetection(d,s.image,config,&s.crosswalk);
        boxes.push_back({
        {"type",d.type},{"label",raceLabel(d.type)},{"score",d.score},
        {"x",d.x},{"y",d.y},{"width",d.width},{"height",d.height},
        {"passes_threshold",d.score>=config.confidence},{"accepted",review.accepted},
        {"checked_stop",review.checkedStop},{"checked_cross",review.checkedCross},
        {"acceptance_reason",review.reason},{"red_fraction",review.redFraction}});
    }
    return {{"frame",s.sequence},{"captured",s.captured},{"age_s",time-s.captured},
        {"failed",s.failed},{"error",s.error},{"confidence_threshold",config.confidence},
        {"stop_candidate_threshold",config.stopCandidateConfidence},{"detections",boxes},
        {"cross_candidate_threshold",config.crossCandidateConfidence},
        {"crosswalk_geometry",raceCrosswalkJson(s,config)},{"direction_checks",raceDirectionJson(s)}};
}
inline nlohmann::json raceDetectionJson(const RaceSnapshot &s,double time,float threshold) {
    RaceConfig config;config.confidence=threshold;return raceDetectionJson(s,time,config);
}
inline cv::Mat racePreviewImage(const RaceSnapshot &s,double time,const RaceConfig &config) {
    cv::Mat canvas=cv::Mat::zeros(600,1280,CV_8UC3);
    const double age=s.sequence?time-s.captured:-1;
    const bool fresh=!s.failed&&s.sequence&&age>=0&&age<=config.maxDetectionAge;
    auto text=[&](std::string value,int y,cv::Scalar color) {
        cv::putText(canvas,value,{650,y*2},cv::FONT_HERSHEY_PLAIN,.85,color,1);
    };
    const cv::Scalar status=fresh?cv::Scalar(0,255,0):cv::Scalar(0,0,255);
    if(!s.image.empty()&&s.image.type()==CV_8UC3&&s.image.size()==cv::Size(640,480))
        s.image.copyTo(canvas(cv::Rect(0,0,640,480)));
    int count=0,line=65;
    for(const auto &d:s.detections) {
        if(!raceBoxValid(d)) continue;
        const auto review=reviewRaceDetection(d,s.image,config,&s.crosswalk);
        if(!review.accepted&&!review.checkedStop&&!review.checkedCross) continue;
        const auto color=review.accepted?status:cv::Scalar(0,140,255);
        if(review.accepted) ++count;
        cv::rectangle(canvas,{d.x,d.y,d.width,d.height},color,1);
        std::ostringstream label; label<<raceLabel(d.type)<<' '<<std::fixed<<std::setprecision(2)<<d.score;
        if(review.checkedStop||review.checkedCross) label<<(review.accepted?" CHECKED":" REJECT");
        cv::putText(canvas,label.str(),{d.x,std::max(12,d.y-3)},cv::FONT_HERSHEY_PLAIN,.8,color,1);
        if(line<280) { text(label.str(),line,color); line+=15; }
    }
    if(raceCrosswalkFallback(s,config)) {
        const auto &c=s.crosswalk;++count;
        cv::rectangle(canvas,c.box,status,1);
        cv::putText(canvas,"cross GEOMETRY",{c.box.x,std::max(12,c.box.y-3)},cv::FONT_HERSHEY_PLAIN,.8,status,1);
        if(line<280) { text("cross GEOMETRY bars="+std::to_string(c.stripes),line,status);line+=15; }
    }
    for(const auto &c:s.directionChecks) {
        const cv::Scalar color=c.finalType<0?cv::Scalar(0,180,255):status;
        if(c.finalType<0&&c.box.width>0&&c.box.height>0) {
            const auto clipped=c.box&cv::Rect(0,0,640,480);
            if(!clipped.empty()) cv::rectangle(canvas,clipped,color,1);
        }
        if(line<250) {
            text(std::string("model: ")+raceLabel(c.rawType),line,color); line+=15;
            text(std::string("shape: ")+c.reason,line,color); line+=15;
            text(std::string("final: ")+(c.finalType<0?"WAIT":raceLabel(c.finalType)),line,color); line+=15;
        }
    }
    std::ostringstream header; header<<"frame="<<s.sequence<<" age="<<std::fixed<<std::setprecision(2)<<age<<"s";
    text(header.str(),15,status);
    text(s.failed?"AI FAILED":fresh?"FRESH":"STALE / WAIT",32,status);
    text("objects="+std::to_string(count),49,status);
    cv::putText(canvas,"s: queue matched sample (all modes)",{5,520},cv::FONT_HERSHEY_PLAIN,.85,{0,255,255},1);
    cv::putText(canvas,"Check box + label + stability, not score alone",{5,560},cv::FONT_HERSHEY_PLAIN,.85,{255,255,255},1);
    return canvas;
}

// GUI 主线程画框和保存；控制线程只发布引用计数快照，不等待远程绘图。
class RacePreview {
    std::mutex mutex;
    RaceSnapshot latest;
    double time=0;
    RaceObstaclePreviewPlan navigation;
public:
    void publish(RaceSnapshot s,double now,RaceObstaclePreviewPlan plan={}) {
        std::unique_lock lock(mutex,std::try_to_lock);
        if(!lock.owns_lock()) return;
        if(s.image.empty()&&s.sequence==latest.sequence) s.image=latest.image;
        latest=std::move(s); time=now;navigation=std::move(plan);
    }
    std::pair<RaceSnapshot,double> snapshot() {
        std::lock_guard lock(mutex); return {latest,time};
    }
    struct View { RaceSnapshot ai;double time;RaceObstaclePreviewPlan navigation; };
    View view() { std::lock_guard lock(mutex);return {latest,time,navigation}; }
};

inline bool saveRaceSample(const std::filesystem::path &directory,const RaceSnapshot &s,
                           double time,const RaceConfig &config,int index,
                           const RaceObstaclePreviewPlan *navigation=nullptr,const RaceNavigationConfig &navConfig={}) {
    if(directory.empty()||s.image.empty()) return false;
    const auto path=directory/("ai-sample-"+std::to_string(index));
    std::filesystem::create_directories(path);
    if(!cv::imwrite((path/"camera.png").string(),s.image)||
       !cv::imwrite((path/"detections.png").string(),racePreviewImage(s,time,config)))
        throw std::runtime_error("Cannot save AI sample image");
    std::ofstream result(path/"detections.json");
    result<<raceDetectionJson(s,time,config).dump(2);
    if(!result) throw std::runtime_error("Cannot save AI sample metadata");
    if(navigation&&navigation->sequence==s.sequence&&navigation->captured==s.captured) {
        auto json=raceObstacleJson(*navigation,navConfig);json["age_s"]=time-s.captured;
        const bool fresh=!s.failed&&time>=s.captured&&time-s.captured<=config.maxDetectionAge;
        json["fresh"]=fresh;
        auto overlay=s.image.clone();
        if(fresh)drawRaceObstaclePreview(overlay,*navigation);
        else { json["candidate"]=false;json["would_stop"]=true;json["state"]="OBS_AI_STALE"; }
        if(!cv::imwrite((path/"navigation.png").string(),overlay))throw std::runtime_error("Cannot save navigation preview");
        std::ofstream output(path/"navigation.json");output<<json.dump(2);
        if(!output)throw std::runtime_error("Cannot save navigation preview metadata");
    }
    return true;
}
