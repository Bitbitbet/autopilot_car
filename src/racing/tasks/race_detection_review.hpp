#pragma once
#include "race_config.hpp"
#include "stop_candidate_review.hpp"
#include "race_crosswalk.hpp"
#include "task_types.hpp"
#include <cmath>

inline bool raceDetectionBoxValid(const PredictResult &d) {
    return d.type>=0&&d.type<15&&std::isfinite(d.score)&&d.score>=0&&d.score<=1&&
           d.x>=0&&d.y>=0&&d.width>0&&d.height>0&&d.x<640&&d.y<480&&
           d.width<=640-d.x&&d.height<=480-d.y;
}
struct RaceDetectionReview {
    bool accepted=false,checkedStop=false,checkedCross=false;
    double redFraction=0;
    std::string reason="INVALID_BOX";
};

// 预览、路牌遮罩和决策共用同一个接受条件。保留实际模型分数，不抬高低分结果。
inline RaceDetectionReview reviewRaceDetection(const PredictResult &d,const cv::Mat &matchedImage,
                                               const RaceConfig &cfg,const CrosswalkGeometry *cross=nullptr) {
    RaceDetectionReview out;
    if(!raceDetectionBoxValid(d)) return out;
    if(d.score>=cfg.confidence) { out.accepted=true;out.reason="MODEL_THRESHOLD";return out; }
    out.reason="BELOW_MODEL_THRESHOLD";
    if(d.type==LABEL_CROSS&&cfg.verifyCrosswalkGeometry&&d.score>=cfg.crossCandidateConfidence) {
        out.checkedCross=true;
        if(matchedImage.size()!=cv::Size(640,480)||matchedImage.type()!=CV_8UC3) {
            out.reason="NO_MATCHED_IMAGE";return out;
        }
        const auto geometry=cross?*cross:detectCrosswalkGeometry(matchedImage);
        out.accepted=crosswalkBoxMatches(geometry,{d.x,d.y,d.width,d.height});
        out.reason=out.accepted?"CROSS_STRIPES_MATCH":"CROSS_STRIPES_NOT_VERIFIED";
        return out;
    }
    if(d.type!=LABEL_STOP||!cfg.verifyStopCandidates||d.score<cfg.stopCandidateConfidence) return out;
    out.checkedStop=true;
    // 必须使用与本次推理结果对应的原图；缺图时不采用其他帧替代。
    if(matchedImage.size()!=cv::Size(640,480)||matchedImage.type()!=CV_8UC3) {
        out.reason="NO_MATCHED_IMAGE";return out;
    }
    const auto review=reviewStopCandidate(matchedImage,{d.x,d.y,d.width,d.height});
    out.accepted=review.matches;out.redFraction=review.redFraction;out.reason=review.reason;
    return out;
}
