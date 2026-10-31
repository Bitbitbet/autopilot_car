#pragma once
#include <nlohmann/json.hpp>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>

// 比赛策略单独配置，原来的 trackview.json 继续只管理巡线和转向。
struct RaceConfig {
    int imageWidth=640,imageHeight=480;
    float confidence=0.65f, maxDetectionAge=0.40f, confirmGap=0.30f;
    int confirmFrames=3;
    float limitedSpeed=0.40f, forkSpeed=0.15f, approachSpeed=0.12f;
    float directionMemory=4.f, forkConfirmTime=0.12f, forkMinTime=0.40f;
    float forkMaxTime=4.f, forkExitTime=0.35f, cooldown=2.f;
    int minBranchWidth=60, minBranchGap=24, minBranchRows=3;
    int triggerRow=290, obstacleRow=310;
    float clearTime=0.60f;
    // 2026 要求越过斑马线后停车。像素阈值只供现场标定，不表示米。
    bool stopAfterCross=true;
    int crossPassedRow=410;
    float crossClearTime=0.25f, stopHoldTime=1.f;
    // 基础版不把“目标消失”当作道闸已经抬起，默认需要同一道闸变为竖直。
    float gateOpenAspect=0.75f;
    bool verifyForkArrows=true; // 对左右岔路类别增加蓝底白箭头几何复核。
    std::string modelDirectory="../res/models/yolov3_mobilenet_v1";
    bool verifyStopCandidates=true;
    float stopCandidateConfidence=.35f; // 仅停车牌的候选采集下限，不能用于放行或其他类别。
    bool verifyCrosswalkGeometry=true;
    float crossCandidateConfidence=.10f; // 仅斑马线：低分结果须同帧条纹复核；远处可单独提供几何证据。
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(RaceConfig,imageWidth,imageHeight,confidence,maxDetectionAge,confirmGap,
        confirmFrames,limitedSpeed,forkSpeed,approachSpeed,directionMemory,
        forkConfirmTime,forkMinTime,forkMaxTime,forkExitTime,cooldown,minBranchWidth,
        minBranchGap,minBranchRows,triggerRow,obstacleRow,clearTime,stopAfterCross,
        crossPassedRow,crossClearTime,stopHoldTime,gateOpenAspect,verifyForkArrows,modelDirectory,
        verifyStopCandidates,stopCandidateConfidence,verifyCrosswalkGeometry,crossCandidateConfidence)
    void validate() const {
        auto range=[](float v,float lo,float hi) { return std::isfinite(v)&&v>=lo&&v<=hi; };
        if(imageWidth!=640||imageHeight!=480||!range(confidence,0.1f,1.f)||!range(maxDetectionAge,0.05f,1.f)||
           !range(confirmGap,0.05f,1.f)||confirmFrames<2||confirmFrames>10||
           !range(limitedSpeed,0.05f,0.49f)||!range(forkSpeed,0.05f,10.f)||
           !range(approachSpeed,0.05f,10.f)||!range(directionMemory,0.5f,15.f)||
           !range(forkConfirmTime,0.05f,1.f)||!range(forkMinTime,0.1f,3.f)||
           !range(forkMaxTime,forkMinTime,60.f)||!range(forkExitTime,0.1f,2.f)||
           !range(cooldown,0.1f,10.f)||minBranchWidth<40||minBranchWidth>200||
           minBranchGap<10||minBranchGap>100||minBranchRows<2||minBranchRows>20||
           triggerRow<100||triggerRow>440||obstacleRow<100||obstacleRow>460||
           !range(clearTime,0.2f,3.f)||crossPassedRow<300||crossPassedRow>470||
           !range(crossClearTime,0.1f,2.f)||!range(stopHoldTime,0.1f,10.f)||
           !range(gateOpenAspect,0.1f,1.f)||modelDirectory.empty()||
           !range(stopCandidateConfidence,.30f,.50f)||!range(crossCandidateConfidence,.05f,.35f))
            throw std::invalid_argument("Invalid race configuration");
    }
    void load(const std::string &path) {
        std::ifstream in(path);
        if(!in) throw std::runtime_error("Cannot read race config: "+path);
        nlohmann::json j; in>>j;
        if(!j.is_object()) throw std::invalid_argument("Race config must be an object");
        
        const int reference=j.value("imageWidth",320);
        if(j.value("imageHeight",reference*3/4)!=reference*3/4)throw std::invalid_argument("Pixel reference must be 4:3");
        if(reference!=320&&reference!=640)throw std::invalid_argument("Unsupported race pixel reference");
        if(reference==320)for(const char *key:{"minBranchWidth","minBranchGap","triggerRow","obstacleRow","crossPassedRow"})if(j.contains(key))j[key]=j[key].get<int>()*2;
        j["imageWidth"]=640;j["imageHeight"]=480;
        nlohmann::json merged=*this;
        for(auto it=j.begin();it!=j.end();++it) {
            if(!merged.contains(it.key())) throw std::invalid_argument("Unknown race key: "+it.key());
            merged[it.key()]=it.value();
        }
        *this=merged.get<RaceConfig>(); validate();
    }
};
