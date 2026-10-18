#pragma once
#include <nlohmann/json.hpp>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>

// 比赛策略单独配置，原来的 trackview.json 继续只管理巡线和转向。
struct RaceConfig {
    float confidence=0.65f, maxDetectionAge=0.40f, confirmGap=0.30f;
    int confirmFrames=3;
    float limitedSpeed=0.40f, forkSpeed=0.15f, approachSpeed=0.12f;
    float directionMemory=4.f, forkConfirmTime=0.12f, forkMinTime=0.40f;
    float forkMaxTime=4.f, forkExitTime=0.35f, cooldown=2.f;
    int minBranchWidth=30, minBranchGap=12, minBranchRows=3;
    int triggerRow=145, obstacleRow=155;
    float clearTime=0.60f;
    // 2026 要求越过斑马线后停车。像素阈值只供现场标定，不表示米。
    bool stopAfterCross=true;
    int crossPassedRow=205;
    float crossClearTime=0.25f, stopHoldTime=1.f;
    // 基础版不把“目标消失”当作道闸已经抬起，默认需要同一道闸变为竖直。
    float gateOpenAspect=0.75f;
    bool verifyForkArrows=true; // 对左右岔路类别增加蓝底白箭头几何复核。
    std::string modelDirectory="../res/models/yolov3_mobilenet_v1";
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(RaceConfig,confidence,maxDetectionAge,confirmGap,
        confirmFrames,limitedSpeed,forkSpeed,approachSpeed,directionMemory,
        forkConfirmTime,forkMinTime,forkMaxTime,forkExitTime,cooldown,minBranchWidth,
        minBranchGap,minBranchRows,triggerRow,obstacleRow,clearTime,stopAfterCross,
        crossPassedRow,crossClearTime,stopHoldTime,gateOpenAspect,verifyForkArrows,modelDirectory)
    void validate() const {
        auto range=[](float v,float lo,float hi) { return std::isfinite(v)&&v>=lo&&v<=hi; };
        if(!range(confidence,0.1f,1.f)||!range(maxDetectionAge,0.05f,1.f)||
           !range(confirmGap,0.05f,1.f)||confirmFrames<2||confirmFrames>10||
           !range(limitedSpeed,0.05f,0.49f)||!range(forkSpeed,0.05f,10.f)||
           !range(approachSpeed,0.05f,10.f)||!range(directionMemory,0.5f,15.f)||
           !range(forkConfirmTime,0.05f,1.f)||!range(forkMinTime,0.1f,3.f)||
           !range(forkMaxTime,forkMinTime,10.f)||!range(forkExitTime,0.1f,2.f)||
           !range(cooldown,0.1f,10.f)||minBranchWidth<20||minBranchWidth>100||
           minBranchGap<5||minBranchGap>50||minBranchRows<2||minBranchRows>20||
           triggerRow<50||triggerRow>220||obstacleRow<50||obstacleRow>230||
           !range(clearTime,0.2f,3.f)||crossPassedRow<150||crossPassedRow>235||
           !range(crossClearTime,0.1f,2.f)||!range(stopHoldTime,0.1f,10.f)||
           !range(gateOpenAspect,0.1f,1.f)||modelDirectory.empty())
            throw std::invalid_argument("Invalid race configuration");
    }
    void load(const std::string &path) {
        std::ifstream in(path);
        if(!in) throw std::runtime_error("Cannot read race config: "+path);
        nlohmann::json j; in>>j;
        if(!j.is_object()) throw std::invalid_argument("Race config must be an object");
        nlohmann::json merged=*this;
        for(auto it=j.begin();it!=j.end();++it) {
            if(!merged.contains(it.key())) throw std::invalid_argument("Unknown race key: "+it.key());
            merged[it.key()]=it.value();
        }
        *this=merged.get<RaceConfig>(); validate();
    }
};
