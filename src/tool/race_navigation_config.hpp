#pragma once
#include <nlohmann/json.hpp>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>

// 像素预览及显式绕行的控制参数；像素宽度比例不代表实测车轮净空。
// 配置独立于 AI 后端，保持已部署的感知 ABI 不变。
struct RaceNavigationConfig {
    std::string obstacleMode="preview";
    float previewVehicleWidthFraction=.55f,previewClearanceFraction=.03f;
    float maxLateralSlope=.8f;
    float bypassWheelbase=.20f,bypassLookahead=.20f,bypassFullLockDegrees=45.f;
    int rowStep=2,minRows=35,minLookaheadRow=35,startTolerance=8;
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(RaceNavigationConfig,obstacleMode,previewVehicleWidthFraction,
        previewClearanceFraction,maxLateralSlope,rowStep,minRows,minLookaheadRow,startTolerance,
        bypassWheelbase,bypassLookahead,bypassFullLockDegrees)
    void validate() const {
        auto range=[](float n,float lo,float hi){return std::isfinite(n)&&n>=lo&&n<=hi;};
        if((obstacleMode!="off"&&obstacleMode!="preview")||
           !range(previewVehicleWidthFraction,.25f,.8f)||!range(previewClearanceFraction,.01f,.1f)||
           previewVehicleWidthFraction+2*previewClearanceFraction>=.95f||
           !range(maxLateralSlope,.1f,1.5f)||rowStep<1||rowStep>4||minRows<20||minRows>100||
           minLookaheadRow<10||minLookaheadRow>100||startTolerance<0||startTolerance>15||
           !range(bypassWheelbase,.1f,.35f)||!range(bypassLookahead,.12f,.4f)||!range(bypassFullLockDegrees,20.f,60.f))
            throw std::invalid_argument("Invalid navigation preview configuration; drive bypass needs ground calibration");
    }
    void load(const std::string &path,bool required=false) {
        std::ifstream in(path);
        if(!in) { if(required)throw std::runtime_error("Cannot read navigation config: "+path);validate();return; }
        nlohmann::json input;in>>input;
        if(!input.is_object())throw std::invalid_argument("Navigation config must be an object");
        nlohmann::json merged=*this;
        for(auto it=input.begin();it!=input.end();++it) {
            if(!merged.contains(it.key()))throw std::invalid_argument("Unknown navigation key: "+it.key());
            merged[it.key()]=it.value();
        }
        *this=merged.get<RaceNavigationConfig>();validate();
    }
};
