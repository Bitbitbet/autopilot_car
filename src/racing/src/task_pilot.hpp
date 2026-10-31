#pragma once
#include "pilot.hpp"
#include "task_track.hpp"
#include "trackview_config.hpp"
#include <nlohmann/json.hpp>
// The master pilot owns path quality, single-edge ground reconstruction and PD.
// This view only adapts task telemetry and supplies task-specific speed policies.
class RaceTaskPilot {
    TrackviewConfig config;
    AutoPilot native{640,480};
    float commandedSpeed=0;
    static const Camera &camera(){static Camera unmapped(cv::Mat{});return Track::camera?*Track::camera:unmapped;}
public:
    struct WidthReading{int row,observations,pendingObservations;float width;double age;bool refreshDue,calibratedReference;};
    std::vector<PointX> centerEdge;
    int center=320,dualPoints=0,singlePoints=0,rejectedPoints=0,missingWidthPoints=0,untrustedPoints=0,widthRangePoints=0,jumpPoints=0;
    float error=0,heading=0,confidence=0,severity=0,rawSeverity=0,curveHeading=0,speedTarget=0;
    bool valid=false,ready=false,lost=false,degraded=false,curveWindowFallback=false;
    const char *pathIssue="NO_POINTS",*speedReason="STOP",*curveCause="NATIVE_MASTER_PILOT";
    uint16_t servo=PWMSERVOMID;
    explicit RaceTaskPilot(const TrackviewConfig &cfg={}):config(cfg){config.validate();native.configure(config.curveSpeed,config.degradedSpeed,config.runP1,config.runP2,config.turnD,config.maxFrameGap);}
    void update(const std::vector<PointX>&left,const std::vector<PointX>&right,float dt,float previous=0){
        NativeTrackRows rows;for(auto p:left)rows[p.x].first=p.y;for(auto p:right)rows[p.x].second=p.y;
        native.update(rows,camera(),dt,previous);
        center=native.center;error=native.error;confidence=native.confidence;severity=rawSeverity=native.severity;servo=native.servo;
        valid=native.valid;ready=valid;lost=native.lost;degraded=native.is_singal_smooth;pathIssue=native.pathIssue;
        dualPoints=degraded?0:native.centerEdge.size();singlePoints=degraded?native.centerEdge.size():0;rejectedPoints=int(rows.size())-int(native.centerEdge.size());
        centerEdge.clear();for(auto p:native.centerEdge)centerEdge.emplace_back(p.y,p.x);
        heading=curveHeading=centerEdge.size()>1?float(centerEdge.back().y-centerEdge.front().y):0;
    }
    float speedFor(float cruise,float dt){const float limit=std::min(cruise,config.maxSpeed);speedTarget=limit;speedReason="MASTER_PILOT";return commandedSpeed=native.speedFor(limit,dt);}
    float speedForDirection(float cruise,float dt,float previous){if(!std::isfinite(dt)||dt<=0||dt>config.maxFrameGap)return 0;speedReason="CONFIRMED_SIGN_SEEK";speedTarget=std::clamp(cruise,0.f,config.maxSpeed);return commandedSpeed=std::min(speedTarget,std::max(0.f,previous)+config.acceleration*dt);}
    void resetControl(){native.resetControl();commandedSpeed=speedTarget=0;servo=PWMSERVOMID;}
    float curveSpeed()const{return config.curveSpeed;}float degradedSpeed()const{return config.degradedSpeed;}
    void setCurveSpeed(float v){native.setCurveSpeed(v);config.curveSpeed=native.curveSpeed();}
    void setDegradedSpeed(float v){native.setDegradedSpeed(v);config.degradedSpeed=native.degradedSpeed();}
    void cycleWeight(){native.cycleWeight();}const char *weightName()const{return native.weightName();}
    float singleFraction()const{return float(singlePoints)/std::max(1,dualPoints+singlePoints);}
    float qualitySpeedLimit()const{return !ready||lost?0:degraded?config.degradedSpeed:config.maxSpeed;}
    const char *qualityLimitReason()const{return !ready?"PATH_WAIT":degraded?"SINGLE_EDGE":"LIMIT";}
    int learnedWidthRows()const{return 0;}int seedCalibratedWidths(const std::vector<std::pair<int,float>>&){return 0;}
    std::vector<WidthReading> widthProfile()const{return {};}
    nlohmann::json runtimeConfig()const{return config.toJson();}
};
template<class Plan>inline bool raceDirectionSeeking(const Plan &p){return !p.stop&&p.fresh&&!p.selectedBranch&&p.guideDirection!=0&&p.guidePwm>0;}
template<class Plan>inline float raceDirectionSpeed(const Plan &p,RaceTaskPilot &pilot,float cruise,float dt,float previous){return raceDirectionSeeking(p)?pilot.speedForDirection(std::min(cruise,p.speedCap),dt,previous):0;}
template<class Plan>inline uint16_t raceDirectionServo(const Plan &p,uint16_t laneServo){
    if(p.stop||!p.fresh||!p.guideDirection)return laneServo;
    if(p.selectedBranch){
        const float blend=std::clamp(p.laneBlend,0.f,1.f);
        const float guided=PWMSERVOMID+p.guideDirection*p.guidePwm;
        return uint16_t(std::lround(std::clamp(guided*(1-blend)+laneServo*blend,float(PWMSERVOMIN),float(PWMSERVOMAX))));
    }
    const float directed=p.guideDirection*(float(laneServo)-PWMSERVOMID);
    return uint16_t(std::clamp(PWMSERVOMID+p.guideDirection*std::max(directed,p.guidePwm),float(PWMSERVOMIN),float(PWMSERVOMAX)));
}
