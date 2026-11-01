#pragma once
#include "pilot.hpp"
#include "task_track.hpp"
#include "trackview_config.hpp"
#include <nlohmann/json.hpp>
struct RaceLaneAlignment {
    bool valid=false,aligned=false;float offset=0,heading=0;uint16_t servo=PWMSERVOMID;
};
// Image-based near-road direction, not a measured vehicle yaw. Unlike a single
// weighted centre, this distinguishes lateral offset from a tilted body.
inline RaceLaneAlignment raceLaneAlignment(const std::vector<PointX> &centers) {
    RaceLaneAlignment a;if(centers.size()<16)return a;
    float nearX=0,nearRow=0,aheadX=0,aheadRow=0;int count=0;
    for(size_t i=0;i<8;++i){nearX+=centers[i].y/8.f;nearRow+=centers[i].x/8.f;}
    for(auto p:centers)if(nearRow-p.x>=64&&nearRow-p.x<=128){aheadX+=p.y;aheadRow+=p.x;++count;}
    if(count<4)return a;
    aheadX/=count;aheadRow/=count;
    a.valid=true;a.offset=nearX-320;a.heading=(aheadX-nearX)*80/(nearRow-aheadRow);
    a.aligned=std::abs(a.offset)<40&&std::abs(a.heading)<12;
    a.servo=uint16_t(std::lround(std::clamp(PWMSERVOMID+1.8f*a.offset+2.2f*a.heading,
        float(PWMSERVOMIN),float(PWMSERVOMAX))));return a;
}
// Runtime settings for the mission/parking window. Keep user settings separate
// from temporary parking/traffic caps so per-frame updates cannot erase them.
struct RaceSpeedSettings {
    float cruise=.10f,curve,degraded,fork,minimum,maximum;
    RaceSpeedSettings(const TrackviewConfig &c,float turn):curve(c.curveSpeed),degraded(c.degradedSpeed),
        fork(turn),minimum(c.minSpeed),maximum(c.maxSpeed){}
    const char *adjust(int key) {
        float *value=nullptr;const char *name=nullptr;float step=.05f;
        switch(key) {
        case '-':case '_':step=-step;[[fallthrough]];
        case '+':case '=':value=&cruise;name="cruiseSpeed";break;
        case '[':step=-step;[[fallthrough]];
        case ']':value=&curve;name="curveSpeed";break;
        case ',':step=-step;[[fallthrough]];
        case '.':value=&degraded;name="degradedSpeed";break;
        case 'j':step=-step;[[fallthrough]];
        case 'k':value=&fork;name="forkSpeed";break;
        default:return nullptr;
        }
        *value=std::clamp(*value+step,minimum,maximum);return name;
    }
};
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
    RaceLaneAlignment alignment;
    explicit RaceTaskPilot(const TrackviewConfig &cfg={}):config(cfg){config.validate();native.configure(config.curveSpeed,config.degradedSpeed,config.runP1,config.runP2,config.turnD,config.maxFrameGap);}
    void update(const std::vector<PointX>&left,const std::vector<PointX>&right,float dt,float previous=0){
        NativeTrackRows rows;for(auto p:left)rows[p.x].first=p.y;for(auto p:right)rows[p.x].second=p.y;
        native.update(rows,camera(),dt,previous);
        if(native.valid&&!native.lost)native.synchronizeAppliedSpeed(previous);
        center=native.center;error=native.error;confidence=native.confidence;severity=rawSeverity=native.severity;servo=native.servo;
        valid=native.valid;ready=valid;lost=native.lost;degraded=native.is_singal_smooth;pathIssue=native.pathIssue;
        dualPoints=degraded?0:native.centerEdge.size();singlePoints=degraded?native.centerEdge.size():0;rejectedPoints=int(rows.size())-int(native.centerEdge.size());
        centerEdge.clear();for(auto p:native.centerEdge)centerEdge.emplace_back(p.y,p.x);
        heading=curveHeading=centerEdge.size()>1?float(centerEdge.back().y-centerEdge.front().y):0;
        alignment=raceLaneAlignment(centerEdge);
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
template<class Plan>inline uint16_t raceLaneRecoveryServo(const Plan &p,const RaceTaskPilot &pilot,uint16_t previous,float dt){
    if(p.alignRoad&&pilot.alignment.valid)return pilot.alignment.servo;
    if(pilot.valid)return pilot.servo;
    if(!p.alignRoad||!std::isfinite(dt)||dt<=0)return previous;
    // The sign pulse has ended: missing geometry must not retain its last
    // same-direction demand indefinitely. Unwind without fabricating a road.
    const float step=600*std::min(dt,.1f);
    return uint16_t(std::lround(std::clamp(float(PWMSERVOMID),previous-step,previous+step)));
}
template<class Plan>inline float raceDirectionSpeed(const Plan &p,RaceTaskPilot &pilot,float cruise,float dt,float previous){return raceDirectionSeeking(p)?pilot.speedForDirection(std::min(cruise,p.speedCap),dt,previous):0;}
template<class Plan>inline uint16_t raceDirectionServo(const Plan &p,uint16_t laneServo){
    if(p.stop||!p.fresh||!p.guideDirection)return laneServo;
    if(p.selectedBranch){
        // Once the selected road asks for countersteer, old sign guidance must
        // not cancel it while the timed blend is still ramping up.
        if(p.guideDirection*(float(laneServo)-PWMSERVOMID)<-20)return laneServo;
        const float blend=std::clamp(p.laneBlend,0.f,1.f);
        const float guided=PWMSERVOMID+p.guideDirection*p.guidePwm;
        return uint16_t(std::lround(std::clamp(guided*(1-blend)+laneServo*blend,float(PWMSERVOMIN),float(PWMSERVOMAX))));
    }
    const float directed=p.guideDirection*(float(laneServo)-PWMSERVOMID);
    return uint16_t(std::clamp(PWMSERVOMID+p.guideDirection*std::max(directed,p.guidePwm),float(PWMSERVOMIN),float(PWMSERVOMAX)));
}
