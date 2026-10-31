#pragma once
#include "race_obstacle_plan.hpp"
#include "race_detection_review.hpp"
#include "race_perception.hpp"
#include "control.hpp"

inline uint16_t raceBypassServo(uint16_t requested,uint16_t previous,float dt) {
    if(!std::isfinite(dt)||dt<=0||dt>.25f)return PWMSERVOMID;
    previous=std::clamp(previous,uint16_t(PWMSERVOMIN),uint16_t(PWMSERVOMAX));
    const float target=std::clamp(float(requested),float(PWMSERVOMIN),float(PWMSERVOMAX));
    return uint16_t(std::lround(std::clamp(target,float(previous)-600*dt,float(previous)+600*dt)));
}

struct RaceBypassSteering {
    bool valid=false;float targetX=0,targetZ=0,curvature=0;
    uint16_t requested=PWMSERVOMID,pwm=PWMSERVOMID;
};

// 用米制前向预瞄，避免远处的小像素偏差在普通巡线PD中被平均成很小的转向。
// 路径原点仍是车头地面投影；这是前向预瞄控制，不冒充已校准的后轴运动学。
// FullLockDegrees为角度/PWM映射的可调实验初值，尚非实测轮角标定。
inline RaceBypassSteering raceGroundSteering(const RaceObstaclePreviewPlan &p,const RaceNavigationConfig &cfg,
                                            float speed,uint16_t previous,float dt) {
    RaceBypassSteering out;
    if(!p.metric||!p.fresh||!p.motionValid||!p.candidate||p.stop||p.groundCenters.size()<2||
       !std::isfinite(speed)||!std::isfinite(dt)||dt<=0||dt>.25f)return out;
    for(size_t i=0;i<p.groundCenters.size();++i) {
        const auto &a=p.groundCenters[i];
        if(!std::isfinite(a[0])||!std::isfinite(a[1])||a[1]<0||
           (i&&a[1]<=p.groundCenters[i-1][1]))return out;
    }
    const float z=std::min(cfg.bypassLookahead+std::min(.1f,.15f*std::abs(speed)),p.groundCenters.back()[1]);
    if(z<.10f)return out;
    for(size_t i=1;i<p.groundCenters.size();++i) {
        const auto &a=p.groundCenters[i-1],&b=p.groundCenters[i];
        if(a[1]>z||b[1]<z)continue;
        const float t=(z-a[1])/(b[1]-a[1]);out.targetX=a[0]+t*(b[0]-a[0]);out.targetZ=z;
        out.curvature=2*out.targetX/(out.targetX*out.targetX+z*z);
        const float radians=std::atan(cfg.bypassWheelbase*out.curvature),fullLock=cfg.bypassFullLockDegrees*float(CV_PI)/180.f;
        const float span=radians>=0?float(PWMSERVOMAX-PWMSERVOMID):float(PWMSERVOMID-PWMSERVOMIN);
        out.requested=uint16_t(std::lround(std::clamp(float(PWMSERVOMID)+span*radians/fullLock,float(PWMSERVOMIN),float(PWMSERVOMAX))));
        out.pwm=raceBypassServo(out.requested,previous,dt);out.valid=true;return out;
    }
    return out;
}

struct RaceBypassDecision {
    bool involved=false,follow=false,stop=false,authorizeCone=false;
    const char *state="BYPASS_IDLE",*reason="LANE_CONTROL";
};

inline bool raceBypassArmReady(const RaceObstaclePreviewPlan &p,double now,const RaceConfig &cfg) {
    return p.metric&&p.fresh&&p.sequence&&now>=p.captured&&now-p.captured<=std::min(.18f,cfg.maxDetectionAge)&&
        p.hasTarget&&!p.passed&&p.confirmedTargets>0&&p.motionValid&&p.candidate&&!p.stop&&
        !p.centers.empty()&&p.groundCenters.size()>=2;
}

// 绕行授权与其他比赛任务分开仲裁。只有米制、鲜活、连续确认的路线能接管转向。
// 短暂失效零速重算；坐标系重建后必须重新看到锥桶并确认路线，不能直接退回巡线。
class RaceBypassControl {
    bool active=false,fault=false,paused=false,recovering=false;
    int epoch=-1,stable=0,aligned=0,stableEpoch=-1;
    uint64_t last=0;
    double began=0,lastCaptured=-1;
    // 恢复计时按"失效原因"分开：计划/帧过期、运动拟合无效、保留参考图验证。
    // 窗口在本轮开始或原因切换时确定，不因后续导航变慢（fresh 变化）而缩短；
    // 同一轮总时长另有上限，避免原因来回切换把停车无限延长。
    double invalidSince=-1,episodeSince=-1,pauseBegan=0,resetSince=0;
    int invalidCause=0,resets=0;
    double causeWindow=0;
    static constexpr int kCausePlan=1,kCauseMotion=2,kCauseHeld=3;
public:
    bool engaged() const {return active||recovering;}
    RaceBypassDecision update(const RaceObstaclePreviewPlan &p,const RaceSnapshot &s,const RaceConfig &cfg,
                              double now,bool automatic) {
        RaceBypassDecision out;
        if(!automatic){recovering=recovering||active;active=paused=false;stable=aligned=0;last=0;invalidSince=-1;return out;}
        if(fault){out.involved=out.stop=true;out.state="BYPASS_FAULT";out.reason="RESTART_AFTER_INSPECTION";return out;}
        bool cone=false,human=false,unmapped=false;
        for(const auto &d:s.detections)if(reviewRaceDetection(d,s.image,cfg,&s.crosswalk).accepted) {
            if(d.type==LABEL_PERSON)human=true;
            if(d.type!=LABEL_CONE)continue;
            cone=true;const cv::Rect box(d.x,d.y,d.width,d.height);
            const bool match=std::any_of(p.obstacles.begin(),p.obstacles.end(),[&](const cv::Rect &r){
                const float intersection=(r&box).area();return intersection/std::max(1,r.area()+box.area()-int(intersection))>.15f;
            });
            if(!match)unmapped=true;
        }
        const bool interested=human||active||recovering||(cone&&(p.state!="BYPASS_NO_TARGET"||unmapped))||(p.hasTarget&&!p.passed);
        if(!interested)return out;
        out.involved=true;out.stop=true;out.state="BYPASS_WAIT";out.reason="MATCHED_ROUTE_REQUIRED";
        const bool fresh=!s.failed&&s.sequence&&now>=s.captured&&now-s.captured<=cfg.maxDetectionAge&&
            p.sequence&&now>=p.captured&&now-p.captured<=std::min(.18f,cfg.maxDetectionAge)&&p.fresh;
        const bool valid=fresh&&p.metric&&p.motionValid&&p.candidate&&!p.stop&&!p.centers.empty()&&!unmapped;
        if(human){out.state="BYPASS_PERSON_STOP";out.reason="PERSON_PRIORITY";stable=0;return out;}
        // 视觉里程重初始化（单帧光流抖动或跳帧）：立即停车并丢弃旧米制地图，回到"重新确认路线"，
        // 不锁死；只有 15s 内反复重初始化（>4 次）才锁停，交由人检查。
        if((active||recovering)&&fresh&&p.metric&&p.motionEpoch!=epoch) {
            active=paused=false;recovering=true;stable=aligned=0;last=0;lastCaptured=-1;stableEpoch=-1;epoch=p.motionEpoch;invalidSince=-1;
            if(now-resetSince>15.0){resets=0;resetSince=now;}
            if(++resets>4){fault=true;out.state="BYPASS_FAULT";out.reason="REPEATED_ODOMETRY_RESET";return out;}
            out.state="BYPASS_REPLAN_WAIT";out.reason="ODOMETRY_RESET_REACQUIRE";return out;
        }
        if(recovering&&(!cone||!p.hasTarget||p.passed)) {
            stable=0;out.state="BYPASS_REPLAN_WAIT";out.reason="REACQUIRE_VISIBLE_CONE";return out;
        }
        auto pause=[&](){if(!paused)pauseBegan=now;paused=true;};
        const bool timeout=active&&!paused&&now-began>25;
        if(!valid||timeout) {
            stable=0;
            // 几何、短期目标丢失与里程失效分别处理。有效坐标系里的无路结果
            // 持续零速重算，不让交错的一帧漏检继承旧的0.4秒失效计时。
            if(active&&fresh&&p.metric&&p.motionValid&&p.motionEpoch==epoch&&!unmapped&&
               !timeout&&(p.state=="BYPASS_NO_CLEAR_PATH"||p.state=="BYPASS_TARGET_LOST_BEFORE_CLEARANCE"||
                   p.state=="BYPASS_STRAIGHT_LANE_REQUIRED"||p.state=="BYPASS_CONFIRMING"||
                   p.state=="BYPASS_CONE_OUTSIDE_CALIBRATED_RANGE")) {
                pause();invalidSince=-1;out.state="BYPASS_REPLAN_WAIT";
                out.reason=p.state=="BYPASS_TARGET_LOST_BEFORE_CLEARANCE"?"STOP_AND_REACQUIRE_TARGET":"STOP_AND_REPLAN";
                return out;
            }
            // 瞬时失效（AI 帧略慢/单帧里程无效/导航成果晚于 0.18s）：此刻已是零速。
            // 旧实现把恢复窗口绑在 fresh 上：导航一慢过 0.18s，同一轮恢复从 0.8s
            // 被切回 0.4s，于是"还没到 0.8s"就被判锁停。这里按原因分开计时，
            // 窗口只取决于失效类型，不取决于后续帧是否还新鲜。
            if(active) {
                pause();
                if(timeout){fault=true;out.state="BYPASS_FAULT";out.reason="BYPASS_EXECUTION_TIMEOUT";return out;}
                // 原因判定不看 fresh：计划晚到不代表换了失效类型。
                const bool motionBad=p.metric&&!p.motionValid&&p.motionEpoch==epoch&&!unmapped;
                const int cause=motionBad?(p.motionCoordinatesHeld?kCauseHeld:kCauseMotion):kCausePlan;
                if(episodeSince<0){episodeSince=now;invalidCause=0;}
                if(cause!=invalidCause) {
                    invalidCause=cause;
                    causeWindow=cause==kCauseHeld?.8:(cause==kCauseMotion?.5:.6);
                    invalidSince=now;          // 只重置本原因的计时，总轮时长由 episodeSince 限制
                }
                // 同一轮无论原因怎么切换，最多停车 1.2s，之后必须由人检查。
                if(now-episodeSince>1.2){fault=true;out.state="BYPASS_FAULT";out.reason="RECOVERY_TIMEOUT";return out;}
                if(now-invalidSince<causeWindow) {
                    out.stop=true;out.follow=false;out.state="BYPASS_REPLAN_WAIT";
                    out.reason=cause==kCauseHeld?"VERIFY_LAST_GROUND_REFERENCE":
                        (cause==kCauseMotion?"TRANSIENT_STOP":"PLAN_STALE_WAIT");
                    return out;
                }
                fault=true;out.state="BYPASS_FAULT";
                out.reason=cause==kCauseHeld?"HELD_REFERENCE_TIMEOUT":
                    (cause==kCauseMotion?"ROUTE_OR_VISUAL_MOTION_LOST":"PLAN_STALE_TIMEOUT");
            }
            return out;
        }
        invalidSince=-1;episodeSince=-1;invalidCause=0;
        if(active&&paused) {
            if(p.sequence!=last){
                if(p.motionEpoch!=stableEpoch||p.sequence<last||p.captured-lastCaptured>.18)stable=0;
                stableEpoch=p.motionEpoch;lastCaptured=p.captured;last=p.sequence;++stable;
            }
            if(stable<3){out.state="BYPASS_REPLAN_WAIT";out.reason="CONFIRM_REPLANNED_ROUTE";return out;}
            began+=now-pauseBegan;paused=false;stable=0;
        }
        if(!active) {
            if(p.passed)return RaceBypassDecision{};
            if(p.sequence!=last) {
                if(p.motionEpoch!=stableEpoch||p.sequence<last||p.captured-lastCaptured>.18)stable=0;
                stableEpoch=p.motionEpoch;lastCaptured=p.captured;last=p.sequence;++stable;
            }
            if(stable<3)return out;
            active=true;recovering=false;epoch=p.motionEpoch;began=now;aligned=0;
        }
        out.follow=out.authorizeCone=true;out.stop=false;
        out.state=p.passed?"BYPASS_RETURN":p.direction>0?"BYPASS_RIGHT":"BYPASS_LEFT";
        out.reason="GROUND_ROUTE";
        if(p.sequence!=last) {
            last=p.sequence;
            aligned=p.passed&&std::abs(p.laneOffset)<.02f&&std::abs(p.laneHeading)<.08f?aligned+1:0;
            if(aligned>=3){active=false;stable=aligned=0;out.follow=false;out.state="BYPASS_COMPLETE";out.reason="REAR_CLEAR_AND_CENTERED";}
        }
        return out;
    }
};
