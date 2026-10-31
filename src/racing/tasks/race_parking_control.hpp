#pragma once
#include "race_parking_geometry.hpp"
#include <string>
#include <cstdint>

// Front-bumper pose in the taught bay frame. Runtime command fallback explicitly
// marks estimated observations; the independent visual tracker is never altered.
struct RaceParkingObservation {
    RaceParkingPose pose;
    double captured=0;
    uint64_t sequence=0;
    int epoch=0;
    bool valid=false;
    bool estimated=false;
};
// Parking-only fallback. Integrate APPLIED commands while current camera frames
// remain fresh. This is an estimate, never measured wheel odometry. Keep the
// ground tracker unchanged so it can independently recover its original frame.
class RaceParkingPoseFallback {
    RaceParkingObservation last;
public:
    void reset(){last={};}
    RaceParkingObservation update(RaceParkingObservation raw,double now,float appliedSpeed,float appliedCurvature,
                                  const RaceParkingDimensions &d) {
        const double gap=raw.captured-last.captured;
        if(!last.valid){if(raw.valid)last=raw;return raw;}
        if(raw.epoch!=last.epoch||raw.sequence<=last.sequence||gap<=0||gap>.25||
           now<raw.captured||now-raw.captured>.18||!std::isfinite(appliedSpeed)||!std::isfinite(appliedCurvature))return raw;
        if(!raw.valid||last.estimated) {
            auto predicted=raceParkingAdvance(last.pose,appliedSpeed*float(gap),appliedCurvature,d);
            if(raw.valid) {
                const float distance=std::hypot(raw.pose.x-predicted.x,raw.pose.z-predicted.z);
                const float angle=raceParkingAngle(raw.pose.yaw-predicted.yaw);
                if(distance<.01f&&std::abs(angle)<.04f) {last=raw;return raw;}
                const float blend=distance>0?std::min(1.f,.008f/distance):1.f;
                predicted.x+=(raw.pose.x-predicted.x)*blend;predicted.z+=(raw.pose.z-predicted.z)*blend;
                predicted.yaw+=std::clamp(angle,-.03f,.03f);
            }
            raw.pose=predicted;raw.valid=true;raw.estimated=true;
        }
        last=raw;return raw;
    }
};
struct RaceParkingCommand {
    float speed=0,curvature=0;
    const char *state="PARK_READY",*reason="TEACH_BAY_AND_PRESS_A";
    size_t segment=0;
    bool predictedBoundaryWarning=false;
    bool entryHeadingRecovery=false;
    float headingError=0,baseCurvature=0,requestedCurvature=0;
    bool entryMouthRecovery=false;
    bool exitHeadingRecovery=false;
};
class RaceParkingControl {
    enum class Phase {Ready,Entry,Dwell,Exit,Done,Fault};
    Phase phase=Phase::Ready;
    RaceParkingSpace space;
    RaceParkingRoute route;
    std::vector<RaceParkingPose> measuredEntry;
    RaceParkingDwell dwell;
    size_t segment=0;
    int epoch=0;
    uint64_t lastSequence=0;
    double lastCapture=-1,lastTime=-1,badSince=-1,phaseSince=0,progressSince=0;
    RaceParkingPose previous{},progressPose{};
    float lastSpeed=0,steeringAngle=0;
    float taskSpeed=.10f;
    bool reverse=false;
    bool estimateUsed=false;
    bool terminalRecovery=false;
    bool mouthRecoveryActive=false;
    const char *faultReason="PARK_FAULT";
    static cv::Point2f rear(RaceParkingPose p,const RaceParkingDimensions &d) {
        return {p.x+d.frontToRearAxle*std::sin(p.yaw),p.z-d.frontToRearAxle*std::cos(p.yaw)};
    }
    RaceParkingCommand fault(const char *reason) {
        phase=Phase::Fault;faultReason=reason;lastSpeed=0;
        return {0,0,"PARK_FAULT",faultReason,segment};
    }
public:
    explicit RaceParkingControl(RaceParkingSpace s=RaceParkingSpace{},float targetSpeed=.10f):space(s),taskSpeed(targetSpeed){
        if(!std::isfinite(taskSpeed)||taskSpeed<=0||taskSpeed>10)throw std::invalid_argument("Invalid parking target speed");
    }
    bool active()const{return phase==Phase::Entry||phase==Phase::Dwell||phase==Phase::Exit;}
    bool done()const{return phase==Phase::Done;}
    bool failed()const{return phase==Phase::Fault;}
    bool usedPoseEstimate()const{return estimateUsed;}
    bool terminalRecoveryUsed()const{return terminalRecovery;}
    const std::vector<RaceParkingPose> &reverseReference()const{return route.exit;}
    RaceParkingCommand status()const {
        if(failed())return {0,0,"PARK_FAULT",faultReason,segment};
        if(done())return {0,0,"PARK_COMPLETE","RETURNED_TO_AISLE",segment};
        return {};
    }
    void abort(){if(active())fault("OPERATOR_STOP_RETEACH_REQUIRED");}
    bool arm(const RaceParkingRoute &r,const RaceParkingObservation &o,double now,bool reverseVerified) {
        if(phase!=Phase::Ready||!reverseVerified||!r.valid||r.entry.size()<2||r.exit.size()!=r.entry.size()||
           !o.valid||!o.sequence||!std::isfinite(now)||!std::isfinite(o.captured)||now<o.captured||now-o.captured>.18||
           !space.freeObserved(o.pose)||!space.parked(r.entry.back()))return false;
        const auto &start=r.entry.front();
        if(std::hypot(o.pose.x-start.x,o.pose.z-start.z)>.015||std::abs(raceParkingAngle(o.pose.yaw-start.yaw))>.04)return false;
        for(size_t i=0;i<r.entry.size();++i) {
            const auto &a=r.entry[i],&b=r.exit[r.entry.size()-1-i];
            if(!space.free(a)||std::hypot(a.x-b.x,a.z-b.z)>1e-5||std::abs(a.yaw-b.yaw)>1e-5)return false;
        }
        route=r;phase=Phase::Entry;epoch=o.epoch;reverse=true;segment=0;
        measuredEntry={o.pose};
        lastSequence=o.sequence;lastCapture=o.captured;lastTime=now;
        phaseSince=progressSince=now;previous=progressPose=o.pose;return true;
    }
    RaceParkingCommand update(const RaceParkingObservation &o,double now,float dt,bool taskClear,float speedLimit=.10f) {
        if(phase==Phase::Fault)return {0,0,"PARK_FAULT",faultReason,segment};
        if(phase==Phase::Done)return {0,0,"PARK_COMPLETE","RETURNED_TO_AISLE",segment};
        if(!active())return {};
        if(!std::isfinite(now)||!std::isfinite(o.captured)||now<lastTime||!std::isfinite(dt)||dt<=0||dt>.25f||
           !std::isfinite(speedLimit)||speedLimit<0)return fault("PARK_TIME_OR_SPEED_INVALID");
        lastTime=now;
        if(o.epoch!=epoch)return fault("PARK_ODOMETRY_EPOCH_CHANGED");
        if(o.sequence<lastSequence||(o.sequence>lastSequence&&o.captured<=lastCapture))return fault("PARK_OBSERVATION_REORDERED");
        if(!taskClear) {
            lastSpeed=0;progressSince=now;dwell.ready(now,false,false,false,reverse);
            if(o.valid&&o.sequence>lastSequence&&now>=o.captured&&now-o.captured<=.18) {
                previous=o.pose;lastCapture=o.captured;lastSequence=o.sequence;badSince=-1;
            }
            return {0,0,"PARK_HOLD","TRAFFIC_STOP_PRIORITY",segment};
        }
        if(!o.valid||!o.sequence||now<o.captured||now-o.captured>.18) {
            lastSpeed=0;dwell.ready(now,false,false,false,reverse);
            if(badSince<0)badSince=now;
            if(now-badSince>.65)return fault("PARK_MOTION_OR_TASK_INTERRUPTED_RETEACH");
            return {0,0,"PARK_HOLD",taskClear?"FRESH_CONTINUOUS_MOTION_REQUIRED":"TRAFFIC_STOP_PRIORITY",segment};
        }
        const bool newFrame=o.sequence>lastSequence;
        if(!newFrame)return {0,0,"PARK_HOLD","WAIT_NEW_MOTION_FRAME",segment};
        const float gap=float(o.captured-lastCapture),move=std::hypot(o.pose.x-previous.x,o.pose.z-previous.z);
        const float turn=std::abs(raceParkingAngle(o.pose.yaw-previous.yaw));
        // A recovered image is measured against the last valid image, not the
        // last loop iteration. Accept its complete visually measured bridge;
        // otherwise every recovery after 250ms would fault even with a good fit.
        const float recoveryExtra=badSince>=0?std::clamp(gap-.08f,0.f,.57f):0.f;
        const auto &dimensions=space.dimensions();
        const float appliedCurvature=std::tan(steeringAngle)/dimensions.wheelbase;
        const float moveLimit=std::max(.025f,.008f+std::abs(lastSpeed)*gap*(1+dimensions.frontToRearAxle*std::abs(appliedCurvature))*1.6f);
        const float turnLimit=std::max(.11f,.04f+std::abs(lastSpeed*appliedCurvature)*gap*1.6f);
        if(!std::isfinite(move)||!std::isfinite(turn)||gap<=0||gap>(badSince>=0?.75f:.25f)||
           move>moveLimit+.4f*recoveryExtra||turn>turnLimit+.8f*recoveryExtra)
            return fault("PARK_POSE_DISCONTINUITY");
        previous=o.pose;lastCapture=o.captured;lastSequence=o.sequence;badSince=-1;
        estimateUsed|=o.estimated;
        const bool observedFree=space.freeObserved(o.pose);
        const bool paintPassage=(phase==Phase::Entry||phase==Phase::Exit)&&space.freeEntryMouth(o.pose);
        const bool parkedObservation=(phase==Phase::Dwell||phase==Phase::Exit)&&space.parkedObserved(o.pose);
        const bool outsidePaint=!observedFree&&!paintPassage&&!parkedObservation;
        if(phase!=Phase::Entry||space.free(o.pose))mouthRecoveryActive=false;
        else if(!observedFree)mouthRecoveryActive=true;
        const bool mouthRecovery=mouthRecoveryActive;
        if(phase==Phase::Entry&&(std::hypot(o.pose.x-measuredEntry.back().x,o.pose.z-measuredEntry.back().z)>.004f||
           std::abs(raceParkingAngle(o.pose.yaw-measuredEntry.back().yaw))>.02f))measuredEntry.push_back(o.pose);
        if(phase==Phase::Dwell) {
            // Arrival is latched: loss of painted containment must never restart
            // forward entry. Stop for three stationary seconds, then reverse the
            // actual entry path. Terminal recovery is not a valid-bay success.
            const bool stationary=move/gap<.012f&&turn/gap<.06f;
            terminalRecovery|=!space.parkedObserved(o.pose);
            if(dwell.ready(o.captured,true,stationary,true,reverse,space.dimensions().dwellSeconds)) {
                measuredEntry.push_back(o.pose);route.exit=raceParkingReverseSamples(measuredEntry,space.dimensions());
                phase=Phase::Exit;phaseSince=progressSince=now;progressPose=o.pose;segment=0;
                lastSpeed=0;return {0,0,"PARK_EXIT","DWELL_COMPLETE_REVERSE_NEXT_FRAME",0};
            }
            lastSpeed=0;return {0,0,"PARK_DWELL",stationary?
                (terminalRecovery?"TERMINAL_RECOVERY_THREE_SECONDS":"THREE_SECONDS_STATIONARY_INSIDE"):"WAIT_PHYSICAL_STOP",segment};
        }
        const bool leaving=phase==Phase::Exit;const auto &path=leaving?route.exit:route.entry;
        const auto &goal=path.back();const auto &d=space.dimensions();
        const float goalError=std::hypot(o.pose.x-goal.x,o.pose.z-goal.z);
        if(leaving&&std::abs(raceParkingAngle(o.pose.yaw-goal.yaw))<.15f) {
            bool inAisle=true;
            for(auto q:space.body(o.pose))if(std::abs(q.x)>d.aisle/2)inAisle=false;
            // Entry may include a long straight approach. Once the full body
            // is back in the aisle and aligned, do not replay that straight all
            // the way to the original start (75cm behind the mouth in the run).
            if(inAisle&&std::hypot(o.pose.x-path.front().x,o.pose.z-path.front().z)>.10f) {
                phase=Phase::Done;lastSpeed=0;
                return {0,0,"PARK_COMPLETE","BODY_CLEAR_ALIGNED_IN_AISLE",segment};
            }
        }
        // A fully contained vehicle near the end is already parked; continuing
        // to chase one exact pose can push an otherwise valid body toward a line.
        // Exit finishes once the complete body is in the aisle near its starting
        // position; do not keep reversing toward the gate to match one exact point.
        if((goalError<(leaving?.045f:.05f)||(!leaving&&space.parked(o.pose)))&&
           std::abs(raceParkingAngle(o.pose.yaw-goal.yaw))<(leaving?.12f:.20f)) {
            if(!leaving&&space.parked(o.pose)) {
                phase=Phase::Dwell;phaseSince=now;lastSpeed=0;steeringAngle=0;
                return {0,0,"PARK_DWELL","FULL_BODY_INSIDE_STOPPING",segment};
            }
            if(leaving) {
                // Full body back in the aisle, parallel to the taught direction.
                bool inside=true;for(auto q:space.body(o.pose))if(std::abs(q.x)>d.aisle/2)inside=false;
                if(inside){phase=Phase::Done;lastSpeed=0;return {0,0,"PARK_COMPLETE","RETURNED_TO_AISLE",segment};}
            }
        }
        const auto axle=rear(o.pose,d);float best=1e9f,projection=0;size_t selected=segment;
        for(size_t i=segment;i+1<path.size();++i) {
            const auto a=rear(path[i],d),b=rear(path[i+1],d),delta=b-a;
            const float length2=delta.dot(delta);if(length2<1e-10f)continue;
            const float t=std::clamp((axle-a).dot(delta)/length2,0.f,1.f);
            const float distance=cv::norm(axle-(a+delta*t));
            if(distance<best){best=distance;selected=i;projection=t;}
        }
        const bool routeRecovery=best>.04f;
        segment=selected;
        if(!leaving&&segment+5>=path.size()&&o.pose.yaw>route.entry.front().yaw+.35f) {
            const float dx=o.pose.x-goal.x,dz=o.pose.z-goal.z;
            const float alongGoal=-std::sin(goal.yaw)*dx+std::cos(goal.yaw)*dz;
            const float acrossGoal=std::cos(goal.yaw)*dx+std::sin(goal.yaw)*dz;
            const bool depthReached=o.pose.x<=goal.x&&std::abs(dz)<.18f;
            const bool endPassed=alongGoal>=0&&std::abs(acrossGoal)<.20f;
            // Do not pursue a point behind the car after reaching the planned
            // bay depth/terminal plane. This complements strict parked success;
            // lateral error must not turn the last segment into endless circles.
            if(depthReached||endPassed) {
                terminalRecovery=!space.parked(o.pose);
                phase=Phase::Dwell;phaseSince=now;lastSpeed=0;steeringAngle=0;
                return {0,0,"PARK_DWELL",terminalRecovery?"ENTRY_TERMINAL_REACHED_STOPPING":"FULL_BODY_INSIDE_STOPPING",segment};
            }
        }
        const auto refYaw=path[segment].yaw+projection*raceParkingAngle(path[segment+1].yaw-path[segment].yaw);
        const bool headingRecovery=std::abs(raceParkingAngle(o.pose.yaw-refYaw))>.35f;
        auto target=rear(path[segment],d)+(rear(path[segment+1],d)-rear(path[segment],d))*projection;
        // 5.5cm forward lookahead amplified millimetres of visual cross-track
        // noise into alternating full-lock commands on the real straight approach.
        float remaining=.10f;
        for(size_t i=segment;i+1<path.size();++i) {
            const auto end=rear(path[i+1],d),delta=end-target;const float length=cv::norm(delta);
            if(length>=remaining){target+=delta*(remaining/length);break;}
            remaining-=length;target=end;
        }
        const auto delta=target-axle;const float lateral=std::cos(o.pose.yaw)*delta.x+std::sin(o.pose.yaw)*delta.y;
        const float limit=std::tan(d.fullLockDegrees*float(CV_PI)/180)/d.wheelbase;
        const float pursuit=-2*lateral/std::max(.0004f,delta.dot(delta));
        const auto segmentRearA=rear(path[segment],d),segmentRearB=rear(path[segment+1],d);
        const float signedLength=(leaving?-1:1)*std::max(.001f,float(cv::norm(segmentRearB-segmentRearA)));
        float feedforward=raceParkingAngle(path[segment+1].yaw-path[segment].yaw)/signedLength;
        const float curvaturePreview=leaving?.10f:route.entryCurvaturePreview;
        if(curvaturePreview>0) {
            // Average the next 10cm of reference curvature, so wheel motion
            // starts before the bend reaches the rear axle. Local-segment
            // feedforward waits too late for real steering response.
            float distance=0,angle=0;
            for(size_t i=segment;i+1<path.size()&&distance<curvaturePreview;++i) {
                const float fraction=i==segment?1-projection:1;
                const float length=float(cv::norm(rear(path[i+1],d)-rear(path[i],d)))*fraction;
                if(length<1e-6f)continue;
                const float used=std::min(length,curvaturePreview-distance);
                angle+=raceParkingAngle(path[i+1].yaw-path[i].yaw)*fraction*(used/length);distance+=used;
            }
            if(distance>.001f)feedforward=angle/((leaving?-1.f:1.f)*distance);
        }
        const float headingError=raceParkingAngle(refYaw-o.pose.yaw);
        const float baseCurvature=.6f*pursuit+.4f*feedforward+(leaving?-2.f:2.f)*headingError;
        float requested=baseCurvature;bool entryHeadingRecovery=false;
        // Pursuit already mixes lateral displacement and heading. At the
        // measured entry, a few cm of lateral error cancelled both the bend
        // feedforward and lagging heading, commanding right while turning left.
        // Once the reference has established its left turn, recover missing
        // heading with a smooth lower bound. Release it on heading overshoot,
        // a right-turn reference, near the final heading, and during reverse.
        // The two Oct-30 runs reached full lock only after the mouth. Use the
        // already-previewed bend to lead wheel response by 6cm, rather than
        // waiting for the nearest reference heading to be 0.06rad into it.
        // The lead disappears on straight sections, near the goal, on heading
        // overshoot, and in reverse; do not globally amplify lane/exit steering.
        const float entryLead=route.entryCurvaturePreview>0?.06f*std::max(0.f,feedforward):0.f;
        const float entryError=headingError+entryLead;
        if(!leaving&&(refYaw>route.entry.front().yaw+.06f||entryLead>.02f)&&
           goal.yaw>refYaw+.08f&&feedforward>=-.05f&&entryError>.02f) {
            const float blend=std::clamp((entryError-.02f)/.04f,0.f,1.f);
            const float recovery=1.1f*std::max(0.f,feedforward)+5.f*std::max(0.f,headingError);
            requested+=blend*std::max(0.f,recovery-requested);
            entryHeadingRecovery=requested>baseCurvature+1e-5f;
        }
        bool exitHeadingRecovery=false;
        if(leaving&&std::abs(headingError)>.04f) {
            // Reverse yaw response has the opposite sign to forward motion.
            // Prevent lateral pursuit from sustaining full counter-steer after
            // heading has already crossed to the other side of the reference.
            const float blend=std::clamp((std::abs(headingError)-.04f)/.08f,0.f,1.f);
            const float recovery=feedforward-4.f*headingError;
            const float bounded=headingError<0?std::max(requested,recovery):std::min(requested,recovery);
            exitHeadingRecovery=std::abs(bounded-requested)>1e-5f;
            requested+=blend*(bounded-requested);
        }
        requested=std::clamp(requested,-limit,limit);
        if(mouthRecovery&&goal.yaw>o.pose.yaw+.03f) {
            // Only continue across mouth paint when a forward left arc can
            // return to the original observation envelope. Validate the whole
            // sweep with finite steering slew before asking for stronger steer.
            bool recoverable=false;
            for(float factor:{1.f,.9f,.8f}) {
                auto predicted=o.pose;float angle=steeringAngle;
                const float desired=std::atan(d.wheelbase*limit*factor);
                const float rate=d.fullLockDegrees*float(CV_PI)/180*600/400;
                for(int step=0;step<50;++step) {
                    const float seconds=.005f/std::max(.01f,std::min(speedLimit,.05f));
                    angle+=std::clamp(desired-angle,-rate*seconds,rate*seconds);
                    predicted=raceParkingAdvance(predicted,.005f,std::tan(angle)/d.wheelbase,d);
                    if(!space.freeEntryMouth(predicted)||predicted.yaw>goal.yaw+.12f)break;
                    if(space.free(predicted)) {requested=limit*factor;recoverable=true;break;}
                }
                if(recoverable)break;
            }
            // A painted-envelope rejection is advisory during execution. If
            // no contained arc exists, keep route/heading correction moving.
        }
        const float lock=d.fullLockDegrees*float(CV_PI)/180;
        // Generate a continuous wheel-angle target within the existing command
        // slew, instead of repeatedly alternating full speed and a steering wait.
        const float angleRate=lock*600/400;
        const float targetAngle=std::atan(d.wheelbase*requested);
        steeringAngle+=std::clamp(targetAngle-steeringAngle,-angleRate*dt,angleRate*dt);
        const float curvature=std::tan(steeringAngle)/d.wheelbase;
        const float creep=std::min(.12f,taskSpeed*.5f);
        const float curveCap=taskSpeed<=.10001f?taskSpeed:std::max(creep,taskSpeed/(1+.5f*std::abs(curvature)));
        float magnitude=std::min({speedLimit,leaving?taskSpeed*.8f:taskSpeed,curveCap,std::max(creep,goalError*.8f)});
        if(mouthRecovery)magnitude=std::min(magnitude,creep);
        // Validate a short swept-body stopping envelope before issuing motion.
        auto sweepClear=[&](float speed) {
            auto prediction=o.pose;const float sweep=speed*.25f+.005f;float predictedAngle=steeringAngle;
            for(float s=0;s<sweep;s+=.0025f) {
                const float ds=std::min(.0025f,sweep-s),horizonStep=ds/std::max(.01f,speed);
                predictedAngle+=std::clamp(targetAngle-predictedAngle,-angleRate*horizonStep,angleRate*horizonStep);
                prediction=raceParkingAdvance(prediction,(leaving?-1:1)*ds,std::tan(predictedAngle)/d.wheelbase,d);
                if(!space.free(prediction))return false;
            }
            return true;
        };
        // Close to a divider, choose a shorter stopping envelope by reducing
        // speed within the existing parking range, retaining the same geometry.
        bool boundaryWarning=outsidePaint||routeRecovery||headingRecovery;
        while(!sweepClear(magnitude)) {
            // Operator requested a less restrictive field trial: a predicted
            // margin crossing is advisory. Keep the existing parking creep
            // command, while current observed body/people/device stops remain.
            if(magnitude<=creep+.00001f){boundaryWarning=true;break;}
            magnitude=std::max(creep,magnitude-.01f);
        }
        if(taskSpeed>.10001f)magnitude=std::min(magnitude,std::abs(lastSpeed)+.5f*dt);
        const float speed=leaving?-magnitude:magnitude;
        if(std::hypot(o.pose.x-progressPose.x,o.pose.z-progressPose.z)>.008f||
           std::abs(raceParkingAngle(o.pose.yaw-progressPose.yaw))>.04f) {progressPose=o.pose;progressSince=now;}
        if(now-progressSince>4)return fault("PARK_NO_MEASURED_PROGRESS");
        lastSpeed=speed;
        return {speed,curvature,leaving?"PARK_EXIT":"PARK_ENTRY",
            o.estimated?"COMMAND_POSE_ESTIMATE_CONTINUE":routeRecovery?"ROUTE_ERROR_CORRECTING_CONTINUE":
            headingRecovery?"HEADING_ERROR_CORRECTING_CONTINUE":outsidePaint?"PAINT_CROSSING_CORRECTING_CONTINUE":
            mouthRecovery?"ENTRY_MOUTH_TURN_RECOVERY":boundaryWarning?"PREDICTED_BOUNDARY_WARNING_CONTINUE":
            leaving?(exitHeadingRecovery?"EXIT_HEADING_RECOVERY":"VISUAL_REVERSE_ROUTE"):
            entryHeadingRecovery?"ENTRY_HEADING_RECOVERY":"VISUAL_FORWARD_ROUTE",segment,boundaryWarning,
            entryHeadingRecovery,headingError,baseCurvature,requested,mouthRecovery,exitHeadingRecovery};
    }
};

// Extra model feasibility filter, not calibration or evidence of physical success.
// Reject a geometrically valid route if the current controller cannot track it
// with command steering slew and the same footprint/rear-boundary checks.
inline bool raceParkingModelCompletes(const RaceParkingSpace &space,const RaceParkingRoute &route,float taskSpeed=.10f) {
    if(!route.valid||route.entry.empty())return false;
    RaceParkingControl control(space,taskSpeed);RaceParkingObservation observation{route.entry.front(),1,1,1,true};
    if(!control.arm(route,observation,1,true))return false;
    const auto &d=space.dimensions();float wheel=0;double time=1;
    for(int step=0;step<3500&&!control.done()&&!control.failed();++step) {
        time+=.04;observation.captured=time;++observation.sequence;
        const auto command=control.update(observation,time,.04f,true,taskSpeed);
        const float target=std::atan(d.wheelbase*command.curvature);
        const float rate=d.fullLockDegrees*float(CV_PI)/180*600/400;
        wheel+=std::clamp(target-wheel,-rate*.04f,rate*.04f);
        const float speed=std::abs(target-wheel)>d.fullLockDegrees*float(CV_PI)/180*35/400?0:command.speed;
        observation.pose=raceParkingAdvance(observation.pose,speed*.04f,std::tan(wheel)/d.wheelbase,d);
        if(!space.free(observation.pose))return false;
    }
    // Runtime terminal recovery must not make an invalid parking trajectory
    // pass the planner's nominal full-containment feasibility filter.
    return control.done()&&!control.terminalRecoveryUsed();
}
inline RaceParkingRoute racePlanTrackableParking(const RaceParkingSpace &space,RaceParkingPose start,float taskSpeed=.10f) {
    RaceParkingRoute result;std::vector<RaceParkingSearchTrial> searches;
    int expanded=0,candidates=0,rejected=0;
    // Wheel-angle state enlarged the lattice. Search toward a contained goal
    // more directly before falling back to the original cost ordering. This
    // changes search priority only; every footprint and model guard is retained.
    // Prefer early entry. For an already-close start, retain the original
    // feasible controller/search profile rather than forcing an early turn
    // that crosses the near divider and losing all previously valid routes.
    for(float preview:{.10f,0.f})for(float weight:{1.5f,1.f})for(float scale:{1.f,.9f,.8f,.7f}) {
        auto candidate=racePlanParking(space,start,scale,[&](const RaceParkingRoute &route){
            auto checked=route;checked.entryCurvaturePreview=preview;return raceParkingModelCompletes(space,checked,taskSpeed);
        },weight,preview>0);
        candidate.entryCurvaturePreview=preview;
        searches.push_back({scale,weight,candidate.expanded,candidate.geometricCandidates,candidate.modelRejected,candidate.reason,preview});
        expanded+=candidate.expanded;candidates+=candidate.geometricCandidates;rejected+=candidate.modelRejected;
        candidate.expanded=expanded;candidate.geometricCandidates=candidates;candidate.modelRejected=rejected;candidate.searches=searches;
        if(candidate.valid) {
            candidate.reason="MODEL_CHECKED_FORWARD_ROUTE";return candidate;
        }
        result=std::move(candidate);
    }
    result.valid=false;result.reason=candidates?"GEOMETRIC_ROUTES_REJECTED_BY_CONTROL_MODEL":
        std::any_of(searches.begin(),searches.end(),[](auto s){return std::string(s.reason)=="SEARCH_BUDGET_EXHAUSTED";})?
        "PARK_SEARCH_BUDGET_EXHAUSTED":"NO_FOOTPRINT_CLEAR_PATH";return result;
}
