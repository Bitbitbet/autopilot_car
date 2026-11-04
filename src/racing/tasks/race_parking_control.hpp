#pragma once
#include "race_parking_geometry.hpp"
#include "race_parking_inset.hpp"
#include <string>
#include <string_view>
#include <cstdint>
#include <deque>

// Front-bumper pose in the taught bay frame. Runtime command fallback explicitly
// marks estimated observations, including visual increments after estimated reanchoring.
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
// propagated pose available for explicitly labelled relative visual recovery.
class RaceParkingPoseFallback {
    RaceParkingObservation last;
    double absoluteCaptured=-1;
    float blindTravel=0;
    bool exhausted=false;
public:
    static constexpr double maxEstimateSeconds=.65;
    static constexpr float maxEstimateTravel=.10f;
    void reset(){last={};absoluteCaptured=-1;blindTravel=0;exhausted=false;}
    bool budgetExhausted()const{return exhausted;}
    float estimateTravel()const{return blindTravel;}
    double estimateAge()const{return absoluteCaptured<0?0:last.captured-absoluteCaptured;}
    RaceParkingObservation update(RaceParkingObservation raw,double now,float appliedSpeed,float appliedCurvature,
                                  const RaceParkingDimensions &d) {
        const auto invalid=[&]{raw.valid=false;return raw;};
        if(!std::isfinite(now)||!std::isfinite(raw.captured)||now<raw.captured||!raw.sequence)return invalid();
        if(last.sequence&&(raw.epoch!=last.epoch||raw.sequence<=last.sequence||raw.captured<=last.captured))return invalid();
        const bool fresh=now-raw.captured<=.18;
        if(raw.valid&&!raw.estimated&&fresh) {
            // The tracker verified a link to the original visual coordinates.
            // Preserve its full correction for the controller's jump check.
            last=raw;absoluteCaptured=raw.captured;blindTravel=0;exhausted=false;return raw;
        }
        if(absoluteCaptured<0||!std::isfinite(appliedSpeed)||!std::isfinite(appliedCurvature))return invalid();
        const double gap=raw.captured-last.captured;
        blindTravel+=std::abs(appliedSpeed)*float(gap);
        exhausted|=now-absoluteCaptured>=maxEstimateSeconds||blindTravel>=maxEstimateTravel;
        if(exhausted) {
            raw.pose=last.pose;raw.valid=false;raw.estimated=true;last=raw;return raw;
        }
        auto predicted=raceParkingAdvance(last.pose,appliedSpeed*float(gap),appliedCurvature,d);
        if(raw.valid&&raw.estimated) {
            const float distance=std::hypot(raw.pose.x-predicted.x,raw.pose.z-predicted.z);
            const float angle=raceParkingAngle(raw.pose.yaw-predicted.yaw);
            const float blend=distance>0?std::min(1.f,.008f/distance):1.f;
            predicted.x+=(raw.pose.x-predicted.x)*blend;predicted.z+=(raw.pose.z-predicted.z)*blend;
            predicted.yaw+=std::clamp(angle,-.03f,.03f);
        }
        raw.pose=predicted;raw.valid=fresh;raw.estimated=true;
        last=raw;
        return raw;
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
// Require near-parallel aisle alignment before ordinary lane control takes over.
// Keep the paint allowance; do not confuse it with a heading allowance.
inline bool raceParkingReadyToRejoin(const RaceParkingSpace &space,RaceParkingPose pose) {
    if(!std::isfinite(pose.x)||!std::isfinite(pose.z)||!std::isfinite(pose.yaw))return false;
    if(std::abs(raceParkingAngle(pose.yaw))>.20f)return false;
    for(auto point:space.body(pose,0))if(std::abs(point.x)>space.dimensions().aisle/2+.025f)return false;
    // A near-boundary body can be inside now yet point out of the aisle. Check
    // a short straight continuation before releasing the alignment controller.
    const auto projected=raceParkingAdvance(pose,.25f,0,space.dimensions());
    for(auto point:space.body(projected,0))if(std::abs(point.x)>space.dimensions().aisle/2+.01f)return false;
    return true;
}
class RaceParkingControl {
    enum class Phase {Ready,Entry,Braking,InsetBack,Inset,Dwell,Exit,Done,Fault};
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
    float lastSpeed=0,steeringAngle=0,previousAppliedSpeed=0;
    std::deque<std::pair<double,float>> appliedHistory;
    double suspectSince=-1,suspectCapture=-1,verifySeconds=0;
    uint64_t suspectSequence=0;
    float resumeRadius=0,resumeAngle=0;
    bool confirmingPose=false,coastObserved=false,previousVerifiable=false;
    RaceParkingPose confirmationPose{},lastAbsolutePose{};
    float forwardSinceAbsolute=0,reverseSinceAbsolute=0,verifyForward=0,verifyReverse=0;
    const char *continuityReason="NORMAL";
    float taskSpeed=.10f;
    bool reverse=false;
    bool estimateUsed=false;
    bool joiningStart=false;
    float joinTravel=0,startDepartureTravel=-1;
    bool previousAbsolute=false;
    bool terminalRecovery=false;
    bool insetUsed=false,exitEdgeAlign=false,exitCoastStopped=false,entryEdgeGuard=false;
    float insetWheel=0,insetTravel=0;
    double insetWheelReadySince=-1;
    bool insetBackUsed=false,settlingBack=false;
    RaceParkingPose insetStart{};
    static constexpr float insetLimit=.16f,insetSpeed=.04f;
    bool exitDistanceRecovery=false;
    float reversedDistance=0,reverseDistanceBudget=0;
    float forwardAlignDistance=0,forwardAlignTime=0;
    double rejoinSince=-1;
    int rejoinFrames=0;
    const char *completionReason="RETURNED_TO_AISLE";
    const char *faultReason="PARK_FAULT";
    static cv::Point2f rear(RaceParkingPose p,const RaceParkingDimensions &d) {
        return {p.x+d.frontToRearAxle*std::sin(p.yaw),p.z-d.frontToRearAxle*std::cos(p.yaw)};
    }
    bool safeVisualBridge(RaceParkingPose from,RaceParkingPose to)const {
        if(!space.free(from)||!space.free(to))return false;
        const float yaw=raceParkingAngle(to.yaw-from.yaw);
        const float sweep=std::hypot(to.x-from.x,to.z-from.z)+space.dimensions().length*std::abs(yaw);
        const int steps=std::max(1,int(std::ceil(sweep/.005f)));
        for(int i=1;i<steps;++i){
            const float t=float(i)/steps;
            if(!space.free({from.x+t*(to.x-from.x),from.z+t*(to.z-from.z),from.yaw+t*yaw}))return false;
        }
        return true;
    }
    cv::Point2f rearStepInBodyFrame(RaceParkingPose from,RaceParkingPose to)const {
        const auto delta=rear(to,space.dimensions())-rear(from,space.dimensions());
        const float heading=from.yaw+.5f*raceParkingAngle(to.yaw-from.yaw);
        return {delta.x*std::cos(heading)+delta.y*std::sin(heading),
                delta.y*std::cos(heading)-delta.x*std::sin(heading)};
    }
    RaceParkingCommand fault(const char *reason) {
        phase=Phase::Fault;faultReason=reason;lastSpeed=0;
        return {0,0,"PARK_FAULT",faultReason,segment};
    }
    RaceParkingCommand finishExit(const char *reason,bool estimated=false) {
        phase=Phase::Done;lastSpeed=0;completionReason=reason;exitDistanceRecovery|=estimated;
        return {0,0,"PARK_COMPLETE",reason,segment};
    }
    RaceParkingCommand brakeForPaint(double captured,const char *reason="ENTRY_PAINT_STOPPING_ENVELOPE") {
        phase=Phase::Braking;phaseSince=captured;lastSpeed=0;steeringAngle=0;
        return {0,0,"PARK_BRAKE",reason,segment,true};
    }
public:
    static constexpr float forwardAlignDistanceLimit=.60f,forwardAlignTimeLimit=6.f;
    explicit RaceParkingControl(RaceParkingSpace s=RaceParkingSpace{},float targetSpeed=.10f):space(s),taskSpeed(targetSpeed){
        if(!std::isfinite(taskSpeed)||taskSpeed<=0||taskSpeed>10)throw std::invalid_argument("Invalid parking target speed");
    }
    bool active()const{return phase==Phase::Entry||phase==Phase::Braking||phase==Phase::InsetBack||phase==Phase::Inset||phase==Phase::Dwell||phase==Phase::Exit;}
    bool done()const{return phase==Phase::Done;}
    bool failed()const{return phase==Phase::Fault;}
    bool usedPoseEstimate()const{return estimateUsed;}
    bool acceptedCoastObservation()const{return coastObserved;}
    const char *poseContinuityReason()const{return continuityReason;}
    bool usedFinalInset()const{return insetUsed;}
    bool terminalRecoveryUsed()const{return terminalRecovery;}
    bool exitRecoveryUsed()const{return exitDistanceRecovery;}
    bool rejoiningAfterReverse()const{return reversedDistance>0&&
        (done()||(phase==Phase::Exit&&(exitDistanceRecovery||rejoinFrames>0)));}
    float reverseDistanceEstimate()const{return reversedDistance;}
    float reverseBudget()const{return reverseDistanceBudget;}
    float forwardAlignCommandM()const{return forwardAlignDistance;}
    float forwardAlignMovingSeconds()const{return forwardAlignTime;}
    const std::vector<RaceParkingPose> &reverseReference()const{return route.exit;}
    RaceParkingCommand status()const {
        if(failed())return {0,0,"PARK_FAULT",faultReason,segment};
        if(done())return {0,0,"PARK_COMPLETE",completionReason,segment};
        return {};
    }
    void abort(){if(active())fault("OPERATOR_STOP_RETEACH_REQUIRED");}
    const char *armBlockReason(const RaceParkingRoute &r,const RaceParkingObservation &o,double now,bool reverseVerified)const {
        if(phase!=Phase::Ready)return "PARK_START_NOT_READY";
        if(!reverseVerified)return "PARK_START_REVERSE_UNVERIFIED";
        if(!r.valid||r.entry.size()<2||r.exit.size()!=r.entry.size())return "PARK_START_ROUTE_INVALID";
        if(!o.valid||o.estimated||!o.sequence||!std::isfinite(now)||!std::isfinite(o.captured)||now<o.captured||now-o.captured>.18)
            return "PARK_START_WAIT_FRESH_POSE";
        const auto &start=r.entry.front();
        if(!RaceParkingSpace::nearRouteStart(o.pose,start))
            return "PARK_START_POSE_MOVED_REPLAN";
        if(!space.freeStartObserved(o.pose)&&!space.canJoinStartForward(o.pose,start))return "PARK_START_BODY_OUTSIDE_SPACE";
        if(!space.parked(r.entry.back()))return "PARK_START_GOAL_OUTSIDE_BAY";
        for(size_t i=0;i<r.entry.size();++i) {
            const auto &a=r.entry[i],&b=r.exit[r.entry.size()-1-i];
            if(!space.free(a))return "PARK_START_ROUTE_OUTSIDE_SPACE";
            if(std::hypot(a.x-b.x,a.z-b.z)>1e-5||std::abs(a.yaw-b.yaw)>1e-5)return "PARK_START_REVERSE_ROUTE_MISMATCH";
        }
        return nullptr;
    }
    bool arm(const RaceParkingRoute &r,const RaceParkingObservation &o,double now,bool reverseVerified) {
        if(armBlockReason(r,o,now,reverseVerified))return false;
        route=r;phase=Phase::Entry;epoch=o.epoch;reverse=true;segment=0;
        joiningStart=!space.freeStartObserved(o.pose);
        measuredEntry={o.pose};previousAbsolute=true;
        lastSequence=o.sequence;lastCapture=o.captured;lastTime=now;
        phaseSince=progressSince=now;previous=progressPose=lastAbsolutePose=o.pose;return true;
    }
    RaceParkingCommand update(const RaceParkingObservation &o,double now,float dt,bool taskClear,float speedLimit=.10f,
                              float appliedSpeed=std::numeric_limits<float>::quiet_NaN(),
                              float appliedCurvature=std::numeric_limits<float>::quiet_NaN()) {
        coastObserved=false;continuityReason="NORMAL";
        if(phase==Phase::Fault)return {0,0,"PARK_FAULT",faultReason,segment};
        if(phase==Phase::Done)return {0,0,"PARK_COMPLETE",completionReason,segment};
        if(!active())return {};
        if(!std::isfinite(now)||!std::isfinite(o.captured)||now<lastTime||!std::isfinite(dt)||dt<=0||dt>.75f||
           !std::isfinite(speedLimit)||speedLimit<0)return fault("PARK_TIME_OR_SPEED_INVALID");
        lastTime=now;
        const float applied=std::isfinite(appliedSpeed)?appliedSpeed:lastSpeed;
        forwardSinceAbsolute+=std::max(0.f,applied)*dt;
        reverseSinceAbsolute+=std::max(0.f,-applied)*dt;
        // Previous applied commands describe the short coast after a speed cut.
        // They are never fed back into the pose or issued again as speeds.
        while(!appliedHistory.empty()&&now-appliedHistory.front().first>.5)appliedHistory.pop_front();
        appliedHistory.emplace_back(now,applied);
        if(joiningStart)joinTravel+=std::abs(applied)*dt;
        if(startDepartureTravel>=0)startDepartureTravel+=std::abs(applied)*dt;
        // A lower new command does not stop physical motion instantly. Retain
        // one preceding applied interval for continuity checks only; this is
        // not a speed command and cannot authorize or fabricate movement.
        const float motionSpeedBound=std::max(std::abs(applied),previousAppliedSpeed);
        previousAppliedSpeed=std::abs(applied);
        // The actual previous PWM is the steering state, not the previous
        // requested target. Runtime and model use the same 100ms slew horizon.
        if(std::isfinite(appliedCurvature))steeringAngle=std::atan(space.dimensions().wheelbase*appliedCurvature);
        if(phase==Phase::Inset)insetTravel+=std::max(0.f,applied)*dt;
        if(phase==Phase::InsetBack)insetTravel+=std::abs(applied)*dt;
        if(phase==Phase::Exit) {
            reversedDistance+=std::max(0.f,-applied)*dt;
            // Count the full preceding command interval, including the last
            // moving interval before a stop. Zero-speed waits preserve the budget.
            if(exitDistanceRecovery&&applied>0) {
                forwardAlignDistance+=applied*dt;
                forwardAlignTime+=dt;
            }
        }
        // Match the bounded history recovery window, but do not turn one slow
        // iteration into a large steering or acceleration step.
        dt=std::min(dt,.25f);
        if(o.epoch!=epoch)return fault("PARK_ODOMETRY_EPOCH_CHANGED");
        if(o.sequence<lastSequence||(o.sequence>lastSequence&&o.captured<=lastCapture))return fault("PARK_OBSERVATION_REORDERED");
        if(suspectSince>=0) {
            if(o.sequence<suspectSequence||(o.sequence>suspectSequence&&o.captured<=suspectCapture))
                return fault("PARK_OBSERVATION_REORDERED");
            if(o.sequence==suspectSequence)return {0,0,"PARK_HOLD","WAIT_NEW_MOTION_FRAME",segment};
            const float interval=float(o.captured-suspectCapture);
            suspectSequence=o.sequence;suspectCapture=o.captured;
            const auto &d=space.dimensions();
            const bool freshAbsolute=o.valid&&!o.estimated&&now>=o.captured&&now-o.captured<=.18;
            if(!taskClear||!freshAbsolute){
                previousVerifiable=confirmingPose=false;continuityReason="WAIT_CLEAR_ABSOLUTE";
                return {0,0,"PARK_HOLD",taskClear?"PARK_VERIFY_WAIT_ABSOLUTE":"TRAFFIC_STOP_PRIORITY",segment};
            }
            // Freeze the last absolute anchor and the signed motion budget at the
            // first suspect frame. Time waiting at a gate adds no distance.
            if(previousVerifiable)verifySeconds+=std::min(interval,.25f);
            previousVerifiable=true;
            const auto step=rearStepInBodyFrame(lastAbsolutePose,o.pose);
            const float displacement=cv::norm(step);
            const float angle=std::abs(raceParkingAngle(o.pose.yaw-lastAbsolutePose.yaw));
            if(!(displacement<=.16f)||!(angle<=.45f)){
                continuityReason="GROSS_ABSOLUTE_JUMP";return fault("PARK_POSE_DISCONTINUITY");
            }
            const bool backAtAnchor=displacement<=resumeRadius&&angle<=resumeAngle;
            const bool continued=step.y<=verifyForward&&-step.y<=verifyReverse&&
                std::abs(step.x)<=.012f+.09f*angle&&angle<=.08f+displacement*std::tan(d.fullLockDegrees*float(CV_PI)/180)/d.wheelbase;
            const bool returned=(backAtAnchor||continued)&&safeVisualBridge(lastAbsolutePose,o.pose);
            if(returned&&confirmingPose&&interval<=.35f&&cv::norm(rear(o.pose,d)-rear(confirmationPose,d))<=.020f&&
               std::abs(raceParkingAngle(o.pose.yaw-confirmationPose.yaw))<=.06f) {
                previous=progressPose=lastAbsolutePose=o.pose;previousAbsolute=true;lastCapture=o.captured;lastSequence=o.sequence;
                forwardSinceAbsolute=reverseSinceAbsolute=0;continuityReason=backAtAnchor?"ORIGINAL_POSE_CONFIRMED":"APPLIED_MOTION_CONFIRMED";
                progressSince=now;badSince=suspectSince=-1;confirmingPose=false;
                return {0,0,"PARK_HOLD","PARK_ORIGINAL_POSE_CONFIRMED",segment};
            }
            if(verifySeconds>=1.){continuityReason="PERSISTENT_UNSUPPORTED_POSE";return fault("PARK_POSE_DISCONTINUITY");}
            confirmingPose=returned;if(returned)confirmationPose=o.pose;
            continuityReason=returned?"WAIT_SECOND_ABSOLUTE":"OUTSIDE_CONFIRMED_CORRIDOR";
            return {0,0,"PARK_HOLD","PARK_VERIFY_POSE_DISCONTINUITY",segment};
        }
        if(!taskClear) {
            previousAbsolute=false;
            rejoinSince=-1;rejoinFrames=0;
            lastSpeed=0;progressSince=now;dwell.ready(now,false,false,false,reverse);
            if(o.valid&&o.sequence>lastSequence&&now>=o.captured&&now-o.captured<=.18) {
                // A red light cannot silently adopt a jumped coordinate frame.
                if(!o.estimated&&cv::norm(rear(o.pose,space.dimensions())-rear(previous,space.dimensions()))<=.035f&&
                   std::abs(raceParkingAngle(o.pose.yaw-previous.yaw))<=.14f) {
                    previous=lastAbsolutePose=o.pose;lastCapture=o.captured;lastSequence=o.sequence;
                    forwardSinceAbsolute=reverseSinceAbsolute=0;
                }
            }
            if(badSince<0)badSince=now;
            return {0,0,"PARK_HOLD","TRAFFIC_STOP_PRIORITY",segment};
        }
        if(!o.valid||!o.sequence||now<o.captured||now-o.captured>.18) {
            previousAbsolute=false;
            rejoinSince=-1;rejoinFrames=0;
            lastSpeed=0;progressSince=now;dwell.ready(now,false,false,false,reverse);
            if(badSince<0)badSince=now;
            // A temporary image/pose delay is resumable. Do not destroy the
            // route after 650ms and require an operator to teach it again.
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
        const float wheelCurvature=std::tan(steeringAngle)/dimensions.wheelbase;
        // Compare rear-axle translation: bumper motion also contains rotation,
        // which cannot be inferred from the newly requested wheel angle.
        const float rearMove=cv::norm(rear(o.pose,dimensions)-rear(previous,dimensions));
        const float moveLimit=std::max(.025f,.008f+motionSpeedBound*gap*1.6f);
        const float turnLimit=std::max(.11f,.04f+motionSpeedBound*std::abs(wheelCurvature)*gap*1.6f);
        if(!std::isfinite(move)||!std::isfinite(turn)||gap<=0){continuityReason="INVALID_POSE_OR_INTERVAL";return fault("PARK_POSE_DISCONTINUITY");}
        const bool returningAbsolute=!previousAbsolute&&!o.estimated&&(rearMove>.035f||turn>.14f);
        if(gap>.75f||returningAbsolute||rearMove>moveLimit+.4f*recoveryExtra||turn>turnLimit+.8f*recoveryExtra) {
            const auto step=rearStepInBodyFrame(previous,o.pose);
            float forward=0,backward=0;
            for(const auto &item:appliedHistory){forward=std::max(forward,item.second);backward=std::max(backward,-item.second);}
            const float peak=step.y>=0?forward:backward;
            const bool entryOrExit=phase==Phase::Entry||phase==Phase::Exit;
            const bool softStep=entryOrExit&&previousAbsolute&&!o.estimated&&gap<=.25f&&rearMove<=.09f&&peak>=.035f&&
                std::abs(step.x)<=.012f+.09f*turn&&turn<=std::min(.22f,.08f+(rearMove+.008f)*
                    std::tan(dimensions.fullLockDegrees*float(CV_PI)/180)/dimensions.wheelbase)&&
                rearMove<=std::max(.035f,.015f+1.6f*peak*gap)&&safeVisualBridge(previous,o.pose);
            coastObserved=softStep&&peak>motionSpeedBound;
            if(softStep)continuityReason=coastObserved?"DECELERATION_COAST":"BOUNDED_VISUAL_MOTION";
            else {
                if(cv::norm(rear(o.pose,dimensions)-rear(lastAbsolutePose,dimensions))>.16f||
                   std::abs(raceParkingAngle(o.pose.yaw-lastAbsolutePose.yaw))>.45f){
                    continuityReason="GROSS_ABSOLUTE_JUMP";return fault("PARK_POSE_DISCONTINUITY");
                }
                suspectSince=now;suspectSequence=o.sequence;suspectCapture=o.captured;confirmingPose=false;verifySeconds=0;
                previousVerifiable=!o.estimated;
                resumeRadius=std::min(.025f,moveLimit);resumeAngle=std::min(.08f,turnLimit);
                verifyForward=std::min(.14f,.012f+1.6f*forwardSinceAbsolute+.40f*forward);
                verifyReverse=std::min(.14f,.012f+1.6f*reverseSinceAbsolute+.40f*backward);
                continuityReason=returningAbsolute?"VERIFY_ABSOLUTE_RETURN":"VERIFY_STEP";
                previousAbsolute=false;lastSpeed=0;rejoinSince=-1;rejoinFrames=0;
                dwell.ready(now,false,false,false,reverse);
                return {0,0,"PARK_HOLD","PARK_VERIFY_POSE_DISCONTINUITY",segment};
            }
        }
        const auto priorPose=previous;
        const bool visualStep=previousAbsolute&&!o.estimated&&gap<=.25f;
        previousAbsolute=!o.estimated;
        previous=o.pose;lastCapture=o.captured;lastSequence=o.sequence;badSince=-1;
        if(!o.estimated){lastAbsolutePose=o.pose;forwardSinceAbsolute=reverseSinceAbsolute=0;}
        estimateUsed|=o.estimated;
        const bool atStart=phase==Phase::Entry&&segment==0&&RaceParkingSpace::nearRouteStart(o.pose,route.entry.front());
        const bool observedFree=space.freeObserved(o.pose)||(atStart&&space.freeStartObserved(o.pose));
        const bool paintPassage=(phase==Phase::Entry||phase==Phase::Exit)&&space.freeEntryMouth(o.pose);
        const bool parkedObservation=(phase==Phase::Dwell||phase==Phase::Exit)&&space.parkedObserved(o.pose);
        const bool outsidePaint=!observedFree&&!paintPassage&&!parkedObservation;
        const bool mouthRecovery=phase==Phase::Entry&&space.needsEntryMouthRecovery(o.pose);
        auto waitForAbsolute=[&](){
            lastSpeed=0;progressSince=now;rejoinSince=-1;rejoinFrames=0;dwell.ready(now,false,false,false,reverse);
            return RaceParkingCommand{0,0,"PARK_HOLD","WAIT_ABSOLUTE_VISUAL_CONFIRMATION",segment};
        };
        if((phase==Phase::Entry||(phase==Phase::Braking&&!settlingBack)||phase==Phase::Inset)&&(std::hypot(o.pose.x-measuredEntry.back().x,o.pose.z-measuredEntry.back().z)>.004f||
           std::abs(raceParkingAngle(o.pose.yaw-measuredEntry.back().yaw))>.02f))measuredEntry.push_back(o.pose);
        if(joiningStart) {
            if(!visualStep)return waitForAbsolute();
            const auto origin=measuredEntry.front();
            const float dx=o.pose.x-origin.x,dz=o.pose.z-origin.z;
            const float along=dz*std::cos(origin.yaw)-dx*std::sin(origin.yaw);
            // Motor command travel can substantially exceed the measured crawl.
            // Once the car has advanced into the ordinary start window, let a
            // verified forward route sweep finish the remaining millimetres.
            const bool measuredJoin=atStart&&space.freeStartObserved(o.pose)&&
                along>=RaceParkingSpace::observedRearTolerance&&joinTravel-along>=RaceParkingSpace::startRearTolerance;
            const bool insidePlan=space.free(o.pose);
            if(insidePlan||measuredJoin) {
                // A camera interval can straddle the plane. Check travel in
                // the commanded straight corridor, rather than requiring the
                // moving car to be back in the stopped-pose matching window.
                const float across=dx*std::cos(origin.yaw)+dz*std::sin(origin.yaw);
                if(!space.insideAisle(o.pose)||along<0||along>std::min(.04f,joinTravel)+RaceParkingSpace::startPositionTolerance||
                   std::abs(across)>RaceParkingSpace::startPositionTolerance||
                   std::abs(raceParkingAngle(o.pose.yaw-route.entry.front().yaw))>RaceParkingSpace::startHeadingTolerance)
                    return fault("PARK_START_FORWARD_JOIN_REPLAN");
                joiningStart=false;
                if(!insidePlan)startDepartureTravel=0;
                progressSince=now;progressPose=o.pose;
            } else {
                if(!space.canJoinStartForward(o.pose,route.entry.front())||joinTravel>=.04f||now-progressSince>4)
                    return fault("PARK_START_FORWARD_JOIN_REPLAN");
                // Neutralize steering before using the straight swept footprint.
                // Applied PWM is read back on the next frame. Both travel and
                // visual confirmation bound this one-time forward command.
                lastSpeed=std::abs(steeringAngle)>.0001f?0.f:
                    std::min({RaceParkingSpace::startJoinSpeed,taskSpeed,speedLimit,(.04f-joinTravel)/.5f});
                return {lastSpeed,0,"PARK_ENTRY","FORWARD_TO_PLANNED_START",segment};
            }
        }
        // A half second of observed bumper translation and yaw covers the
        // residual motion seen after zero-speed commands in the field.
        auto coastClear=[&](bool inset=false){
            const float yawStep=raceParkingAngle(o.pose.yaw-priorPose.yaw);
            for(int i=1;i<=5;++i) {
                const float t=(.10f*i)/gap;
                const RaceParkingPose predicted{o.pose.x+(o.pose.x-priorPose.x)*t,
                    o.pose.z+(o.pose.z-priorPose.z)*t,o.pose.yaw+yawStep*t};
                if(!(inset?raceParkingInsetClear(space,predicted):space.free(predicted)))return false;
            }
            return true;
        };
        if(phase==Phase::Braking) {
            if(o.estimated)return waitForAbsolute();
            // Zero speed on the wire preceded up to 6.4cm of bumper motion.
            // Classify the actual resting footprint, never the predicted one.
            if(!visualStep||o.captured-phaseSince<.45||move/gap>=.012f||turn/gap>=.06f)
                return {0,0,"PARK_BRAKE","WAIT_ENTRY_COAST_STOP",segment};
            if(settlingBack) {
                // Drop the small segment that the car has already backed over;
                // the later reverse reference must contain only the forward entry.
                const auto &d=space.dimensions();
                const auto axle=rear(o.pose,d);
                size_t best=measuredEntry.size();float distance=.05f;
                for(size_t i=measuredEntry.size();i-->0;) {
                    const auto delta=axle-rear(measuredEntry[i],d);
                    if(cv::norm(delta)>.08f)break;
                    const float forward=delta.y*std::cos(o.pose.yaw)-delta.x*std::sin(o.pose.yaw);
                    const float gapM=cv::norm(delta);
                    if(forward>=0&&gapM<distance&&std::abs(raceParkingAngle(o.pose.yaw-measuredEntry[i].yaw))<.18f){
                        best=i;distance=gapM;
                    }
                }
                if(best==measuredEntry.size())return fault("PARK_INSET_RETURN_REFERENCE_LOST");
                measuredEntry.resize(best+1);measuredEntry.push_back(o.pose);settlingBack=false;
            }
            const auto wheel=!insetUsed&&!raceParkingInsetContained(space,o.pose)?
                raceParkingInsetWheel(space,o.pose,steeringAngle,insetLimit):std::nullopt;
            if(wheel) {
                phase=Phase::Inset;phaseSince=o.captured;insetUsed=true;insetStart=o.pose;
                insetWheel=*wheel;insetTravel=0;insetWheelReadySince=-1;
            } else if(!insetUsed&&!insetBackUsed&&reverse&&raceParkingCanStageInset(space,o.pose)) {
                phase=Phase::InsetBack;phaseSince=o.captured;insetBackUsed=settlingBack=true;
                insetStart=o.pose;insetTravel=0;insetWheelReadySince=-1;
            } else {
                phase=Phase::Dwell;
                terminalRecovery=!raceParkingInsetContained(space,o.pose);
            }
        }
        if(phase==Phase::InsetBack) {
            lastSpeed=0;
            if(!visualStep){insetWheelReadySince=-1;return waitForAbsolute();}
            const auto &d=space.dimensions();
            const auto delta=rear(o.pose,d)-rear(insetStart,d);
            const float back=delta.x*std::sin(insetStart.yaw)-delta.y*std::cos(insetStart.yaw);
            if(back>=.010f||back<-.005f||cv::norm(delta)>.035f||insetTravel>=.045f||o.captured-phaseSince>=3.||
               !coastClear(true)||!raceParkingInsetClear(space,o.pose)||!raceParkingInsetArcClear(space,o.pose,-.025f,0))
                return brakeForPaint(o.captured);
            const bool centered=std::abs(steeringAngle)<.02f;
            if(!centered)insetWheelReadySince=-1;
            else if(insetWheelReadySince<0)insetWheelReadySince=o.captured;
            steeringAngle=raceParkingSteeringStep(steeringAngle,0,dt,d);
            if(centered&&o.captured-insetWheelReadySince>=.45)lastSpeed=-std::min({insetSpeed,taskSpeed,speedLimit});
            return {lastSpeed,std::tan(steeringAngle)/d.wheelbase,"PARK_INSET_ADJUST","SHORT_REVERSE_BEFORE_FINAL_INSET",segment};
        }
        if(phase==Phase::Inset) {
            if(!visualStep){insetWheelReadySince=-1;return waitForAbsolute();}
            const auto &d=space.dimensions();
            const float travelled=cv::norm(rear(o.pose,d)-rear(insetStart,d));
            const float remaining=insetLimit-std::max(insetTravel,travelled);
            if(!raceParkingInsetClear(space,o.pose)||!coastClear(true)||raceParkingInsetContained(space,o.pose)||
               remaining<=0||o.captured-phaseSince>=7.)return brakeForPaint(o.captured);
            const bool preset=std::abs(steeringAngle-insetWheel)<.02f;
            if(!preset)insetWheelReadySince=-1;
            else if(insetWheelReadySince<0)insetWheelReadySince=o.captured;
            steeringAngle=raceParkingSteeringStep(steeringAngle,insetWheel,dt,d);
            const float curvature=std::tan(steeringAngle)/d.wheelbase;
            // Set the wheel while stationary before asking for the short arc.
            // The applied PWM is read again next frame. No unseen forward step
            // can clear the first brake or consume an additional attempt.
            lastSpeed=0;
            if(preset&&o.captured-insetWheelReadySince>=.45) {
                const float v=std::min({insetSpeed,speedLimit,taskSpeed,remaining/.5f});
                if(!raceParkingInsetArcClear(space,o.pose,v*.5f+.005f,steeringAngle))return brakeForPaint(o.captured);
                lastSpeed=v;
            }
            return {lastSpeed,curvature,"PARK_ENTRY","BOUNDED_FINAL_INSET",segment};
        }
        if(phase==Phase::Dwell) {
            if(o.estimated)return waitForAbsolute();
            // Arrival is latched: loss of painted containment must never restart
            // forward entry. Stop for the configured stationary dwell, then reverse the
            // actual entry path. Terminal recovery is not a valid-bay success.
            const bool stationary=move/gap<.012f&&turn/gap<.06f;
            terminalRecovery|=!space.parkedObserved(o.pose);
            if(dwell.ready(o.captured,true,stationary,true,reverse,space.dimensions().dwellSeconds)) {
                measuredEntry.push_back(o.pose);route.exit=raceParkingReverseSamples(measuredEntry,space.dimensions());
                // Remove the long approach straight from the reverse goal.
                // Keep only enough recorded entry to return roughly to the aisle.
                float length=0;
                for(size_t i=1;i<route.exit.size();++i) {
                    length+=cv::norm(rear(route.exit[i],space.dimensions())-rear(route.exit[i-1],space.dimensions()));
                    if(length>.10f&&raceParkingReadyToRejoin(space,route.exit[i])){route.exit.resize(i+1);break;}
                }
                reversedDistance=0;reverseDistanceBudget=std::max(.15f,length+.12f);
                phase=Phase::Exit;phaseSince=progressSince=now;progressPose=o.pose;segment=0;
                lastSpeed=0;return {0,0,"PARK_EXIT","DWELL_COMPLETE_REVERSE_NEXT_FRAME",0};
            }
            lastSpeed=0;return {0,0,"PARK_DWELL",stationary?
                (terminalRecovery?"TERMINAL_RECOVERY_DWELL":"STATIONARY_INSIDE_DWELL"):"WAIT_PHYSICAL_STOP",segment};
        }
        const bool leaving=phase==Phase::Exit;const auto &path=leaving?route.exit:route.entry;
        const auto &goal=path.back();const auto &d=space.dimensions();
        const float goalError=std::hypot(o.pose.x-goal.x,o.pose.z-goal.z);
        // In the deep part of the bay, predict the observed bumper translation
        // AND yaw. The final forward command is a poor estimate of either while
        // the car is still turning/sliding. Two field stops took ~0.4-0.5s
        // of residual motion; use a half-second horizon and sample its sweep.
        const bool deepEntry=!leaving&&o.pose.x<-.5f*d.aisle-.35f*d.depth;
        if(deepEntry) {
            if(!visualStep)return waitForAbsolute();
            if(!space.freeEntryMouth(o.pose)||!coastClear())return brakeForPaint(o.captured);
        }
        const float edgeMargin=space.oppositeEdgeClearance(o.pose);
        // Physical motion can continue after zero PWM. Predict the measured
        // outermost corner, which changes from the nose to the tail on exit.
        const float coastMargin=space.oppositeEdgeCoastClearance(o.pose,priorPose,gap);
        bool nearExitEdge=false;
        const bool readyToRejoin=leaving&&raceParkingReadyToRejoin(space,o.pose)&&
            std::hypot(o.pose.x-path.front().x,o.pose.z-path.front().z)>.10f;
        if(leaving) {
            const auto edge=space.exitEdge(o.pose,priorPose,gap);
            nearExitEdge=edge.near;
            if((nearExitEdge||exitEdgeAlign)&&!visualStep)return waitForAbsolute();
            if(edge.align&&!readyToRejoin) {
                exitEdgeAlign=true;exitDistanceRecovery=true;
            }
        }
        if(readyToRejoin) {
            if(o.estimated)return waitForAbsolute();
            if(rejoinSince<0)rejoinSince=o.captured;
            ++rejoinFrames;lastSpeed=0;
            if(o.captured-rejoinSince>=.20&&rejoinFrames>=3)
                return finishExit(exitDistanceRecovery?"CORRECTED_AISLE_ALIGNMENT_FORWARD_REJOIN":"CONFIRMED_AISLE_ALIGNMENT_FORWARD_REJOIN",exitDistanceRecovery);
            return {0,0,"PARK_EXIT","AISLE_ALIGNMENT_CONFIRMING",segment};
        }
        rejoinSince=-1;rejoinFrames=0;
        // Bound reversing without falsely reporting an unaligned pose as done.
        // If the distance estimate is exhausted, retain parking control and
        // correct forward toward the taught aisle before handing over to Pilot.
        if(leaving&&reversedDistance>=reverseDistanceBudget)
            exitDistanceRecovery=true;
        if(leaving&&exitDistanceRecovery) {
            if(forwardAlignDistance>=forwardAlignDistanceLimit||forwardAlignTime>=forwardAlignTimeLimit)
                return fault("PARK_EXIT_ALIGNMENT_LIMIT");
            if(o.estimated)return waitForAbsolute();
            const auto axle=rear(o.pose,d);
            const float desiredYaw=std::atan2(axle.x,.45f);
            const float error=raceParkingAngle(o.pose.yaw-desiredYaw);
            const float lock=d.fullLockDegrees*float(CV_PI)/180;
            const float angle=std::atan(d.wheelbase*(-4.f*error));
            const float target=std::clamp(angle,-lock,lock);
            steeringAngle=raceParkingSteeringStep(steeringAngle,target,dt,d);
            lastSpeed=std::min({speedLimit,taskSpeed,.12f});
            if(exitEdgeAlign) {
                // Allow the reverse steering to unwind before driving
                // forward, so the bumper does not swing into the bay side.
                lastSpeed=std::min(lastSpeed,.065f);
                if(applied>=0&&rearMove/gap<=.035f&&turn/gap<=.12f)exitCoastStopped=true;
                if(!exitCoastStopped||std::abs(target-steeringAngle)>.06f)lastSpeed=0;
                if(lastSpeed>0){
                    const float sweep=lastSpeed*.5f+.005f;
                    for(float ds=0;ds<sweep;ds+=.0025f){
                        const auto next=raceParkingAdvance(o.pose,std::min(sweep,ds+.0025f),std::tan(steeringAngle)/d.wheelbase,d);
                        if(!space.free(next)){lastSpeed=0;break;}
                    }
                }
            }
            return {lastSpeed,std::tan(steeringAngle)/d.wheelbase,"PARK_EXIT_ALIGN",
                exitEdgeAlign?"OPPOSITE_EDGE_FORWARD_ALIGNMENT":"REVERSE_BUDGET_FORWARD_ALIGNMENT",segment};
        }
        // A fully contained vehicle near the end is already parked; continuing
        // to chase one exact pose can push an otherwise valid body toward a line.
        // Exit finishes once the complete body is in the aisle near its starting
        // position; do not keep reversing toward the gate to match one exact point.
        if((goalError<(leaving?.045f:.05f)||(!leaving&&space.parked(o.pose)))&&
            std::abs(raceParkingAngle(o.pose.yaw-goal.yaw))<.20f) {
            if(!leaving&&space.parked(o.pose)) {
                if(o.estimated)return waitForAbsolute();
                phase=Phase::Dwell;phaseSince=now;lastSpeed=0;steeringAngle=0;
                return {0,0,"PARK_DWELL","FULL_BODY_INSIDE_STOPPING",segment};
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
                if(o.estimated)return waitForAbsolute();
                if(!raceParkingInsetContained(space,o.pose))
                    return brakeForPaint(o.captured,"ENTRY_TERMINAL_REACHED_STOPPING");
                terminalRecovery=false;
                phase=Phase::Dwell;phaseSince=now;lastSpeed=0;steeringAngle=0;
                return {0,0,"PARK_DWELL","FULL_BODY_INSIDE_STOPPING",segment};
            }
        }
        const auto refYaw=path[segment].yaw+projection*raceParkingAngle(path[segment+1].yaw-path[segment].yaw);
        const bool headingRecovery=std::abs(raceParkingAngle(o.pose.yaw-refYaw))>.35f;
        auto target=rear(path[segment],d)+(rear(path[segment+1],d)-rear(path[segment],d))*projection;
        // 5.5cm forward lookahead amplified millimetres of visual cross-track
        // noise into alternating full-lock commands on the real straight approach.
        float remaining=.10f,minPreviewYaw=refYaw;
        for(size_t i=segment;i+1<path.size();++i) {
            const auto end=rear(path[i+1],d),delta=end-target;const float length=cv::norm(delta);
            if(length>1e-6f){
                const float fromYaw=i==segment?refYaw:path[i].yaw;
                minPreviewYaw=std::min(minPreviewYaw,fromYaw+std::min(1.f,remaining/length)*raceParkingAngle(path[i+1].yaw-fromYaw));
            }
            if(length>=remaining){target+=delta*(remaining/length);break;}
            remaining-=length;target=end;
        }
        const auto delta=target-axle;const float lateral=std::cos(o.pose.yaw)*delta.x+std::sin(o.pose.yaw)*delta.y;
        const float limit=std::tan(d.fullLockDegrees*float(CV_PI)/180)/d.wheelbase;
        // At the end of the path the target gets closer than the nominal
        // 10cm horizon. Do not amplify centimetre errors into full counterlock.
        const float pursuit=-2*lateral/std::max(.01f,delta.dot(delta));
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
        // Retain previewed bend support with signed heading feedback. This
        // parking-entry correction does not change ordinary lane steering.
        if(!leaving&&route.entryCurvaturePreview>0) {
            // The former heading-error threshold toggled this lower bound
            // from left lock to full right correction around 5 degrees of
            // overshoot (field frames 110-120, 150-160). Keep the bend support
            // continuous through zero heading error; signed feedback releases
            // it gradually as heading overshoots. Taper at straight/right
            // references and the goal instead of switching it off abruptly.
            const float blend=std::clamp(feedforward/.5f,0.f,1.f)*
                std::clamp((goal.yaw-refYaw)/.15f,0.f,1.f);
            const float recovery=1.1f*std::max(0.f,feedforward)+5.f*headingError;
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
        const float lock=d.fullLockDegrees*float(CV_PI)/180;
        const float angleRate=lock*600/400;
        if(!leaving&&visualStep&&requested<0&&steeringAngle<0&&refYaw<=path.front().yaw+.04f&&
           minPreviewYaw<path.front().yaw-.04f) {
            // The approach can swing right before the planned left turn.
            // Limit lateral pursuit by how far even a neutralising wheel will
            // carry the measured yaw past the rightmost planned heading.
            const float yawRate=raceParkingAngle(o.pose.yaw-priorPose.yaw)/gap;
            if(yawRate<-.10f){
                const float yawAtNeutral=o.pose.yaw+std::clamp(yawRate*(.18f+.5f*std::abs(steeringAngle)/angleRate),-.45f,0.f);
                const float bounded=std::max(requested,(minPreviewYaw-yawAtNeutral-.025f)/.07f);
                entryHeadingRecovery|=bounded>requested+.001f;requested=bounded;
            }
        }
        if(!leaving&&!deepEntry){
            if(visualStep&&edgeMargin<.10f&&coastMargin<.025f&&o.pose.yaw<-.04f)entryEdgeGuard=true;
            if(entryEdgeGuard&&!visualStep)return waitForAbsolute();
            if(entryEdgeGuard&&edgeMargin>.025f&&o.pose.yaw>=0)entryEdgeGuard=false;
            if(entryEdgeGuard)requested=limit;
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
        // Generate a continuous wheel-angle target within the existing command
        // slew, instead of repeatedly alternating full speed and a steering wait.
        const float targetAngle=std::atan(d.wheelbase*requested);
        steeringAngle=raceParkingSteeringStep(steeringAngle,targetAngle,dt,d);
        const float curvature=std::tan(steeringAngle)/d.wheelbase;
        const float creep=std::min(.12f,taskSpeed*.5f);
        const float curveCap=taskSpeed<=.10001f?taskSpeed:std::max(creep,taskSpeed/(1+.5f*std::abs(curvature)));
        float magnitude=std::min({speedLimit,leaving?taskSpeed*.8f:taskSpeed,curveCap,std::max(creep,goalError*.8f)});
        if(!leaving) {
            // Let the front wheels build the commanded turn before carrying
            // the rear axle past its reference bend on a short approach.
            const float wheelLag=std::abs(targetAngle-steeringAngle);
            if(feedforward>.5f&&(wheelLag>.09f||headingError>.08f))
                magnitude=std::min(magnitude,creep);
            // Remove the creep floor as the bumper approaches its terminal
            // pose. Less speed here leaves room for the measured coast.
            if(deepEntry)
                magnitude=std::min(magnitude,std::max(.035f,goalError*.55f));
        }
        if(nearExitEdge)magnitude=std::min(magnitude,.08f);
        if(entryEdgeGuard){
            magnitude=std::min(magnitude,.065f);
            if(std::abs(targetAngle-steeringAngle)>.06f)magnitude=0;
        }
        if(mouthRecovery)magnitude=std::min(magnitude,creep);
        // In the accepted start window the live rear bumper can still be
        // millimetres behind the frozen planning plane. Only a measured forward
        // departure may use that strip, and its sweep must end inside the plan.
        const bool startSweep=startDepartureTravel>=0&&!space.free(o.pose);
        auto departureSpace=space;
        if(startSweep) {
            if(!visualStep)return waitForAbsolute();
            if(!atStart||!space.insideAisle(o.pose)||!space.freeStartObserved(o.pose)||startDepartureTravel>=.04f)
                return fault("PARK_START_FORWARD_JOIN_REPLAN");
            // The actuator watchdog can carry this command for 250 ms.
            magnitude=std::min(magnitude,(.04f-startDepartureTravel)/.25f);
            if(taskSpeed>.10001f)magnitude=std::min(magnitude,std::abs(lastSpeed)+.5f*dt);
            departureSpace.keepRearWithinStart(o.pose);
        } else startDepartureTravel=-1;
        // Validate a short swept-body stopping envelope before issuing motion.
        auto sweepClear=[&](float speed) {
            auto prediction=o.pose;const float sweep=speed*.25f+.005f;float predictedAngle=steeringAngle;
            for(float s=0;s<sweep;s+=.0025f) {
                const float ds=std::min(.0025f,sweep-s),horizonStep=ds/std::max(.01f,speed);
                predictedAngle+=std::clamp(targetAngle-predictedAngle,-angleRate*horizonStep,angleRate*horizonStep);
                prediction=raceParkingAdvance(prediction,(leaving?-1:1)*ds,std::tan(predictedAngle)/d.wheelbase,d);
                if(!space.free(prediction)&&!(startSweep&&space.insideAisle(prediction)&&departureSpace.free(prediction)))return false;
            }
            return !startSweep||space.free(prediction);
        };
        // Close to a divider, choose a shorter stopping envelope by reducing
        // speed within the existing parking range, retaining the same geometry.
        bool boundaryWarning=outsidePaint||routeRecovery||headingRecovery;
        while(!sweepClear(magnitude)) {
            if(startSweep)return fault("PARK_START_FORWARD_JOIN_REPLAN");
            if(entryEdgeGuard){magnitude=0;break;}
            if(deepEntry) {
                if(magnitude<=.03501f)return brakeForPaint(o.captured);
                magnitude=std::max(.035f,magnitude-.01f);
                continue;
            }
            // Keep the existing traversable mouth and reverse-correction policy.
            if(magnitude<=creep+.00001f){boundaryWarning=true;break;}
            magnitude=std::max(creep,magnitude-.01f);
        }
        if(taskSpeed>.10001f)magnitude=std::min(magnitude,std::abs(lastSpeed)+.5f*dt);
        const float speed=leaving?-magnitude:magnitude;
        if(!o.estimated&&(std::hypot(o.pose.x-progressPose.x,o.pose.z-progressPose.z)>.008f||
           std::abs(raceParkingAngle(o.pose.yaw-progressPose.yaw))>.04f)) {progressPose=o.pose;progressSince=now;}
        if(now-progressSince>4)return fault("PARK_NO_MEASURED_PROGRESS");
        lastSpeed=speed;
        return {speed,curvature,leaving?"PARK_EXIT":"PARK_ENTRY",
            entryEdgeGuard?"OPPOSITE_EDGE_STEERING_RECOVERY":o.estimated?"COMMAND_POSE_ESTIMATE_CONTINUE":routeRecovery?"ROUTE_ERROR_CORRECTING_CONTINUE":
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
inline bool raceParkingModelCompletes(const RaceParkingSpace &space,const RaceParkingRoute &route,float taskSpeed=.10f,float dt=.04f,
                                      bool delayedEntry=false) {
    if(!route.valid||route.entry.empty())return false;
    RaceParkingControl control(space,taskSpeed);RaceParkingObservation observation{route.entry.front(),1,1,1,true};
    if(!control.arm(route,observation,1,true))return false;
    const auto &d=space.dimensions();float wheel=0,appliedSpeed=0,rollingSpeed=0,rollingAngle=0;double time=1;
    for(int step=0;step<3500&&!control.done()&&!control.failed();++step) {
        time+=dt;observation.captured=time;++observation.sequence;
        const auto command=control.update(observation,time,dt,true,taskSpeed,appliedSpeed,std::tan(wheel)/d.wheelbase);
        // A planned reference itself must fit. Reserve the final inset for
        // unexpected real tracking error after braking to rest.
        if(control.usedFinalInset())return false;
        const float target=std::atan(d.wheelbase*command.curvature);
        wheel=raceParkingSteeringStep(wheel,target,dt,d);
        // Runtime keeps moving during slew; do not validate using an artificial
        // steering wait that never occurs on the vehicle.
        const float pwm=std::round(1500-wheel/(d.fullLockDegrees*float(CV_PI)/180)*400);
        wheel=(1500-pwm)/400*d.fullLockDegrees*float(CV_PI)/180;
        appliedSpeed=command.speed;
        if(delayedEntry) {
            // A second feasibility scenario, not a calibration: the two right
            // bays showed steering lag, coast and lateral displacement of the
            // visually reconstructed rear axle while yaw was changing.
            if(std::string_view(command.state)=="PARK_EXIT")
                return !control.terminalRecoveryUsed()&&space.parkedObserved(observation.pose);
            rollingSpeed+=(appliedSpeed-rollingSpeed)*(1-std::exp(-dt/.38f));
            rollingAngle+=(wheel-rollingAngle)*(1-std::exp(-dt/.25f));
            const float curvature=std::tan(rollingAngle)/d.wheelbase;
            const float yawStep=rollingSpeed*dt*curvature;
            observation.pose=raceParkingAdvance(observation.pose,rollingSpeed*dt,curvature,d);
            observation.pose.x-=.09f*yawStep*std::cos(observation.pose.yaw);
            observation.pose.z-=.09f*yawStep*std::sin(observation.pose.yaw);
        } else {
            observation.pose=raceParkingAdvance(observation.pose,appliedSpeed*dt,std::tan(wheel)/d.wheelbase,d);
        }
        if(!space.free(observation.pose))return false;
    }
    // Runtime terminal recovery must not make an invalid parking trajectory
    // pass the planner's nominal full-containment feasibility filter.
    return control.done()&&!control.terminalRecoveryUsed()&&!control.exitRecoveryUsed();
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
    auto search=[&](float preview,float weight,float scale,float outwardYawLimit) {
        auto candidate=racePlanParking(space,start,scale,[&](const RaceParkingRoute &route){
            auto checked=route;checked.entryCurvaturePreview=preview;
            return raceParkingModelCompletes(space,checked,taskSpeed)&&raceParkingModelCompletes(space,checked,taskSpeed,.10f)&&
                raceParkingModelCompletes(space,checked,taskSpeed,.08f,true);
        },weight,preview>0,outwardYawLimit);
        candidate.entryCurvaturePreview=preview;
        searches.push_back({scale,weight,candidate.expanded,candidate.geometricCandidates,candidate.modelRejected,candidate.reason,preview,outwardYawLimit});
        expanded+=candidate.expanded;candidates+=candidate.geometricCandidates;rejected+=candidate.modelRejected;
        candidate.expanded=expanded;candidate.geometricCandidates=candidates;candidate.modelRejected=rejected;candidate.searches=searches;
        result=std::move(candidate);
        if(result.valid)result.reason="MODEL_CHECKED_FORWARD_ROUTE";
        return result.valid;
    };
    // With over a body length of approach, first seek a gentler bend:
    // half the permitted outward heading and steering authority in reserve.
    // Tight approaches retain the original search and all three model checks.
    if(start.z<-space.dimensions().length&&search(0.f,1.f,.8f,.12f))return result;
    for(float preview:{.10f,0.f})for(float weight:{1.5f,1.f})for(float scale:{1.f,.9f,.8f,.7f})
        if(search(preview,weight,scale,.24f))return result;
    result.valid=false;result.reason=candidates?"GEOMETRIC_ROUTES_REJECTED_BY_CONTROL_MODEL":
        std::any_of(searches.begin(),searches.end(),[](auto s){return std::string(s.reason)=="SEARCH_BUDGET_EXHAUSTED";})?
        "PARK_SEARCH_BUDGET_EXHAUSTED":"NO_FOOTPRINT_CLEAR_PATH";return result;
}
