#pragma once
#include "race_ground_arrow.hpp"
#include "race_ground.hpp"
#include "race_parking_sign.hpp"
// Route stages are explicit; repeated arrows cannot identify a lap or finish.
// On the final return STOP arms a finish beyond the crosswalk; n remains the manual finish.
class RaceMission {
public:
    enum class Stage {Ready,LabLoop,FirstFork,ParkingApproach,ParkingSearch,ParkingRun,ParkingExit,FirstReturn,SecondFork,SecondReturn,Done,Fault};
private:
    enum class ParkingResult { NotReturned, ReturnedUnverified, ParkedAndReturned };
    ParkingResult parkingResult=ParkingResult::NotReturned;
    Stage stage=Stage::Ready;bool lab=false,branch=false;int branchDirection=0;uint64_t sequence=0;int parkCount=0;
    std::string reason="PRESS_A_START";bool parkingSign=false;
    int exitArrowCount=0;uint64_t exitArrowFrame=0;bool exitArrowConfirmed=false;
    cv::Rect lastExitArrowBox;
    double lastExitArrowCapture=-1;
    float parkModelScore=0;cv::Rect parkShapeBox;const char *parkEvidence="NONE";
    double parkCaptured=0;cv::Rect previousParkBox;
    int parkNearCount=0;bool parkTurnNear=false;
    bool entryGateSeen=false,entryGateReleased=false,exitGateSeen=false,exitGateReleased=false;
    bool exitGateSeenDuringRejoin=false;
    bool entranceRoadAcquired=false;
    float baySearchCommandM=0,exitGateCommandM=0;
    float exitSearchCommandM=0,exitSearchLimitM;
    cv::Rect previousFinishBox;
    int finishStopFrames=0;
    uint64_t finishSequence=0;
    double finishCapture=-1;
    bool finishStopArmed=false,finishMoving=false;
    cv::Rect finishCrossBox;
    int firstCrossBottom=0,crossSamples=0,closeCrossSamples=0;
    double crossCapture=-1,clearSince=-1,finishClearSeconds;
    float clearDistanceM=0,afterNearCommandM=0,finishSearchCommandM=0;
    void observeFinishStop(const RaceSnapshot &ai,const RacePlan &traffic,const RaceGroundMap *ground,float length) {
        if(!traffic.fresh){if(!finishStopArmed)finishStopFrames=0;previousFinishBox={};closeCrossSamples=0;clearSince=-1;return;}
        if(!ai.sequence||ai.sequence<=finishSequence||!std::isfinite(ai.captured)||ai.captured<=finishCapture)return;
        const double gap=ai.captured-finishCapture;
        finishSequence=ai.sequence;finishCapture=ai.captured;
        if(!finishStopArmed){
            const auto &box=traffic.acceptedStopBox;
            const bool same=box.area()>0&&previousFinishBox.area()>0&&gap<=.30&&
                (box&previousFinishBox).area()>=.30*std::min(box.area(),previousFinishBox.area());
            previousFinishBox=box;
            finishStopFrames=box.area()>0?(same?finishStopFrames+1:1):0;
            if(finishStopFrames>=3){finishStopArmed=true;reason="FINISH_STOP_ARMED_WAIT_CROSSWALK";}
        }
        const auto &band=traffic.acceptedCrossBox;
        if(band.area()>0){
            const int bottom=band.br().y;
            const int overlap=std::min(band.br().x,finishCrossBox.br().x)-std::max(band.x,finishCrossBox.x);
            const double crossGap=ai.captured-crossCapture;
            const bool samePosition=overlap>=.40f*std::min(band.width,finishCrossBox.width)&&bottom+32>=finishCrossBox.br().y;
            const bool matches=crossSamples>0&&samePosition&&
                (crossGap<=1.2||(clearDistanceM>0&&traffic.acceptedCrossAtPassRow));
            // A newly visible far band is not the close band under the car.
            // Only another observation of the tracked band may reset the tail distance.
            if(matches||clearDistanceM<=0||traffic.acceptedCrossAtPassRow) {
                if(!matches){firstCrossBottom=bottom;crossSamples=0;clearDistanceM=afterNearCommandM=0;}
                closeCrossSamples=traffic.acceptedCrossAtPassRow?
                    (matches&&crossGap<=.30?std::min(3,closeCrossSamples+1):1):0;
                ++crossSamples;crossCapture=ai.captured;finishCrossBox=band;clearSince=-1;
                if(finishStopArmed&&traffic.acceptedCrossAtPassRow&&crossSamples>=2&&
                   (bottom-firstCrossBottom>=24||closeCrossSamples>=3)&&ground&&std::isfinite(length)&&length>0) {
                    // The top of the matched box is the far edge. The calibrated map
                    // has its origin at the front bumper. Use the farther of the two ends.
                    const float left=ground->toGround({float(band.x),float(band.y)}).y;
                    const float right=ground->toGround({float(band.br().x-1),float(band.y)}).y;
                    if(std::isfinite(left)&&std::isfinite(right)&&std::max(left,right)>=0&&std::max(left,right)<=1.f) {
                        clearDistanceM=std::max(left,right)+length+.02f;afterNearCommandM=0;
                        reason="FINISH_CROSSWALK_NEAR_CLEAR_REAR";
                    }
                }
                return;
            }
        } else closeCrossSamples=0;
        if(!finishStopArmed||clearDistanceM<=0||!finishMoving||traffic.stop||gap>.30){
            clearSince=-1;return;
        }
        if(clearSince<0)clearSince=ai.captured;
        if(ai.captured-clearSince>=finishClearSeconds&&afterNearCommandM>=clearDistanceM){
            stage=Stage::Done;reason="FINISH_CROSSWALK_CLEARED";
        }
    }
public:
    explicit RaceMission(bool laboratory=false,float exitLimitM=RaceConfig::defaultExitSearchLimitM,double crossClearSeconds=.25):
        lab(laboratory),exitSearchLimitM(exitLimitM),finishClearSeconds(crossClearSeconds){}
    const char *name()const {
        switch(stage){case Stage::Ready:return "MISSION_READY";case Stage::LabLoop:return "LAB_SINGLE_LOOP";
        case Stage::FirstFork:return "LAP1_FLOW_THEN_SIGNED_FORK";case Stage::ParkingApproach:return "PARKING_APPROACH";
        case Stage::ParkingSearch:return "PARKING_FIND_BAY";case Stage::ParkingRun:return "PARKING_MANEUVER";
        case Stage::ParkingExit:return "PARKING_EXIT_LEFT";case Stage::FirstReturn:return "LAP1_RETURN_PRESS_N_AT_START";
        case Stage::SecondFork:return parkingSign?"LAP2_LEFT_TO_PARKING":"LAP2_WAIT_PARK_SIGN";case Stage::SecondReturn:return "LAP2_RETURN_PRESS_N_TO_FINISH";
        case Stage::Done:return "MISSION_COMPLETE";default:return "MISSION_FAULT";}
    }
    bool running()const{return stage!=Stage::Ready&&stage!=Stage::Done&&stage!=Stage::Fault;}
    bool parkingSearch()const{return stage==Stage::ParkingSearch;}
    bool parkingRun()const{return stage==Stage::ParkingRun;}
    bool parkingEntrancePending()const{return stage==Stage::SecondFork&&parkingSign;}
    bool parkingEntranceNear()const{return parkTurnNear;}
    // Bounds on applied-command travel, not measured distances or a speed cap.
    // A missing bay must not turn into an unbounded drive through the car park.
    static constexpr float baySearchCommandLimit=1.f;
    static constexpr float exitGatePassCommandM=.40f;
    void advance(const RacePlan &traffic,float dt,float appliedSpeed) {
        finishMoving=false;
        if(!running()||lab||!std::isfinite(dt)||dt<=0||dt>.25f||!std::isfinite(appliedSpeed)){clearSince=-1;return;}
        const float distance=std::max(0.f,appliedSpeed)*dt;
        if(stage==Stage::SecondReturn&&finishStopArmed){
            finishMoving=traffic.fresh&&!traffic.stop&&distance>0;
            if(finishMoving){
                finishSearchCommandM+=distance;
                if(clearDistanceM>0)afterNearCommandM+=distance;
            }else clearSince=-1;
        }
        if(parkingEntrancePending()&&traffic.fresh&&traffic.selectedBranch&&
           traffic.guideDirection==-1&&traffic.laneBlend>=.95f)entranceRoadAcquired=true;
        if((parkingEntrancePending()&&entranceRoadAcquired)||stage==Stage::ParkingApproach)
            baySearchCommandM+=distance;
        if(stage==Stage::ParkingExit&&exitGateReleased)exitGateCommandM+=distance;
        if(stage==Stage::ParkingExit&&!exitGateSeen&&!exitArrowConfirmed)exitSearchCommandM+=distance;
    }
    bool exitSearchTravelExhausted()const{return stage==Stage::ParkingExit&&!exitGateSeen&&!exitArrowConfirmed&&
        exitSearchCommandM>=exitSearchLimitM;}
    bool baySearchTravelExhausted()const{return parkingResult==ParkingResult::NotReturned&&
        (stage==Stage::ParkingApproach||(parkingEntrancePending()&&entranceRoadAcquired))&&
        baySearchCommandM>=baySearchCommandLimit;}
    bool canScanParkingBays(const RacePlan &traffic)const {
        return (stage==Stage::ParkingApproach||parkingSearch()||
            (parkingEntrancePending()&&(traffic.completedDirection==-1||entranceRoadAcquired||
                (traffic.selectedBranch&&traffic.guideDirection==-1&&traffic.laneBlend>=.95f))))&&
            !traffic.stop&&!traffic.gateWaiting;
    }
    bool exitArrowTurnAuthorized()const{return stage==Stage::ParkingExit&&exitArrowConfirmed;}
    bool exitTurnAuthorized()const{return stage==Stage::ParkingExit&&
        (exitGateSeen?(exitGateReleased&&exitGateCommandM>=exitGatePassCommandM):exitArrowConfirmed);}
    bool followParkingAisle()const{return stage==Stage::ParkingApproach||parkingSearch()||parkingRun()||
        (stage==Stage::ParkingExit&&!exitTurnAuthorized());}
    void observeExitArrow(const RaceGroundArrow &arrow,uint64_t frame,bool fresh,double captured){
        if(stage!=Stage::ParkingExit||lab||exitArrowConfirmed||frame<=exitArrowFrame)return;
        exitArrowFrame=frame;
        // Repeated Y/choice markings occur inside the parking aisle. They only
        // support exit direction after its gate releases. A near left-only
        // arrow also supports an exit whose gate was already open/unobserved.
        const bool left=(arrow.allowed&RoadLeft)&&(arrow.allowed==RoadLeft||exitGateReleased)&&
            arrow.box.area()>0&&arrow.box.y+arrow.box.height>=180;
        if(!fresh||!left||!std::isfinite(captured)||captured<=lastExitArrowCapture){
            exitArrowCount=0;return;
        }
        // At the near-arrow image height the same painted marking should
        // overlap between frames. Disjoint arrows cannot confirm each other.
        const int intersection=(lastExitArrowBox&arrow.box).area();
        const bool same=exitArrowCount>0&&captured-lastExitArrowCapture<=.30&&
            intersection>=.30*std::min(lastExitArrowBox.area(),arrow.box.area());
        exitArrowCount=same?exitArrowCount+1:1;
        lastExitArrowCapture=captured;lastExitArrowBox=arrow.box;
        if(exitArrowCount>=3){exitArrowConfirmed=true;reason="PARKING_EXIT_GROUND_LEFT_TURN_AUTHORIZED";}
    }
    bool parkingZone()const{return (stage==Stage::SecondFork&&parkingSign)||stage==Stage::ParkingApproach||parkingSearch()||parkingRun()||stage==Stage::ParkingExit;}
    bool parkingApproach()const{return stage==Stage::ParkingApproach||
        (stage==Stage::SecondFork&&parkingSign)||
        ((stage==Stage::FirstReturn||stage==Stage::SecondFork)&&parkCount>0);}
    bool parkingSpeedLimited()const{return !lab&&running()&&
        (!exitGateReleased||(exitGateSeenDuringRejoin&&exitGateCommandM<exitGatePassCommandM))&&
        (parkingZone()||parkingApproach());}
    float speedLimit(float cruise,float parkingSpeed)const{return parkingSpeedLimited()?std::min(cruise,parkingSpeed):cruise;}
    void observeGate(const RacePlan &traffic,bool rejoiningAfterReverse=false) {
        if(!running()||lab||!traffic.fresh)return;
        // A gate reached while aligning forward after the reverse belongs to
        // the exit. Retain its verified cycle across handoff. ParkingExit still
        // owns both passage-distance accumulation and permission to turn.
        // The traffic hold interrupts the controller's alignment observations.
        // Keep ownership of a gate whose stop was seen during rejoin until its
        // own verified opening, even if the alignment timer has been reset.
        const bool rejoining=parkingRun()&&(rejoiningAfterReverse||(exitGateSeenDuringRejoin&&!exitGateReleased));
        if(stage==Stage::ParkingExit||rejoining) {
            if(traffic.gateWaiting&&(!rejoining||traffic.stop)) {
                exitGateSeen=true;exitGateSeenDuringRejoin|=rejoining;
                exitGateReleased=false;exitGateCommandM=0;
            }
            if(exitGateSeen&&traffic.gateOpened){exitGateReleased=true;reason=rejoining?
                "EXIT_GATE_OPEN_DURING_AISLE_REJOIN":"EXIT_GATE_OPEN_RESTORE_CRUISE";}
        } else if(parkingZone()||parkingApproach()) {
            if(traffic.gateWaiting)entryGateSeen=true;
            if(entryGateSeen&&traffic.gateOpened){entryGateReleased=true;reason="ENTRY_GATE_OPEN_PARKING_AREA_SPEED";}
        }
    }
    bool complete()const{return stage==Stage::Done;}
    bool finalReturn()const{return stage==Stage::SecondReturn;}
    // A missing crossing may not produce unlimited autonomous travel. This is
    // an unresolved landmark, never a successful finish. n remains available.
    bool finishSearchHeld()const{return finalReturn()&&finishStopArmed&&clearDistanceM<=0&&finishSearchCommandM>=2.5f;}
    bool failed()const{return stage==Stage::Fault;}
    const char *statusReason()const{return reason.c_str();}
    bool parkStageEligible()const{return !lab&&parkingResult==ParkingResult::NotReturned&&
        (stage==Stage::FirstFork||stage==Stage::FirstReturn||stage==Stage::SecondFork);}
    // FLOW and Y signs belong to the local traffic manager. A permanent
    // straight override would suppress the later left/right Y sign.
    int routeDirection()const{return (stage==Stage::SecondFork&&parkingSign)||exitTurnAuthorized()?-1:0;}
    void observeSigns(const RaceSnapshot &ai,bool fresh) {
        if(!running()||lab||!fresh||ai.sequence<=sequence)return;sequence=ai.sequence;
        parkModelScore=0;cv::Rect modelBox;
        for(auto d:ai.detections)if(d.type==LABEL_PARK&&d.score>parkModelScore){
            parkModelScore=d.score;modelBox={d.x,d.y,d.width,d.height};
        }
        parkShapeBox=raceParkingPShape(ai.image);
        parkEvidence=parkModelScore>=.65f?"MODEL_PARK":parkShapeBox.area()>0?"BLUE_P_SHAPE":"NONE";
        const bool park=parkModelScore>=.65f||parkShapeBox.area()>0;
        const auto box=parkModelScore>=.65f?modelBox:parkShapeBox;
        const bool continuous=previousParkBox.area()>0&&std::abs(box.x+box.width/2-previousParkBox.x-previousParkBox.width/2)<100&&
            std::abs(box.y+box.height/2-previousParkBox.y-previousParkBox.height/2)<80&&
            (parkCaptured<=0||ai.captured-parkCaptured<=.30);
        parkCount=park?(continuous?std::min(3,parkCount+1):1):0;
        previousParkBox=park?box:cv::Rect{};parkCaptured=ai.captured;
        // Sign size starts the approach-distance estimate, never an immediate
        // turn. Near road geometry or bounded approach fallback authorizes entry.
        if(parkStageEligible()&&!parkTurnNear){
            const bool close=park&&((box.width>=96&&box.height>=48)||(box.width>=48&&box.height>=64));
            parkNearCount=close?(continuous?std::min(2,parkNearCount+1):1):0;
            if(parkNearCount>=2&&parkCount>=3)parkTurnNear=true;
        }
        // Beijing's switched PARK is the lap-two route landmark. Do not require
        // the image-based Y completion latch: it can remain pending on a bend.
        // Still require three fresh consistent observations and never retrigger
        // after parking or outside an active competition approach.
        if(parkStageEligible()&&parkCount>=3&&!parkingSign){
            stage=Stage::SecondFork;branch=false;branchDirection=0;parkingSign=true;
            reason="LAP2_PARK_SIGN_APPROACH_BEFORE_TURN";
        }
    }
    void start(){if(stage==Stage::Ready){stage=lab?Stage::LabLoop:Stage::FirstFork;reason="ROUTE_STARTED";}}
    void stop(){if(running()){stage=Stage::Fault;reason="OPERATOR_STOP_RESTART_REQUIRED";}}
    void markStart(){if(stage==Stage::LabLoop||stage==Stage::SecondReturn){stage=Stage::Done;reason="MANUAL_FINISH_CONFIRMED";}
        else if(stage==Stage::FirstReturn){stage=Stage::SecondFork;branch=false;branchDirection=0;parkCount=0;reason="MANUAL_LAP1_CONFIRMED";}}
    void parkingStarted(){if(parkingSearch()){stage=Stage::ParkingRun;reason="AUTO_PARKING_ROUTE_ARMED";}}
    void nextParkingBay(const char *why="PLAN_FAILED_FIND_NEXT_BAY"){if(parkingSearch()){stage=Stage::ParkingApproach;branch=false;reason=why;}}
    void parkingFinished(bool enteredBay=true){if(parkingRun()){
        // The maneuver has completed its dwell and reverse, and the driver
        // has confirmed the aisle handoff. Score bay containment separately.
        parkingResult=enteredBay?ParkingResult::ParkedAndReturned:ParkingResult::ReturnedUnverified;
        stage=Stage::ParkingExit;branch=false;branchDirection=0;
        exitArrowCount=0;exitArrowFrame=0;exitArrowConfirmed=false;exitGateCommandM=0;exitSearchCommandM=0;
        lastExitArrowCapture=-1;lastExitArrowBox={};
        reason=enteredBay?"PARKED_DWELLED_AND_RETURNED_TO_AISLE":"BAY_UNCONFIRMED_RETURNED_TO_AISLE";
    }}
    void fault(const std::string &why){stage=Stage::Fault;reason=why;}
    void observe(const RaceSnapshot &ai,const RacePlan &traffic,bool fullMouth,const RaceGroundMap *finishGround=nullptr,float length=.31f) {
        if(!running())return;
        if(stage==Stage::SecondReturn){observeFinishStop(ai,traffic,finishGround,length);return;}
        if(!traffic.fresh)return;
        // Adjacent, parallel bay mouths confirm the acquired parking aisle.
        // Do not wait for its repeated bay dividers to become a straight road,
        // or for a gate that may already have been raised when we entered.
        if(parkingEntrancePending()&&fullMouth&&canScanParkingBays(traffic)) {
            stage=Stage::ParkingSearch;branch=false;branchDirection=0;
            reason="ACQUIRED_ENTRANCE_VISIBLE_BAY_STOP_AND_PLAN";return;
        }
        const bool turning=traffic.selectedBranch;
        const bool finished=!traffic.stop&&!turning&&(branch||traffic.completedDirection!=0)&&
            (std::string(traffic.state)=="LANE"||std::string(traffic.state)=="LIMIT_ZONE");
        if(traffic.completedDirection)branchDirection=traffic.completedDirection;
        if(finished) {
            if(stage==Stage::FirstFork&&(branchDirection==-1||branchDirection==1)){
                stage=Stage::FirstReturn;parkCount=0;reason="LAP1_SIGNED_FORK_COMPLETED";
            }
            else if(stage==Stage::SecondFork&&parkingSign&&branchDirection==-1){stage=Stage::ParkingApproach;reason="PARKING_ENTRANCE_LEFT_COMPLETED";}
            else if(stage==Stage::ParkingExit&&exitTurnAuthorized()&&traffic.completedDirection==-1){
                stage=Stage::SecondReturn;reason="LAP2_PARKING_EXIT_TURN_COMPLETED";
                finishSequence=ai.sequence;finishCapture=ai.captured;
            }
        }
        if(turning){branch=true;branchDirection=traffic.guideDirection;}else if(finished){branch=false;branchDirection=0;}
        if(stage==Stage::ParkingApproach&&fullMouth&&!turning&&!traffic.stop&&!traffic.gateWaiting){stage=Stage::ParkingSearch;reason="VISIBLE_BAY_STOP_AND_PLAN";}
        // A park sign supplies context, but never substitutes for a visible bay or completed turn.
    }
    nlohmann::json json()const{return {{"state",name()},{"reason",reason},{"lab_single_loop",lab},
        {"parking_completed",int(parkingResult==ParkingResult::ParkedAndReturned)},
        {"parking_returned_to_aisle",parkingResult!=ParkingResult::NotReturned},
        {"park_model_score",parkModelScore},{"park_evidence",parkEvidence},{"park_evidence_frames",parkCount},
        {"park_evidence_sequence",sequence},
        {"parking_entrance_near",parkTurnNear},{"parking_near_frames",parkNearCount},
        {"parking_turn_trigger","near_left_opening_or_approach_command_distance_fallback"},
        {"park_shape_box",{parkShapeBox.x,parkShapeBox.y,parkShapeBox.width,parkShapeBox.height}},
        {"park_stage_eligible",parkStageEligible()},
        {"route_policy","beijing_lap1_flow_signed_fork_lap2_parking"},
        {"lap_finish_source",lab?"finish_operator_n":"confirmed_stop_then_crosswalk_clear_or_operator_n"},{"parking_sign_confirmed",parkingSign},
        {"finish_stop_frames",finishStopFrames},{"finish_stop_sequence",finishSequence},
        {"finish_stop_armed",finishStopArmed},{"finish_cross_bottom",finishCrossBox.br().y},
        {"finish_cross_clearance_required_m",clearDistanceM},{"finish_cross_forward_command_m",afterNearCommandM},
        {"finish_cross_search_hold",finishSearchHeld()},
        {"route_direction",routeDirection()},{"parking_exit_arrow_authorized",exitArrowTurnAuthorized()},
        {"parking_exit_turn_authorized",exitTurnAuthorized()},{"exit_gate_command_m",exitGateCommandM},
        {"exit_search_command_m",exitSearchCommandM},{"exit_search_limit_m",exitSearchLimitM},
        {"exit_search_travel_exhausted",exitSearchTravelExhausted()},{"exit_arrow_confirmation_frames",exitArrowCount},
        {"entrance_road_acquired",entranceRoadAcquired},{"bay_search_command_m",baySearchCommandM},
        {"bay_search_command_limit_m",baySearchCommandLimit},{"bay_search_travel_exhausted",baySearchTravelExhausted()},
        {"parking_speed_limited",parkingSpeedLimited()},{"entry_gate_seen",entryGateSeen},{"entry_gate_released",entryGateReleased},
        {"exit_gate_seen",exitGateSeen},{"exit_gate_released",exitGateReleased},{"exit_gate_seen_during_rejoin",exitGateSeenDuringRejoin},
        {"ground_arrows_are_route_constraints_not_global_location",true}};}
};
