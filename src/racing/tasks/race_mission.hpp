#pragma once
#include "race_ground_arrow.hpp"
#include "race_parking_sign.hpp"
// Route stages are explicit; repeated arrows cannot identify a lap or finish.
// n at the physical start is the user's currently selected lap/finish landmark.
class RaceMission {
public:
    enum class Stage {Ready,LabLoop,FirstFork,ParkingApproach,ParkingSearch,ParkingRun,ParkingExit,FirstReturn,SecondFork,SecondReturn,Done,Fault};
private:
    Stage stage=Stage::Ready;bool lab=false,branch=false;int branchDirection=0;uint64_t sequence=0;int parkCount=0;
    std::string reason="PRESS_A_START";int completedParking=0;bool parkingSign=false;
    int exitArrowCount=0;uint64_t exitArrowFrame=0;bool exitArrowConfirmed=false;
    float parkModelScore=0;cv::Rect parkShapeBox;const char *parkEvidence="NONE";
    double parkCaptured=0;cv::Rect previousParkBox;
    int parkNearCount=0;bool parkTurnNear=false;
    bool entryGateSeen=false,entryGateReleased=false,exitGateSeen=false,exitGateReleased=false;
public:
    explicit RaceMission(bool laboratory=false):lab(laboratory){}
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
    bool exitArrowTurnAuthorized()const{return stage==Stage::ParkingExit&&exitArrowConfirmed;}
    void observeExitArrow(const RaceGroundArrow &arrow,uint64_t frame,bool fresh){
        if(stage!=Stage::ParkingExit||lab||!fresh||frame<=exitArrowFrame)return;
        exitArrowFrame=frame;
        const bool left=(arrow.allowed&RoadLeft)&&!(arrow.allowed&(RoadRight|RoadStraight));
        exitArrowCount=left?std::min(3,exitArrowCount+1):0;
        if(exitArrowCount>=3){exitArrowConfirmed=true;reason="PARKING_EXIT_GROUND_LEFT_TURN_AUTHORIZED";}
    }
    bool parkingZone()const{return (stage==Stage::SecondFork&&parkingSign)||stage==Stage::ParkingApproach||parkingSearch()||parkingRun()||stage==Stage::ParkingExit;}
    bool parkingApproach()const{return stage==Stage::ParkingApproach||
        (stage==Stage::SecondFork&&parkingSign)||
        ((stage==Stage::FirstReturn||stage==Stage::SecondFork)&&parkCount>0);}
    bool parkingSpeedLimited()const{return !lab&&running()&&!exitGateReleased&&
        (parkingZone()||parkingApproach()||completedParking>0);}
    float speedLimit(float cruise,float parkingSpeed)const{return parkingSpeedLimited()?std::min(cruise,parkingSpeed):cruise;}
    void observeGate(const RacePlan &traffic) {
        if(!running()||lab||!traffic.fresh)return;
        if(completedParking>0) {
            if(traffic.gateWaiting)exitGateSeen=true;
            if(exitGateSeen&&traffic.gateOpened){exitGateReleased=true;reason="EXIT_GATE_OPEN_RESTORE_CRUISE";}
        } else if(parkingZone()||parkingApproach()) {
            if(traffic.gateWaiting)entryGateSeen=true;
            if(entryGateSeen&&traffic.gateOpened){entryGateReleased=true;reason="ENTRY_GATE_OPEN_PARKING_AREA_SPEED";}
        }
    }
    bool complete()const{return stage==Stage::Done;}
    bool failed()const{return stage==Stage::Fault;}
    bool parkStageEligible()const{return !lab&&completedParking==0&&
        (stage==Stage::FirstFork||stage==Stage::FirstReturn||stage==Stage::SecondFork);}
    // FLOW and Y signs belong to the local traffic manager. A permanent
    // straight override would suppress the later left/right Y sign.
    int routeDirection()const{return (stage==Stage::SecondFork&&parkingSign)||stage==Stage::ParkingExit?-1:0;}
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
        // Recognition supplies route intent, not a measured turn location.
        // Native640 experimental proximity cue: the field's 66x34 distant
        // plate must not immediately command full left lock.
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
    void nextParkingBay(){if(parkingSearch()){stage=Stage::ParkingApproach;branch=false;reason="PLAN_FAILED_FIND_NEXT_BAY";}}
    void parkingFinished(){if(parkingRun()){stage=Stage::ParkingExit;++completedParking;exitArrowCount=0;exitArrowFrame=0;exitArrowConfirmed=false;reason="PARKED_DWELLED_AND_RETURNED_TO_AISLE";}}
    void fault(const std::string &why){stage=Stage::Fault;reason=why;}
    void observe(const RaceSnapshot &ai,const RacePlan &traffic,bool fullMouth) {
        if(!running()||!traffic.fresh)return;
        const bool turning=traffic.selectedBranch;
        const bool finished=!traffic.stop&&!turning&&(branch||traffic.completedDirection!=0)&&
            (std::string(traffic.state)=="LANE"||std::string(traffic.state)=="LIMIT_ZONE");
        if(traffic.completedDirection)branchDirection=traffic.completedDirection;
        if(finished) {
            if(stage==Stage::FirstFork&&(branchDirection==-1||branchDirection==1)){
                stage=Stage::FirstReturn;parkCount=0;reason="LAP1_SIGNED_FORK_COMPLETED";
            }
            else if(stage==Stage::SecondFork&&parkingSign&&branchDirection==-1){stage=Stage::ParkingApproach;reason="PARKING_ENTRANCE_LEFT_COMPLETED";}
            else if(stage==Stage::ParkingExit&&branchDirection==-1){stage=Stage::SecondReturn;reason="LAP2_PARKING_EXIT_TURN_COMPLETED";}
        }
        if(turning){branch=true;branchDirection=traffic.guideDirection;}else if(finished){branch=false;branchDirection=0;}
        if(stage==Stage::ParkingApproach&&fullMouth&&!turning&&!traffic.stop){stage=Stage::ParkingSearch;reason="VISIBLE_BAY_STOP_AND_PLAN";}
        // A park sign supplies context, but never substitutes for a visible bay or completed turn.
    }
    nlohmann::json json()const{return {{"state",name()},{"reason",reason},{"lab_single_loop",lab},{"parking_completed",completedParking},
        {"park_model_score",parkModelScore},{"park_evidence",parkEvidence},{"park_evidence_frames",parkCount},
        {"park_evidence_sequence",sequence},
        {"parking_entrance_near",parkTurnNear},{"parking_near_frames",parkNearCount},
        {"park_shape_box",{parkShapeBox.x,parkShapeBox.y,parkShapeBox.width,parkShapeBox.height}},
        {"park_stage_eligible",parkStageEligible()},
        {"route_policy","beijing_lap1_flow_signed_fork_lap2_parking"},
        {"lap_finish_source","confirmed_switched_park_independent_of_fork_completion; finish_operator_n"},{"parking_sign_confirmed",parkingSign},
        {"route_direction",routeDirection()},{"parking_exit_arrow_authorized",exitArrowTurnAuthorized()},
        {"parking_speed_limited",parkingSpeedLimited()},{"entry_gate_seen",entryGateSeen},{"entry_gate_released",entryGateReleased},
        {"exit_gate_seen",exitGateSeen},{"exit_gate_released",exitGateReleased},
        {"ground_arrows_are_route_constraints_not_global_location",true}};}
};
