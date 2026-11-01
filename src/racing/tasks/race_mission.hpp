#pragma once
#include "race_ground_arrow.hpp"
// Route stages are explicit; repeated arrows cannot identify a lap or finish.
// n at the physical start is the user's currently selected lap/finish landmark.
class RaceMission {
public:
    enum class Stage {Ready,LabLoop,FirstFork,ParkingApproach,ParkingSearch,ParkingRun,ParkingExit,FirstReturn,SecondFork,SecondReturn,Done,Fault};
private:
    Stage stage=Stage::Ready;bool lab=false,branch=false;int branchDirection=0;uint64_t sequence=0;int parkCount=0;
    std::string reason="PRESS_A_START";int completedParking=0;bool parkingSign=false;
    int exitArrowCount=0;uint64_t exitArrowFrame=0;bool exitArrowConfirmed=false;
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
    // FLOW and Y signs belong to the local traffic manager. A permanent
    // straight override would suppress the later left/right Y sign.
    int routeDirection()const{return (stage==Stage::SecondFork&&parkingSign)||stage==Stage::ParkingExit?-1:0;}
    void observeSigns(const RaceSnapshot &ai,bool fresh) {
        if(!running()||lab||!fresh||ai.sequence<=sequence)return;sequence=ai.sequence;
        bool park=false;
        for(auto d:ai.detections)if(d.score>=.65f) {
            park|=d.type==LABEL_PARK;
        }
        parkCount=park?std::min(3,parkCount+1):0;
        // Beijing: FLOW on lap one, PARK on lap two. After completing the
        // signed Y fork, the newly switched PARK is route evidence for lap two;
        // it is not a global position measurement. n at the start also works.
        if((stage==Stage::FirstReturn||stage==Stage::SecondFork)&&parkCount>=3&&!parkingSign){
            stage=Stage::SecondFork;branch=false;branchDirection=0;parkingSign=true;
            reason="LAP2_PARK_SIGN_TURN_LEFT_NOW";
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
        const bool finished=branch&&!turning&&(std::string(traffic.state)=="LANE"||std::string(traffic.state)=="LIMIT_ZONE");
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
        {"route_policy","beijing_lap1_flow_signed_fork_lap2_parking"},
        {"lap_finish_source","after_signed_fork_new_park_or_operator_n; finish_operator_n"},{"parking_sign_confirmed",parkingSign},
        {"route_direction",routeDirection()},{"parking_exit_arrow_authorized",exitArrowTurnAuthorized()},
        {"parking_speed_limited",parkingSpeedLimited()},{"entry_gate_seen",entryGateSeen},{"entry_gate_released",entryGateReleased},
        {"exit_gate_seen",exitGateSeen},{"exit_gate_released",exitGateReleased},
        {"ground_arrows_are_route_constraints_not_global_location",true}};}
};
