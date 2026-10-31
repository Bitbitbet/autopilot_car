#pragma once
#include "race_ground_arrow.hpp"
// Route stages are explicit; repeated arrows cannot identify a lap or finish.
// n at the physical start is the user's currently selected lap/finish landmark.
class RaceMission {
public:
    enum class Stage {Ready,LabLoop,FirstFork,ParkingApproach,ParkingSearch,ParkingRun,ParkingExit,FirstReturn,SecondFork,SecondReturn,Done,Fault};
private:
    Stage stage=Stage::Ready;bool lab=false,branch=false;uint64_t sequence=0;int parkCount=0;
    std::string reason="PRESS_A_START";int completedParking=0;bool parkingSign=false;int flowCount=0;
public:
    explicit RaceMission(bool laboratory=false):lab(laboratory){}
    const char *name()const {
        switch(stage){case Stage::Ready:return "MISSION_READY";case Stage::LabLoop:return "LAB_SINGLE_LOOP";
        case Stage::FirstFork:return parkingSign?"LAP1_LEFT_TO_PARKING":"LAP1_WAIT_PARK_SIGN";case Stage::ParkingApproach:return "PARKING_APPROACH";
        case Stage::ParkingSearch:return "PARKING_FIND_BAY";case Stage::ParkingRun:return "PARKING_MANEUVER";
        case Stage::ParkingExit:return "PARKING_EXIT_LEFT";case Stage::FirstReturn:return "LAP1_RETURN_PRESS_N_AT_START";
        case Stage::SecondFork:return "LAP2_STRAIGHT_OUTER";case Stage::SecondReturn:return "LAP2_RETURN_PRESS_N_TO_FINISH";
        case Stage::Done:return "MISSION_COMPLETE";default:return "MISSION_FAULT";}
    }
    bool running()const{return stage!=Stage::Ready&&stage!=Stage::Done&&stage!=Stage::Fault;}
    bool parkingSearch()const{return stage==Stage::ParkingSearch;}
    bool parkingRun()const{return stage==Stage::ParkingRun;}
    bool parkingZone()const{return (stage==Stage::FirstFork&&parkingSign)||stage==Stage::ParkingApproach||parkingSearch()||parkingRun()||stage==Stage::ParkingExit;}
    bool parkingApproach()const{return stage==Stage::ParkingApproach||(stage==Stage::FirstFork&&parkCount>0);}
    bool complete()const{return stage==Stage::Done;}
    bool failed()const{return stage==Stage::Fault;}
    int routeDirection()const{return stage==Stage::FirstFork?(parkingSign?-1:2):stage==Stage::ParkingExit?-1:stage==Stage::SecondFork?2:0;}
    void observeSigns(const RaceSnapshot &ai,bool fresh) {
        if(!running()||lab||!fresh||ai.sequence<=sequence)return;sequence=ai.sequence;
        bool park=false,flow=false,stop=false;
        for(auto d:ai.detections)if(d.score>=.65f) {
            park|=d.type==LABEL_PARK;flow|=d.type==LABEL_FLOW;stop|=d.type==LABEL_STOP;
        }
        parkCount=park?std::min(3,parkCount+1):0;
        flowCount=flow&&!stop?std::min(3,flowCount+1):0;
        if(stage==Stage::FirstFork&&parkCount>=3){parkingSign=true;reason="PARK_SIGN_TURN_LEFT_NOW";}
        // The same start FLOW sign cannot skip the parking task. Only after
        // parking and the exit turn can a new FLOW confirm the outer-lap route.
        if(stage==Stage::FirstReturn&&flowCount>=3){stage=Stage::SecondFork;branch=false;reason="POST_PARK_FLOW_STRAIGHT_OUTER";}
    }
    void start(){if(stage==Stage::Ready){stage=lab?Stage::LabLoop:Stage::FirstFork;reason="ROUTE_STARTED";}}
    void stop(){if(running()){stage=Stage::Fault;reason="OPERATOR_STOP_RESTART_REQUIRED";}}
    void markStart(){if(stage==Stage::LabLoop||stage==Stage::SecondReturn){stage=Stage::Done;reason="MANUAL_FINISH_CONFIRMED";}
        else if(stage==Stage::FirstReturn){stage=Stage::SecondFork;reason="MANUAL_LAP1_CONFIRMED";}}
    void parkingStarted(){if(parkingSearch()){stage=Stage::ParkingRun;reason="AUTO_PARKING_ROUTE_ARMED";}}
    void nextParkingBay(){if(parkingSearch()){stage=Stage::ParkingApproach;branch=false;reason="PLAN_FAILED_FIND_NEXT_BAY";}}
    void parkingFinished(){if(parkingRun()){stage=Stage::ParkingExit;++completedParking;reason="PARKED_DWELLED_AND_RETURNED_TO_AISLE";}}
    void fault(const std::string &why){stage=Stage::Fault;reason=why;}
    void observe(const RaceSnapshot &ai,const RacePlan &traffic,bool fullMouth) {
        if(!running()||!traffic.fresh)return;
        const bool turning=traffic.selectedBranch;
        const bool finished=branch&&!turning&&(std::string(traffic.state)=="LANE"||std::string(traffic.state)=="LIMIT_ZONE");
        if(finished) {
            if(stage==Stage::FirstFork&&parkingSign){stage=Stage::ParkingApproach;reason="FIRST_FORK_COMPLETED";}
            else if(stage==Stage::ParkingExit){stage=Stage::FirstReturn;reason="PARKING_EXIT_TURN_COMPLETED";}
            else if(stage==Stage::SecondFork){stage=Stage::SecondReturn;reason="OUTER_ROUTE_SELECTED";}
        }
        if(turning)branch=true;else if(finished)branch=false;
        if(stage==Stage::ParkingApproach&&fullMouth&&!turning&&!traffic.stop){stage=Stage::ParkingSearch;reason="VISIBLE_BAY_STOP_AND_PLAN";}
        // A park sign supplies context, but never substitutes for a visible bay or completed turn.
    }
    nlohmann::json json()const{return {{"state",name()},{"reason",reason},{"lab_single_loop",lab},{"parking_completed",completedParking},
        {"lap_finish_source","post_parking_flow_or_operator_n; finish_operator_n"},{"parking_sign_confirmed",parkingSign},
        {"route_direction",routeDirection()},{"ground_arrows_are_route_constraints_not_global_location",true}};}
};
