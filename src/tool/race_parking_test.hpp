#pragma once
#include <nlohmann/json.hpp>
#include <cstdint>
#include <string>
#include <vector>

// Reports reached physical-control stages, never a percentage inferred from a
// successful planner. Motion and containment still come from the controller.
class RaceParkingTestReport {
    struct Attempt {
        int id;
        bool route=false,started=false,entry=false,inside=false,dwell=false,reverse=false,returned=false,lane=false;
        uint64_t movingFrames=0,steeringWaitFrames=0;
        std::string outcome="WAITING",reason="WAIT_ROUTE";
    };
    std::vector<Attempt> attempts{{1}};
    int nextId=2;
public:
    void reset() {
        auto &a=attempts.back();
        if(a.outcome=="RUNNING"||a.outcome=="WAITING")a.outcome="RESET";
        attempts.push_back({nextId++});
        if(attempts.size()>32)attempts.erase(attempts.begin());
    }
    void planned(bool valid,const std::string &reason) {auto &a=attempts.back();a.route=valid;a.reason=reason;}
    void start() {auto &a=attempts.back();a.started=true;a.outcome="RUNNING";a.reason="STARTED";}
    void startBlocked(const std::string &reason) {if(!attempts.back().started)attempts.back().reason=reason;}
    void observe(const std::string &state,const std::string &reason,float speed,const std::string &actuator) {
        auto &a=attempts.back();
        if(!a.started)return;
        a.movingFrames+=speed!=0;
        a.steeringWaitFrames+=actuator=="STEERING_SLEW_WAIT";
        a.entry|=state=="PARK_ENTRY"&&speed>0;
        a.inside|=state=="PARK_DWELL";
        a.dwell|=state=="PARK_EXIT";
        a.reverse|=state=="PARK_EXIT"&&speed<0;
        a.returned|=state=="PARK_COMPLETE";
        a.lane|=state=="PARK_REJOIN_LANE"&&speed>0;
        if(a.returned)a.outcome="COMPLETE";
        if(state=="PARK_FAULT")a.outcome="FAULT";
        a.reason=reason;
    }
    void finish(const std::string &reason) {
        auto &a=attempts.back();
        if(a.outcome=="RUNNING"){a.outcome="INTERRUPTED";a.reason=reason;}
        else if(a.outcome=="WAITING")a.outcome="NOT_STARTED";
    }
    std::string progress()const {
        const auto &a=attempts.back();
        auto flag=[](bool ok){return ok?"OK":"--";};
        return std::string("PLAN ")+flag(a.route)+" | IN "+flag(a.inside)+" | 3S "+flag(a.dwell)+" | OUT "+flag(a.returned);
    }
    nlohmann::json toJson()const {
        auto list=nlohmann::json::array();
        for(const auto &a:attempts)list.push_back({{"attempt",a.id},{"route_generated",a.route},{"started",a.started},
            {"entry_motion_commanded",a.entry},{"full_body_inside_observed",a.inside},{"three_second_dwell_confirmed",a.dwell},
            {"reverse_motion_commanded",a.reverse},{"returned_to_aisle_observed",a.returned},{"lane_motion_resumed",a.lane},
            {"moving_command_frames",a.movingFrames},{"steering_wait_frames",a.steeringWaitFrames},
            {"outcome",a.outcome},{"reason",a.reason}});
        return {{"version",1},{"scope","parking aisle entry, 3s dwell, reverse exit and lane rejoin"},
            {"entrance_and_exit_arrow_turns_included",false},{"independent_physical_measurement",false},
            {"stages_based_on","live controller and visual odometry; motion commands alone are not physical completion"},
            {"retained_attempts",list}};
    }
};
