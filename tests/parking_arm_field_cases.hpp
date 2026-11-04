#pragma once

static void parkingArmFieldChecks(const fs::path &dir) {
    nlohmann::json saved,teaching,summary;
    std::ifstream(dir/"parking-route.json")>>saved;
    std::ifstream(dir/"parking-teaching-attempt.json")>>teaching;
    std::ifstream(dir/"parking-test-summary.json")>>summary;
    const auto &attempt=summary.at("retained_attempts").at(0);
    require(teaching.at("frame")==749&&teaching.at("automatic")==true&&
        teaching.at("planner_reason")=="MODEL_CHECKED_FORWARD_ROUTE"&&
        attempt.at("route_generated")==true&&attempt.at("started")==false&&
        attempt.at("reason")=="PARK_START_BODY_OUTSIDE_SPACE","Wrong blocked automatic parking field run");
    RaceParkingRoute route;
    require(saved.at("reverse_verified")==true&&saved.at("parking_side")=="left",
        "Missing reverse verification or wrong recorded bay");
    route.valid=true;
    route.entryCurvaturePreview=saved.at("entry_curvature_preview_m");
    for(const auto &p:saved.at("entry"))route.entry.push_back({p.at(0),p.at(1),p.at(2)});
    require(route.entry.size()>=3,"Incomplete parking route");
    route.exit.assign(route.entry.rbegin(),route.entry.rend());
    const auto start=route.entry.front();
    RaceParkingSpace space;
    space.keepRearWithinStart(start);
    require(std::abs(space.rearLimit()-saved.at("rear_boundary_m").get<float>())<.00001f,
        "Incorrect reconstructed rear boundary");

    const auto readFrames=[&](const char *name) {
        std::ifstream input(dir/name);
        std::vector<nlohmann::json> rows;
        for(std::string line;std::getline(input,line);) {
            auto row=nlohmann::json::parse(line);
            const int frame=row.at("frame");
            if(frame>=750&&frame<=752)rows.push_back(std::move(row));
            if(frame>=752)break;
        }
        require(rows.size()==3&&rows[0].at("frame")==750&&rows[1].at("frame")==751&&rows[2].at("frame")==752,
            "Missing consecutive parking start field frames");
        return rows;
    };
    const auto mission=readFrames("mission-state.jsonl");
    const auto control=readFrames("parking-control.jsonl");
    require(mission[0].at("traffic_reason")=="AI_STALE"&&mission[0].at("speed_command")==0,
        "Wrong pre-start traffic hold");
    for(size_t i=1;i<control.size();++i) {
        require(mission[i].at("state")=="PARKING_FIND_BAY"&&mission[i].at("no_control")==false&&
            mission[i].at("traffic_reason")=="PARKING_AISLE_CURRENT_ROAD"&&mission[i].at("control_timeout")==false&&
            control[i].at("pose_source")=="VISUAL_ODOMETRY"&&control[i].at("armed")==false&&
            control[i].at("command_speed")==0&&control[i].at("reason")=="PARK_START_BODY_OUTSIDE_SPACE",
            "Expected a fresh stopped field observation after traffic cleared");
    }
    const auto observation=[](const nlohmann::json &row) {
        const auto &p=row.at("visual_pose");
        return RaceParkingObservation{{p.at(0),p.at(1),p.at(2)},row.at("captured"),row.at("frame"),
            row.at("motion_epoch"),row.at("motion_valid"),row.at("motion_estimated_origin")};
    };
    const auto first=observation(control[1]),next=observation(control[2]);
    float rear=std::numeric_limits<float>::infinity();
    for(auto corner:space.body(first.pose)) {
        require(std::abs(corner.x)<space.dimensions().aisle/2,"Recorded body touches a physical aisle edge");
        rear=std::min(rear,corner.y);
    }
    const float rearError=space.rearLimit()-rear;
    require(rearError>.013f&&rearError<.014f&&!space.free(first.pose),
        "Recorded pose no longer reproduces the old 10 mm virtual rear-plane rejection");

    const float cruise=control[1].at("task_cruise_limit");
    RaceParkingControl controller(space,cruise);
    const double armTime=control[1].at("time"),nextTime=control[2].at("time");
    const float dt=control[2].at("control_dt_s");
    const float limit=std::min(cruise,control[2].at("traffic_speed_cap").get<float>());
    require(controller.arm(route,first,armTime,true),"Field frame 751 cannot arm the checked route");
    auto paused=controller;
    const auto hold=paused.update(next,nextTime,dt,false,limit,0,0);
    require(hold.speed==0&&std::string_view(hold.reason)=="TRAFFIC_STOP_PRIORITY",
        "Traffic stop permits a parking start command");
    auto visuallyUnconfirmed=controller;
    auto relative=next;relative.estimated=true;
    require(visuallyUnconfirmed.update(relative,nextTime,dt,true,limit,0,0).speed==0,
        "Estimated motion permits the initial forward correction");
    auto wheelsTurning=controller;
    require(wheelsTurning.update(next,nextTime,dt,true,limit,0,1.f).speed==0,
        "Initial correction drives before applied steering returns to centre");
    const auto same=controller.update(first,armTime,control[1].at("control_dt_s"),true,limit,0,0);
    require(same.speed==0&&std::string_view(same.reason)=="WAIT_NEW_MOTION_FRAME",
        "Parking moves before a second visual frame");
    const auto cmd=controller.update(next,nextTime,dt,true,limit,0,0);
    require(std::string_view(cmd.state)=="PARK_ENTRY"&&std::string_view(cmd.reason)=="FORWARD_TO_PLANNED_START"&&
        cmd.speed>0&&cmd.speed<=limit&&cmd.curvature==0&&
        !cmd.entryMouthRecovery&&!cmd.predictedBoundaryWarning,
        "Fresh field frame 752 cannot begin a contained forward entry");

    // Only frames through the first changed command are real. Step the subsequent
    // motion with the controller's model; archived later poses belong to a stopped car.
    constexpr float modelDt=.08f;
    auto model=next;
    auto movement=cmd;
    float modelTravel=0;
    int normalEntrySteps=0,joinSteps=0;
    for(int step=0;step<24&&normalEntrySteps<4;++step) {
        const float ds=movement.speed*modelDt;
        modelTravel+=ds;
        model.pose=raceParkingAdvance(model.pose,ds,movement.curvature,space.dimensions());
        ++model.sequence;model.captured+=modelDt;
        movement=controller.update(model,nextTime+(step+1)*modelDt,modelDt,true,limit,
            movement.speed,movement.curvature);
        require(std::string_view(movement.state)=="PARK_ENTRY"&&movement.speed>0,
            "Model stops before completing the startup transition");
        if(std::string_view(movement.reason)=="FORWARD_TO_PLANNED_START") {
            require(!space.free(model.pose)&&movement.curvature==0&&movement.speed<=.04f&&normalEntrySteps==0,
                "Model re-entered startup correction or moved too fast");
            ++joinSteps;
        } else {
            require(space.free(model.pose)&&!movement.predictedBoundaryWarning,
                "Model resumed normal entry outside the strict planning bounds");
            ++normalEntrySteps;
        }
    }
    require(joinSteps>0&&normalEntrySteps==4,"Model did not sustain normal entry after passing the rear plane");

    RaceParkingControl ready(space,cruise);
    const auto blocked=[&](const RaceParkingRoute &r,const RaceParkingObservation &o,double now,bool reverse,const char *why) {
        const char *reason=ready.armBlockReason(r,o,now,reverse);
        require(reason&&std::string_view(reason)==why&&!ready.arm(r,o,now,reverse),why);
    };
    blocked(route,first,armTime,false,"PARK_START_REVERSE_UNVERIFIED");
    blocked(route,first,first.captured+.20,true,"PARK_START_WAIT_FRESH_POSE");
    auto unavailable=first;unavailable.valid=false;
    blocked(route,unavailable,armTime,true,"PARK_START_WAIT_FRESH_POSE");
    auto estimated=first;estimated.estimated=true;
    blocked(route,estimated,armTime,true,"PARK_START_WAIT_FRESH_POSE");
    auto moved=first;moved.pose.z=start.z-.03f;
    blocked(route,moved,armTime,true,"PARK_START_POSE_MOVED_REPLAN");
    auto turned=first;turned.pose.yaw=start.yaw+.08f;
    blocked(route,turned,armTime,true,"PARK_START_POSE_MOVED_REPLAN");
    // A 20 mm and 0.05 rad perturbation is outside both old pose tolerances.
    // Use a coarse 150 ms interval so the model crosses the virtual plane
    // between camera frames; strict aisle clearance still governs handover.
    auto offset=first;
    offset.pose=start;offset.pose.z-=.020f;offset.pose.yaw+=.05f;
    RaceParkingControl wider(space,cruise);
    require(!space.freeStartObserved(offset.pose)&&space.insideAisle(offset.pose)&&wider.arm(route,offset,armTime,true),
        "Contained 20 mm start discrepancy or 0.05 rad heading discrepancy still blocks");
    constexpr float coarseDt=.15f;
    RaceParkingCommand moving;
    bool widerJoined=false;
    for(int step=1;step<=12&&!widerJoined;++step) {
        offset.pose=raceParkingAdvance(offset.pose,moving.speed*coarseDt,moving.curvature,space.dimensions());
        ++offset.sequence;offset.captured+=coarseDt;
        moving=wider.update(offset,armTime+step*coarseDt,coarseDt,true,limit,moving.speed,moving.curvature);
        require(std::string_view(moving.state)=="PARK_ENTRY"&&moving.speed>0&&space.insideAisle(offset.pose),
            "Enlarged start window permits arming but fails to rejoin safely");
        widerJoined=std::string_view(moving.reason)!="FORWARD_TO_PLANNED_START";
    }
    require(widerJoined&&space.free(offset.pose),"Perturbed start never rejoined original strict planning space");

    // Park the reference just inside either side of the aisle before the bay.
    // The small observed lateral shift then crosses real paint within the start pose window.
    for(float side:{-1.f,1.f}) {
        float outer=-std::numeric_limits<float>::infinity();
        for(auto p:space.body(start))outer=std::max(outer,side*p.x);
        auto nearEdge=route;
        nearEdge.entry.front().x+=side*(space.dimensions().aisle/2-.003f-outer);
        nearEdge.exit.back()=nearEdge.entry.front();
        require(space.free(nearEdge.entry.front()),"Synthetic edge reference already outside aisle");
        auto crossing=first;crossing.pose=nearEdge.entry.front();crossing.pose.x+=side*.008f;
        crossing.pose.z-=.011f;
        blocked(nearEdge,crossing,armTime,true,"PARK_START_BODY_OUTSIDE_SPACE");
    }
    // The live-pose exception must never be used to validate route geometry.
    auto unsafe=route;
    unsafe.entry[1]=first.pose;unsafe.exit[unsafe.exit.size()-2]=unsafe.entry[1];
    blocked(unsafe,first,armTime,true,"PARK_START_ROUTE_OUTSIDE_SPACE");
    unsafe=route;unsafe.exit.front().x+=.03f;
    blocked(unsafe,first,armTime,true,"PARK_START_REVERSE_ROUTE_MISMATCH");
    std::cout<<"PASS parking arm field "<<dir.filename().string()<<" arm_frame="<<first.sequence
             <<" first_forward_frame="<<next.sequence<<" old_rear_deficit_mm="<<1000*rearError
             <<" replay_command_mps="<<cmd.speed<<" modeled_travel_m="<<modelTravel
             <<" modeled_normal_entry_steps="<<normalEntrySteps<<" physical retest pending\n";
}
