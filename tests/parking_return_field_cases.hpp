#pragma once

// The fault snapshot is already lens-corrected. Use the saved calibration for
// the lane pilot and require a forward command on that actual aisle image.
static float parkingReturnLaneSpeed(const fs::path &directory,const RaceMission &mission,const RaceConfig &config,unsigned faultFrame) {
    std::ifstream resultFile(directory/"parking-result.json");nlohmann::json result;resultFile>>result;
    require(result.at("frame")==faultFrame,"Result image is not from the recorded fault frame");
    const auto image=cv::imread((directory/"parking-result-camera.png").string());
    require(image.size()==cv::Size(640,480),"Missing fault-frame camera image");
    TrackviewConfig vehicle;vehicle.load((directory/"trackview-config.json").string(),true);
    RaceGroundCalibration ground;ground.load((directory/"ground-calibration.json").string(),false);
    const float half=ground.roadWidth/2;
    cv::Point2f src[4],dst[4]={{-half,ground.nearDistance},{half,ground.nearDistance},{-half,ground.farDistance},{half,ground.farDistance}};
    for(int i=0;i<4;++i)src[i]={ground.imagePoints[i][0],ground.imagePoints[i][1]};
    Camera camera(cv::getPerspectiveTransform(src,dst));camera.blackRoad=vehicle.roadSurface=="black";
    require(vehicle.binary<0,"This replay requires the recorded automatic threshold");
    struct RestoreCamera {Camera *saved;~RestoreCamera(){Track::camera=saved;}} restore{Track::camera};
    Track::camera=&camera;
    Track track;track.rowCutUp=vehicle.rowCutUp;track.rowCutBottom=vehicle.rowCutBottom;track.handle(camera.binarize(image));
    RaceTaskPilot pilot(vehicle);
    const float limit=mission.speedLimit(result.at("cruise_setting"),config.parkingSpeed);
    const float dt=result.at("control_dt_s");
    pilot.setCurveSpeed(limit);pilot.setDegradedSpeed(result.at("degraded_setting"));pilot.resetControl();
    pilot.update(track.pointsEdgeLeft,track.pointsEdgeRight,dt,0);
    require(pilot.ready&&!pilot.lost&&pilot.alignment.valid,"Recorded aisle image cannot support lane following");
    return pilot.speedFor(std::min(limit,result.at("traffic_speed_cap").get<float>()),dt);
}

// Replay recorded absolute poses and the commands that were actually applied.
// Stop at the original fault. Later stationary frames cannot predict a new path.
static void parkingReturnFieldCheck(const fs::path &directory) {
    RaceConfig config;config.load((directory/"race-config.json").string());
    std::ifstream routeFile(directory/"parking-route.json"),log(directory/"parking-control.jsonl");
    require(bool(routeFile)&&bool(log),"Missing recorded parking return");
    nlohmann::json saved;routeFile>>saved;
    RaceParkingRoute route;route.valid=true;route.entryCurvaturePreview=saved.at("entry_curvature_preview_m");
    for(const auto &p:saved.at("entry"))route.entry.push_back({p.at(0),p.at(1),p.at(2)});
    require(route.entry.size()>=2,"Missing recorded entry route");
    route.exit.assign(route.entry.rbegin(),route.entry.rend());
    RaceParkingSpace space;space.keepRearWithinStart(route.entry.front());
    RaceParkingControl controller(space,config.parkingSpeed);
    RaceParkingAisleHandoff handoff;
    auto mission=parkingMissionAtEntrance(config.parkingExitSearchLimitM);
    RacePlan entrance;entrance.fresh=true;entrance.state="FORK_LEFT";entrance.selectedBranch=true;
    entrance.guideDirection=-1;entrance.laneBlend=1;
    mission.observe(RaceSnapshot{},entrance,true);
    const float handedness=saved.at("canonical_frame_handedness");
    bool started=false;float previousSpeed=0,previousCurvature=0;
    unsigned terminal=0,dwell=0,firstReverse=0,returned=0,oldFault=0;
    std::string line;
    while(std::getline(log,line)) {
        const auto row=nlohmann::json::parse(line);
        if(!started&&!row.at("armed").get<bool>())continue;
        const auto &p=row.at("visual_pose");
        const unsigned frame=row.at("frame");
        if(row.at("state")=="MISSION_FAULT"&&row.at("reason")=="PARK_INCOMPLETE_RETURNED_TO_AISLE")oldFault=frame;
        if(returned) {
            if(oldFault)break;
            continue;
        }
        const double now=row.at("time");
        RaceParkingObservation observation{{p.at(0),p.at(1),p.at(2)},row.at("captured"),frame,
            row.at("motion_epoch"),row.at("motion_valid"),row.at("pose_estimated")};
        if(!started) {
            require(controller.arm(route,observation,now,saved.at("reverse_verified")),"Cannot arm the recorded parking route");
            mission.parkingStarted();started=true;
        }
        // There was clear traffic throughout this maneuver.
        require(row.at("traffic_state")=="LANE","Unexpected traffic stop in recorded maneuver");
        const auto command=controller.update(observation,now,row.at("control_dt_s"),true,
            row.at("task_cruise_limit"),previousSpeed,previousCurvature);
        require(!controller.failed(),"Parking controller failed on the recorded maneuver");
        previousSpeed=row.at("command_speed");
        previousCurvature=handedness*row.at("physical_curvature").get<float>();
        if(!terminal&&(std::string_view(command.state)=="PARK_BRAKE"||
           std::string_view(command.reason)=="ENTRY_TERMINAL_REACHED_STOPPING"))terminal=frame;
        if(std::string_view(command.reason)=="DWELL_COMPLETE_REVERSE_NEXT_FRAME")dwell=frame;
        if(previousSpeed<0&&!firstReverse)firstReverse=frame;
        const bool aiReady=row.at("ai_age_s").get<double>()<=config.maxDetectionAge;
        auto before=handoff;
        using State=RaceParkingAisleHandoff::State;
        if(handoff.update(controller,space,observation,now,true,aiReady)!=State::Ready) {
            require(row.at("state")!="MISSION_FAULT","New logic still holds at the original fault");
            continue;
        }
        require(row.at("state")=="PARK_COMPLETE"||oldFault,"Handoff preceded the recorded aisle confirmation");
        require(controller.terminalRecoveryUsed()&&controller.reverseDistanceEstimate()>0&&
            raceParkingReadyToRejoin(space,observation.pose)&&row.at("lane_ready").get<bool>(),
            "Missing confirmed aisle return");
        // Interrupt the final confirmation exactly where the good input would release it.
        auto checkWait=[&](RaceParkingObservation bad,bool clear,bool ready) {
            auto interrupted=before;
            require(interrupted.update(controller,space,bad,now,clear,ready)!=State::Ready,"Invalid observation released aisle handoff");
            auto next=observation;++next.sequence;next.captured+=.08;
            require(interrupted.update(controller,space,next,now+.08,true,true)!=State::Ready,"Interrupted confirmation reused its old timer");
        };
        auto stale=observation;stale.captured=now-1;checkWait(stale,true,true);
        auto estimated=observation;estimated.estimated=true;checkWait(estimated,true,true);
        checkWait(observation,false,true);checkWait(observation,true,false);
        mission.parkingFinished(!controller.terminalRecoveryUsed());
        require(mission.running()&&std::string_view(mission.name())=="PARKING_EXIT_LEFT"&&
            mission.followParkingAisle()&&!mission.exitTurnAuthorized()&&mission.json().at("parking_completed")==0,
            "Unverified parking failed to transfer to the guarded exit search");
        returned=frame;
        if(oldFault)break;
    }
    require(terminal&&terminal<dwell&&dwell<firstReverse&&firstReverse<returned&&returned<=oldFault,
        "Missing terminal dwell, reverse or safe aisle handoff by the recorded fault");
    const auto laneSpeed=parkingReturnLaneSpeed(directory,mission,config,oldFault);
    require(laneSpeed>0&&laneSpeed<=config.parkingSpeed,"No bounded forward command on the original stop image");
    std::cout<<"PASS parking return "<<directory.filename().string()<<" terminal_stop="<<terminal<<" dwell_complete="<<dwell
        <<" reverse_start="<<firstReverse<<" handoff="<<returned<<" old_fault="<<oldFault
        <<" parking_completed=0 aisle_image_speed_mps="<<laneSpeed<<"; offline telemetry and image replay\n";
}
