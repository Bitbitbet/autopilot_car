#pragma once

static void parkingGateTurnFieldChecks(const fs::path &root,const fs::path &latest) {
    const auto turnRun=root/"vehicle_evidence/mission-1793786826705";
    const auto frames=[](const fs::path &file,int begin,int end) {
        std::map<int,nlohmann::json> rows;std::ifstream input(file);
        for(std::string line;std::getline(input,line);) {
            auto r=nlohmann::json::parse(line);const int frame=r.at("frame");
            if(frame>=begin)rows.emplace(frame,std::move(r));
            if(frame>=end)break;
        }
        require(int(rows.size())==end-begin+1,"Incomplete gate/turn field telemetry");
        return rows;
    };
    const auto control=frames(latest/"parking-control.jsonl",580,969);
    const auto missionLog=frames(latest/"mission-state.jsonl",580,969);
    const auto turns=frames(turnRun/"mission-state.jsonl",924,973);
    const auto turnControl=frames(turnRun/"parking-control.jsonl",924,973);
    const auto photo=[](const fs::path &name) {
        auto image=cv::imread(name.string());require(image.size()==cv::Size(640,480),"Missing field image");
        return image;
    };
    const auto closed=photo(latest/"parking-result-camera.png");
    const auto atTurn=photo(turnRun/"parking-exit-decision/frame-925.png");
    const auto onCurve=photo(turnRun/"parking-result-camera.png");
    nlohmann::json curveMeta;
    std::ifstream(turnRun/"parking-result.json")>>curveMeta;
    require(curveMeta.at("frame")==961&&curveMeta.at("reason")=="EXIT_ROAD_NOT_REACQUIRED_WITHIN_TRAVEL",
        "Exit curve image does not correspond to the recorded stop");
    RaceConfig cfg;cfg.load((latest/"race-config.json").string());
    RaceGroundCalibration ground;ground.load((turnRun/"ground-calibration.json").string(),false);
    cv::Point2f source[4],target[4]={{-ground.roadWidth/2,ground.nearDistance},{ground.roadWidth/2,ground.nearDistance},
        {-ground.roadWidth/2,ground.farDistance},{ground.roadWidth/2,ground.farDistance}};
    for(size_t i=0;i<4;++i)source[i]={ground.imagePoints[i][0],ground.imagePoints[i][1]};
    Camera camera(cv::getPerspectiveTransform(source,target));camera.blackRoad=true;
    struct RestoreCamera {Camera *old;~RestoreCamera(){Track::camera=old;}} restore{Track::camera};
    Track::camera=&camera;
    Track track;
    // The stopped frame is real. No camera recording was retained while the bar
    // was raised, so expose the underlying aisle with pavement sampled below it.
    // Cover the high, former false ROI to distinguish a fixed passage from the old moving one.
    auto raised=closed.clone();
    closed(cv::Rect(60,280,366,82)).copyTo(raised(cv::Rect(60,180,366,82)));
    cv::rectangle(raised,cv::Rect(70,40,175,76),cv::Scalar(225,225,225),-1);
    const auto road=camera.binarize(raised),bar=camera.binarize(closed);
    track.handle(road);
    const cv::Rect oldRegion{85,46,148,64};
    require(cv::countNonZero(road(oldRegion))<oldRegion.area()*.5,"Old displaced ROI unexpectedly contains road");
    const auto tick=[&](RaceManager &manager,const nlohmann::json &r,const cv::Mat &image,const cv::Mat &bin) {
        const double now=r.at("time");
        RaceSnapshot s;s.image=image;s.sequence=r.at("ai_sequence");s.captured=now-r.at("ai_age_s").get<double>();
        for(const auto &d:r.at("ai_detections")) {
            const auto &b=d.at("box");auto detection=box(d.at("type"),b[0],b[1],b[2],b[3]);
            detection.score=d.at("score");s.detections.push_back(detection);
        }
        return manager.update(s,now,r.at("control_dt_s"),true,bin,track,image,true,true);
    };
    RaceManager entry(cfg);
    int firstEntryStop=0,entryOpen=0;
    for(int f=580;f<=630;++f) {
        const bool clear=f>=608;
        const auto p=tick(entry,control.at(f),clear?raised:closed,clear?road:bar);
        if(!firstEntryStop&&p.gateWaiting&&p.stop)firstEntryStop=f;
        if(p.gateOpened){entryOpen=f;break;}
    }
    require(missionLog.at(589).at("gate_waiting")==true&&missionLog.at(623).at("gate_opened")==true&&
        firstEntryStop==589&&entryOpen>=618&&entryOpen<=627,"Existing entrance stop/open dwell regressed");
    require(missionLog.at(853).at("state")=="PARKING_MANEUVER"&&missionLog.at(969).at("gate_waiting")==true&&
        control.at(969).at("command_speed")==0,"Wrong latest stuck gate");
    RaceManager gate(cfg);RacePlan p;
    for(int f=846;f<=865;++f) {
        p=tick(gate,control.at(f),closed,bar);
        require(!p.gateOpened&&(f<853||p.stop),"Gate releases while the recorded bar is still detected");
    }
    auto missed=gate,lost=gate;
    int released=0;
    for(int f=866;f<=912;++f) {
        const auto &r=control.at(f);
        for(const auto &d:r.at("ai_detections"))require(d.at("type")!=LABEL_GATE||d.at("accepted")==false,
            "Expected the recorded interval without an accepted gate box");
        const auto stay=tick(missed,r,closed,bar);
        require(stay.stop&&!stay.gateOpened,"A missed detection opens a still-visible horizontal bar");
        const auto unknown=tick(lost,r,cv::Mat{},cv::Mat{});
        require(unknown.stop&&!unknown.gateOpened,"Missing gate imagery authorizes motion");
        if(!released) {
            p=tick(gate,r,raised,road);
            require(p.gateClearRegion.y>170&&p.gateRoadFraction>=.85f&&p.gateYellowFraction<.08f,
                "Opening checks a lifted box instead of the vacated road");
            if(p.gateOpened) {
                require(!p.stop&&r.at("time").get<double>()-control.at(866).at("time").get<double>()>=cfg.gateHoldTime,
                    "Opening skips the stopped dwell");
                released=f;
            }
        }
    }
    require(released>866&&released<890,"Confirmed open passage cannot release before the bar returned at 913");

    nlohmann::json saved;std::ifstream(latest/"parking-route.json")>>saved;
    RaceParkingRoute route;route.valid=true;route.entryCurvaturePreview=saved.at("entry_curvature_preview_m");
    for(const auto &q:saved.at("entry"))route.entry.push_back({q.at(0),q.at(1),q.at(2)});
    route.exit.assign(route.entry.rbegin(),route.entry.rend());
    RaceParkingSpace space;space.keepRearWithinStart(route.entry.at(0));
    RaceParkingControl parking(space,cfg.parkingSpeed);RaceParkingAisleHandoff handoff;
    int resumed=0;
    for(int f=653;f<=released+12&&!resumed;++f) {
        const auto &r=control.at(f),&prior=control.at(f-1),&q=r.at("pose");
        RaceParkingObservation o{{q.at(0),q.at(1),q.at(2)},r.at("captured"),uint64_t(f),r.at("motion_epoch"),
            r.at("control_pose_valid"),r.at("pose_estimated")};
        const double now=r.at("time");
        if(f==653)require(parking.arm(route,o,now,saved.at("reverse_verified")),"Recorded parking route did not arm");
        const bool clear=f<853||f>=released;
        parking.update(o,now,r.at("control_dt_s"),clear,std::min(cfg.parkingSpeed,r.at("traffic_speed_cap").get<float>()),
            prior.at("command_speed"),saved.at("canonical_frame_handedness").get<float>()*prior.at("physical_curvature").get<float>());
        if(f==852)require(parking.done()&&parking.rejoiningAfterReverse(),"Completed parking was not awaiting aisle handoff");
        if(parking.done()&&handoff.update(parking,space,o,now,clear,r.at("ai_age_s").get<float>()<=cfg.maxDetectionAge)==
           RaceParkingAisleHandoff::State::Ready)resumed=f;
    }
    require(resumed>released&&resumed<released+12,"Gate release leaves the completed parking maneuver stuck");

    const auto turnBin=camera.binarize(atTurn);track.handle(turnBin);
    RaceManager manager(cfg);RaceSnapshot snap;snap.image=atTurn;
    RacePlan initial;
    for(int i=1;i<=3;++i) {
        snap.sequence=i;snap.captured=i*.08;
        initial=manager.update(snap,snap.captured,.08f,true,turnBin,track,atTurn,true,true,
            -1,true,true,cfg.parkingSpeed,{true,true,0,0,0,1500},false,false,false,true);
        if(raceDirectionSeeking(initial))break;
    }
    require(raceDirectionSeeking(initial)&&initial.parkingTurnCommit&&raceDirectionServo(initial,1500)<1500,
        "Fresh authorized exit image cannot start left steering");
    const auto &mixed=turns.at(934);
    require(mixed.at("traffic_reason")=="TARGET_ROAD_GRADUAL_HANDOFF"&&mixed.at("pwm").get<int>()>1500,
        "Missing field countersteer during the initial blend");
    RacePlan blend=initial;blend.selectedBranch=true;blend.laneBlend=mixed.at("lane_blend");blend.guidePwm=mixed.at("guide_pwm");
    require(raceDirectionServo(blend,mixed.at("pwm"))<1500,"First opposite lane command still cancels exit turn");
    blend.parkingTurnCommit=false;
    require(raceDirectionServo(blend,mixed.at("pwm"))==mixed.at("pwm").get<int>(),
        "Parking change delays normal fork countersteering");
    const auto curveBin=camera.binarize(onCurve);track.handle(curveBin);
    require(raceBranchPath(curveBin,track,0,cfg).singleRoadAhead&&turns.at(961).at("single_road_ahead")==true,
        "Recorded stop image and its current-road geometry disagree");
    // The manager reads the last applied command and the pilot's previous
    // alignment, then inspects current geometry. Replay that one-frame delay.
    RaceYForkProgress progress,noTurn,unseen,paused;
    int accepted=0,stationary=0;
    for(int f=925;f<=973;++f) {
        const auto &state=turns.at(f),&prior=turns.at(f-1);
        RaceForkFeedback feedback{true,prior.at("lane_alignment_valid"),prior.at("lane_alignment_offset_px"),
            prior.at("lane_alignment_heading_px"),prior.at("speed_command"),prior.at("pwm")};
        const float dt=state.at("control_dt_s");
        const bool selected=f>=944; // Original transition to COMMITTED_ROAD_CENTER_TRACKING.
        const bool single=state.at("single_road_ahead").get<bool>()&&state.at("geometry_split_rows")==0;
        const auto &ai=turnControl.at(f);
        const double captured=ai.at("time").get<double>()-ai.at("ai_age_s").get<double>();
        const bool fresh=ai.at("ai_sequence")!=turnControl.at(f-1).at("ai_sequence")&&
            ai.at("ai_age_s").get<float>()<=cfg.maxDetectionAge;
        progress.update(dt,fresh,false,feedback);
        if(progress.observeExitRoad(feedback,fresh,dt,selected,single,-1,captured)&&!accepted)accepted=f;
        paused.update(dt,fresh,false,feedback);
        if(paused.observeExitRoad(feedback,fresh,dt,selected,single&&f>=961,-1,captured)&&!stationary)stationary=f;
        auto straight=feedback;straight.appliedPwm=1500;
        noTurn.update(dt,fresh,false,straight);unseen.update(dt,fresh,false,feedback);
        require(!noTurn.observeExitRoad(straight,fresh,dt,selected,single,-1,captured)&&
            !unseen.observeExitRoad(feedback,fresh,dt,selected,false,-1,captured),
            "Unsteered or unconfirmed road hands off the parking exit");
    }
    require(accepted>=957&&accepted<=962&&stationary>961&&stationary<=969&&
        turns.at(stationary-1).at("speed_command")==0,"Turn cannot reacquire the visible curve before or while stopped");
    RaceMission mission;mission.start();
    RaceSnapshot sign;sign.detections={box(LABEL_PARK,500,2,110,80)};
    for(int i=1;i<=3;++i){sign.sequence=i;sign.captured=.07*i;mission.observeSigns(sign,true);}
    RacePlan lane;lane.fresh=true;lane.state="LANE";lane.completedDirection=-1;
    mission.observe({},lane,false);lane.completedDirection=0;mission.observe({},lane,true);mission.parkingStarted();
    auto shut=lane;shut.stop=shut.gateWaiting=true;mission.observeGate(shut,true);
    auto open=lane;open.gateOpened=true;mission.observeGate(open,true);
    mission.parkingFinished(false);
    for(int i=0;i<18;++i)mission.advance(lane,.10f,.25f);
    require(mission.exitTurnAuthorized(),"A verified exit cannot begin its integrated left turn");
    // The intermediate images were not saved. Exercise the complete manager and
    // mission on the actual approach and curve keyframes, with archived feedback
    // and a staged image switch. This is a counterfactual integration test.
    RaceManager integrated(cfg);
    int completed=0;bool left=false,selectedRoad=false;
    for(int f=925;f<=973&&!completed;++f) {
        const bool curve=f>=934;
        const auto &image=curve?onCurve:atTurn,&bin=curve?curveBin:turnBin;
        track.handle(bin);
        const auto &state=turns.at(f),&prior=turns.at(f-1),&ai=turnControl.at(f);
        const double time=ai.at("time");const float dt=state.at("control_dt_s");
        snap.sequence=ai.at("ai_sequence");snap.captured=time-ai.at("ai_age_s").get<double>();snap.image=image;
        RaceForkFeedback feedback{true,prior.at("lane_alignment_valid"),prior.at("lane_alignment_offset_px"),
            prior.at("lane_alignment_heading_px"),prior.at("speed_command"),prior.at("pwm")};
        const auto decision=integrated.update(snap,time,dt,true,bin,track,image,true,true,mission.routeDirection(),
            true,mission.exitTurnAuthorized(),cfg.parkingSpeed,feedback,false,false,false,mission.exitTurnAuthorized());
        left|=raceDirectionSeeking(decision)&&raceDirectionServo(decision,1500)<1500;
        selectedRoad|=decision.selectedBranch;
        mission.observe(snap,decision,false);
        if(decision.completedDirection==-1)completed=f;
    }
    require(left&&selectedRoad&&completed&&std::string_view(mission.name())=="LAP2_RETURN_PRESS_N_TO_FINISH",
        "Actual approach and curve keyframes cannot hand the exit turn to the return mission");
    std::cout<<"PASS parking gate turn entrance_stop="<<firstEntryStop<<" entrance_open="<<entryOpen
             <<" exit_gate_open="<<released<<" aisle_handoff="<<resumed<<" turn_handoff="<<accepted
             <<" stopped_handoff="<<stationary<<" integrated_handoff="<<completed
             <<" staged opening and keyframe transition; field retest pending\n";
}
