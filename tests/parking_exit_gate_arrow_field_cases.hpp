#pragma once

static void parkingExitGateArrowFieldChecks(const fs::path &run) {
    RacePlan lane;lane.fresh=true;lane.state="LANE";lane.completedDirection=-1;
    auto gate=parkingMissionAtEntrance();gate.observe({},lane,false);lane.completedDirection=0;
    gate.observe({},lane,true);gate.parkingStarted();
    auto arrow=gate;arrow.parkingFinished(false);
    auto orphan=gate;auto opened=lane;opened.gateOpened=true;
    orphan.observeGate(opened,true);orphan.parkingFinished(false);
    require(orphan.json().at("exit_gate_seen")==false&&!orphan.exitTurnAuthorized(),
        "Isolated gate-open event during rejoin authorizes exit");
    auto earlier=gate;auto closed=lane;closed.stop=closed.gateWaiting=true;
    earlier.observeGate(closed);earlier.observeGate(opened,true);earlier.parkingFinished(false);
    require(earlier.json().at("exit_gate_seen")==false&&!earlier.exitTurnAuthorized(),
        "A wait before reversing is reused for the exit");
    auto interrupted=gate;interrupted.observeGate(closed,true);
    interrupted.observeGate(closed,false);interrupted.observeGate(opened,false);
    require(interrupted.json().at("exit_gate_released")==true&&!interrupted.exitTurnAuthorized()&&
            interrupted.json().at("entry_gate_seen")==false,
            "Gate ownership lost when its stop interrupted aisle confirmation");
    interrupted.parkingFinished(false);
    require(!interrupted.exitTurnAuthorized(),"Interrupted alignment skipped the post-handoff gate passage");

    std::ifstream control(run/"parking-control.jsonl"),mission(run/"mission-state.jsonl");
    require(bool(control)&&bool(mission),"Missing field gate and handoff logs");
    std::map<int,double> cameraTimes;
    float previousSpeed=0;bool rejoining=false;int waiting=0,released=0,handoff=0,gateTurn=0;
    std::string l,m;
    while(std::getline(control,l)&&std::getline(mission,m)) {
        const auto row=nlohmann::json::parse(l),state=nlohmann::json::parse(m);
        const int f=row.at("frame");require(state.at("frame")==f,"Misaligned parking logs");
        if(f>=972&&f<=975)cameraTimes[f]=row.at("captured");
        RacePlan actual;actual.fresh=true;actual.gateWaiting=state.at("gate_waiting");
        actual.stop=actual.gateWaiting&&row.at("command_speed").get<float>()==0;
        actual.gateOpened=state.at("gate_opened");
        gate.advance(actual,row.at("control_dt_s"),previousSpeed);gate.observeGate(actual,rejoining);
        if(actual.gateWaiting&&rejoining&&!waiting)waiting=f;
        if(actual.gateOpened&&rejoining)released=f;
        if(gate.parkingRun()) {
            require(gate.routeDirection()==0&&gate.json().at("exit_gate_command_m")==0&&gate.speedLimit(.55f,.3f)==.3f,
                "Gate turn, passage travel or restored speed during the parking maneuver");
            if(state.at("state")=="PARKING_EXIT_LEFT") {
                gate.parkingFinished(false);handoff=f;
                require(gate.json().at("exit_gate_seen_during_rejoin")==true&&gate.json().at("exit_gate_released")==true&&
                        gate.routeDirection()==0&&gate.speedLimit(.55f,.3f)==.3f,"Early open lost at handoff or passage skipped");
            }
        }
        if(gate.exitTurnAuthorized()&&!gateTurn)gateTurn=f;
        previousSpeed=row.at("command_speed");
        rejoining=row.at("armed").get<bool>()&&row.at("reverse_command_distance_m").get<float>()>0&&
            (row.at("exit_distance_recovery").get<bool>()||row.at("reason")=="AISLE_ALIGNMENT_CONFIRMING"||
             row.at("state")=="PARK_COMPLETE");
        if(f>=975&&gateTurn)break;
    }
    require(waiting==924&&released==940&&handoff==957&&gateTurn>handoff&&gateTurn<=975&&
        gate.json().at("exit_gate_command_m")>=RaceMission::exitGatePassCommandM,"Recorded exit gate not carried through aisle handoff");

    TrackviewConfig vehicle;vehicle.load((run/"trackview-config.json").string(),true);
    RaceGroundCalibration ground;ground.load((run/"ground-calibration.json").string(),false);
    cv::Point2f src[4],dst[4]={{-ground.roadWidth/2,ground.nearDistance},{ground.roadWidth/2,ground.nearDistance},
        {-ground.roadWidth/2,ground.farDistance},{ground.roadWidth/2,ground.farDistance}};
    for(int i=0;i<4;++i)src[i]={ground.imagePoints[i][0],ground.imagePoints[i][1]};
    Camera camera(cv::getPerspectiveTransform(src,dst));camera.blackRoad=vehicle.roadSurface=="black";
    require(camera.blackRoad&&vehicle.binary<0,"Unexpected field road color/threshold");
    struct Restore {Camera *old;~Restore(){Track::camera=old;}} restore{Track::camera};Track::camera=&camera;
    Track t;t.rowCutUp=vehicle.rowCutUp;t.rowCutBottom=vehicle.rowCutBottom;
    const auto readArrow=[&](const cv::Mat &img){
        require(img.size()==cv::Size(640,480),"Missing corrected field arrow image");
        t.handle(camera.binarize(img));return raceGroundArrow(img,t,true);
    };
    const auto choices=readArrow(cv::imread((run/"taught-bay.png").string()));
    require(choices.allowed==(RoadLeft|RoadRight),"Parking aisle choice marking misclassified as exit");
    auto closedGate=arrow;closedGate.observeGate(closed);
    for(int f=972;f<=975;++f) {
        const auto img=cv::imread((run/"parking-exit-arrow"/("frame-"+std::to_string(f)+".png")).string());
        const auto recognized=readArrow(img);require(recognized.allowed==RoadLeft,"Bent field arrow did not produce pure left");
        cv::Mat mirrored;cv::flip(img,mirrored,1);
        require(readArrow(mirrored).allowed==RoadRight,"Mirrored bent arrow did not produce pure right");
        arrow.observeExitArrow(recognized,f,true,cameraTimes.at(f));
        closedGate.observeExitArrow(recognized,f,true,cameraTimes.at(f));
        require(arrow.exitTurnAuthorized()==(f>=974)&&!closedGate.exitTurnAuthorized(),
            "Three-frame arrow confirmation or closed-gate priority failed");
    }
    require(readArrow(cv::imread((run/"parking-exit-decision/frame-1035.png").string())).allowed!=RoadLeft,
            "View after the missed exit produces a left arrow");
    // Exercise the production steering handoff on the saved approach image
    // with fresh inputs. The archived 974/975 loops included a camera stall.
    const auto view=cv::imread((run/"parking-exit-arrow/frame-975.png").string());
    const auto binary=camera.binarize(view);t.handle(binary);
    RaceConfig cfg;cfg.load((run/"race-config.json").string());RaceManager manager(cfg);RaceSnapshot snapshot;
    RaceTaskPilot pilot(vehicle);RacePlan turn;
    for(int i=1;i<=3;++i) {
        snapshot.sequence=i;snapshot.captured=i*.08;snapshot.image=view;
        turn=manager.update(snapshot,snapshot.captured,.08,true,binary,t,view,true,true,
            arrow.routeDirection(),true,arrow.exitTurnAuthorized(),cfg.forkSpeed,{true,true,0,0,0},
            false,false,arrow.followParkingAisle(),arrow.exitTurnAuthorized());
        if(raceDirectionSeeking(turn))break;
    }
    require(raceDirectionSeeking(turn)&&raceDirectionServo(turn,PWMSERVOMID)<PWMSERVOMID&&
            raceDirectionSpeed(turn,pilot,cfg.parkingSpeed,.08,0)>0,
            "Authorized exit on the saved approach image does not request a forward left turn");
    std::cout<<"PASS parking exit gate arrow "<<run.filename().string()<<" gate_wait="<<waiting<<" open="<<released
        <<" handoff="<<handoff<<" gate_authorized="<<gateTurn
        <<" arrow_only_authorized=974; four real left arrows, four mirrored right, aisle choice and closed gate; offline replay\n";
}
