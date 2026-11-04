#pragma once

static RaceMission parkingMissionAtEntrance(float exitLimit=RaceConfig{}.parkingExitSearchLimitM) {
    RaceMission mission(false,exitLimit);mission.start();RaceSnapshot s;
    s.detections={box(LABEL_PARK,500,2,110,80)};
    for(int i=1;i<=3;++i){s.sequence=i;s.captured=i*.07;mission.observeSigns(s,true);}
    require(mission.parkingEntrancePending(),"Cannot establish parking entrance context");
    return mission;
}

static void parkingMissionRejoinChecks() {
    RacePlan aisle;aisle.fresh=true;aisle.state="LANE";aisle.completedDirection=-1;
    auto mission=parkingMissionAtEntrance(.75f);
    mission.observe({},aisle,false);aisle.completedDirection=0;
    mission.observe({},aisle,true);
    mission.parkingFinished(false);
    require(mission.parkingSearch()&&!mission.json().at("parking_returned_to_aisle").get<bool>(),
        "Unstarted maneuver can claim an aisle handoff");
    mission.parkingStarted();auto interrupted=mission;interrupted.stop();interrupted.parkingFinished(false);
    require(interrupted.failed()&&!interrupted.json().at("parking_returned_to_aisle").get<bool>(),
        "Operator stop can be cleared by parking handoff");
    auto validated=mission;validated.parkingFinished(true);
    require(validated.json().at("parking_completed")==1&&validated.json().at("parking_returned_to_aisle")==true,
        "Confirmed parking no longer counts");

    mission.parkingFinished(false);mission.parkingFinished(true);
    require(mission.running()&&mission.json().at("parking_completed")==0&&
        mission.json().at("parking_returned_to_aisle")==true&&!mission.parkStageEligible()&&
        mission.followParkingAisle()&&mission.routeDirection()==0&&mission.speedLimit(.8f,.3f)==.3f,
        "Recovered aisle return either stops the mission, counts a valid bay, or turns without a landmark");
    for(int i=0;i<12;++i)mission.advance(aisle,.125f,.5f);
    require(mission.exitSearchTravelExhausted()&&!mission.exitTurnAuthorized(),
        "Recovered return bypasses the unmarked exit search limit");

    RaceGroundArrow left;left.allowed=RoadLeft;left.box={120,228,88,60};
    auto gated=mission;auto gate=aisle;gate.stop=gate.gateWaiting=true;gated.observeGate(gate);
    for(int i=1;i<=3;++i){gated.observeExitArrow(left,i,true,.07*i);mission.observeExitArrow(left,i,true,.07*i);}
    require(!gated.exitTurnAuthorized()&&gated.json().at("exit_gate_seen")==true,
        "Unverified bay return misclassifies the exit gate or turns through its closed bar");
    gate.stop=gate.gateWaiting=false;gate.gateOpened=true;gated.observeGate(gate);
    for(int i=0;i<6;++i)gated.advance(aisle,.125f,.5f);
    require(!gated.exitTurnAuthorized(),"Recovered return turns before clearing the exit gate");
    gated.advance(aisle,.125f,.5f);
    require(gated.exitTurnAuthorized()&&gated.routeDirection()==-1,
        "Recovered return cannot turn after a verified exit gate passage");
    require(!mission.exitSearchTravelExhausted()&&mission.routeDirection()==-1,
        "No-gate recovered return cannot use a verified exit arrow");
    aisle.completedDirection=-1;mission.observe({},aisle,false);
    require(std::string(mission.name())=="LAP2_RETURN_PRESS_N_TO_FINISH"&&mission.json().at("parking_completed")==0,
        "Left exit completion loses the unverified parking result");
    std::cout<<"PASS verified aisle handoff continues without granting an unverified bay; stop, exit limit, gate and arrow protections\n";
}

static void parkingLatestFieldChecks(const fs::path &root) {
    RaceConfig cfg;cv::Mat blank=cv::Mat::zeros(480,640,CV_8UC1);Track track;track.handle(blank);
    {
        // The two latest missions released the gate while still FORK_SEEK_LEFT,
        // with an expired pulse and a centered current road, not FORK_LEFT.
        RaceManager manager(cfg);RaceSnapshot s;RacePlan p;double time=0;
        auto tick=[&](bool roadReady,float speed,int gate){
            ++s.sequence;s.captured=time+=.07;s.detections.clear();
            if(gate==1)s.detections={box(LABEL_GATE,100,200,440,70)};
            if(gate==2)s.detections={box(LABEL_GATE,100,180,40,100)};
            return manager.update(s,time,.07,true,blank,track,{},true,true,-1,true,true,.3,
                {true,roadReady,0,-15,speed},true,true);
        };
        for(int i=0;i<42;++i)p=tick(false,.3,0);
        require(std::string(p.state)=="FORK_SEEK_LEFT"&&p.guidePwm==0,"Cannot reproduce exhausted entrance seek");
        for(int i=0;i<3;++i)p=tick(true,0,1);
        require(p.stop&&p.gateWaiting,"Entrance gate test did not stop");
        bool released=false;
        for(int i=0;i<22;++i){p=tick(true,0,2);if(p.gateOpened){released=true;break;}}
        require(released&&p.completedDirection==-1&&!p.stop&&!p.selectedBranch,
            "Gate release from completed guided seek still delays bay search");
        auto mission=parkingMissionAtEntrance();mission.observeGate(p);mission.observe(s,p,false);
        require(mission.canScanParkingBays(p)&&mission.followParkingAisle(),"Released seek cannot scan current aisle");
        RaceManager unguided(cfg);s={};
        for(int i=1;i<10;++i){s.sequence=i;s.captured=i*.07;
            auto q=unguided.update(s,s.captured,.07,true,blank,track,{},true,true,-1,true,true,.3,
                {true,true,0,0,.3},true,true);
            require(q.completedDirection==0,"Unstarted entry completed from a centered road");}
        std::cout<<"PASS latest gate-release seek handoff; scan immediately; unstarted entrance protected\n";
    }
    const auto base=root/"vehicle_evidence/parking-20261102-130423";
    {
        RaceGroundCalibration calibration;calibration.load((base/"parking-1793624548684/ground-calibration.json").string(),false);
        RaceGroundMap map(calibration);const auto image=cv::imread((base/"parking-1793624548684/taught-bay.png").string());
        const auto mouths=raceDetectParkingMouths(image,map);
        require(!mouths.empty()&&mouths.front().side==-1&&std::abs(mouths.front().nearGround.y-.4425f)<.03f,
            "Latest real standalone bay is no longer detected");
        int visible=0;const auto folder=base/"mission-1793624333030/parking-stop-evidence-1";
        for(int frame=770;frame<=781;++frame){
            const auto view=cv::imread((folder/("frame-"+std::to_string(frame)+".png")).string());
            require(!view.empty(),"Missing latest mission stop image");visible+=!raceDetectParkingMouths(view,map).empty();
        }
        std::cout<<"PASS saved real parking images: standalone bay detected; mission visible_frames="<<visible
            <<"/12; image check only, no inference about unseen gate-release frames\n";
    }
    {
        // Use the actual detections from the stopped run: one transient at
        // frame 781 must not stop the aisle, and no cached point may be planned.
        std::ifstream input(base/"mission-1793624333030/parking-bay-search.jsonl");
        require(bool(input),"Missing latest bay search telemetry");
        RaceParkingMouthConfirmation confirmation;std::string line;int candidates=0,confirmed=0;
        while(std::getline(input,line)) {
            const auto r=nlohmann::json::parse(line);if(!r.contains("candidates"))continue;
            std::vector<RaceParkingMouthCandidate> mouths;
            for(const auto &m:r.at("candidates")){auto n=m.at("near"),f=m.at("far");
                mouths.push_back({{},{},{n[0],n[1]},{f[0],f[1]},m.at("support"),m.at("side")});}
            candidates+=mouths.size();const int frame=r.at("frame");
            confirmed+=confirmation.update(mouths,frame*.07,frame,0);
        }
        require(candidates==1&&confirmed==0,"Transient field bay still latches stationary search");
        auto mission=parkingMissionAtEntrance();RacePlan p;p.fresh=true;p.completedDirection=-1;p.state="LANE";
        mission.observe({},p,false);
        for(int i=0;i<34;++i)mission.advance(p,.1,.3);
        mission.observe({},p,true);require(mission.parkingSearch(),"Cannot establish stopped search");
        mission.nextParkingBay("BAY_CONFIRMATION_LOST_FIND_NEXT");
        require(mission.baySearchTravelExhausted()&&!mission.exitTurnAuthorized(),"Confirmation retry resets search bound or exits");
        std::cout<<"PASS latest single-detection loss never latches stop; confirmation retry retains search bound\n";
    }
    {
        const auto folder=base/"parking-1793624548684";nlohmann::json saved;
        std::ifstream(folder/"parking-route.json")>>saved;
        RaceParkingRoute route;route.valid=true;route.entryCurvaturePreview=saved.at("entry_curvature_preview_m");
        for(const auto &p:saved.at("entry"))route.entry.push_back({p[0],p[1],p[2]});
        route.exit=route.entry;std::reverse(route.exit.begin(),route.exit.end());
        RaceParkingSpace space;space.keepRearWithinStart(route.entry.front());RaceParkingControl control(space,.3);
        std::ifstream input(folder/"parking-control.jsonl");std::string line;bool armed=false;
        int oldSign=0,newSign=0,oldReversals=0,newReversals=0,rows=0;float oldAngle=0,newAngle=0,oldVariation=0,newVariation=0;
        auto count=[&](float k,int &sign,int &changes,float &angle,float &variation){
            const int next=k>.5f?1:k<-.5f?-1:0;if(next){if(sign&&next!=sign)++changes;sign=next;}
            const float a=std::atan(.2f*std::clamp(k,-3.247038f,3.247038f));variation+=std::abs(a-angle);angle=a;
        };
        while(std::getline(input,line)) {
            const auto r=nlohmann::json::parse(line);const int frame=r.at("frame");if(frame<38)continue;if(frame>=172)break;
            const auto p=r.at("pose");RaceParkingObservation o{{p[0],p[1],p[2]},r.at("captured"),r.at("frame"),r.at("motion_epoch"),r.at("motion_valid")};
            if(!armed){require(control.arm(route,o,r.at("time"),true),"Recorded parking route no longer arms");armed=true;continue;}
            const auto command=control.update(o,r.at("time"),r.at("control_dt_s"),true,.1,r.at("command_speed"));
            require(!control.failed(),"Recorded parking observation replay faults");
            count(r.at("requested_curvature"),oldSign,oldReversals,oldAngle,oldVariation);
            count(command.requestedCurvature,newSign,newReversals,newAngle,newVariation);++rows;
        }
        require(rows>120&&newReversals<oldReversals&&newVariation<oldVariation,
            "Continuous entry support does not reduce recorded steering reversals");
        std::cout<<"PASS latest observation replay: steering_reversals="<<oldReversals<<"->"<<newReversals
            <<" angle_variation="<<oldVariation<<"->"<<newVariation<<"; fixed recorded poses, not predicted vehicle path\n";
        const auto planned=racePlanTrackableParking(space,route.entry.front(),.3);
        require(planned.valid&&raceParkingModelCompletes(space,planned,.3),"Latest bay start has no contained trackable parking route");
        std::cout<<"PASS latest real bay start replans with full-body entry, dwell, reverse and aisle rejoin in model\n";
    }
}

static void manualGateStopChecks() {
    RaceConfig cfg;require(cfg.gateStopRow==245,"Default manual gate stop row changed");
    auto previous=cfg;previous.gateStopRow=270;
    RaceManager manual(cfg),old(previous);
    cv::Mat scene(480,640,CV_8UC3,cv::Scalar(65,65,65));
    cv::Mat binary(480,640,CV_8UC1,cv::Scalar(0));cv::rectangle(binary,{80,0,480,480},{255},-1);
    Track track;track.handle(binary);
    RaceSnapshot s;s.image=scene;RacePlan current,prior;
    auto tick=[&](int bottom){
        ++s.sequence;s.captured=s.sequence*.125;s.detections.clear();
        if(bottom)s.detections={box(LABEL_GATE,100,bottom-60,440,60)};
        current=manual.update(s,s.captured,.125f,true,binary,track,scene,true,true);
        prior=old.update(s,s.captured,.125f,true,binary,track,scene,true,true);
    };
    tick(0);
    require(!current.stop&&!current.gateWaiting&&current.speedCap>cfg.gateApproachSpeed,
        "Clear road spuriously enters manual gate control");
    tick(160);require(!current.stop&&current.speedCap==cfg.gateApproachSpeed,"Distant gate lost its approach cap");
    tick(244);require(!current.stop&&current.gateWaiting&&current.speedCap==cfg.gateApproachSpeed,
        "Manual gate stopped before the configured bounding-box bottom");
    tick(245);require(current.stop&&current.gateWaiting&&!prior.stop&&prior.gateWaiting,
        "Manual gate did not stop earlier than the previous 270-row threshold");
    tick(235);require(current.stop&&!current.gateOpened,"Jittering closed bar restarted the car");
    tick(270);require(current.stop&&prior.stop,"Previous stop row was not respected on the same input");
    tick(0);require(current.stop&&!current.gateOpened,"One missed gate detection released the car");
    tick(245);
    for(int i=0;i<12;++i){tick(245);require(current.stop&&!current.gateOpened,"Persistent lowered bar released the car");}
    for(int i=0;i<8;++i){tick(0);require(current.stop&&!current.gateOpened,"Clear passage during lift released before one second");}
    tick(0);require(current.gateOpened&&!current.stop&&!current.gateWaiting,"Manually cleared passage never released");
    std::cout<<"PASS manual gate stop: 245 vs 270 on same box; approach, closed-bar latch and one-second clear release\n";
}

static void parkingMissionChecks(const fs::path &root) {
    parkingMissionRejoinChecks();manualGateStopChecks();
    RaceConfig cfg;
    cv::Mat binary(480,640,CV_8UC1,cv::Scalar(0));cv::rectangle(binary,{80,0,480,480},{255},-1);
    Track track;track.handle(binary);
    RacePlan acquired;acquired.fresh=true;acquired.state="FORK_LEFT";acquired.selectedBranch=true;
    acquired.guideDirection=-1;acquired.laneBlend=1;acquired.geometrySplitRows=20;
    RaceSnapshot empty;
    auto mission=parkingMissionAtEntrance();
    {
        RacePlan seeking=acquired;seeking.selectedBranch=false;seeking.laneBlend=0;
        require(!mission.canScanParkingBays(seeking),"Unstarted entrance can consume a bay");
        auto merging=acquired;merging.laneBlend=.5;
        require(!mission.canScanParkingBays(merging),"Initial entrance pulse can consume a bay");
        require(mission.canScanParkingBays(acquired),"Acquired entrance still waits for a gate to scan bays");
        auto closed=acquired;closed.gateWaiting=true;
        require(!mission.canScanParkingBays(closed),"Closed gate approach permits bay maneuver");
        auto budget=mission;budget.advance(acquired,.07,0);
        for(int i=0;i<100;++i)budget.advance(acquired,.1,0);
        require(!budget.baySearchTravelExhausted(),"Stationary search consumes command distance");
        budget.advance(acquired,1,1);
        require(!budget.baySearchTravelExhausted(),"Invalid interval consumes command distance");
        for(int i=0;i<34;++i)budget.advance(acquired,.1,.3);
        require(budget.baySearchTravelExhausted(),"No-bay search can drive through the whole parking area");
        budget.observe(empty,acquired,true);
        require(budget.parkingSearch()&&!budget.baySearchTravelExhausted(),"Stopped search cannot resume on a current bay");
        mission.observe(empty,acquired,true);
        require(mission.parkingSearch()&&mission.followParkingAisle()&&mission.routeDirection()==0,
            "Bay does not transfer entrance steering to stationary planning");
        auto unstarted=parkingMissionAtEntrance();unstarted.parkingFinished();
        require(unstarted.json()["parking_completed"]==0,"Unstarted parking counted as completed");
        mission.parkingStarted();
        mission.parkingFinished();
        require(mission.routeDirection()==0&&mission.followParkingAisle(),"Reverse rejoin immediately repeats the entrance left turn");
        auto oldTurn=acquired;oldTurn.selectedBranch=false;oldTurn.completedDirection=-1;oldTurn.state="LANE";
        mission.observe(empty,oldTurn,false);
        require(std::string(mission.name())=="PARKING_EXIT_LEFT","Stale entrance completion skips exit turn");
        std::cout<<"PASS acquired entrance scans before gate; closed-gate exclusion; bounded search; reverse rejoin awaits exit landmark\n";
    }
    {
        RaceManager manager(cfg);RaceSnapshot s;RacePlan p;
        for(int i=1;i<=8;++i){s.sequence=i;s.captured=i*.07;
            p=manager.update(s,s.captured,.07,true,binary,track,{},true,true,-1,true,true);}
        s.sequence++;s.captured+=.07;
        p=manager.update(s,s.captured,.07,true,binary,track,{},true,true,0,true,false,.3,{},false,false,true);
        require(!p.selectedBranch&&p.guideDirection==0&&!p.stop,"Aisle retains the old entrance steering");
        for(int i=0;i<8;++i){++s.sequence;s.captured+=.07;s.detections={box(LABEL_GATE,100,210,440,60)};
            p=manager.update(s,s.captured,.07,true,binary,track,{},true,true,0,true,false,.3,{},false,false,true);}
        require(p.stop&&p.gateWaiting,"Aisle handoff suppresses traffic gate stop");
    }
    {
        auto arrowExit=mission;RaceGroundArrow arrow;arrow.allowed=RoadLeft|RoadRight;arrow.box={240,80,100,60};
        for(int i=1;i<=3;++i)arrowExit.observeExitArrow(arrow,i,true,i*.07);
        require(!arrowExit.exitTurnAuthorized(),"Distant aisle marking starts exit turn");
        arrow.box.y=180;
        for(int i=4;i<=6;++i)arrowExit.observeExitArrow(arrow,i,true,i*.07);
        require(!arrowExit.exitTurnAuthorized(),"Repeated aisle choice markings trigger premature exit");
        arrow.allowed=RoadLeft;arrowExit.observeExitArrow(arrow,7,true,.49);
        arrowExit.observeExitArrow(arrow,7,true,.49);arrowExit.observeExitArrow(arrow,8,false,.56);
        require(!arrowExit.exitTurnAuthorized(),"Duplicate or stale arrow confirms exit");
        for(int i=9;i<=11;++i)arrowExit.observeExitArrow(arrow,i,true,i*.07);
        require(arrowExit.exitTurnAuthorized()&&arrowExit.routeDirection()==-1,"Near left arrow cannot authorize required left exit");
        auto gateExit=mission;RacePlan gate;gate.fresh=true;gate.state="GATE_WAIT";gate.stop=gate.gateWaiting=true;
        gateExit.observeGate(gate);
        for(int i=0;i<30;++i)gateExit.advance(gate,.1,0);
        require(!gateExit.exitTurnAuthorized(),"Waiting at closed exit gate starts turn");
        gate.stop=gate.gateWaiting=false;gate.gateOpened=true;gate.state="LANE";gateExit.observeGate(gate);
        arrow.allowed=RoadLeft|RoadRight;
        for(int i=1;i<=3;++i)gateExit.observeExitArrow(arrow,i,true,i*.07);
        require(gateExit.exitArrowTurnAuthorized(),"Post-gate choice arrow rejects required left direction");
        require(!gateExit.exitTurnAuthorized(),"Gate release starts left lock before passage");
        auto noArrow=mission;noArrow.observeGate(RacePlan{});
        auto closed=gate;closed.gateOpened=false;closed.stop=closed.gateWaiting=true;
        noArrow.observeGate(closed);noArrow.observeGate(gate);
        gate.gateOpened=false;
        for(int i=0;i<13;++i)gateExit.advance(gate,.1,.3);
        require(!gateExit.exitTurnAuthorized(),"Gate passage command bound bypassed");
        gateExit.advance(gate,.1,.3);
        require(gateExit.exitTurnAuthorized()&&gateExit.routeDirection()==-1,"Exit without arrow continues straight forever");
        for(int i=0;i<14;++i)noArrow.advance(gate,.1,.3);
        require(noArrow.exitTurnAuthorized()&&!noArrow.exitArrowTurnAuthorized(),"Missing arrow blocks verified gate passage exit");
        std::cout<<"PASS near choice-arrow left exit; duplicate/stale rejection; gate-passage fallback without arrow\n";
    }
    {
        RacePlan lane;lane.state="LANE";lane.fresh=true;
        RaceGroundArrow left;left.allowed=RoadLeft;left.box={120,228,88,60};
        auto other=left;other.box.x=435;
        auto choice=left;choice.allowed=RoadLeft|RoadRight;
        require(cfg.parkingExitSearchLimitM==1.50f,"Default unmarked exit search limit changed");
        auto stopped=parkingMissionAtEntrance(.75f);
        stopped.observe(empty,acquired,true);stopped.parkingStarted();
        for(int i=0;i<24;++i)stopped.advance(lane,.125f,.5f);
        stopped.parkingFinished(true);
        require(!stopped.exitSearchTravelExhausted(),"Parking maneuver consumes subsequent exit search distance");
        for(int i=0;i<11;++i)stopped.advance(lane,.125f,.5f); // 0.6875 m commanded
        for(int i=0;i<50;++i)stopped.advance(lane,.125f,0);
        stopped.advance(lane,.125f,-.5f);stopped.advance(lane,1.f,.5f);
        require(!stopped.exitSearchTravelExhausted(),"Zero speed or invalid interval consumes exit search distance");
        stopped.advance(lane,.125f,.5f); // configured 0.7500 m
        require(stopped.exitSearchTravelExhausted()&&!stopped.exitTurnAuthorized()&&stopped.routeDirection()==0,
            "Travel limit authorizes blind left turn, misses the hold, or reverses accumulated distance");
        auto geometryOnly=stopped;
        for(int i=1;i<=20;++i) {
            geometryOnly.advance(lane,.125f,0);geometryOnly.observe(empty,acquired,true);
            geometryOnly.observeExitArrow(choice,i,true,i*.07);
            require(geometryOnly.exitSearchTravelExhausted()&&!geometryOnly.exitTurnAuthorized(),
                "Waiting with a bay mouth, reported fork or choice arrow releases unmarked exit");
        }
        auto obscured=stopped;
        obscured.observeExitArrow(left,1,true,.10);
        obscured.observeExitArrow(left,2,false,.17);
        obscured.observeExitArrow(left,3,true,.24);
        obscured.observeExitArrow(left,4,true,.31);
        require(obscured.exitSearchTravelExhausted(),"Stale frame bridges exit-arrow observations");
        auto disjoint=stopped;
        disjoint.observeExitArrow(left,1,true,.10);disjoint.observeExitArrow(left,2,true,.17);
        disjoint.observeExitArrow(other,3,true,.24);disjoint.observeExitArrow(other,3,true,.24);
        disjoint.observeExitArrow(other,4,true,.31);
        require(disjoint.exitSearchTravelExhausted(),"Different markings or duplicate observations form an exit arrow");
        disjoint.observeExitArrow(other,5,true,1.60);
        require(disjoint.exitSearchTravelExhausted(),"A long time gap completes old exit arrow evidence");
        disjoint.observeExitArrow(other,6,true,1.67);
        require(!disjoint.exitTurnAuthorized(),"Two fresh arrows after a long gap authorize exit");
        disjoint.observeExitArrow(other,7,true,1.74);
        require(!disjoint.exitSearchTravelExhausted()&&disjoint.exitTurnAuthorized()&&disjoint.routeDirection()==-1&&
            disjoint.speedLimit(.8f,.3f)==.3f,"Confirmed near left arrow cannot release stopped exit at parking speed");
        disjoint.observe(empty,acquired,false);disjoint.observe(empty,lane,false);
        require(disjoint.exitTurnAuthorized()&&disjoint.speedLimit(.8f,.3f)==.3f,
            "Disappearing selected branch fabricates left completion or releases cruise");
        auto finished=lane;finished.completedDirection=1;disjoint.observe(empty,finished,false);
        require(disjoint.exitTurnAuthorized(),"Explicit right completion finishes the required left exit");
        finished.completedDirection=-1;disjoint.observe(empty,finished,false);
        require(std::string(disjoint.name())=="LAP2_RETURN_PRESS_N_TO_FINISH"&&disjoint.routeDirection()==0&&
            disjoint.speedLimit(.8f,.3f)==.8f,"No-gate left completion leaves parking speed cap on return");
        auto gate=lane;gate.stop=true;gate.gateWaiting=true;stopped.observeGate(gate);
        for(int i=1;i<=6;++i)stopped.observeExitArrow(left,i,true,.07*i);
        require(!stopped.exitTurnAuthorized(),"Near exit arrow overrides an observed closed gate");
        gate.stop=gate.gateWaiting=false;gate.gateOpened=true;stopped.observeGate(gate);
        for(int i=0;i<6;++i)stopped.advance(lane,.125f,.5f); // 0.375 m
        require(!stopped.exitTurnAuthorized()&&!stopped.exitSearchTravelExhausted(),
            "Stopped search prevents gate passage or rushes the verified gate distance");
        stopped.advance(lane,.125f,.5f);
        require(stopped.exitTurnAuthorized()&&stopped.routeDirection()==-1,
            "Verified opening beyond the previous search bound cannot start the left exit");
        std::cout<<"PASS parking exit bounded search and return speed: unmarked hold, independent fresh arrow or gate release, explicit left return\n";
    }
    {
        RaceManager manager(cfg);RaceSnapshot s;RacePlan p;bool steered=false,completed=false;
        for(int i=1;i<=65;++i){s.sequence=i;s.captured=i*.07;
            p=manager.update(s,s.captured,.07,true,binary,track,{},true,true,-1,true,true,.3,
                {true,true,0,0,.2},false,false,false,true);
            steered|=p.guideDirection==-1&&p.guidePwm>0;
            if(p.completedDirection==-1){completed=true;break;}
        }
        require(steered&&completed&&!p.stop,"Finite exit left turn cannot hand over to centered road");
        RaceManager lost(cfg);cv::Mat blank=cv::Mat::zeros(480,640,CV_8UC1);Track noRoad;noRoad.handle(blank);
        bool pulseEnded=false,held=false;float applied=.2;
        for(int i=1;i<=80;++i){s.sequence=i;s.captured=i*.07;
            p=lost.update(s,s.captured,.07,true,blank,noRoad,{},true,true,-1,true,true,.3,
                {true,false,0,0,applied},false,false,false,true);
            if(i>25&&!p.stop&&p.guidePwm==0)pulseEnded=true;
            if(p.stop){held=true;applied=0;}
        }
        require(pulseEnded&&held&&p.stop,"Missing exit road permits endless turn or straight runaway");
        std::cout<<"PASS finite exit steering; centered-road handoff; bounded hold when exit road stays missing\n";
    }
    {
        RaceParkingMouthCandidate left{{},{},{-.225f,.3f},{-.225f,.71f},1,-1};
        auto right=left;right.side=1;right.nearGround.x=right.farGround.x=.225f;
        RaceParkingMouthConfirmation confirmation;
        std::vector<RaceParkingMouthCandidate> mouths{left,right};
        require(!confirmation.update(mouths,1,1,0),"One bay observation is sufficient");
        mouths={};require(!confirmation.update(mouths,1.14,2,0),"Cached bay fabricates fresh planning corners");
        left.nearGround.x+=.02f;left.farGround.x+=.02f;mouths={right,left};
        require(!confirmation.update(mouths,1.28,3,0)&&mouths.front().side==-1,"Candidate reordering loses same bay");
        require(!confirmation.update(mouths,1.29,3,0),"Duplicate frame confirms bay");
        mouths={right,left};require(confirmation.update(mouths,1.42,4,0)&&mouths.front().side==-1,
            "Three current matches with short dropout cannot confirm bay");
        mouths={left};require(!confirmation.update(mouths,2,5,0),"Old bay reused after long loss");
        auto next=left;next.nearGround.y+=.41f;next.farGround.y+=.41f;mouths={next};
        require(!confirmation.update(mouths,2.14,6,0)&&confirmation.count()==1,"Different adjacent bay inherits confirmation");
        std::cout<<"PASS same-bay confirmation across rank swaps and short dropout; no stale or adjacent-bay reuse\n";
    }
    {
        const auto folder=root/"vehicle_evidence/parking-20261102-123006/mission-1793620274097";
        nlohmann::json saved;std::ifstream(folder/"parking-route.json")>>saved;
        RaceParkingRoute route;route.valid=true;route.entryCurvaturePreview=saved.at("entry_curvature_preview_m");
        for(auto p:saved.at("entry"))route.entry.push_back({p[0],p[1],p[2]});
        route.exit=route.entry;std::reverse(route.exit.begin(),route.exit.end());
        RaceParkingSpace space;space.keepRearWithinStart(route.entry.front());RaceParkingControl control(space,.3);
        std::ifstream input(folder/"parking-control.jsonl");std::string line;bool checked=false;
        while(std::getline(input,line)) {
            const auto r=nlohmann::json::parse(line);
            if(r.at("frame")!=390)continue;
            const auto p=r.at("visual_pose");RaceParkingObservation o{{p[0],p[1],p[2]},r.at("captured"),r.at("frame"),r.at("motion_epoch"),r.at("motion_valid")};
            require(!space.freeObserved(o.pose)&&space.freeStartObserved(o.pose),"Field pose does not exercise start-only rear tolerance");
            const char *blocked=control.armBlockReason(route,o,r.at("time"),true);
            if(blocked)throw std::runtime_error(std::string("Recorded startup still blocked: ")+blocked);
            auto moved=o;moved.pose.z-=.03f;
            require(control.armBlockReason(route,moved,r.at("time"),true)!=nullptr,"Startup tolerance permits a moved vehicle");
            require(control.arm(route,o,r.at("time"),true),"Recorded millimetre drift cannot arm");checked=true;break;
        }
        require(checked,"Missing field start-rejection frame");
        std::cout<<"PASS recorded frame390 start drift accepted; moved pose still rejected; plan geometry unchanged\n";
    }
    {
        const auto folder=root/"vehicle_evidence/parking-20261102-123006/mission-1793621944148";
        std::ifstream stateFile(folder/"mission-state.jsonl"),controlFile(folder/"parking-control.jsonl");
        require(bool(stateFile)&&bool(controlFile),"Latest field logs missing");
        RaceMission replay;replay.start();std::string line,controlLine;float previousSpeed=0;int firstScan=0,holdFrame=0;
        while(std::getline(stateFile,line)&&std::getline(controlFile,controlLine)) {
            const auto r=nlohmann::json::parse(line),c=nlohmann::json::parse(controlLine);
            RaceSnapshot s;s.sequence=c.at("ai_sequence");s.captured=c.at("time").get<double>()-c.at("ai_age_s").get<double>();
            for(const auto &d:c.at("ai_detections"))if(d.at("accepted").get<bool>()){
                const auto b=d.at("box");auto v=box(d.at("type"),b[0],b[1],b[2],b[3]);v.score=d.at("score");s.detections.push_back(v);}
            replay.observeSigns(s,c.at("ai_age_s").get<float>()<=cfg.maxDetectionAge);
            RacePlan p;const auto state=r.at("traffic_state").get<std::string>();p.state=state.c_str();p.fresh=true;
            p.selectedBranch=state=="FORK_LEFT"&&r.at("guide_direction")==-1;
            p.guideDirection=r.at("guide_direction");p.laneBlend=r.at("lane_blend");
            p.gateWaiting=r.at("gate_waiting");p.gateOpened=r.at("gate_opened");p.completedDirection=r.at("fork_completed_direction");
            p.stop=p.gateWaiting&&state!="GATE_APPROACH";
            replay.advance(p,c.at("control_dt_s"),previousSpeed);replay.observeGate(p);
            if(!firstScan&&replay.canScanParkingBays(p))firstScan=r.at("frame");
            replay.observe(s,p,false);previousSpeed=c.at("command_speed");
            if(replay.baySearchTravelExhausted()){holdFrame=r.at("frame");break;}
        }
        require(firstScan>=700&&firstScan<730,"Field replay still waits until frame883 to scan");
        require(holdFrame>firstScan&&holdFrame<865,"Field no-bay case is allowed to reach the later gate");
        require(replay.json()["parking_completed"]==0&&!replay.exitTurnAuthorized(),"Field no-bay replay fabricates parking success");
        std::cout<<"PASS latest telemetry policy replay: scan_frame="<<firstScan<<" no_bay_hold_frame="<<holdFrame
            <<"; original scan_frame=883; no images available, not detection or vehicle validation\n";
    }
    parkingLatestFieldChecks(root);
}
