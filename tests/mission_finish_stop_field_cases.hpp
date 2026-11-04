#pragma once

static void missionFinishStopFieldChecks(const fs::path &root,const fs::path &latest) {
    const auto stages=[](bool parked) {
        std::vector<RaceMission> result;
        RaceMission mission;mission.start();result.push_back(mission);
        RacePlan lane;lane.fresh=true;lane.state="LANE";lane.completedDirection=-1;
        mission.observe({},lane,false);result.push_back(mission);
        mission.markStart();result.push_back(mission);
        RaceSnapshot ai;ai.detections={box(LABEL_PARK,500,2,110,80)};
        for(int i=1;i<=3;++i){ai.sequence=i;ai.captured=.08*i;mission.observeSigns(ai,true);}
        mission.observe(ai,lane,false);result.push_back(mission);
        lane.completedDirection=0;mission.observe(ai,lane,true);result.push_back(mission);
        mission.parkingStarted();result.push_back(mission);
        mission.parkingFinished(parked);result.push_back(mission);
        auto shut=lane;shut.stop=shut.gateWaiting=true;mission.observeGate(shut);
        auto open=lane;open.gateOpened=true;mission.observeGate(open);
        for(int i=0;i<18;++i)mission.advance(lane,.1f,.25f);
        require(mission.exitTurnAuthorized(),"Cannot authorize test exit");
        result.push_back(mission);lane.completedDirection=-1;
        mission.observe(ai,lane,false);result.push_back(mission);
        require(std::string_view(mission.name())=="LAP2_RETURN_PRESS_N_TO_FINISH","Cannot enter final return");
        return result;
    };
    const auto finalOutput=[](const RaceMission &m,bool stopped) {
        float speed=.45f;uint16_t servo=1320;RaceParkingCommand c;
        raceApplyMissionFinish(m,c,speed,servo);
        if(stopped)require(speed==0&&servo==PWMSERVOMID&&std::string_view(c.state)=="MISSION_COMPLETE"&&
            std::string_view(c.reason)==m.statusReason(),"Final wire command does not hold the confirmed finish");
        else require(speed==.45f&&servo==1320,"Finish override stops normal driving");
    };
    for(const auto &dir:{root/"vehicle_evidence/mission-1793794667967",root/"vehicle_evidence/mission-1793794867037",latest}) {
        const bool earlyStop=dir==latest;
        const bool completeTrace=dir.filename()=="mission-1793794667967";
        const int rejoined=earlyStop?1268:completeTrace?1158:1310;
        RaceConfig cfg;cfg.load((dir/"race-config.json").string());
        RaceGroundCalibration cal;cal.load((dir/"ground-calibration.json").string(),false);
        RaceGroundMap map(cal);
        if(earlyStop) {
            nlohmann::json stopped;std::ifstream(dir/"parking-result.json")>>stopped;
            const auto photo=cv::imread((dir/"parking-result-camera.png").string());
            require(photo.size()==cv::Size(640,480)&&stopped.at("frame")==1342&&stopped.at("command_speed")==0,
                "Missing the earlier implementation's actual stopping image");
            // The saved control camera image is adjacent to the matched AI frame.
            // Independently check the visible light stripes high in that image.
            int bands=0,width=0;
            for(int x=200;x<=401;++x) {
                const auto pixel=photo.at<cv::Vec3b>(77,x);
                if(x<401&&std::min({pixel[0],pixel[1],pixel[2]})>165)++width;
                else{bands+=width>=7&&width<=26;width=0;}
            }
            require(bands>=5&&bands<=8,"The old stop image does not show the distant stripe band");
        }
        auto phases=stages(completeTrace||earlyStop);
        auto normal=phases.back(),reviewed=normal;RaceManager manager(cfg);Track track;RaceSnapshot ai;RacePlan plan;
        int armed=0,reviewedArmed=0,near=0,finish=0,reviewedFinish=0,lastFrame=0;
        float previous=0,nearTravel=0,completedTravel=0,requiredTravel=0;
        std::ifstream control(dir/"parking-control.jsonl"),state(dir/"mission-state.jsonl");
        std::string cl,sl;
        while(std::getline(control,cl)&&std::getline(state,sl)) {
            const auto c=nlohmann::json::parse(cl),m=nlohmann::json::parse(sl);
            const int f=c.at("frame");require(m.at("frame")==f,"Mismatched finish frame logs");
            if(f<rejoined)continue;
            const bool previouslyStopped=earlyStop&&f>=1342;
            require(m.at("state")==(previouslyStopped?"MISSION_COMPLETE":"LAP2_RETURN_PRESS_N_TO_FINISH")&&
                m.at("parking_returned_to_aisle")==true,"Wrong recorded finish-stage window");
            if(f==rejoined)require(m.at("fork_completed_direction")==-1&&m.at("parking_completed")==int(completeTrace||earlyStop),
                "Incorrect parking score or recorded left-turn completion");
            const double now=c.at("time");
            ai.sequence=c.at("ai_sequence");ai.captured=now-c.at("ai_age_s").get<double>();ai.detections.clear();
            cv::Rect storedStop,storedCross;
            for(const auto &d:c.at("ai_detections")) {
                const auto &b=d.at("box");
                auto det=box(d.at("type"),b[0],b[1],b[2],b[3]);det.score=d.at("score");ai.detections.push_back(det);
                if(det.type==LABEL_STOP&&d.at("accepted")==true) {
                    require(det.y+det.height<cfg.triggerRow,"Field finish sign unexpectedly reached the traffic threshold");
                    storedStop={det.x,det.y,det.width,det.height};
                }
                if(det.type==LABEL_CROSS&&d.at("accepted")==true)
                    storedCross={det.x,det.y,det.width,det.height};
            }
            // No matched terminal camera images were retained. First use the normal
            // manager with model scores alone. Separately replay saved review decisions.
            const float dt=c.at("control_dt_s");
            plan=manager.update(ai,now,dt,true,{},track,{},true,true,0,true,false,cfg.forkSpeed,
                {},false,false,false,false,normal.finalReturn());
            normal.advance(plan,dt,previous);normal.observe(ai,plan,false,&map,cal.vehicleLength);
            auto evidence=plan;evidence.acceptedStopBox=storedStop;evidence.acceptedCrossBox=storedCross;
            evidence.acceptedCrossAtPassRow=storedCross.area()>0&&storedCross.br().y>=cfg.crossPassedRow;
            reviewed.advance(evidence,dt,previous);reviewed.observe(ai,evidence,false,&map,cal.vehicleLength);
            if(!armed&&normal.json().at("finish_stop_armed")==true)armed=f;
            if(!reviewedArmed&&reviewed.json().at("finish_stop_armed")==true)reviewedArmed=f;
            if(near)nearTravel+=std::max(0.f,previous)*dt;
            if(armed&&!near&&plan.acceptedCrossAtPassRow) {
                near=f;
                const auto b=plan.acceptedCrossBox;
                const float far=std::max(map.toGround({float(b.x),float(b.y)}).y,
                    map.toGround({float(b.br().x-1),float(b.y)}).y);
                requiredTravel=far+cal.vehicleLength+.02f;
            }
            if(normal.complete()&&!finish) {
                require(near&&f>near&&nearTravel>=requiredTravel-.001f&&std::string_view(normal.statusReason())=="FINISH_CROSSWALK_CLEARED",
                    "Sign, disappearance or an incomplete forward clearance caused an immediate stop");
                finish=f;completedTravel=nearTravel;
            }
            if(reviewed.complete()&&!reviewedFinish)reviewedFinish=f;
            finalOutput(normal,normal.complete());
            finalOutput(reviewed,reviewed.complete());previous=c.at("command_speed");lastFrame=f;
        }
        require(armed==(earlyStop?1342:completeTrace?1236:1390)&&reviewedArmed==(earlyStop?1342:completeTrace?1236:1387),
            "Lost field confirmation of the independent STOP sign");
        if(completeTrace) {
            require(near==1258&&finish>1262&&finish<1300&&reviewedFinish==finish&&lastFrame-finish>=20,
                "Full field crossing did not finish after near stripe and sustained motion with STOP absent");
            require(requiredTravel>.49f&&requiredTravel<.51f,"Field clearance is not based on the band's far edge plus the whole car");
            std::cout<<"PASS mission finish crosswalk "<<dir.filename().string()<<" recorded_stop_armed="<<armed
                     <<" recorded_near="<<near<<" replay_finish="<<finish<<" applied_forward_m="<<completedTravel<<"\n";
            continue;
        }
        require(!near&&!finish&&!reviewedFinish&&normal.finalReturn(),"Far crosswalk or stationary tail completed the course");
        const double tick=.08;
        const auto follow=[&](RaceMission &m,RaceSnapshot &s,cv::Rect crossing,float applied,bool fresh) {
            RacePlan p;p.state="LANE";p.fresh=fresh;p.stop=!fresh;
            p.acceptedCrossBox=crossing;p.acceptedCrossAtPassRow=crossing.area()>0&&crossing.br().y>=cfg.crossPassedRow;
            ++s.sequence;s.captured+=tick;m.advance(p,tick,applied);m.observe(s,p,false,&map,cal.vehicleLength);
            finalOutput(m,m.complete());
        };
        // Neither truncated recording saw a near stripe. Continue with explicitly
        // synthetic fresh stripe geometry, with STOP absent on every new frame.
        auto gone=normal;auto sGone=ai;
        for(int i=0;i<90&&!gone.finishSearchHeld();++i)follow(gone,sGone,{},.45f,true);
        require(!gone.complete()&&gone.finishSearchHeld(),"Lost distant crosswalk ends the course or permits an unbounded search");
        follow(gone,sGone,{12,321,100,105},0,true);
        for(int i=0;i<8;++i)follow(gone,sGone,{},0,true);
        require(gone.finishSearchHeld()&&!gone.complete(),"One unrelated close patch releases the search hold");
        for(int i=0;i<3;++i)follow(gone,sGone,{85,295,480,135},0,true);
        require(!gone.finishSearchHeld()&&!gone.complete(),"A stopped vehicle cannot re-confirm the near stripe");
        for(int i=0;i<25;++i)follow(gone,sGone,{},.45f,true);
        require(gone.complete(),"Forward clearance did not resume after re-confirming the crossing");
        auto close=normal;auto sClose=ai;
        for(int i=0;i<=23;++i) {
            const int w=210+12*i,h=42+4*i,bottom=110+14*i;
            follow(close,sClose,{320-w/2,bottom-h,w,h},.45f,true);
            require(!close.complete(),"Near stripe still visible when finish was declared");
        }
        auto unmoved=close,stale=close,sameBand=close;auto sStill=sClose,sStale=sClose,sSame=sClose;
        for(int i=0;i<35;++i) {
            follow(unmoved,sStill,{},0,true);follow(stale,sStale,{},.45f,false);
            follow(sameBand,sSame,{80,295,480,135},.45f,true);
            require(!unmoved.complete()&&!stale.complete()&&!sameBand.complete(),
                "Waiting, stale data or the same visible close stripe counted as a completed crossing");
        }
        int clearSteps=0;
        for(;clearSteps<45&&!close.complete();++clearSteps) {
            // A farther band appearing after this crossing must not be
            // confused with the near band that already passed underneath.
            const cv::Rect far=clearSteps>=2&&clearSteps<=6?cv::Rect(220,18,160,24):cv::Rect{};
            follow(close,sClose,far,.45f,true);
        }
        require(close.complete()&&clearSteps*.45f*tick>=cal.vehicleLength&&
            std::string_view(close.statusReason())=="FINISH_CROSSWALK_CLEARED",
            "Synthetic complete crossing with vanished STOP cannot finish");
        for(int i=0;i<4;++i)follow(close,sClose,{},.45f,true);
        std::cout<<"PASS mission finish crosswalk "<<dir.filename().string()<<" recorded_stop_armed="<<reviewedArmed
                 <<" recorded_end="<<lastFrame<<" recorded_near=0 synthetic_clear_forward_m="<<clearSteps*.45f*tick<<"\n";
    }
    auto phases=stages(true);
    const auto returned=phases.back();phases.pop_back();
    RacePlan accepted;accepted.fresh=true;accepted.state="LANE";accepted.acceptedStopBox={410,4,66,51};
    RaceSnapshot raw;
    for(auto phase:phases) {
        const std::string initial=phase.name();
        for(int i=1;i<=5;++i){raw.sequence=30+i;raw.captured=2+.08*i;phase.observe(raw,accepted,false);}
        require(!phase.complete()&&initial==phase.name(),"Finish sign skipped a lap or incomplete parking-exit turn");
        finalOutput(phase,false);
    }
    auto manual=returned;manual.markStart();
    require(manual.complete()&&std::string_view(manual.statusReason())=="MANUAL_FINISH_CONFIRMED","Operator n finish regressed");
    finalOutput(manual,true);
    auto guarded=returned;
    const auto feed=[&](int seq,double capture,cv::Rect rectangle,bool fresh) {
        raw.sequence=seq;raw.captured=capture;accepted.acceptedStopBox=rectangle;accepted.fresh=fresh;
        guarded.observe(raw,accepted,false);return guarded.json().at("finish_stop_armed").get<bool>();
    };
    const cv::Rect a{410,3,70,50},other{20,3,72,51};
    require(!feed(40,3,a,true)&&!feed(40,3,a,true)&&!feed(39,2.95,a,true)&&!feed(41,3.08,a,false)&&
        !feed(42,3.16,a,true)&&!feed(43,3.24,{},true)&&!feed(44,3.32,a,true)&&
        !feed(45,3.40,other,true)&&!feed(46,3.48,a,true)&&!feed(47,3.90,a,true)&&
        !feed(48,3.98,a,true)&&feed(49,4.06,a,true),
        "Duplicate, stale, missing, unrelated or separated observations can arm the finish");
    require(!guarded.complete(),"STOP without a crosswalk completed the course");

    // A real traffic stop mid-course must still wait for FLOW before releasing.
    RaceConfig cfg;cfg.load((latest/"race-config.json").string());RaceManager traffic(cfg);Track track;
    RaceMission mid;mid.start();RaceSnapshot s;RacePlan decision;double now=0;
    const auto tick=[&](std::vector<PredictResult> detections) {
        ++s.sequence;s.captured=now+=.08;s.detections=std::move(detections);
        decision=traffic.update(s,now,.08f,true,{},track,{},true,true);
        mid.observe(s,decision,false);require(!mid.complete(),"Ordinary traffic stop ended the mission");
    };
    for(int i=0;i<5;++i)tick({box(LABEL_STOP,270,290,80,76)});
    require(decision.stop&&std::string_view(decision.state)=="SIGNAL_WAIT","Middle-course STOP no longer stops");
    for(int i=0;i<25;++i){tick({});require(decision.stop,"Traffic STOP releases without FLOW");}
    for(int i=0;i<4;++i)tick({box(LABEL_FLOW,270,290,80,76)});
    require(!decision.stop&&mid.running(),"Traffic release after confirmed FLOW is broken");
}
