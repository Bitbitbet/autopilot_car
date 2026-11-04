#pragma once

static void parkingEntranceGateFieldChecks(const fs::path &dir) {
    std::ifstream input(dir/"parking-control.jsonl");
    std::map<int,nlohmann::json> rows;
    for(std::string line;std::getline(input,line);) {
        auto row=nlohmann::json::parse(line);const int frame=row.at("frame");
        if(frame>=655)rows.emplace(frame,std::move(row));
        if(frame==778)break;
    }
    require(rows.size()==124&&rows.at(726).at("reason")=="F_FREEZE_AT_BAY_2"&&
        rows.at(778).at("reason")=="PARK_START_FORWARD_JOIN_REPLAN","Wrong entrance/start field run");

    // Use the actual detector boxes and timing. Images of the raised interval
    // were not retained: only the closed/clear passage masks here are synthetic.
    RaceConfig cfg;cfg.load((dir/"race-config.json").string());
    RaceManager manager(cfg);Track track;
    cv::Mat view(480,640,CV_8UC3,cv::Scalar(70,70,70));
    const cv::Mat closed(480,640,CV_8UC1,cv::Scalar(0)),clear(480,640,CV_8UC1,cv::Scalar(255));
    int released=0;
    for(int f=655;f<=725;++f) {
        const auto &r=rows.at(f);const double now=r.at("time");
        RaceSnapshot s;s.image=view;s.sequence=r.at("ai_sequence");s.captured=now-r.at("ai_age_s").get<double>();
        for(const auto &d:r.at("ai_detections"))if(d.at("type")==LABEL_GATE) {
            const auto &b=d.at("box");auto det=box(LABEL_GATE,b[0],b[1],b[2],b[3]);
            det.score=d.at("score");s.detections.push_back(det);
        }
        const auto p=manager.update(s,now,r.at("control_dt_s"),true,f>=697?clear:closed,track,view,true,true);
        if(f==661)require(p.stop&&std::string_view(p.reason)=="WAIT_FOR_GATE_VIEW","Missed distant bar did not hold for a fresh view");
        if(f==678)require(p.stop&&std::string_view(p.state)=="GATE_WAIT","Closed bar did not stop at the approach row");
        if(p.gateOpened){require(!released,"Same entrance gate reopened twice");released=f;}
        if(released)require(!p.stop&&!p.gateWaiting,"Cleared entrance stopped the car again");
    }
    require(released>=710&&released<=714&&rows.at(725).at("command_speed")>0,
        "Entrance did not release after the continuous clear dwell");

    nlohmann::json saved;std::ifstream(dir/"parking-route.json")>>saved;
    RaceParkingRoute route;route.valid=true;route.entryCurvaturePreview=saved.at("entry_curvature_preview_m");
    for(const auto &p:saved.at("entry"))route.entry.push_back({p[0],p[1],p[2]});
    route.exit.assign(route.entry.rbegin(),route.entry.rend());
    RaceParkingSpace space;space.keepRearWithinStart(route.entry.front());
    const auto observation=[](const nlohmann::json &r) {
        const auto &p=r.at("visual_pose");
        return RaceParkingObservation{{p[0],p[1],p[2]},r.at("captured"),r.at("frame"),r.at("motion_epoch"),
            r.at("motion_valid"),r.at("motion_estimated_origin")};
    };
    RaceParkingControl control(space,rows.at(734).at("task_cruise_limit"));
    auto o=observation(rows.at(734));
    require(!space.freeStartObserved(o.pose)&&control.arm(route,o,rows.at(734).at("time"),true),
        "Field start does not require a limited straight correction");
    RaceParkingCommand cmd;
    for(int f=735;f<=777;++f) {
        const auto &r=rows.at(f),&prev=rows.at(f-1);o=observation(r);
        cmd=control.update(o,r.at("time"),r.at("control_dt_s"),true,r.at("traffic_speed_cap"),prev.at("command_speed"),0);
        require(cmd.speed>0&&!control.failed(),"Start correction stopped before a safe departure");
        if(std::string_view(cmd.reason)!="FORWARD_TO_PLANNED_START")break;
    }
    require(!space.free(o.pose)&&space.freeStartObserved(o.pose)&&std::string_view(cmd.reason)!="FORWARD_TO_PLANNED_START"&&
        !cmd.predictedBoundaryWarning,"Controller starved in the measured startup window");
    auto stalled=control;auto stationary=o;auto stationaryCommand=cmd;
    for(int i=0;i<55&&!stalled.failed();++i) {
        stationary.captured+=.10;++stationary.sequence;
        stationaryCommand=stalled.update(stationary,stationary.captured+.05,.10f,true,.30f,stationaryCommand.speed,0);
    }
    require(stalled.failed()&&stationaryCommand.speed==0,"Stationary car accumulated unchecked departure travel");
    auto traffic=control;auto stopped=o;stopped.captured+=.1;++stopped.sequence;
    require(traffic.update(stopped,stopped.captured+.05,.1f,false,.30f,cmd.speed,0).speed==0,
        "Traffic hold was bypassed near the virtual starting plane");
    // Later field frames followed the old command. Simulate only after divergence.
    for(int i=0;i<12&&!space.free(o.pose);++i) {
        o.pose=raceParkingAdvance(o.pose,cmd.speed*.08f,cmd.curvature,space.dimensions());o.captured+=.08;++o.sequence;
        cmd=control.update(o,o.captured+.04,.08f,true,.30f,cmd.speed,cmd.curvature);
        require(cmd.speed>0&&!control.failed()&&space.insideAisle(o.pose),"Safe forward departure did not reach the original route");
    }
    require(space.free(o.pose),"Measured departure did not enter strict planning space");
    std::cout<<"PASS parking entrance gate hold, release, bay stop, and measured start correction\n";
}
