#pragma once

static void parkingFinalEntryFieldChecks(const fs::path &dir) {
    nlohmann::json file;std::ifstream(dir/"parking-route.json")>>file;
    RaceParkingRoute saved;saved.valid=true;saved.entryCurvaturePreview=file.at("entry_curvature_preview_m");
    for(const auto &p:file.at("entry"))saved.entry.push_back({p.at(0),p.at(1),p.at(2)});
    saved.exit.assign(saved.entry.rbegin(),saved.entry.rend());
    RaceParkingSpace space;space.keepRearWithinStart(saved.entry.front());const auto &d=space.dimensions();
    const auto planned=racePlanTrackableParking(space,saved.entry.front(),.30f);
    require(planned.valid&&raceParkingModelCompletes(space,planned,.30f)&&
        raceParkingModelCompletes(space,planned,.30f,.08f,true),"New long-approach plan cannot park without an inset");
    auto clearance=[&](const RaceParkingRoute &r) {
        float edge=1,yaw=0;
        for(auto p:r.entry) {
            yaw=std::min(yaw,p.yaw);
            for(auto corner:space.body(p,0))edge=std::min(edge,d.aisle*.5f-corner.x);
        }
        return std::pair{edge,yaw};
    };
    const auto [oldEdge,oldYaw]=clearance(saved);const auto [edge,yaw]=clearance(planned);
    require(edge>oldEdge+.03f&&yaw>oldYaw+.1f,"New approach still swings against the opposite aisle edge");

    RaceParkingControl control(space,.30f);
    RaceParkingObservation o;RaceParkingCommand cmd;float speed=0,curvature=0;
    bool armed=false;uint64_t brake=0,started=0;
    std::ifstream input(dir/"parking-control.jsonl");std::string line;
    while(std::getline(input,line)) {
        const auto row=nlohmann::json::parse(line);
        if(!row.at("armed").get<bool>())continue;
        const auto p=row.at("visual_pose");
        o={{p[0],p[1],p[2]},row.at("captured"),row.at("frame"),row.at("motion_epoch"),
           row.at("motion_valid"),row.at("pose_estimated")};
        if(!armed){armed=control.arm(saved,o,o.captured,true);require(armed,"Recorded entry did not arm");}
        cmd=control.update(o,o.captured,row.at("control_dt_s"),true,.30f,speed,curvature);
        if(std::string_view(cmd.state)=="PARK_BRAKE"&&!brake)brake=o.sequence;
        if(control.usedFinalInset()) {
            require(row.at("state")=="PARK_DWELL"&&row.at("terminal_recovery").get<bool>()&&
                !space.parkedObserved(o.pose)&&space.free(o.pose)&&cmd.speed==0,
                "Inset did not start with the recorded short, stopped, paint-clear pose");
            started=o.sequence;break; // New commands change the future poses.
        }
        speed=row.at("command_speed");
        curvature=file.at("canonical_frame_handedness").get<float>()*row.at("physical_curvature").get<float>();
    }
    require(brake&&started>brake,"Missing protective brake and measured short stop");
    constexpr float dt=.08f;
    auto paused=control;auto unavailable=o;++unavailable.sequence;unavailable.captured+=dt;
    unavailable.estimated=true;
    require(paused.update(unavailable,unavailable.captured,dt,true,.30f,0,0).speed==0,
        "Estimated pose permits a final correcting motion");
    unavailable.estimated=false;++unavailable.sequence;unavailable.captured+=dt;
    require(paused.update(unavailable,unavailable.captured,dt,false,.30f,0,0).speed==0,
        "Traffic stop permits a final correcting motion");
    require(!raceParkingInsetWheel(space,o.pose,0,.03f),"A short budget permits an incomplete inset");
    auto nearDivider=o.pose;nearDivider.z-=.07f;
    require(!raceParkingInsetWheel(space,nearDivider,0,.14f),"Unsafe divider position permits further entry");

    // Closed-loop continuation from the real stopped pose. Model steering
    // response and zero-command coast; subsequent archived poses would be wrong.
    float appliedWheel=0,rollingWheel=0,rollingSpeed=0,travel=0;bool reverse=false;
    auto advancePlant=[&](RaceParkingObservation &obs,RaceParkingCommand command,float drift) {
        const float request=std::atan(d.wheelbase*command.curvature);
        appliedWheel=raceParkingSteeringStep(appliedWheel,request,dt,d);
        rollingWheel+=(appliedWheel-rollingWheel)*(1-std::exp(-dt/.25f));
        rollingSpeed+=(command.speed-rollingSpeed)*(1-std::exp(-dt/.38f));
        const float curvature=std::tan(rollingWheel)/d.wheelbase,yawStep=rollingSpeed*dt*curvature;
        obs.pose=raceParkingAdvance(obs.pose,rollingSpeed*dt,curvature,d);
        obs.pose.x-=drift*yawStep*std::cos(obs.pose.yaw);obs.pose.z-=drift*yawStep*std::sin(obs.pose.yaw);
        obs.captured+=dt;++obs.sequence;
    };
    for(int i=0;i<350&&!control.done();++i) {
        advancePlant(o,cmd,0);
        if(!reverse)travel+=rollingSpeed*dt;
        require(space.free(o.pose),"Inset, reverse or forward alignment crosses a painted boundary");
        cmd=control.update(o,o.captured,dt,true,.30f,cmd.speed,std::tan(appliedWheel)/d.wheelbase);
        if(std::string_view(cmd.reason)=="DWELL_COMPLETE_REVERSE_NEXT_FRAME") {
            require(raceParkingInsetContained(space,o.pose)&&!control.terminalRecoveryUsed()&&
                !control.reverseReference().empty()&&std::hypot(o.pose.x-control.reverseReference().front().x,
                o.pose.z-control.reverseReference().front().z)<.003f,"Inset did not finish fully inside at the recorded reverse origin");
            reverse=true;
        }
        require(!control.failed()&&(reverse||(cmd.speed>=0&&cmd.speed<=.04001f)),"Unbounded or failed final inset");
    }
    require(reverse&&travel>.085f&&travel<=.14f&&control.done()&&raceParkingReadyToRejoin(space,o.pose),
        "Recorded short stop cannot complete a bounded inset, dwell and aisle return");
    for(float drift:{0.f,.09f}) {
        RaceParkingControl fromStart(space,.30f);RaceParkingObservation obs{planned.entry.front(),1,1,1,true};
        require(fromStart.arm(planned,obs,1,true),"New route does not arm");
        appliedWheel=rollingWheel=rollingSpeed=0;RaceParkingCommand command;
        for(int i=0;i<600&&!fromStart.done();++i) {
            advancePlant(obs,command,drift);require(space.free(obs.pose),"New route crosses paint in the delayed model");
            command=fromStart.update(obs,obs.captured,dt,true,.30f,command.speed,std::tan(appliedWheel)/d.wheelbase);
            require(!fromStart.failed()&&!fromStart.usedFinalInset(),"New route relies on field correction");
        }
        require(fromStart.done()&&!fromStart.terminalRecoveryUsed()&&raceParkingReadyToRejoin(space,obs.pose),
            "Delayed model did not fully park and rejoin the aisle");
    }
    std::cout<<"PASS parking final entry "<<dir.filename().string()<<" brake="<<brake<<" stopped_correction="<<started
        <<" planned_outward_deg="<<oldYaw*180/CV_PI<<"->"<<yaw*180/CV_PI
        <<" opposite_clearance_cm="<<100*oldEdge<<"->"<<100*edge<<" modeled_inset_cm="<<100*travel
        <<"; measured prefix, modeled continuation; 5mm swept-body clearance through dwell, reverse and aisle rejoin\n";
}
