#pragma once

static void parkingPoseFieldChecks(const fs::path &dir) {
    nlohmann::json saved;
    std::ifstream(dir/"parking-route.json")>>saved;
    require(saved.at("parking_side")=="right"&&saved.at("reverse_verified")==true,"Wrong field parking route");
    RaceParkingRoute route;route.valid=true;route.entryCurvaturePreview=saved.at("entry_curvature_preview_m");
    for(const auto &p:saved.at("entry"))route.entry.push_back({p.at(0),p.at(1),p.at(2)});
    require(route.entry.size()>=2,"Missing field route");
    route.exit.assign(route.entry.rbegin(),route.entry.rend());
    RaceParkingSpace space;space.keepRearWithinStart(route.entry.front());
    const auto &dims=space.dimensions();
    const float mirror=saved.at("canonical_frame_handedness");
    std::ifstream input(dir/"parking-control.jsonl");
    std::vector<nlohmann::json> rows;
    for(std::string line;std::getline(input,line);) {
        auto r=nlohmann::json::parse(line);
        const int frame=r.at("frame");
        if(frame>=942&&frame<=989)rows.push_back(std::move(r));
        if(frame>=989)break;
    }
    require(rows.size()==48,"Incomplete parking field sequence");
    const auto row=[&](int frame)->const nlohmann::json & {
        const auto &r=rows.at(frame-942);
        require(r.at("frame")==frame,"Unexpected field frame number");
        return r;
    };
    const auto observation=[](const nlohmann::json &r) {
        const auto &p=r.at("visual_pose");
        return RaceParkingObservation{{p.at(0),p.at(1),p.at(2)},r.at("captured"),r.at("frame"),
            r.at("motion_epoch"),r.at("motion_valid"),r.at("motion_estimated_origin")};
    };
    const float cruise=row(943).at("task_cruise_limit");
    RaceParkingControl controller(space,cruise);
    RaceParkingPoseFallback fallback;
    RaceParkingObservation trusted;
    RaceParkingCommand command;
    require(row(942).at("armed")==false&&row(943).at("armed")==true&&row(986).at("armed")==false&&
        row(986).at("reason")=="PARK_POSE_DISCONTINUITY","Unexpected timing of recorded arming or fault");
    for(int frame=943;frame<=985;++frame) {
        const auto &r=row(frame),&prior=row(frame-1);
        const auto raw=observation(r);
        const double now=r.at("time");
        const float speed=prior.at("command_speed"),curvature=mirror*prior.at("physical_curvature").get<float>();
        if(frame==943)require(controller.arm(route,raw,now,true),"Cannot arm recorded parking start");
        trusted=fallback.update(raw,now,speed,curvature,dims);
        const auto &archived=r.at("pose");
        require(trusted.valid&&trusted.estimated==r.at("pose_estimated").get<bool>()&&
            std::hypot(trusted.pose.x-archived.at(0).get<float>(),trusted.pose.z-archived.at(1).get<float>())<.0001f&&
            std::abs(raceParkingAngle(trusted.pose.yaw-archived.at(2).get<float>()))<.0001f,
            "Replayed tracker fallback diverged before the original fault");
        command=controller.update(trusted,now,r.at("control_dt_s"),true,
            std::min(cruise,r.at("traffic_speed_cap").get<float>()),speed,curvature);
        require(!controller.failed()&&std::abs(command.speed-r.at("requested_speed").get<float>())<.0001f,
            "Control diverged before original discontinuity");
    }
    require(row(952).at("pose_estimated")==true&&!trusted.estimated,"Wrong estimated-frame replay");
    const auto &fault=row(986);
    const auto current=observation(fault);
    const double now=fault.at("time");
    const float dt=fault.at("control_dt_s"),applied=row(985).at("command_speed");
    const float wheelCurvature=mirror*row(985).at("physical_curvature").get<float>();
    const float cap=std::min(cruise,fault.at("traffic_speed_cap").get<float>());
    const auto rear=[&](RaceParkingPose p) {
        return cv::Point2f(p.x+dims.frontToRearAxle*std::sin(p.yaw),p.z-dims.frontToRearAxle*std::cos(p.yaw));
    };
    const float travel=float(cv::norm(rear(current.pose)-rear(trusted.pose)));
    const float oldLimit=std::max(.025f,.008f+1.6f*std::max(applied,row(984).at("command_speed").get<float>())*
        float(current.captured-trusted.captured));
    require(space.free(current.pose)&&fault.at("pose_source")=="VISUAL_ODOMETRY"&&fault.at("motion_inliers")==88&&
        fault.at("motion_residual_m").get<float>()<.0031f&&row(983).at("command_speed").get<float>()>.29f&&
        applied<=.12f&&travel-oldLimit>.00045f&&travel-oldLimit<.00055f,
        "Wrong field deceleration, registration or rear-axle discontinuity");
    const auto beforeFault=controller;
    command=controller.update(current,now,dt,true,cap,applied,wheelCurvature);
    require(controller.acceptedCoastObservation()&&std::string_view(command.state)=="PARK_ENTRY"&&command.speed>0&&command.speed<=cap,
        "Field frame 986 no longer follows the checked parking route");

    // Frames 987–989 followed the original stop. Keep their real coast separate
    // from the counterfactual trajectory after frame 986 receives a new command.
    for(int frame=986;frame<=989;++frame)require(row(frame).at("command_speed")==0,
        "Field coast contains a new positive command");
    const float coast=float(cv::norm(rear(observation(row(989)).pose)-rear(current.pose)));
    require(coast>.045f&&coast<.08f,"Wrong observed zero-command coast");
    auto model=current;
    const float replaySpeed=command.speed;
    float modelTravel=0;
    constexpr float modelDt=.08f;
    for(int i=1;i<=6;++i) {
        const float ds=command.speed*modelDt;
        modelTravel+=ds;
        model.pose=raceParkingAdvance(model.pose,ds,command.curvature,dims);
        ++model.sequence;model.captured+=modelDt;
        command=controller.update(model,now+i*modelDt,modelDt,true,cap,command.speed,command.curvature);
        require(space.free(model.pose)&&std::string_view(command.state)=="PARK_ENTRY"&&command.speed>0&&command.speed<=cap,
            "Model cannot sustain forward entry after the recovered frame");
    }
    require(modelTravel>.02f,"Insufficient modeled forward progression");

    // Flush the recorded higher speed: this control case has only steady
    // 0.12 m/s commands, with a contained forward fit 2 mm above the old cutoff.
    auto steady=beforeFault;
    auto measured=trusted;
    const float steadyDt=.09f,steadySpeed=.12f;
    for(int i=0;i<8;++i) {
        measured.pose=raceParkingAdvance(measured.pose,steadySpeed*steadyDt,0,dims);
        measured.captured+=steadyDt;++measured.sequence;
        const auto c=steady.update(measured,measured.captured+.05,steadyDt,true,cap,steadySpeed,0);
        require(c.speed>0&&space.free(measured.pose),"Failed to build steady-speed test history");
    }
    auto small=measured;small.captured+=steadyDt;++small.sequence;
    const float steadyLimit=std::max(.025f,.008f+1.6f*steadySpeed*steadyDt);
    small.pose=raceParkingAdvance(measured.pose,steadyLimit+.002f,0,dims);
    auto steadyCopy=steady;
    const auto steadyCommand=steadyCopy.update(small,small.captured+.05,steadyDt,true,cap,steadySpeed,0);
    require(steadyCommand.speed>0&&space.free(small.pose),"A two-millimetre steady-speed measurement overshoot still stops entry");
    small.estimated=true;
    require(steady.update(small,small.captured+.05,steadyDt,true,cap,steadySpeed,0).speed==0,
        "Estimated-origin fit uses the allowance for absolute measurements");

    // Unlike the small measured correction at 952–953, a larger change from
    // an estimated origin requires new absolute agreement before motion resumes.
    auto returnProbe=beforeFault;
    auto recovered=trusted;recovered.estimated=true;
    recovered.pose=raceParkingAdvance(recovered.pose,applied*modelDt,0,dims);
    ++recovered.sequence;recovered.captured+=modelDt;
    require(returnProbe.update(recovered,recovered.captured+.05,modelDt,true,cap,applied,wheelCurvature).speed>0,
        "Short existing fallback cannot bridge a missing frame");
    recovered.estimated=false;recovered.pose=raceParkingAdvance(recovered.pose,.045f,0,dims);
    for(int i=0;i<4;++i) {
        ++recovered.sequence;recovered.captured+=modelDt;
        const auto c=returnProbe.update(recovered,recovered.captured+.05,modelDt,true,cap,i==0?applied:0.f,0);
        if(i<2)require(!returnProbe.failed()&&c.speed==0&&std::string_view(c.reason)=="PARK_VERIFY_POSE_DISCONTINUITY",
            "Large estimate-to-absolute correction accepted without two independent confirmations");
        if(i==2)require(c.speed==0&&std::string_view(c.reason)=="PARK_ORIGINAL_POSE_CONFIRMED",
            "Bounded forward return cannot rejoin absolute coordinates");
        if(i==3)require(c.speed>0,"Verified absolute return did not resume normal entry");
    }

    const auto checkStopped=[&](RaceParkingObservation o,double time,bool clear,bool mustFault) {
        auto probe=beforeFault;
        const auto c=probe.update(o,time,dt,clear,cap,applied,wheelCurvature);
        require(c.speed==0&&probe.failed()==mustFault,"Unexpected response to unsafe pose or timing");
    };
    auto jump=current;jump.pose.x+=.30f;
    require(!space.free(jump.pose),"Large jump does not cross physical aisle edge");
    checkStopped(jump,now,true,true);
    auto epoch=current;++epoch.epoch;checkStopped(epoch,now,true,true);
    auto reordered=current;reordered.captured=trusted.captured;checkStopped(reordered,now,true,true);
    auto stale=current;checkStopped(stale,current.captured+.25,true,false);
    checkStopped(current,now,false,false);

    auto lateral=current;lateral.pose=trusted.pose;lateral.pose.x+=.030f;
    auto backward=current;backward.pose=trusted.pose;backward.pose.z-=.030f;
    auto yaw=current;yaw.pose=trusted.pose;yaw.pose.yaw+=.55f;
    checkStopped(backward,now,true,false);
    checkStopped(yaw,now,true,true);
    auto estimated=current;estimated.estimated=true;
    checkStopped(estimated,now,true,false);
    auto suspect=beforeFault;
    require(std::string_view(suspect.update(lateral,now,dt,true,cap,applied,wheelCurvature).reason)==
        "PARK_VERIFY_POSE_DISCONTINUITY","Unexplained sideways jump was accepted as coast");

    // Multiple stable frames in a displaced coordinate frame must never validate
    // the displacement. A repeated sequence or estimated origin cannot confirm return.
    auto displaced=suspect;
    auto sample=lateral;
    for(int i=1;i<=18&&!displaced.failed();++i) {
        sample.captured+=modelDt;++sample.sequence;
        require(displaced.update(sample,now+i*modelDt,modelDt,true,cap,0,0).speed==0,
            "Persistent lateral displacement was accepted as a new origin");
    }
    require(displaced.failed(),"Persistent outlier never locks parking");
    require(suspect.update(lateral,now,dt,true,cap,0,0).speed==0,"Duplicate frame releases pose protection");
    auto interrupted=suspect;
    sample=lateral;
    for(int i=1;i<=33;++i) {
        ++sample.sequence;sample.captured+=modelDt;
        sample.pose=trusted.pose;
        // More than one second each behind a closed gate and without usable vision.
        const bool trafficClear=i>15;
        sample.valid=i<=15||i>30;
        const auto c=interrupted.update(sample,now+i*modelDt,modelDt,trafficClear,cap,0,0);
        require(!interrupted.failed(),"Traffic or missing vision consumed the pose verification timeout");
        if(i<32)require(c.speed==0&&std::string_view(c.reason)!="PARK_ORIGINAL_POSE_CONFIRMED",
            "Stopped or unavailable frames confirmed a suspect pose");
        if(i==32)require(c.speed==0&&std::string_view(c.reason)=="PARK_ORIGINAL_POSE_CONFIRMED",
            "Two fresh absolute frames do not restore the route after a long stop");
        if(i==33)require(c.speed>0,"Cannot resume after traffic and visual recovery");
    }
    sample=current;sample.pose=trusted.pose;
    for(int i=1;i<=5;++i) {
        ++sample.sequence;sample.captured+=modelDt;
        // Frame 2 imitates a rigid visual fit relative to an estimated origin.
        // It must restart confirmation even though its coordinates equal the trusted ones.
        sample.estimated=i==2;
        const auto c=suspect.update(sample,now+i*modelDt,modelDt,true,cap,0,0);
        if(i<4)require(c.speed==0&&!suspect.failed()&&std::string_view(c.reason)!="PARK_ORIGINAL_POSE_CONFIRMED",
            "Pose protection used an estimate or a single confirmation");
        if(i==4)require(c.speed==0&&std::string_view(c.reason)=="PARK_ORIGINAL_POSE_CONFIRMED",
            "Second fresh absolute confirmation did not restore original route while stopped");
        if(i==5)require(c.speed>0&&std::string_view(c.state)=="PARK_ENTRY",
            "Parking failed to resume on the next confirmed frame");
    }
    // Move a copy of the controller up to the aisle edge in small, accepted steps.
    // Give the same deceleration history to two adjacent endpoints: only the one
    // whose full footprint stays inside may use the coast exception.
    auto edge=beforeFault;
    sample=trusted;
    float outer=-std::numeric_limits<float>::infinity();
    for(auto p:space.body(sample.pose))outer=std::max(outer,p.x);
    const float toEdge=space.dimensions().aisle/2-.001f-outer;
    constexpr float edgeDt=.09f;
    for(int i=1;i<=10;++i) {
        sample.pose.x+=toEdge/10;sample.pose.z+=.006f;
        ++sample.sequence;sample.captured+=edgeDt;
        edge.update(sample,sample.captured+.04,edgeDt,true,cruise,i<10?.30f:.12f,0);
        require(space.free(sample.pose)&&!edge.failed(),"Synthetic reference cannot approach aisle edge");
    }
    auto inside=edge;
    auto endpoint=sample;endpoint.pose.z+=.030f;++endpoint.sequence;endpoint.captured+=edgeDt;
    require(space.free(endpoint.pose)&&inside.update(endpoint,endpoint.captured+.04,edgeDt,true,cap,.12f,0).speed>0&&
        inside.acceptedCoastObservation(),"Contained control case lost deceleration context");
    endpoint.pose.x+=.003f;
    require(!space.free(endpoint.pose)&&edge.update(endpoint,endpoint.captured+.04,edgeDt,true,cap,.12f,0).speed==0&&
        !edge.acceptedCoastObservation(),"Deceleration exception accepts a footprint outside the real aisle edge");
    std::cout<<"PASS parking pose field "<<dir.filename().string()<<" replay=943-986 old_limit_mm="<<1000*oldLimit
             <<" rear_travel_mm="<<1000*travel<<" replay_command_mps="<<replaySpeed<<" actual_coast_m="<<coast
             <<" modeled_travel_m="<<modelTravel<<" physical retest pending\n";
}
