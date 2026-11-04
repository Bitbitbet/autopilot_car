#pragma once

// Last right-bay failure: saved observations are evidence, never a simulation
// of the new physical response. Closed-loop checks below are explicitly models.
static void parkingEntryFieldChecks(const fs::path &root) {
    const auto folder=root/"vehicle_evidence/latest-20261102-213015/mission-1793625776685";
    nlohmann::json saved;std::ifstream(folder/"parking-route.json")>>saved;
    RaceParkingRoute route;route.valid=true;route.entryCurvaturePreview=saved.at("entry_curvature_preview_m");
    for(const auto &p:saved.at("entry"))route.entry.push_back({p[0],p[1],p[2]});
    route.exit=route.entry;std::reverse(route.exit.begin(),route.exit.end());
    RaceParkingSpace space;space.keepRearWithinStart(route.entry.front());
    std::vector<nlohmann::json> rows;std::ifstream input(folder/"parking-control.jsonl");std::string line;
    while(std::getline(input,line)) {
        auto r=nlohmann::json::parse(line);const int frame=r.at("frame");
        if(frame>=967&&frame<=1001)rows.push_back(std::move(r));
    }
    require(rows.size()==35,"Latest parking entry evidence is incomplete");
    auto observation=[](const nlohmann::json &r){
        const auto p=r.at("pose");return RaceParkingObservation{{p[0],p[1],p[2]},r.at("captured"),
            r.at("frame"),r.at("motion_epoch"),r.at("motion_valid")};
    };
    const auto initial=observation(rows.front());
    require(!space.free(initial.pose)&&!space.freeObserved(initial.pose)&&space.freeStartObserved(initial.pose),
        "Latest initial pose no longer exercises startup-only rear tolerance");
    require(std::abs(initial.pose.z-route.entry.front().z)>.004f&&
        std::abs(initial.pose.z-route.entry.front().z)<.005f,"Wrong latest stopped-frame rear drift");
    for(size_t i=0;i<16;++i)require(std::abs(route.entry[i].yaw-route.entry.front().yaw)<1e-6f,
        "Latest field route does not begin with the recorded straight approach");
    {
        RaceParkingControl control(space,.3f);
        require(control.arm(route,initial,rows.front().at("time"),true),"Latest stopped pose cannot arm");
        float appliedSpeed=0,appliedCurvature=0;
        for(size_t i=1;i<=6;++i) {
            const auto &r=rows[i];const auto c=control.update(observation(r),r.at("time"),r.at("control_dt_s"),
                true,.3f,appliedSpeed,appliedCurvature);
            require(!control.failed()&&!c.entryMouthRecovery&&std::abs(c.requestedCurvature)<.3f,
                "Millimetre rear drift still invokes mouth full-lock on a straight approach");
            appliedSpeed=c.speed;appliedCurvature=c.curvature;
        }
        require(!space.free(initial.pose),"Startup handling silently enlarges planner space");
        std::cout<<"PASS latest right-bay frames967-973: startup rear drift does not trigger mouth steering; planning boundary retained\n";
    }
    const std::array<float,16> fieldDt{.0835f,.0792f,.0994f,.1017f,.1049f,.0995f,.0787f,.0819f,
        .0868f,.0983f,.1278f,.1410f,.1475f,.1431f,.1179f,.1160f};
    for(int sampling=0;sampling<3;++sampling) {
        RaceParkingControl control(space,.3f);auto o=initial;o.captured=1;o.sequence=1;
        require(control.arm(route,o,1,true),"Recorded-drift closed-loop model cannot arm");
        float appliedSpeed=0,appliedCurvature=0,wheel=0;double time=1;bool inside=false,dwelled=false,reversed=false;
        for(int step=0;step<4000&&!control.done()&&!control.failed();++step) {
            const float dt=sampling==0?.04f:sampling==1?.10f:fieldDt[step%fieldDt.size()];
            time+=dt;o.captured=time;++o.sequence;
            const auto c=control.update(o,time,dt,true,.3f,appliedSpeed,appliedCurvature);
            inside|=space.parkedObserved(o.pose);dwelled|=std::string(c.state)=="PARK_DWELL";
            reversed|=c.speed<0;appliedSpeed=c.speed;
            const auto &d=space.dimensions();
            wheel=raceParkingSteeringStep(wheel,std::atan(d.wheelbase*c.curvature),dt,d);
            const float pwm=std::round(1500-wheel/(d.fullLockDegrees*float(CV_PI)/180)*400);
            wheel=(1500-pwm)/400*d.fullLockDegrees*float(CV_PI)/180;
            appliedCurvature=std::tan(wheel)/d.wheelbase;
            o.pose=raceParkingAdvance(o.pose,appliedSpeed*dt,appliedCurvature,space.dimensions());
            const bool initialRearNoise=o.pose.z<route.entry.front().z+.03f&&
                std::abs(raceParkingAngle(o.pose.yaw-route.entry.front().yaw))<.04f&&space.freeStartObserved(o.pose);
            if(!space.free(o.pose)&&!initialRearNoise)std::cerr<<"latest-model sampling="<<sampling
                <<" step="<<step<<" pose="<<o.pose.x<<','<<o.pose.z<<','<<o.pose.yaw<<" state="<<c.state<<'\n';
            require(space.free(o.pose)||initialRearNoise,"Latest-start closed-loop model crosses parking boundary");
        }
        if(!control.done()||control.terminalRecoveryUsed()||control.exitRecoveryUsed())std::cerr
            <<"latest-model sampling="<<sampling<<" state="<<control.status().state<<" reason="<<control.status().reason<<'\n';
        require(control.done()&&!control.terminalRecoveryUsed()&&!control.exitRecoveryUsed()&&inside&&dwelled&&reversed,
            "Latest-start closed-loop model fails full entry, dwell or measured reverse/rejoin");
        std::cout<<"PASS latest stopped start closed-loop model sampling="<<(sampling==0?".04":sampling==1?".10":"recorded .079-.1475")
            <<": contained entry, dwell, reverse and aisle rejoin; not physical validation\n";
    }
    {
        RaceParkingControl control(space,.3f);
        require(control.arm(route,initial,rows.front().at("time"),true),"Continuity replay cannot arm");
        for(size_t i=1;i<rows.size();++i) {
            const auto &r=rows[i],&previous=rows[i-1];
            // Right-bay mirror: positive canonical curvature corresponds to
            // physical right steering. Use APPLIED PWM, not requested PWM.
            const float wheel=(previous.at("pwm").get<float>()-1500)/400*
                space.dimensions().fullLockDegrees*float(CV_PI)/180;
            control.update(observation(r),r.at("time"),r.at("control_dt_s"),true,.3f,
                previous.at("command_speed"),std::tan(wheel)/space.dimensions().wheelbase);
            require(!control.failed(),"Latest continuous rigid-motion observations still cause a false discontinuity");
        }
        std::cout<<"PASS latest frames967-1001 continuous observation replay with applied right-bay PWM; not a new vehicle trajectory\n";
    }
    {
        RaceParkingControl control(space,.3f);auto o=initial;o.captured=1;o.sequence=1;
        require(control.arm(route,o,1,true),"Discontinuity test cannot arm");
        o.captured+=.1;++o.sequence;o.pose.x+=.30f;
        const auto c=control.update(o,1.1,.1f,true,.3f,0,0);
        require(c.speed==0&&control.failed(),"A genuine 30cm observation jump is allowed to move");
        std::cout<<"PASS genuine pose discontinuity still stops parking\n";
    }
}
