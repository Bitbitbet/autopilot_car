#pragma once

static void parkingClearanceFieldChecks(const fs::path &dir) {
    nlohmann::json saved,summary;
    std::ifstream(dir/"parking-route.json")>>saved;
    std::ifstream(dir/"parking-test-summary.json")>>summary;
    const auto &attempt=summary.at("retained_attempts").at(0);
    require(saved.at("parking_side")=="left"&&attempt.at("full_body_inside_observed")==false&&
        attempt.at("terminal_recovery_used")==true&&attempt.at("reverse_motion_commanded")==true,
        "Wrong incomplete parked-body field run");
    RaceParkingRoute route;route.valid=true;route.entryCurvaturePreview=saved.at("entry_curvature_preview_m");
    for(const auto &p:saved.at("entry"))route.entry.push_back({p.at(0),p.at(1),p.at(2)});
    route.exit.assign(route.entry.rbegin(),route.entry.rend());
    RaceParkingSpace space;space.keepRearWithinStart(route.entry.at(0));const auto &d=space.dimensions();
    const auto observation=[](const nlohmann::json &r) {
        const auto &p=r.at("pose");
        return RaceParkingObservation{{p.at(0),p.at(1),p.at(2)},r.at("captured"),r.at("frame"),r.at("motion_epoch"),
            r.at("control_pose_valid"),r.at("pose_estimated")};
    };
    std::map<int,nlohmann::json> rows;
    std::ifstream input(dir/"parking-control.jsonl");
    for(std::string line;std::getline(input,line);) {
        auto r=nlohmann::json::parse(line);const int frame=r.at("frame");
        if(frame>=774)rows.emplace(frame,std::move(r));
        if(frame>=912)break;
    }
    require(rows.size()==139,"Missing parking clearance field records");
    // Signed distances to the opposite aisle edge, mouth and the three closed bay
    // sides, using the physical corners. Negative mouth clearance is expected in transit.
    const auto margins=[&](RaceParkingPose p) {
        std::array<float,5> m;m.fill(10);
        for(auto corner:space.body(p,0)) {
            const float depth=-d.aisle/2-corner.x;
            const float height=(corner.y-depth*d.stagger()/d.depth)*d.depth/d.side;
            m[0]=std::min(m[0],d.aisle/2-corner.x);m[1]=std::min(m[1],depth);
            m[2]=std::min(m[2],d.depth-depth);m[3]=std::min(m[3],height);m[4]=std::min(m[4],d.depth-height);
        }
        return m;
    };
    const auto stopped=observation(rows.at(868));
    const float entryEdge=margins(observation(rows.at(827)).pose)[0],exitEdge=margins(observation(rows.at(907)).pose)[0];
    const auto end=margins(stopped.pose);
    require(rows.at(827).at("pose_source")=="VISUAL_ODOMETRY"&&rows.at(907).at("pose_source")=="VISUAL_ODOMETRY"&&
        entryEdge<-.023f&&entryEdge>-.024f&&exitEdge<-.0065f&&exitEdge>-.0075f&&
        end[1]<-.074f&&end[1]>-.075f&&end[3]>.0055f&&end[3]<.006f,
        "Wrong field opposite-edge crossings, rear shortfall or near-divider clearance");
    float plannedMargin=1;
    for(auto p:route.entry)plannedMargin=std::min(plannedMargin,margins(p)[0]);
    require(plannedMargin>.073f&&plannedMargin<.074f,"Saved nominal path does not match the field route");

    const float cruise=rows.at(775).at("task_cruise_limit"),mirror=saved.at("canonical_frame_handedness");
    RaceParkingControl replay(space,cruise);RaceParkingCommand cmd;RaceParkingObservation sample;
    int counter=0,entryStop=0,inset=0;
    for(int f=775;f<=878;++f) {
        const auto &r=rows.at(f),&prior=rows.at(f-1);sample=observation(r);
        if(f==775)require(replay.arm(route,sample,r.at("time"),saved.at("reverse_verified")),"Field left-bay route failed to arm");
        cmd=replay.update(sample,r.at("time"),r.at("control_dt_s"),true,
            std::min(cruise,r.at("traffic_speed_cap").get<float>()),prior.at("command_speed"),
            mirror*prior.at("physical_curvature").get<float>());
        const float requested=r.at("requested_curvature");
        if(f>=794&&f<=813&&requested<-.5f&&cmd.requestedCurvature>requested+.3f&&!counter)counter=f;
        if(std::string_view(cmd.reason)=="OPPOSITE_EDGE_STEERING_RECOVERY"&&cmd.speed==0&&!entryStop)entryStop=f;
        if(f>=868) {
            // Isolate the terminal policy on the actual rest pose. Earlier
            // guard decisions already change the real future of the approach.
            if(std::string_view(cmd.state)=="PARK_INSET_ADJUST")inset=f;
            break;
        }
    }
    require(counter&&counter<813&&entryStop>=counter&&entryStop<824&&
        margins(observation(rows.at(entryStop)).pose)[0]>.015f,
        "Controller does not unwind right steering and brake before field entry crosses the opposite line");
    int reverseGuard=0;
    for(int f=879;f<=907;++f) {
        const auto o=observation(rows.at(f)),prior=observation(rows.at(f-1));
        if(space.exitEdge(o.pose,prior.pose,float(o.captured-prior.captured)).align){reverseGuard=f;break;}
    }
    require(reverseGuard>=890&&reverseGuard<=900&&margins(observation(rows.at(reverseGuard)).pose)[0]>.035f&&
        !space.exitEdge(stopped.pose,stopped.pose,.08f).align,
        "Reverse guard misses the closing aisle edge or activates while still in the bay");
    require(inset==868,"Incomplete parking did not start its guarded correction at measured standstill");
    std::cout<<"parking clearance input replay counter="<<counter<<" entrance_brake="<<entryStop
             <<" reverse_edge_guard="<<reverseGuard<<" stopped_correction="<<inset<<'\n';
    require(raceParkingInsetClear(space,stopped.pose)&&!raceParkingInsetContained(space,stopped.pose)&&
        !raceParkingInsetWheel(space,stopped.pose,0,.16f)&&raceParkingCanStageInset(space,stopped.pose)&&
        raceParkingCanStageInset(space,observation(rows.at(877)).pose),
        "Recorded rest poses do not admit a guarded setback and insertion");
    auto outside=stopped.pose;outside.z-=.018f;
    require(margins(outside)[3]<-.01f&&!raceParkingInsetClear(space,outside)&&
        !raceParkingInsetWheel(space,outside,0,.16f)&&!raceParkingCanStageInset(space,outside),
        "Body ten millimetres over the real divider can start a correction");
    const auto tooShort=raceParkingAdvance(stopped.pose,-.05f,0,d);
    require(raceParkingInsetClear(space,tooShort)&&!raceParkingCanStageInset(space,tooShort),
        "A distant unparked body can start the short final correction");
    struct TerminalResult {float back=0,forward=0,minEdge=1;bool dwelled=false;};
    TerminalResult terminal;
    for(float slip:{0.f,.09f}) {
        auto corrected=replay;auto o=stopped;auto c=cmd;
        float wheel=0,rollingWheel=0,rollingSpeed=0;
        constexpr float tick=.08f;
        double now=rows.at(868).at("time");
        TerminalResult model;bool preset=false,advanced=false;int exitFrames=0;
        const auto advance=[&] {
            wheel=raceParkingSteeringStep(wheel,std::atan(d.wheelbase*c.curvature),tick,d);
            rollingWheel+=(wheel-rollingWheel)*(1-std::exp(-tick/.25f));
            rollingSpeed+=(c.speed-rollingSpeed)*(1-std::exp(-tick/.38f));
            const float ds=rollingSpeed*tick,k=std::tan(rollingWheel)/d.wheelbase;
            o.pose=raceParkingAdvance(o.pose,ds,k,d);
            o.pose.x-=slip*ds*k*std::cos(o.pose.yaw);o.pose.z-=slip*ds*k*std::sin(o.pose.yaw);
            o.captured+=tick;++o.sequence;now+=tick;
            c=corrected.update(o,now,tick,true,cruise,c.speed,std::tan(wheel)/d.wheelbase);
            return ds;
        };
        for(int i=0;i<260&&!model.dwelled&&!corrected.failed();++i) {
            require(c.speed>=-.041f&&c.speed<=.041f,"Terminal correction exceeds its low speed limit");
            const float ds=advance();
            model.back+=std::max(0.f,-ds);model.forward+=std::max(0.f,ds);
            const auto m=margins(o.pose);model.minEdge=std::min({model.minEdge,m[0],m[2],m[3],m[4]});
            require(model.minEdge>=.00499f&&raceParkingInsetClear(space,o.pose),"Terminal correction crosses its physical paint reserve");
            if(c.speed==0&&std::abs(wheel)>.08f)preset=true;
            if(c.speed>0) {
                require(model.back>.008f&&preset,"Forward correction starts before a measured setback and steering preset");
                if(!advanced) {
                    auto paused=corrected,estimated=corrected;auto next=o;next.captured+=tick;++next.sequence;
                    require(paused.update(next,now+tick,tick,false,cruise,0,0).speed==0,"Traffic stop permits forward correction");
                    next.estimated=true;
                    require(estimated.update(next,now+tick,tick,true,cruise,0,0).speed==0,"Estimated origin permits terminal correction");
                }
                advanced=true;
            }
            if(std::string_view(c.reason)=="DWELL_COMPLETE_REVERSE_NEXT_FRAME") {
                require(advanced&&raceParkingInsetContained(space,o.pose)&&!corrected.terminalRecoveryUsed(),
                    "An incomplete corrected body passed dwell confirmation");
                model.dwelled=true;
            }
        }
        if(!model.dwelled)std::cerr<<"terminal correction state="<<c.state<<" reason="<<c.reason
            <<" back="<<model.back<<" forward="<<model.forward<<" minimum="<<model.minEdge<<'\n';
        require(model.dwelled&&model.back<.06f&&model.forward<=.18f,
            "Real stopped pose cannot complete bounded setback, insertion and dwell");
        for(int i=0;i<12&&exitFrames<4;++i) {
            advance();
            require(space.free(o.pose)&&!corrected.failed(),"Reverse from corrected berth immediately crosses paint");
            if(std::string_view(c.state)=="PARK_EXIT"&&c.speed<-.025f)++exitFrames;
        }
        require(exitFrames==4,"Corrected parking confirms dwell but does not actually start reversing out");
        terminal=model;
    }

    // A full closed-loop test cannot use the subsequent recorded positions after
    // changing the controls. Use the bicycle model and the actuator's existing
    // response scenario, and check the full body on both entry and reverse.
    const auto plan=racePlanTrackableParking(space,route.entry.front(),cruise);
    require(plan.valid,"Current planner cannot produce a driveable field route");
    float modelEntry=1,modelExit=1;
    constexpr float dt=.08f;
    for(float slip:{0.f,.09f}) {
        RaceParkingControl model(space,cruise);
        RaceParkingObservation o{plan.entry.front(),1,1,1,true};RaceParkingCommand c;
        require(model.arm(plan,o,1,true),"Model cannot arm planned route");
        float wheel=0,rollingWheel=0,rollingSpeed=0;bool contained=false,reversed=false;
        for(int i=0;i<1400&&!model.done()&&!model.failed();++i) {
            const float lock=d.fullLockDegrees*float(CV_PI)/180;
            wheel=raceParkingSteeringStep(wheel,std::atan(d.wheelbase*c.curvature),dt,d);
            wheel=std::round(wheel/lock*400)*lock/400;
            rollingWheel+=(wheel-rollingWheel)*(1-std::exp(-dt/.25f));
            rollingSpeed+=(c.speed-rollingSpeed)*(1-std::exp(-dt/.38f));
            const float k=std::tan(rollingWheel)/d.wheelbase,ds=rollingSpeed*dt;
            o.pose=raceParkingAdvance(o.pose,ds,k,d);
            o.pose.x-=slip*ds*k*std::cos(o.pose.yaw);o.pose.z-=slip*ds*k*std::sin(o.pose.yaw);
            const auto m=margins(o.pose);
            if(reversed)modelExit=std::min(modelExit,m[0]);else modelEntry=std::min(modelEntry,m[0]);
            require(space.free(o.pose),"Delayed field-start model crosses a painted boundary");
            ++o.sequence;o.captured+=dt;
            c=model.update(o,o.captured,dt,true,cruise,c.speed,std::tan(wheel)/d.wheelbase);
            if(std::string_view(c.reason)=="DWELL_COMPLETE_REVERSE_NEXT_FRAME")
                contained=raceParkingInsetContained(space,o.pose);
            if(c.speed<0){require(contained,"Model reverses without confirmed complete containment");reversed=true;}
        }
        require(reversed&&model.done()&&!model.terminalRecoveryUsed(),"Model cannot finish parking and return to the aisle");
    }
    require(modelEntry>.015f&&modelExit>.015f,"Model still grazes the opposite aisle edge");
    std::cout<<"PASS parking clearance field "<<dir.filename().string()<<" field_entry_edge_mm="<<entryEdge*1000
             <<" field_exit_edge_mm="<<exitEdge*1000<<" field_rear_shortfall_mm="<<-end[1]*1000
             <<" model_entry_edge_mm="<<modelEntry*1000<<" model_exit_edge_mm="<<modelExit*1000
             <<" modeled_setback_mm="<<terminal.back*1000<<" modeled_inset_mm="<<terminal.forward*1000
             <<" physical retest pending\n";
}
