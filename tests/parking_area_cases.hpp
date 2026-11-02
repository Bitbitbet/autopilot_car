#pragma once

static void parkingAreaChecks(RaceConfig config,const RaceGroundCalibration &reference,
                              const cv::Mat &binary,const Track &track) {
    RaceParkingDimensions dims;
    // Both off-centre starts map into the same established left-bay model.
    for(int side:{-1,1})for(float offset:{-.08f,.08f})for(float tilt:{0.f,.08f}) {
        RaceParkingFrame frame;const float x=side*dims.aisle/2-offset;
        frame.locate({x,.3f},{x+dims.side*std::sin(tilt),.3f+dims.side*std::cos(tilt)});
        require(frame.side==side,"Parking side is not inferred from the actual mouth");
        auto canonical=frame.local({},dims.aisle);const auto start=canonical;
        RaceParkingPose physical{};
        for(int i=0;i<30;++i) {
            canonical=raceParkingAdvance(canonical,.004f,2.2f,dims);
            physical=raceParkingAdvance(physical,.004f,frame.handedness()*2.2f,dims);
            const auto recovered=frame.local(physical,dims.aisle);
            require(std::hypot(recovered.x-canonical.x,recovered.z-canonical.z)<1e-5&&
                    std::abs(raceParkingAngle(recovered.yaw-canonical.yaw))<1e-5,"Mirrored parking position/curvature/heading disagree");
        }
        const auto returned=frame.physical(start,dims.aisle);
        require(std::hypot(returned.x,returned.z)<1e-5&&std::abs(returned.yaw)<1e-5,"Parking frame cannot round-trip an off-centre start");
    }
    RaceGroundCalibration calibration=reference;
    calibration.nearDistance=.15f;calibration.farDistance=1.f;
    calibration.imagePoints={{{80,440},{560,440},{220,60},{420,60}}};
    RaceGroundMap map(calibration);
    for(float offset:{-.08f,.08f}) {
        cv::Mat image(480,640,CV_8UC3,cv::Scalar(15,15,15));
        for(int row=16;row<440;++row)for(int col=6;col<634;++col) {
            const auto point=map.toGround({float(col),float(row)});bool paint=false;
            for(int side:{-1,1})for(float z:{.30f,.30f+dims.side}) {
                const float x=side<0?point.x:-point.x;
                const float mouth=-dims.aisle/2+(side<0?-offset:offset),dx=x-mouth;
                paint|=(std::abs(dx)<.009f&&std::abs(point.y-z)<.075f)||
                    (dx<0&&dx>-.20f&&std::abs(point.y-z+dx*dims.stagger()/dims.depth)<.009f);
            }
            if(paint)image.at<cv::Vec3b>(row,col)={255,255,255};
        }
        const auto mouths=raceDetectParkingMouths(image,map,dims);
        require(std::any_of(mouths.begin(),mouths.end(),[](auto m){return m.side<0;}),"Off-centre synthetic left mouth not detected");
        require(std::any_of(mouths.begin(),mouths.end(),[](auto m){return m.side>0;}),"Off-centre synthetic right mouth not detected");
        require(!mouths.empty()&&mouths.front().side==(offset>0?1:-1),"Bay selection still always prefers left despite the nearer right bay");
    }
    std::cout<<"PASS bilateral mouth detection; off-centre starts; mirrored curvature/heading; nearer-side selection\n";

    {
        RaceManager approach(config);RaceSnapshot s;RacePlan p;double t=0;uint64_t n=0;
        auto tick=[&](int row){s.sequence=++n;s.captured=t+=.07;
            s.detections=row>0?std::vector<PredictResult>{box(LABEL_GATE,100,row-70,440,70)}:std::vector<PredictResult>{};
            return approach.update(s,t,.07,true,binary,track,{},true,true);};
        for(int i=0;i<4;++i)p=tick(330);
        require(!p.stop&&p.gateWaiting&&p.alignRoad&&p.speedCap<=.12f&&std::string(p.state)=="GATE_APPROACH","Distant gate must approach sensor slowly");
        p=tick(0);require(p.stop&&!p.gateOpened,"Lost bar permits blind sensor approach");
        p=tick(380);require(!p.stop,"Fresh bar cannot resume sensor approach");
        p=tick(config.gateStopRow);require(p.stop&&p.gateWaiting,"Sensor row must stop for closed bar");
        p=tick(380);require(p.stop,"Box jitter restarts stopped gate approach");
        std::cout<<"PASS sensor gate approach; late stop; missing bar hold; no jitter creep\n";
    }
    // Exercise existing release cases at their original synthetic stop row.
    config.gateStopRow=330;

    // Missing detections without a visible passage must still wait. Raised
    // bars and continuously visible empty road are separate release sources.
    RaceManager manager(config);RaceSnapshot ai;RacePlan plan;int sequence=0;double time=0;
    auto step=[&](std::vector<PredictResult> detections) {
        ai.sequence=++sequence;ai.captured=time+=.07;ai.detections=std::move(detections);
        return manager.update(ai,time,.07,true,binary,track,{},true,true,0,false,false,0,{true,true,0,0,.30f});
    };
    for(int i=0;i<4;++i)plan=step({box(LABEL_GATE,100,260,440,70)});
    require(plan.stop&&plan.gateWaiting,"Closed entrance gate does not stop the vehicle");
    for(int i=0;i<12;++i){plan=step({box(LABEL_FLOW)});require(plan.stop&&plan.gateWaiting&&!plan.gateOpened,"Absent imagery or a FLOW sign alone releases a closed gate");}
    bool opened=false;
    for(int i=0;i<24&&!opened;++i){plan=step({box(LABEL_GATE,500,120,40,210)});opened=plan.gateOpened;
        if(i<14)require(plan.stop&&!opened,"Raised gate skips one-second stopped hold");}
    require(opened&&!plan.gateWaiting&&!plan.stop,"Raised gate at the far end never releases");
    require(plan.speedCap<=config.parkingSpeed,"Gate release skips parking area passage cap");
    for(int i=0;i<10;++i){plan=step({});require(plan.speedCap<=config.parkingSpeed,"Gate passage cap releases too early");}
    for(int i=0;i<15;++i)plan=step({});
    require(plan.speedCap>config.parkingSpeed,"Gate passage cap never releases after command progress");

    {
        RaceManager passage(config);RaceSnapshot snapshot;RacePlan result;int frame=0;double clock=0;
        cv::Mat scene(480,640,CV_8UC3,cv::Scalar(120,120,120));
        auto tick=[&](std::vector<PredictResult> detections,const cv::Mat &road,const cv::Mat &view) {
            snapshot.sequence=++frame;snapshot.captured=clock+=.07;snapshot.image=view;
            snapshot.detections=std::move(detections);
            return passage.update(snapshot,clock,.07,true,road,track,view,true,true);
        };
        const auto gate=box(LABEL_GATE,100,260,440,70);
        for(int i=0;i<4;++i)result=tick({gate},binary,scene);
        require(result.stop&&result.gateWaiting,"Missing-gate release case did not latch the closed gate");
        result=tick({},binary,scene);
        require(result.stop&&!result.gateOpened,"One missed gate frame releases immediately");
        result=tick({gate},binary,scene);
        require(result.stop,"A returning closed gate does not reset the clearance interval");
        cv::Mat blocked=binary.clone();cv::rectangle(blocked,{220,260,200,70},{0},-1);
        for(int i=0;i<12;++i){result=tick({},blocked,scene);require(result.stop&&!result.gateOpened,"Missing gate box releases over an occluded road");}
        cv::Mat yellow=scene.clone();cv::rectangle(yellow,{220,260,200,70},{0,220,220},-1);
        for(int i=0;i<12;++i){result=tick({},binary,yellow);require(result.stop&&!result.gateOpened,"Undetected yellow bar is treated as empty white road");}
        for(int i=0;i<12;++i){result=tick({box(LABEL_CONE,290,270,60,50)},binary,scene);require(result.stop&&!result.gateOpened,"Another object in the gate passage does not block release");}
        bool released=false;
        for(int i=0;i<24&&!released;++i){result=tick({},binary,scene);released=result.gateOpened;}
        require(released&&!result.stop&&!result.gateWaiting,"Continuously visible empty road cannot release an undetected/removed gate");
    }

    RaceMission mission;mission.start();RaceSnapshot signs;signs.detections={box(LABEL_PARK)};
    RacePlan firstFork;firstFork.fresh=true;firstFork.state="FORK_RIGHT";firstFork.selectedBranch=true;firstFork.guideDirection=1;
    mission.observe(signs,firstFork,false);firstFork.state="LANE";firstFork.selectedBranch=false;mission.observe(signs,firstFork,false);
    mission.markStart();
    for(int i=1;i<=3;++i){signs.sequence=i;mission.observeSigns(signs,true);}
    RacePlan traffic;traffic.fresh=true;traffic.state="GATE_WAIT";traffic.stop=traffic.gateWaiting=true;
    mission.observeGate(traffic);require(mission.speedLimit(.5,.3)==.3f,"Parking sign does not establish area speed cap");
    traffic.stop=traffic.gateWaiting=false;traffic.gateOpened=true;mission.observeGate(traffic);
    require(mission.speedLimit(.5,.3)==.3f,"Entrance gate opening restores full cruise too early");
    traffic.gateOpened=false;traffic.state="FORK_LEFT";traffic.selectedBranch=true;traffic.guideDirection=-1;mission.observe(signs,traffic,false);
    traffic.state="LANE";traffic.selectedBranch=false;traffic.gateWaiting=true;mission.observe(signs,traffic,true);
    require(!mission.parkingSearch(),"Sensor approach starts bay parking before gate release");
    traffic.gateWaiting=false;mission.observe(signs,traffic,true);
    require(mission.parkingSearch(),"Gate test cannot enter parking search");
    mission.parkingStarted();mission.parkingFinished();
    require(mission.speedLimit(.5,.3)==.3f,"Reversing out of a bay removes the area cap before the exit gate");
    traffic.gateOpened=true;mission.observeGate(traffic);
    require(mission.speedLimit(.5,.3)==.3f,"An old entrance opening is reused as exit gate release");
    traffic.gateOpened=false;traffic.gateWaiting=traffic.stop=true;mission.observeGate(traffic);
    traffic.gateWaiting=traffic.stop=false;traffic.gateOpened=true;mission.observeGate(traffic);
    require(mission.speedLimit(.5,.3)==.5f&&!mission.parkingSpeedLimited(),"Exit gate opening does not restore configured cruise");
    require(std::min(mission.speedLimit(.5,.3),.4f)==.4f,"Restored cruise ignores an active traffic limit");
    std::cout<<"PASS closed/raised gate and visible-clear disappearance release; flicker/occlusion/yellow/obstacle rejection; entry/exit speed phases; cruise restoration under traffic limits\n";

    RaceParkingSpace space(dims);auto route=racePlanTrackableParking(space,{0,-.4f,0},.30f);
    require(route.valid,"No model-trackable parking route at the new target speed");
    {RaceParkingDwell timer;
     require(dims.dwellSeconds==1.f,"Runtime dwell still defaults to three seconds");
     require(!timer.ready(1,true,true,true,true)&&!timer.ready(1.99,true,true,true,true)&&
         timer.ready(2,true,true,true,true),"One-second dwell boundary is incorrect");
     require(!timer.ready(2.1,true,false,true,true)&&!timer.ready(2.2,true,true,true,true)&&
         !timer.ready(3.19,true,true,true,true)&&timer.ready(3.21,true,true,true,true),"Movement does not reset dwell");
     require(!timer.ready(4,false,true,true,true)&&!timer.ready(4.1,true,true,true,true),"Stale frame does not reset dwell");}
    require(raceParkingReadyToRejoin(space,{.04f,.2f,.15f}),"Small residual angle should permit lane rejoin");
    for(float side:{-1.f,1.f})require(!raceParkingReadyToRejoin(space,{side*.11f,.2f,-side*.18f}),
        "Inside but outward-facing body is released into ordinary lane control");
    for(float yaw:{.35f,-.35f,.496f,-.496f})
        require(!raceParkingReadyToRejoin(space,{0,.2f,yaw}),"20-28 degree exit is still accepted as aligned");
    require(!raceParkingReadyToRejoin(space,{0,.2f,.7f}),"A large sideways heading is incorrectly called basic alignment");
    require(!raceParkingReadyToRejoin(space,{-.4f,.2f,.2f}),"A car still in the bay is incorrectly called back in the aisle");
    require(!raceParkingReadyToRejoin(space,{-.16987824f,.1548803f,.496f}),
        "Recorded Nov-1 premature rejoin pose is still accepted");
    RaceParkingControl control(space,.30f);RaceParkingObservation observation{route.entry.front(),1,1,1,true};
    require(control.arm(route,observation,1,true),"Cannot arm target-speed parking");
    RaceParkingFrame rightBay;rightBay.locate({dims.aisle/2,.4f},{dims.aisle/2,.4f+dims.side});
    auto physical=rightBay.physical(observation.pose,dims.aisle);
    double clock=1,dwell=-1,reverse=-1;float peak=0;bool held=false;
    for(int i=0;i<3000&&!control.done();++i) {
        clock+=.04;observation.captured=clock;++observation.sequence;
        if(!held&&i==12) {
            for(int wait=0;wait<100;++wait){clock+=.04;observation.captured=clock;++observation.sequence;
                auto command=control.update(observation,clock,.04,false,.3);
                require(!control.failed()&&command.speed==0,"Gate wait longer than 650ms aborts parking");}
            held=true;clock+=.04;observation.captured=clock;++observation.sequence;
        }
        const auto command=control.update(observation,clock,.04,true,.3);
        require(!control.failed(),"Target-speed parking or gate resume faults");
        peak=std::max(peak,std::abs(command.speed));
        if(std::string(command.state)=="PARK_DWELL"&&dwell<0)dwell=clock;
        if(command.speed<0&&reverse<0)reverse=clock;
        physical=raceParkingAdvance(physical,command.speed*.04,rightBay.handedness()*command.curvature,dims);
        observation.pose=rightBay.local(physical,dims.aisle);
    }
    require(control.done()&&reverse-dwell>=1&&reverse-dwell<1.5&&peak>.15f,"Target-speed entry/dwell/reverse did not finish or stayed at the old crawl speed");
    require(!control.exitRecoveryUsed(),"Nominal right parking relies on command-distance recovery");
    require(control.reverseReference().back().z>route.entry.front().z+.03f,"Reverse reference still follows the long original approach straight");
    std::cout<<"PASS target-speed mirrored right parking model; 1s dwell; long gate pause/resume; peak command="<<peak<<" m/s\n";

    // Exhausted reverse budget must retain heading control, not complete while
    // sideways. Forward correction must converge before ordinary lane handoff.
    RaceParkingControl biased(space,.30f);observation={route.entry.front(),1,1,1,true};
    require(biased.arm(route,observation,1,true),"Cannot arm reverse-budget case");
    clock=1;bool reversing=false;float previousSpeed=0;RaceParkingCommand command;
    for(int i=0;i<3000&&!reversing;++i) {
        clock+=.04;observation.captured=clock;++observation.sequence;
        command=biased.update(observation,clock,.04,true,.3,previousSpeed);
        require(!biased.failed(),"Budget case fails before reverse");
        observation.pose=raceParkingAdvance(observation.pose,command.speed*.04,command.curvature,dims);
        previousSpeed=command.speed;reversing=std::string(command.state)=="PARK_EXIT";
    }
    require(reversing,"Budget case never reaches reverse");
    bool correcting=false,sawUnalignedCorrection=false;double alignedSince=-1;
    for(int i=0;i<1600&&!biased.done();++i) {
        if(i==30) {
            const float before=biased.reverseDistanceEstimate();
            for(int wait=0;wait<50;++wait){clock+=.04;observation.captured=clock;++observation.sequence;
                const auto held=biased.update(observation,clock,.04,false,.3,0);
                require(held.speed==0&&!biased.failed(),"Gate pause during reverse changes the task to a fault");}
            require(std::abs(biased.reverseDistanceEstimate()-before)<1e-6,"Zero-speed gate wait consumes the reverse distance budget");
            previousSpeed=0;
        }
        clock+=.04;observation.captured=clock;++observation.sequence;
        if(raceParkingReadyToRejoin(space,observation.pose)){if(alignedSince<0)alignedSince=clock;}
        else alignedSince=-1;
        command=biased.update(observation,clock,.04,true,.3,previousSpeed);
        require(!biased.failed(),"Biased reverse becomes a terminal race fault instead of forward rejoin");
        if(std::string(command.state)=="PARK_EXIT_ALIGN"){
            correcting=true;sawUnalignedCorrection|=std::abs(observation.pose.yaw)>.20f;
            require(!biased.done()&&command.speed>0,"Budget ends parking before alignment correction");
        }
        if(correcting)observation.pose=raceParkingAdvance(observation.pose,command.speed*.04,command.curvature,dims);
        else observation.pose.z-=std::abs(command.speed)*.04;
        previousSpeed=command.speed;
    }
    require(biased.done()&&biased.exitRecoveryUsed()&&command.speed==0&&sawUnalignedCorrection&&
        std::string(command.reason)=="CORRECTED_AISLE_ALIGNMENT_FORWARD_REJOIN","Reverse-budget correction failed to rejoin");
    require(alignedSince>0&&clock-alignedSince>=.19&&raceParkingReadyToRejoin(space,observation.pose),
        "Reverse budget bypasses continuous alignment confirmation");
    require(biased.reverseDistanceEstimate()<=biased.reverseBudget()+.03f,"Reverse distance budget is substantially exceeded");
    std::cout<<"PASS strict aisle alignment; 28 degree rejection; bounded reverse with forward correction; continuous aligned handoff\n";
}
