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

    // Missing detections without a visible passage must still wait. Raised
    // bars and continuously visible empty road are separate release sources.
    RaceManager manager(config);RaceSnapshot ai;RacePlan plan;int sequence=0;double time=0;
    auto step=[&](std::vector<PredictResult> detections) {
        ai.sequence=++sequence;ai.captured=time+=.07;ai.detections=std::move(detections);
        return manager.update(ai,time,.07,true,binary,track,{},true,true);
    };
    for(int i=0;i<4;++i)plan=step({box(LABEL_GATE,100,260,440,70)});
    require(plan.stop&&plan.gateWaiting,"Closed entrance gate does not stop the vehicle");
    for(int i=0;i<12;++i){plan=step({box(LABEL_FLOW)});require(plan.stop&&plan.gateWaiting&&!plan.gateOpened,"Absent imagery or a FLOW sign alone releases a closed gate");}
    bool opened=false;
    for(int i=0;i<16&&!opened;++i){plan=step({box(LABEL_GATE,500,120,40,210)});opened=plan.gateOpened;}
    require(opened&&!plan.gateWaiting&&!plan.stop,"Raised gate at the far end never releases");

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
        for(int i=0;i<16&&!released;++i){result=tick({},binary,scene);released=result.gateOpened;}
        require(released&&!result.stop&&!result.gateWaiting,"Continuously visible empty road cannot release an undetected/removed gate");
    }

    RaceMission mission;mission.start();RaceSnapshot signs;signs.detections={box(LABEL_PARK)};
    for(int i=1;i<=3;++i){signs.sequence=i;mission.observeSigns(signs,true);}
    RacePlan traffic;traffic.fresh=true;traffic.state="GATE_WAIT";traffic.stop=traffic.gateWaiting=true;
    mission.observeGate(traffic);require(mission.speedLimit(.5,.3)==.3f,"Parking sign does not establish area speed cap");
    traffic.stop=traffic.gateWaiting=false;traffic.gateOpened=true;mission.observeGate(traffic);
    require(mission.speedLimit(.5,.3)==.3f,"Entrance gate opening restores full cruise too early");
    traffic.gateOpened=false;traffic.state="FORK_LEFT";traffic.selectedBranch=true;mission.observe(signs,traffic,false);
    traffic.state="LANE";traffic.selectedBranch=false;mission.observe(signs,traffic,true);
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
    require(raceParkingReadyToRejoin(space,{.04f,.2f,.35f}),"Basic alignment with slight paint overlap still demands precise reverse alignment");
    require(!raceParkingReadyToRejoin(space,{0,.2f,.7f}),"A large sideways heading is incorrectly called basic alignment");
    require(!raceParkingReadyToRejoin(space,{-.4f,.2f,.2f}),"A car still in the bay is incorrectly called back in the aisle");
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
    require(control.done()&&reverse-dwell>=3&&peak>.15f,"Target-speed entry/dwell/reverse did not finish or stayed at the old crawl speed");
    require(!control.exitRecoveryUsed(),"Nominal right parking relies on command-distance recovery");
    require(control.reverseReference().back().z>route.entry.front().z+.03f,"Reverse reference still follows the long original approach straight");
    std::cout<<"PASS target-speed mirrored right parking model; 3s dwell; long gate pause/resume; peak command="<<peak<<" m/s\n";

    // A persistently biased position/heading must not request endless reverse.
    // This is an estimated completion, deliberately distinct from observed exit.
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
    for(int i=0;i<1200&&!biased.done();++i) {
        if(i==30) {
            const float before=biased.reverseDistanceEstimate();
            for(int wait=0;wait<50;++wait){clock+=.04;observation.captured=clock;++observation.sequence;
                const auto held=biased.update(observation,clock,.04,false,.3,0);
                require(held.speed==0&&!biased.failed(),"Gate pause during reverse changes the task to a fault");}
            require(std::abs(biased.reverseDistanceEstimate()-before)<1e-6,"Zero-speed gate wait consumes the reverse distance budget");
            previousSpeed=0;
        }
        clock+=.04;observation.captured=clock;++observation.sequence;
        command=biased.update(observation,clock,.04,true,.3,previousSpeed);
        require(!biased.failed(),"Biased reverse becomes a terminal race fault instead of forward rejoin");
        observation.pose.z-=std::abs(command.speed)*.04;previousSpeed=command.speed;
    }
    require(biased.done()&&biased.exitRecoveryUsed()&&command.speed==0&&
        std::string(command.reason)=="REVERSE_DISTANCE_BUDGET_FORWARD_REJOIN","Bad alignment still causes unbounded reverse");
    require(biased.reverseDistanceEstimate()<=biased.reverseBudget()+.03f,"Reverse distance budget is substantially exceeded");
    std::cout<<"PASS basic aisle alignment; clipped reverse straight; bounded reverse estimate transitions to forward rejoin\n";
}
