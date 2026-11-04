#pragma once
#include "race_ground_motion.hpp"

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
        RaceGroundMotion tracker(calibration);tracker.enableParkingNoiseTolerance();
        cv::Mat texture(480,640,CV_8UC3);cv::RNG random(7616);random.fill(texture,cv::RNG::UNIFORM,0,255);
        RaceSnapshot s;s.image=texture;s.sequence=1;s.captured=1;tracker.update(s,true,false);
        ++s.sequence;s.captured+=.07;require(tracker.update(s,true,false).valid,"Textured parking reference cannot initialize");
        s.image=cv::Mat(480,640,CV_8UC3,cv::Scalar(60,60,60));
        for(int i=0;i<4;++i){++s.sequence;s.captured+=.2;require(!tracker.update(s,true,false).valid,"Blank frame invents visual motion");}
        // A new visible patch has no match to the original absolute anchor.
        // The estimate-seeded local tracker must keep its origin labelled.
        s.image=texture.clone();random.fill(s.image,cv::RNG::UNIFORM,0,255);++s.sequence;s.captured+=.07;
        require(!tracker.update(s,true,false).valid,"Old reference unexpectedly bridged a long gap");
        const RaceGroundPose seed{.12f,.27f,.18f};
        require(tracker.reanchorParkingEstimate(s,seed),"Parking estimated reference cannot recover frozen tracking");
        ++s.sequence;s.captured+=.07;auto recovered=tracker.update(s,true,false);
        require(recovered.valid&&recovered.estimatedOrigin&&recovered.epoch==0&&
            std::abs(recovered.pose.x-seed.x)<.001&&std::abs(recovered.pose.z-seed.z)<.001&&
            std::abs(recovered.pose.angle-seed.angle)<.001,"Recovered visual tracking resets coordinates or hides estimated origin");
        require(!tracker.reanchorParkingEstimate(s,{}),"Valid visual reference was replaced with an estimate");
        RaceGroundMotion ordinary(calibration);ordinary.update(s,true,false);
        require(!ordinary.reanchorParkingEstimate(s,seed),"Parking recovery leaked into ordinary ground tracking");
        std::cout<<"PASS parking relative visual recovery after frozen reference; propagated origin retained and labelled estimated\n";
    }

    {
        RaceManager approach(config);RaceSnapshot s;RacePlan p;double t=0;uint64_t n=0;
        auto tick=[&](int row){s.sequence=++n;s.captured=t+=.07;
            s.detections=row>0?std::vector<PredictResult>{box(LABEL_GATE,100,row-70,440,70)}:std::vector<PredictResult>{};
            return approach.update(s,t,.07,true,binary,track,{},true,true);};
        tick(0);p=tick(154);
        require(!p.stop&&!p.gateWaiting&&p.speedCap<=.12f,"First far horizontal gate fails to slow before sensor approach");
        p=tick(0);require(!p.stop&&p.speedCap<=.12f,"Brief distant gate miss drops slowdown");
        for(int i=0;i<4;++i)p=tick(config.gateStopRow-40);
        require(!p.stop&&p.gateWaiting&&p.alignRoad&&p.speedCap<=.12f&&std::string(p.state)=="GATE_APPROACH","Distant gate must approach sensor slowly");
        p=tick(0);require(p.stop&&!p.gateOpened,"Lost bar permits blind sensor approach");
        p=tick(config.gateStopRow-10);require(!p.stop,"Fresh bar cannot resume sensor approach");
        p=tick(config.gateStopRow);require(p.stop&&p.gateWaiting,"Sensor row must stop for closed bar");
        p=tick(config.gateStopRow-10);require(p.stop,"Box jitter restarts stopped gate approach");
        std::cout<<"PASS sensor gate approach; late stop; missing bar hold; no jitter creep\n";
    }

    {
        RaceManager passage(config);RaceSnapshot s;double t=0;RacePlan p;
        cv::Mat scene(480,640,CV_8UC3,cv::Scalar(50,50,50));
        cv::Mat road(480,640,CV_8UC1,cv::Scalar(255));
        cv::rectangle(scene,{100,390,440,90},{220,130,0},-1);
        cv::rectangle(road,{100,390,440,90},{0},-1);
        auto tick=[&](bool closed,const cv::Mat &view){++s.sequence;s.captured=t+=.07;s.image=view;
            s.detections=closed?std::vector<PredictResult>{box(LABEL_GATE,0,345,599,63)}:std::vector<PredictResult>{};
            return passage.update(s,t,.07,true,road,track,view,true,true);};
        for(int i=0;i<4;++i)p=tick(true,scene);
        cv::Mat bar=scene.clone();cv::rectangle(bar,{220,330,160,25},{0,220,220},-1);
        for(int i=0;i<20;++i){p=tick(false,bar);require(!p.gateOpened,"Unboxed yellow bar releases the near gate");}
        bool released=false;
        for(int i=0;i<24&&!released;++i){p=tick(false,scene);released=p.gateOpened;
            if(i<14)require(p.stop&&!released,"Near gate skips open dwell");}
        require(released&&!p.stop&&!p.gateWaiting,"Bonnet in the old gate box permanently blocks clear exit");
        std::cout<<"PASS far gate early slowdown and short miss hold; near bonnet excluded from clear passage; yellow rejection; exit release\n";
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
    for(auto point:space.body(route.entry.back(),.015f))
        require(space.inBay(point),"Parking goal still accepts a 5mm paint-hugging footprint");
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
    // Simulate a biased reverse that reaches the aisle but retains a heading
    // error. Spread the measured motion across the whole reverse so no pose
    // discontinuity or artificial jump is needed to set up the correction.
    const auto reverseStart=observation.pose;
    const RaceParkingPose nearAisle{-.05f,.2f,.35f};
    for(int i=0;i<1200&&!biased.exitRecoveryUsed();++i) {
        if(i==30) {
            const float before=biased.reverseDistanceEstimate();
            for(int wait=0;wait<50;++wait) {
                clock+=.04;observation.captured=clock;++observation.sequence;
                const auto held=biased.update(observation,clock,.04,false,.3,0);
                require(held.speed==0&&!biased.failed(),"Gate wait during reverse aborts parking");
            }
            require(biased.reverseDistanceEstimate()==before,"Stopped gate wait consumes reverse budget");
            previousSpeed=0;
        }
        const float distance=biased.reverseDistanceEstimate()+std::max(0.f,-previousSpeed)*.04f;
        const float t=std::min(1.f,distance/biased.reverseBudget());
        observation.pose={reverseStart.x+(nearAisle.x-reverseStart.x)*t,
                          reverseStart.z+(nearAisle.z-reverseStart.z)*t,
                          reverseStart.yaw+raceParkingAngle(nearAisle.yaw-reverseStart.yaw)*t};
        clock+=.04;observation.captured=clock;++observation.sequence;
        command=biased.update(observation,clock,.04,true,.3,previousSpeed);
        require(!biased.failed()&&!biased.done(),"Biased reverse hands off before alignment");
        previousSpeed=command.speed;
    }
    require(biased.exitRecoveryUsed()&&std::string(command.state)=="PARK_EXIT_ALIGN"&&command.speed>0,
        "Reverse budget does not begin bounded forward alignment");
    require(biased.reverseDistanceEstimate()<=biased.reverseBudget()+.03f,"Reverse budget substantially exceeded");
    {
        auto recovery=biased;auto o=observation;auto c=command;double alignedSince=-1;
        for(int i=0;i<180&&!recovery.done();++i) {
            o.pose=raceParkingAdvance(o.pose,c.speed*.04f,c.curvature,dims);
            o.captured+=.04;++o.sequence;
            if(raceParkingReadyToRejoin(space,o.pose)){if(alignedSince<0)alignedSince=o.captured;}
            else alignedSince=-1;
            c=recovery.update(o,o.captured,.04f,true,.3f,c.speed,c.curvature);
            require(!recovery.failed(),"Reachable near-aisle correction cannot rejoin");
        }
        require(recovery.done()&&c.speed==0&&std::string(c.reason)=="CORRECTED_AISLE_ALIGNMENT_FORWARD_REJOIN"&&
                alignedSince>0&&o.captured-alignedSince>=.19&&raceParkingReadyToRejoin(space,o.pose),
                "Near-aisle correction bypasses continuous measured alignment");
    }
    // Keep the measured heading unaligned while translation continues. At
    // .12 m/s the travel cap wins; at .04 m/s the moving-time cap wins.
    for(float cap:{.12f,.04f}) {
        auto recovery=biased;auto o=observation;auto c=command;
        float applied=std::min(cap,c.speed);
        for(int i=0;i<210&&!recovery.failed();++i) {
            if(i==25) {
                const float beforeM=recovery.forwardAlignCommandM(),beforeS=recovery.forwardAlignMovingSeconds();
                require(beforeM>0&&beforeS>0,"No active correction budget to test before pause");
                for(int wait=0;wait<175;++wait) {
                    o.captured+=.04;++o.sequence;
                    c=recovery.update(o,o.captured,.04f,false,cap,0);
                    require(c.speed==0&&!recovery.failed(),"Traffic pause consumes forward correction time");
                }
                o.estimated=true;o.captured+=.04;++o.sequence;
                c=recovery.update(o,o.captured,.04f,true,cap,0);
                require(c.speed==0&&std::string(c.reason)=="WAIT_ABSOLUTE_VISUAL_CONFIRMATION",
                        "Forward alignment continues on relative-only position");
                require(recovery.forwardAlignCommandM()==beforeM&&recovery.forwardAlignMovingSeconds()==beforeS,
                        "Zero-speed hold resets or consumes forward correction budget");
                o.estimated=false;applied=0;
            }
            o.pose.z+=applied*.04f;o.captured+=.04;++o.sequence;
            c=recovery.update(o,o.captured,.04f,true,cap,applied);
            require(!recovery.done(),"Unaligned correction is accepted as completion");
            applied=c.speed;
        }
        require(recovery.failed()&&c.speed==0&&std::string(c.reason)=="PARK_EXIT_ALIGNMENT_LIMIT",
                "Unaligned forward correction did not fault at its limit");
        const float usedM=recovery.forwardAlignCommandM(),usedS=recovery.forwardAlignMovingSeconds();
        if(cap>.1f)require(usedM>=RaceParkingControl::forwardAlignDistanceLimit-.0001f&&
            usedM<=RaceParkingControl::forwardAlignDistanceLimit+.005f&&usedS<RaceParkingControl::forwardAlignTimeLimit,
            "Fast alignment did not stop on applied travel");
        else require(usedS>=RaceParkingControl::forwardAlignTimeLimit-.0001f&&
            usedS<=RaceParkingControl::forwardAlignTimeLimit+.041f&&usedM<RaceParkingControl::forwardAlignDistanceLimit,
            "Slow alignment did not stop on moving time");
    }
    std::cout<<"PASS strict aisle alignment; 28 degree rejection; bounded reverse; near-aisle recovery; forward travel/time limits; continuous aligned handoff\n";
}
