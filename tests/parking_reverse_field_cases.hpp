#pragma once

static void parkingReverseFieldChecks(const fs::path &root,const fs::path &latest) {
    const auto earlier=root/"vehicle_evidence/mission-1793787774458";
    const auto readRows=[](const fs::path &dir,int begin,int end) {
        std::ifstream input(dir/"parking-control.jsonl");std::map<int,nlohmann::json> result;
        for(std::string line;std::getline(input,line);) {
            auto r=nlohmann::json::parse(line);const int f=r.at("frame");
            if(f>=begin)result.emplace(f,std::move(r));
            if(f>=end)break;
        }
        require(int(result.size())==end-begin+1,"Incomplete parking field sequence");return result;
    };
    const auto early=readRows(earlier,745,775);
    const auto rows=readRows(latest,690,840);
    nlohmann::json earlierRoute,saved;
    std::ifstream(earlier/"parking-route.json")>>earlierRoute;std::ifstream(latest/"parking-route.json")>>saved;
    RaceGroundCalibration earlyCal,latestCal;
    earlyCal.load((earlier/"ground-calibration.json").string(),false);latestCal.load((latest/"ground-calibration.json").string(),false);
    const auto imageFrame=[](const fs::path &dir,const nlohmann::json &r) {
        RaceSnapshot s;s.sequence=r.at("frame");s.captured=r.at("captured");
        s.image=cv::imread((dir/"parking-stop-evidence-3"/("frame-"+std::to_string(s.sequence)+".png")).string());
        require(s.image.size()==cv::Size(640,480),"Missing real parking image");
        if(r.at("ai_age_s").get<float>()<=.4f)for(const auto &d:r.at("ai_detections")) {
            const auto &b=d.at("box");auto det=box(d.at("type"),b.at(0),b.at(1),b.at(2),b.at(3));
            det.score=d.at("score");s.detections.push_back(det);
        }
        return s;
    };
    require(early.at(759).at("pose_source")=="VISUAL_ODOMETRY"&&early.at(764).at("estimate_budget_exhausted")==true&&
        early.at(764).at("estimate_applied_travel_m").get<float>()>.10f&&early.at(764).at("requested_speed")==0,
        "Wrong earlier entry travel-budget failure");
    RaceGroundMotion earlyTracker(earlyCal);earlyTracker.setMetricScale(earlierRoute.at("ground_scale"));
    earlyTracker.enableParkingNoiseTolerance();int linked=0;RaceGroundMotionResult olderView;
    std::map<int,RaceGroundPose> earlyPoses;
    RaceSnapshot earlierSample;cv::Mat olderImage;
    for(int f=753;f<=764;++f) {
        earlierSample=imageFrame(earlier,early.at(f));const auto motion=earlyTracker.update(earlierSample,true,false);
        if(f==754){olderView=motion;olderImage=earlierSample.image.clone();}
        if(f>=760) {
            linked+=motion.valid&&!motion.estimatedOrigin;
            if(motion.valid&&!motion.estimatedOrigin)earlyPoses.emplace(f,motion.pose);
            std::cout<<" parking entry image "<<f<<' '<<motion.reason<<" original_frame="<<(motion.valid&&!motion.estimatedOrigin)
                <<" matches="<<motion.matches<<" inliers="<<motion.inliers<<" residual_m="<<motion.residual
                <<" scale="<<motion.fitScale<<'\n';
        }
    }
    require(linked==5,"Field images 760-764 cannot all remain in the measured parking frame");
    require(olderView.valid&&!olderView.estimatedOrigin,"No measured entry reference");
    const cv::Mat obscured(olderImage.size(),olderImage.type(),cv::Scalar::all(40));
    bool recoveredEarlier=false;int reference=-1;
    // Counterfactual return to an older view, after the car has lost the scene.
    for(int i=0;i<50&&!recoveredEarlier;++i) {
        earlierSample.captured+=.10;++earlierSample.sequence;
        earlierSample.image=i<9?obscured:olderImage;earlierSample.detections.clear();
        const auto m=earlyTracker.update(earlierSample,true,false);
        if(i<9)require(!m.valid,"Untextured frame invents entry localization");
        if(m.valid&&!m.estimatedOrigin&&m.absoluteRecovered) {
            require(std::hypot(m.pose.x-olderView.pose.x,m.pose.z-olderView.pose.z)<.01f&&
                std::abs(m.pose.angle-olderView.pose.angle)<.04f,"Earlier view recovered at wrong parking position");
            recoveredEarlier=true;reference=m.absoluteReference;
        }
    }
    require(recoveredEarlier&&reference>0,"Returning to an older measured entry view cannot use the historical reference");

    RaceParkingRoute route;route.valid=true;route.entryCurvaturePreview=saved.at("entry_curvature_preview_m");
    for(const auto &p:saved.at("entry"))route.entry.push_back({p.at(0),p.at(1),p.at(2)});
    route.exit.assign(route.entry.rbegin(),route.entry.rend());
    RaceParkingSpace space;space.keepRearWithinStart(route.entry.at(0));
    const auto &dims=space.dimensions();const float handedness=saved.at("canonical_frame_handedness");
    const float cruise=rows.at(691).at("task_cruise_limit");
    RaceParkingControl controller(space,cruise);RaceParkingPoseFallback fallback;
    auto observation=[](const nlohmann::json &r) {
        const auto &p=r.at("visual_pose");
        return RaceParkingObservation{{p.at(0),p.at(1),p.at(2)},r.at("captured"),r.at("frame"),r.at("motion_epoch"),
            r.at("motion_valid"),r.at("motion_estimated_origin")};
    };
    auto inBayFrame=[](RaceParkingPose start,RaceGroundPose measured,float mirror) {
        const RaceGroundPose origin{start.x,start.z,start.yaw};
        const auto p=origin.world({mirror*measured.x,measured.z});
        return RaceParkingPose{p.x,p.y,start.yaw+mirror*measured.angle};
    };
    {
        RaceParkingRoute path;path.valid=true;path.entryCurvaturePreview=earlierRoute.at("entry_curvature_preview_m");
        for(const auto &p:earlierRoute.at("entry"))path.entry.push_back({p.at(0),p.at(1),p.at(2)});
        path.exit.assign(path.entry.rbegin(),path.entry.rend());
        RaceParkingSpace aisle;aisle.keepRearWithinStart(path.entry.at(0));RaceParkingControl entry(aisle,cruise);
        const float mirror=earlierRoute.at("canonical_frame_handedness");
        for(int f=746;f<=764;++f) {
            const auto &r=early.at(f),&prior=early.at(f-1);auto o=observation(r);
            if(f==746)require(entry.arm(path,o,r.at("time"),earlierRoute.at("reverse_verified")),"Earlier entry route cannot arm");
            if(f>=760){o.pose=inBayFrame(observation(early.at(753)).pose,earlyPoses.at(f),mirror);o.valid=true;o.estimated=false;}
            const auto c=entry.update(o,r.at("time"),r.at("control_dt_s"),true,
                std::min(cruise,r.at("traffic_speed_cap").get<float>()),prior.at("command_speed"),
                mirror*prior.at("physical_curvature").get<float>());
            require(!entry.failed(),"Ordinary startup incorrectly consumes the special forward join travel budget");
            if(f>=760) {
                if(c.speed<=0)std::cerr<<"entry replay frame="<<f<<" pose="<<o.pose.x<<','<<o.pose.z<<','<<o.pose.yaw
                    <<" reason="<<c.reason<<" previous_speed="<<prior.at("command_speed")<<'\n';
                require(c.speed>0&&std::string_view(c.state)=="PARK_ENTRY",
                    "Keeping the real image link did not release the former zero-speed entry frame");
            }
        }
    }
    RaceParkingCommand command;int dwell=0,firstReverse=0,reverseFrames=0;
    RaceParkingControl waitingAtDwell=controller;
    for(int f=691;f<=840;++f) {
        const auto &r=rows.at(f),&prior=rows.at(f-1);
        auto raw=observation(r);const double now=r.at("time");
        if(f==691)require(controller.arm(route,raw,now,saved.at("reverse_verified")),"Latest parking route cannot arm");
        const float applied=prior.at("command_speed"),wheel=handedness*prior.at("physical_curvature").get<float>();
        const auto o=fallback.update(raw,now,applied,wheel,dims);
        if(f==813)waitingAtDwell=controller;
        command=controller.update(o,now,r.at("control_dt_s"),true,
            std::min(cruise,r.at("traffic_speed_cap").get<float>()),applied,wheel);
        if(std::string_view(command.reason)=="DWELL_COMPLETE_REVERSE_NEXT_FRAME")dwell=f;
        if(command.speed<-.025f){if(!firstReverse)firstReverse=f;++reverseFrames;}
        if(f>=822)require(!o.valid&&command.speed==0&&!controller.failed(),
            "Exhausted blind estimate moves or permanently faults");
    }
    require(space.parkedObserved(observation(rows.at(813)).pose)&&dwell==813&&firstReverse==814&&reverseFrames>=6&&
        rows.at(814).at("command_speed").get<float>()<-.05f&&rows.at(822).at("estimate_budget_exhausted")==true&&
        rows.at(822).at("estimate_applied_travel_m").get<float>()<.05f,
        "Wrong recorded stationary dwell, reverse start, or blind-time exhaustion");
    const auto &dwellRow=rows.at(813);auto held=observation(dwellRow);held.estimated=true;
    auto protectedDwell=waitingAtDwell;
    require(protectedDwell.update(held,dwellRow.at("time"),dwellRow.at("control_dt_s"),true,cruise,0,0).speed==0&&
        protectedDwell.reverseReference().size()==route.exit.size(),"Estimate completed the physical dwell");
    held.estimated=false;
    const auto gate=waitingAtDwell.update(held,dwellRow.at("time"),dwellRow.at("control_dt_s"),false,cruise,0,0);
    require(gate.speed==0&&std::string_view(gate.reason)=="TRAFFIC_STOP_PRIORITY","Gate permits reverse after dwell");

    // Images at the actual reverse dropout were not archived. Model the return of a
    // previously measured view using genuine 786–797 entry images from this same bay.
    RaceGroundMotion tracker(latestCal);tracker.setMetricScale(saved.at("ground_scale"));tracker.enableParkingNoiseTolerance();
    // The reverse failure had 55 inliers. Its 820 image was not retained.
    // Compare a dominant rigid plane with random outliers against a competing
    // moving surface and optical zoom, using constructed ground correspondences.
    {
        std::vector<cv::Point2f> a,b;cv::RNG noise(820);
        for(int i=0;i<91;++i) {
            cv::Point2f p{-.165f+.055f*(i%7),.26f+.028f*(i/7)};a.push_back(p);
            const float angle=noise.uniform(-float(CV_PI),float(CV_PI));
            b.push_back(i<55?p+cv::Point2f(0,.018f):p+cv::Point2f(.09f*std::cos(angle),.09f*std::sin(angle)));
        }
        const auto accepted=[&]() {
            RaceGroundPose pose;RaceGroundFitDiagnostics info;int n=0;float residual=0;
            return tracker.fitParkingConsensus(a,b,pose,n,residual,info,.13f);
        };
        require(accepted(),"Dominant distributed reverse-motion surface cannot be recovered");
        for(size_t i=55;i<a.size();++i)b[i]=a[i]+cv::Point2f(.05f,-.02f);
        require(!accepted(),"Two competing ground motions accepted as the vehicle trajectory");
        for(size_t i=0;i<a.size();++i)b[i]=a[i]*1.05f;
        require(!accepted(),"Coherent zoom accepted as a rigid reverse step");
    }
    RaceGroundMotionResult last;RaceSnapshot snap;cv::Mat expectedImage;
    for(int f=786;f<=797;++f) {
        snap=imageFrame(latest,rows.at(f));last=tracker.update(snap,true,false);
        if(f==797)expectedImage=snap.image.clone();
    }
    require(last.valid&&!last.estimatedOrigin,"Could not reconstruct the last saved bay image");
    const auto expected=last.pose;
    cv::Mat blank(expectedImage.size(),expectedImage.type(),cv::Scalar::all(50));
    auto tick=[&](const cv::Mat &image) {
        ++snap.sequence;snap.captured+=.10;snap.image=image;snap.detections.clear();
        return tracker.update(snap,true,false);
    };
    cv::Mat unrelated(expectedImage.size(),expectedImage.type());
    cv::RNG rng(804);rng.fill(unrelated,cv::RNG::UNIFORM,0,255);
    // Re-anchor before the first absolute search builds descriptors. Sharing the
    // writable relative frame with the saved absolute image would overwrite it.
    for(int i=0;i<3;++i)require(!tick(unrelated).valid,"Unrelated texture was linked to parking coordinates");
    require(tracker.reanchorParkingEstimate(snap,{.10f,.05f,.20f}),"Cannot create estimated-origin recovery control");
    for(int i=0;i<12;++i) {
        const auto m=tick(unrelated);
        require(!m.valid||m.estimatedOrigin,"Re-anchored foreign image became an absolute measurement");
    }
    for(int i=0;i<8;++i)require(!tick(blank).valid,"Featureless scene returns a measured parking position");
    bool recovered=false;
    for(int i=0;i<24&&!recovered;++i) {
        last=tick(expectedImage);
        recovered=last.valid&&!last.estimatedOrigin&&last.absoluteRecovered;
    }
    if(!recovered)std::cerr<<"parking relocalization: "<<last.reason<<" matches="<<last.absoluteMatches<<'\n';
    require(recovered&&std::hypot(last.pose.x-expected.x,last.pose.z-expected.z)<.008f&&
        std::abs(last.pose.angle-expected.angle)<.04f,"Known view did not recover its measured coordinates");
    const auto startPose=observation(rows.at(786)).pose;
    auto resumed=observation(rows.at(840));resumed.valid=true;resumed.estimated=false;
    resumed.pose=inBayFrame(startPose,last.pose,handedness);
    const auto recorded=observation(rows.at(797));
    require(std::hypot(resumed.pose.x-recorded.pose.x,resumed.pose.z-recorded.pose.z)<.018f,
        "Reacquired image position disagrees with the original bay coordinates");
    int continued=0;float distance=0;double now=rows.at(840).at("time");float speed=0,curvature=0;
    for(int i=0;i<12&&continued<5;++i) {
        constexpr float dt=.08f;
        resumed.pose=raceParkingAdvance(resumed.pose,speed*dt,curvature,dims);
        distance+=std::max(0.f,-speed)*dt;resumed.captured+=dt;++resumed.sequence;now+=dt;
        const auto o=fallback.update(resumed,now,speed,curvature,dims);
        command=controller.update(o,now,dt,true,cruise,speed,curvature);
        require(!controller.failed()&&command.speed<=0,"Restored original position cannot continue the reverse path");
        continued+=command.speed<-.025f;speed=command.speed;curvature=command.curvature;
    }
    require(continued==5&&distance>.01f,"Reverse fails to sustain progress after verified visual return");
    std::cout<<"PASS parking reverse field latest_dwell="<<dwell<<" recorded_reverse="<<firstReverse
             <<" earlier_linked_images="<<linked<<" restored_reference="<<reference
             <<" modeled_reverse_m="<<distance<<" field retest pending\n";
}
