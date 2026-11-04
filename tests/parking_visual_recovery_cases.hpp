#pragma once

static void parkingVisualRecoveryChecks(const fs::path &root) {
    const auto field=root/"vehicle_evidence/mission-1793697429281";
    const auto window=field/"parking-motion-loss-1";
    RaceGroundCalibration calibration;calibration.load((field/"ground-calibration.json").string(),false);
    nlohmann::json route,frames;std::ifstream(field/"parking-route.json")>>route;std::ifstream(window/"frames.json")>>frames;
    RaceGroundMotion tracker(calibration);tracker.setMetricScale(route.at("ground_scale"));tracker.enableParkingNoiseTolerance();
    int afterLoss=0;bool firstRecovered=false;
    for(const auto &row:frames){
        RaceSnapshot s;s.sequence=row.at("frame");s.captured=row.at("captured");
        s.image=cv::imread((window/row.at("image").get<std::string>()).string());
        require(!s.image.empty(),"Missing loss-window camera frame");
        if(row.at("ai_age_s").get<double>()<=.4)for(const auto &d:row.at("ai_detections")){
            auto b=d.at("box");auto det=box(d.at("type"),b[0],b[1],b[2],b[3]);det.score=d.at("score");s.detections.push_back(det);
        }
        auto r=tracker.update(s,true,false);
        if(s.sequence>=854){
            std::cout<<"field frame="<<s.sequence<<" recorded="<<row.at("motion_reason")<<" revised="<<r.reason
                <<" matches="<<r.matches<<" scale="<<r.fitScale<<" residual="<<r.residual<<" z="<<r.pose.z<<'\n';
            if(s.sequence==854)firstRecovered=r.valid&&!r.estimatedOrigin;
            afterLoss+=r.valid&&!r.estimatedOrigin;
        }
    }
    require(firstRecovered&&afterLoss==9,"Latest real-image chain did not cover the full saved loss window");

    const auto old=root/"vehicle_evidence/parking-20261101-134856/parking-1793539130610";
    RaceGroundCalibration oldCal;oldCal.load((old/"ground-calibration.json").string(),false);
    nlohmann::json oldRoute,oldFrames;std::ifstream(old/"parking-route.json")>>oldRoute;
    const auto oldWindow=old/"parking-stop-evidence-1";std::ifstream(oldWindow/"frames.json")>>oldFrames;
    RaceGroundMotion oldReplay(oldCal);oldReplay.setMetricScale(oldRoute.at("ground_scale"));oldReplay.enableParkingNoiseTolerance();
    bool oldRecovered=false;
    for(const auto &row:oldFrames){RaceSnapshot s;s.sequence=row.at("frame");s.captured=row.at("captured");
        s.image=cv::imread((oldWindow/row.at("image").get<std::string>()).string());require(!s.image.empty(),"Missing historical powered-run image");
        const auto r=oldReplay.update(s,true,false);if(s.sequence==118){oldRecovered=r.valid&&!r.estimatedOrigin;
            std::cout<<"powered-run image frame=118 revised="<<r.reason<<" residual="<<r.residual<<" (archived images; prior masks not retained)\n";}}
    require(oldRecovered,"Powered-run frame 118 still lacks a measured bridge");
    std::vector<cv::Point2f> a,b;for(int i=-4;i<=4;++i)for(int j=0;j<9;++j){
        a.emplace_back(i*.065f,.15f+j*.1f);b.push_back(a.back()*1.04f);}
    RaceGroundPose unchanged;RaceGroundFitDiagnostics rejected;int n=0,count=0;float err=0;
    require(!tracker.fitMetricSupport(a,b,unchanged,count,n,err,rejected,.08f),"Uniform zoom accepted as rigid ground motion");

    // A blind interval followed by a small, known ground translation. The
    // old reference must measure that displacement even after an estimated
    // re-anchor, and must not accept a different scene as the original one.
    RaceGroundMotion recover(calibration);recover.enableParkingNoiseTolerance();RaceGroundMap map(calibration);
    cv::Mat texture(480,640,CV_8UC3),foreign(texture.size(),texture.type());cv::RNG rng(2431);
    rng.fill(texture,cv::RNG::UNIFORM,0,255);rng.fill(foreign,cv::RNG::UNIFORM,0,255);
    std::array<cv::Point2f,4> from,to;for(int i=0;i<4;++i){from[i]={calibration.imagePoints[i][0],calibration.imagePoints[i][1]};
        auto ground=map.toGround(from[i]);ground.y-=.018f;to[i]=map.toImage(ground);}
    cv::Mat shifted;cv::warpPerspective(texture,shifted,cv::getPerspectiveTransform(from.data(),to.data()),texture.size());
    RaceSnapshot s;double time=1;auto tick=[&](const cv::Mat &image){s.image=image;s.captured=time+=.08;++s.sequence;return recover.update(s,true,false);};
    tick(texture);require(tick(texture).valid,"Initial absolute anchor missing");
    cv::Mat blank(texture.size(),texture.type(),cv::Scalar::all(50));
    for(int i=0;i<12;++i)require(!tick(blank).valid,"Featureless gap invented a pose");
    require(recover.reanchorParkingEstimate(s,{.12f,.04f,.15f}),"Estimated re-anchor setup failed");
    for(int i=0;i<8;++i){auto r=tick(foreign);require(!r.valid||r.estimatedOrigin,"Unrelated texture restored absolute position");}
    bool linked=false;for(int i=0;i<14&&!linked;++i){const auto r=tick(shifted);if(r.absoluteRecovered){
        require(r.valid&&!r.estimatedOrigin&&std::abs(r.pose.x)<.007f&&std::abs(r.pose.z-.018f)<.007f&&
            std::abs(r.pose.angle)<.025f,"Recovered pose uses the wrong origin or misses gap travel");linked=true;}}
    require(linked,"Original ground did not restore the absolute coordinate system");
    RaceParkingTestReport report;report.start();report.observe("PARK_ENTRY","EST",.1,"TRACKING",false);
    report.observe("PARK_DWELL","FULL_BODY_INSIDE_STOPPING",0,"STOPPED");report.observe("PARK_EXIT","DWELL_COMPLETE_REVERSE_NEXT_FRAME",0,"STOPPED");
    report.observe("PARK_COMPLETE","CONFIRMED_AISLE_ALIGNMENT_FORWARD_REJOIN",0,"STOPPED");
    require(report.toJson()["retained_attempts"][0]["parking_validated"].get<bool>(),"Earlier estimate invalidates subsequently observed success");
    std::cout<<"PASS parking visual recovery: nine real loss-window frames; measured-anchor return after dropout; unrelated texture rejected\n";
}
