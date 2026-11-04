#pragma once

static void parkingBayAcquisitionFieldChecks(const fs::path &dir) {
    RaceGroundCalibration calibration;calibration.load((dir/"ground-calibration.json").string(),false);
    RaceGroundMap map(calibration);RaceParkingDimensions dims;
    RaceConfig cfg;cfg.load((dir/"race-config.json").string());
    const auto fork=cv::imread((dir/"signed-fork-selected/frame-555.png").string());
    require(fork.size()==cv::Size(640,480)&&raceDetectParkingMouths(fork,map,dims).empty(),
        "An ordinary fork is accepted as a parking bay");
    require(raceDetectParkingMouths(cv::Mat(480,640,CV_8UC3,cv::Scalar::all(255)),map,dims).empty(),
        "Blank white surface is accepted as a bay");
    std::ifstream control(dir/"parking-control.jsonl"),search(dir/"parking-bay-search.jsonl");
    require(bool(control)&&bool(search),"Missing bay-acquisition telemetry");
    std::map<unsigned,nlohmann::json> scans;
    for(std::string line;std::getline(search,line);) {
        const auto row=nlohmann::json::parse(line);
        if(row.at("event")=="LIVE_BAY_SCAN")scans.emplace(row.at("frame").get<unsigned>(),row);
    }
    RaceParkingMouthConfirmation originalCadence,withImages;
    float applied=0,curvature=0;
    unsigned recovered=0,confirmed=0,stationaryPlan=0;
    double lastScan=-1,stopTime=-1;
    RaceParkingMouthCandidate stopped;
    const auto planAvailable=[&](const RaceParkingMouthCandidate &m) {
        const auto corners=raceCheckParkingMouth(map.toGround(m.nearImage),map.toGround(m.farImage),dims);
        if(!corners.valid)return false;
        RaceParkingFrame bay;bay.locate(corners.near,corners.far);
        const auto start=bay.local({},dims.aisle);
        RaceParkingSpace space(dims);space.keepRearWithinStart(start);
        return racePlanTrackableParking(space,start,cfg.parkingSpeed).valid;
    };
    for(std::string line;std::getline(control,line);) {
        const auto row=nlohmann::json::parse(line);
        const auto frame=row.at("frame").get<unsigned>();
        const float dt=row.at("control_dt_s"),distance=applied*dt;
        const double now=row.at("captured");
        originalCadence.advance(distance,distance*curvature);withImages.advance(distance,distance*curvature);
        if(frame>=1080&&frame<=1098) {
            if(auto it=scans.find(frame);it!=scans.end()) {
                std::vector<RaceParkingMouthCandidate> current;
                for(const auto &c:it->second.at("candidates")){
                    const auto a=c.at("near"),b=c.at("far");
                    current.push_back({{},{},{a[0],a[1]},{b[0],b[1]},c.at("support"),c.at("side")});
                }
                if(originalCadence.update(current,now,frame,applied)&&!recovered) {
                    require(it->second.at("confirmation_frames")==1,"Wrong recorded association reset");
                    recovered=frame;
                }
            }
            if(withImages.tracking(now)||now-lastScan>=.12) {
                lastScan=now;
                auto path=dir/(frame<=1091?"parking-stop-evidence-1":"parking-stop-evidence-2")/("frame-"+std::to_string(frame)+".png");
                const auto image=cv::imread(path.string());
                require(image.size()==cv::Size(640,480),"Missing current bay image");
                auto current=raceDetectParkingMouths(image,map,dims);
                if(confirmed) {
                    current.erase(std::remove_if(current.begin(),current.end(),[&](const auto &m){
                        return !m.withinStoppedBay(stopped);
                    }),current.end());
                }
                if(withImages.update(current,now,frame,applied)) {
                    if(!confirmed){
                        confirmed=frame;stopped=current.front();stopTime=now;
                        require(planAvailable(stopped),"Bay cannot be reached from first confirmed image");
                    }
                    if(!stationaryPlan&&withImages.settledAfter(stopTime)) {
                        require(frame>1091&&applied==0,"Parking was planned while the recorded car still advanced");
                        require(now-stopTime<.8,"Stationary confirmation exceeds the mission's candidate retry timeout");
                        require(planAvailable(current.front()),"Stationary bay cannot produce a trackable parking route");
                        stationaryPlan=frame;
                    }
                }
            }
        }
        applied=row.at("command_speed");curvature=row.at("physical_curvature");
        if(frame>=1098)break;
    }
    require(recovered==1092&&confirmed&&stationaryPlan>confirmed,"Missing field bay confirmation or stationary plan");
    std::cout<<"PASS parking bay acquisition "<<dir.filename().string()<<" recorded_samples_reassociated="<<recovered
             <<" image_confirmation="<<confirmed<<" stationary_image_plan="<<stationaryPlan<<" model checked; field retest pending\n";
}
