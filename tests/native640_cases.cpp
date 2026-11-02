#include "race_mission.hpp"
#include "race_runtime.hpp"
#include "task_pilot.hpp"
#include "task_preprocess.hpp"
#include "race_parking_control.hpp"
#include "race_parking_detection.hpp"
#include "race_virtual_boundary.hpp"
#include "race_ui.hpp"
#include <filesystem>
#include <iostream>

namespace fs=std::filesystem;
static void require(bool b,const char *why){if(!b)throw std::runtime_error(why);}
static PredictResult box(int type,int x=270,int y=250,int w=60,int h=70){PredictResult d{};d.type=type;d.score=.95;d.x=x;d.y=y;d.width=w;d.height=h;return d;}
#include "parking_area_cases.hpp"
int main(int argc,char **argv){try{
    if(argc==3&&std::string(argv[2])=="--park-entry-field") {
        RaceMission mission;mission.start();std::ifstream input(fs::path(argv[1])/"parking-control.jsonl");
        require(bool(input),"Missing entrance telemetry");std::string line;bool checked=false;
        while(std::getline(input,line)){
            const auto r=nlohmann::json::parse(line);RaceSnapshot s;s.sequence=r.at("ai_sequence");
            s.captured=r.at("time").get<double>()-r.at("ai_age_s").get<double>();
            for(const auto &v:r.at("ai_detections")){
                auto b=v.at("box");auto d=box(v.at("type"),b[0],b[1],b[2],b[3]);d.score=v.at("score");s.detections.push_back(d);
            }
            mission.observeSigns(s,r.at("ai_age_s").get<float>()<=.30);
            if(r.at("frame")==779){require(mission.parkingEntrancePending()&&!mission.parkingEntranceNear(),
                "Recorded distant 66x34 PARK still authorizes immediate entry turn");checked=true;break;}
        }
        require(checked,"Missing field frame779");
        std::cout<<"PASS recorded frame779 PARK confirmed for approach, no premature entry turn (telemetry only)\n";return 0;
    }
    if(argc==3&&std::string(argv[2])=="--fork-exit-field") {
        std::ifstream input(fs::path(argv[1])/"parking-control.jsonl");require(bool(input),"Missing fork telemetry");
        RaceYForkProgress progress;std::string line;bool started=false;uint64_t previousSequence=0,completed=0;
        RaceForkFeedback previous;double pulseEndDistance=0;
        while(std::getline(input,line)) {
            const auto r=nlohmann::json::parse(line);const std::string state=r.at("traffic_state");
            if(state=="FORK_SEEK_LEFT"||state=="FORK_SEEK_RIGHT")started=true;
            const auto sequence=r.at("ai_sequence").get<uint64_t>();
            bool sign=false;for(const auto &d:r.at("ai_detections"))
                if(d.at("accepted").get<bool>()&&(d.at("type")==LABEL_YLEFT||d.at("type")==LABEL_YRIGHT||d.at("type")==LABEL_LEFT))sign=true;
            const auto dt=r.at("control_dt_s").get<float>();
            if(started&&state.find("FORK")!=std::string::npos&&dt>0&&dt<=.25&&r.at("ai_age_s").get<float>()<=.30) {
                progress.update(dt,sequence!=previousSequence,sign,previous);
                if(!pulseEndDistance&&progress.pulsePhase>=1.4f)pulseEndDistance=progress.commandDistance;
                if(progress.complete()){completed=r.at("frame");break;}
            }
            previousSequence=sequence;
            previous={true,r.at("lane_ready").get<bool>()&&r.at("lane_alignment_valid").get<bool>(),
                r.at("lane_alignment_offset_px"),r.at("lane_alignment_heading_px"),r.at("command_speed")};
        }
        require(completed!=0,"Recorded recovered road never releases fork cap");
        std::cout<<"PASS recorded fork exit policy frame="<<completed<<" pulse_command_m="<<pulseEndDistance
            <<" tracking_s="<<progress.trackingTime<<" total_command_m="<<progress.commandDistance
            <<" (telemetry policy replay, not image or vehicle validation)\n";return 0;
    }
    if(argc==3&&std::string(argv[2])=="--park-switch-field") {
        RaceMission mission;mission.start();std::ifstream input(fs::path(argv[1])/"parking-control.jsonl");
        require(bool(input),"Missing recorded parking-sign telemetry");std::string line;uint64_t triggered=0;
        while(std::getline(input,line)) {
            const auto r=nlohmann::json::parse(line);RaceSnapshot s;s.sequence=r.at("ai_sequence");
            s.captured=r.at("time").get<double>()-r.at("ai_age_s").get<double>();
            for(const auto &v:r.at("ai_detections")) {
                auto b=v.at("box");auto d=box(v.at("type"),b[0],b[1],b[2],b[3]);d.score=v.at("score");s.detections.push_back(d);
            }
            // No synthetic Y completion: this is exactly the blocked condition.
            mission.observeSigns(s,r.at("ai_age_s").get<double>()<=.30);
            if(mission.parkingZone()){triggered=r.at("frame");break;}
        }
        require(triggered==1245&&mission.routeDirection()==-1,"Recorded confirmed PARK does not switch at frame 1245");
        std::cout<<"PASS recorded MODEL_PARK switches at frame 1245 despite unfinished Y (decision replay, not vehicle motion)\n";return 0;
    }
    if(argc==3&&std::string(argv[2])=="--park-sign-photo") {
        auto image=cv::imread(argv[1]);require(!image.empty(),"Missing supplied parking plate photo");
        const float scale=std::min(640.f/image.cols,480.f/image.rows);cv::Mat resized,canvas(480,640,CV_8UC3,cv::Scalar(80,80,80));
        cv::resize(image,resized,{},scale,scale);resized.copyTo(canvas(cv::Rect(0,0,resized.cols,resized.rows)));
        const auto found=raceParkingPShape(canvas);std::cout<<"P shape "<<found<<'\n';
        require(found.area()>0,"Supplied circular P/car/text plate was not matched");
        std::cout<<"PASS supplied parking plate photo shape (phone image, not vehicle inference)\n";return 0;
    }
    if(argc==3&&std::string(argv[2])=="--parking-stale-field") {
        const fs::path run(argv[1]);nlohmann::json saved;std::ifstream(run/"parking-route.json")>>saved;
        RaceParkingRoute route;route.valid=true;route.entryCurvaturePreview=saved.at("entry_curvature_preview_m");
        for(auto p:saved.at("entry"))route.entry.push_back({p[0].get<float>(),p[1].get<float>(),p[2].get<float>()});
        route.exit=route.entry;std::reverse(route.exit.begin(),route.exit.end());
        RaceParkingSpace space;space.keepRearWithinStart(route.entry.front());RaceParkingControl control(space);
        RaceParkingPoseFallback fallback;bool started=false,sawDelay=false,resumed=false;double last=0;
        float appliedSpeed=0,appliedCurvature=0;RaceParkingCommand command;
        std::ifstream input(run/"parking-control.jsonl");std::string line;
        while(std::getline(input,line)) {
            auto r=nlohmann::json::parse(line);if(!started&&!r.at("armed").get<bool>())continue;
            auto p=r.at("visual_pose");RaceParkingObservation raw{{p[0],p[1],p[2]},r.at("captured"),r.at("frame"),r.at("motion_epoch"),r.at("motion_valid")};
            const double now=r.at("time");
            if(!started){require(control.arm(route,raw,now,true),"Cannot arm field delay replay");started=true;last=raw.captured;fallback.update(raw,now,0,0,space.dimensions());}
            else {
                const auto o=fallback.update(raw,now,appliedSpeed,appliedCurvature,space.dimensions());
                // Runtime dt is measured before optical flow; telemetry time
                // is after it. Older logs lack dt, so use capture intervals.
                const float dt=r.value("control_dt_s",float(raw.captured-last));last=raw.captured;
                command=control.update(o,now,dt,true,.1f,appliedSpeed);
                if(control.failed())throw std::runtime_error("Field delay frame="+std::to_string(raw.sequence)+" reason="+command.reason);
                if(now-raw.captured>.18){sawDelay=true;require(!o.valid&&command.speed==0,"Expired frame authorized motion");}
                if(r.at("state")=="PARK_FAULT") {
                    require(sawDelay&&o.valid&&o.estimated&&command.speed>0,"Fresh camera cannot resume after recorded permanent fault");
                    resumed=true;break;
                }
            }
            // Recorded commands are the only motion actually applied in this replay.
            appliedSpeed=r.at("command_speed");appliedCurvature=r.at("physical_curvature");
        }
        require(resumed,"Field fault frame was not reached");
        std::cout<<"PASS recorded 204-246ms parking delay: stale frames hold; fresh frame resumes entry without reteach; estimate labelled\n";
        return 0;
    }
    if(argc==3&&std::string(argv[2])=="--reject-old-backend") {
        try {auto perception=createRacePerception(RaceConfig{});require(false,"Old configuration ABI was accepted");}
        catch(const std::exception &error){require(std::string(error.what()).find("backend version mismatch")!=std::string::npos,"Old backend was not rejected at the ABI boundary");}
        std::cout<<"PASS ABI4 backend rejected before creation by the ABI5 loader\n";return 0;
    }
    require(argc==2,"Pass workspace root");const fs::path root=argv[1];
    const auto temp=fs::temp_directory_path()/"race-native640-regression";fs::create_directories(temp);
    cv::Mat image(480,640,CV_8UC3,cv::Scalar(0));raceNormalizeFrame(image);
    require(image.size()==cv::Size(640,480),"Native frame was resized");
    bool rejected=false;try{cv::Mat small(240,320,CV_8UC3);raceNormalizeFrame(small);}catch(const std::invalid_argument&){rejected=true;}
    require(rejected,"Live 320 frame silently accepted");

    TrackviewConfig pilotConfig;pilotConfig.load((root/"autopilot_car/config/trackview.json").string(),true);
    RaceConfig config;config.load((root/"autopilot_car/config/race.json").string());config.forkMaxTime=12;config.validate();
    RaceNavigationConfig navigation;navigation.load((root/"autopilot_car/config/navigation.json").string(),true);
    require(pilotConfig.imageWidth==640&&config.triggerRow>=280&&navigation.rowStep==4,"Old configs not converted to native pixels");
    {std::ofstream out(temp/"native-pilot.json");out<<pilotConfig.toJson().dump();}
    TrackviewConfig twice;twice.load((temp/"native-pilot.json").string(),true);
    require(twice.rowCutBottom==pilotConfig.rowCutBottom&&twice.runP1==pilotConfig.runP1,"Config conversion repeated");

    RaceGroundCalibration ground;
    ground.load((root/"vehicle_evidence/parking-20261030-135042/parking-1793367911255/ground-calibration.json").string(),false);
    require(ground.version==2&&ground.imageWidth==640,"Legacy ground coordinates not converted");
    RaceGroundMap map(ground);
    for(size_t i=0;i<4;++i){auto p=map.toGround({ground.imagePoints[i][0],ground.imagePoints[i][1]});
        require(std::abs(p.x-(i%2?ground.roadWidth/2:-ground.roadWidth/2))<1e-5&&std::abs(p.y-(i<2?ground.nearDistance:ground.farDistance))<1e-5,"Native ground metric changed");}
    {std::ofstream out(temp/"native-ground.json");out<<nlohmann::json(ground).dump();}
    RaceGroundCalibration again;again.load((temp/"native-ground.json").string(),false);
    require(again.imagePoints==ground.imagePoints,"Ground points doubled twice");
    std::cout<<"PASS native frame contract; old/new config and ground conversion; 12s mission seek; metric invariance\n";

    // The master camera's lens remap must agree for equivalent 320/640 K.
    const auto previous=fs::current_path();fs::create_directories(temp/"res/calibration");fs::create_directories(temp/"bin");fs::current_path(temp/"bin");
    cv::Mat k=(cv::Mat_<double>(3,3)<<300,0,160,0,300,120,0,0,1),dist=cv::Mat::zeros(1,5,CV_64F);
    {cv::FileStorage f("../res/calibration/calibration.xml",cv::FileStorage::WRITE);f<<"cameraMatrix"<<k<<"distCoeffs"<<dist<<"image_width"<<320<<"image_height"<<240;}
    Camera lens320(Camera::CalibrationOnly{});cv::Mat texture(480,640,CV_8UC3);cv::randu(texture,0,255);auto corrected320=texture.clone();lens320.correct(corrected320);
    k.row(0)*=2;k.row(1)*=2;
    {cv::FileStorage f("../res/calibration/calibration.xml",cv::FileStorage::WRITE);f<<"cameraMatrix"<<k<<"distCoeffs"<<dist<<"image_width"<<640<<"image_height"<<480;}
    Camera lens640(Camera::CalibrationOnly{});auto corrected640=texture.clone();lens640.correct(corrected640);
    fs::current_path(previous);
    require(cv::norm(corrected320,corrected640,cv::NORM_INF)==0,"Lens coordinate conversion changes image geometry");
    std::cout<<"PASS master Camera equivalent lens reference scaling at full resolution\n";

    cv::Mat binary=cv::Mat::zeros(480,640,CV_8UC1);cv::rectangle(binary,{80,0,480,480},{255},-1);
    Track::camera=nullptr;Track track;track.handle(binary);
    require(track.pointsEdgeLeft.size()>100&&track.pointsEdgeLeft.front().x>400,"Master Track native row coordinates missing");
    RaceTaskPilot pilot(pilotConfig);
    for(int i=0;i<8;++i)pilot.update(track.pointsEdgeLeft,track.pointsEdgeRight,.07,0);
    require(pilot.ready&&std::abs(pilot.center-320)<4,"Master Pilot not tracking native midpoint");
    pilot.update(track.pointsEdgeLeft,track.pointsEdgeRight,.08,.25);
    const float beforeGap=pilot.speedFor(.25,.08);
    pilot.update({}, {},.08,beforeGap);require(!pilot.ready,"Blank road unexpectedly valid");
    const float recovery=pilot.speedForDirection(.25,.08,beforeGap);
    pilot.update(track.pointsEdgeLeft,track.pointsEdgeRight,.08,recovery);
    require(pilot.speedFor(.25,.08)>=recovery-.04,"Fresh road restarts speed ramp at zero after moving recovery");
    pilot.update(track.pointsEdgeLeft,track.pointsEdgeRight,.08,0);
    require(pilot.speedFor(.25,.08)<.04,"Actual stop fails to reset acceleration ramp");
    {RacePlan p;p.alignRoad=true;pilot.update({}, {},.08,0);
     require(raceLaneRecoveryServo(p,pilot,1200,.08)>1200&&raceLaneRecoveryServo(p,pilot,1800,.08)<1800,
        "No-road recovery retains the expired turn pulse");}
    std::cout<<"PASS moving geometry recovery preserves applied speed; actual stop resets ramp\n";
    std::cout<<"PASS master Track/Pilot full-image row map; no compatibility edge engine\n";

    const std::vector<cv::Point> left{{190,210},{280,150},{280,186},{400,186},{400,234},{280,234},{280,270}};
    const std::vector<cv::Point> bent{{140,180},{260,80},{260,140},{400,140},{460,200},{460,360},{400,360},{400,220},{260,220},{260,280}};
    for(const auto &points:{left,bent}){
        cv::Mat arrows(480,640,CV_8UC3,cv::Scalar(120,120,120));cv::rectangle(arrows,{80,0,480,480},{20,20,20},-1);cv::fillPoly(arrows,std::vector<std::vector<cv::Point>>{points},{255,255,255});
        const auto found=raceGroundArrow(arrows,track,true);
        std::cout<<"arrow allowed="<<found.allowed<<" vertices="<<points.size()<<'\n';
        require(found.allowed==RoadLeft,"Native straight-tail/bent-tail left ground arrow not recognized");
    }
    RaceMission mission;mission.start();RaceSnapshot ai;RacePlan traffic;traffic.fresh=true;traffic.state="LANE";
    {cv::Mat plate(480,640,CV_8UC3,cv::Scalar(90,90,90));
     cv::circle(plate,{320,220},90,{150,30,10},-1);
     cv::putText(plate,"P",{273,244},cv::FONT_HERSHEY_SIMPLEX,3,{255,255,255},13);
     require(raceParkingPShape(plate).area()>0,"Blue P shape not recognized");
     RaceMission m;RaceSnapshot s;s.image=plate;
     for(int i=1;i<=4;++i){s.sequence=i;s.captured=i*.07;m.observeSigns(s,true);}
     require(!m.parkingZone(),"P shape starts an unarmed mission");m.start();
     for(int i=1;i<=4;++i){s.sequence=i;s.captured=i*.07;m.observeSigns(s,true);}
     require(m.json()["park_evidence"]=="BLUE_P_SHAPE"&&m.parkingZone()&&m.routeDirection()==-1,
        "Confirmed switched P shape remains blocked by unfinished Y");
     for(const char *letter:{"O","B","R"}){
        cv::circle(plate,{320,220},90,{150,30,10},-1);
        cv::putText(plate,letter,{273,244},cv::FONT_HERSHEY_SIMPLEX,3,{255,255,255},13);
        require(raceParkingPShape(plate).empty(),"Non-P letter accepted as parking");}
     cv::circle(plate,{320,220},90,{150,30,10},-1);cv::arrowedLine(plate,{360,220},{270,220},{255,255,255},16);
     require(raceParkingPShape(plate).empty(),"Blue direction arrow accepted as parking");
     require(raceParkingPShape({}).empty(),"Missing frame fabricates P sign");}
    std::cout<<"PASS blue P shape; non-P rejection; unarmed isolation; switched sign entry without Y completion\n";
    RaceGroundArrow arrow;arrow.allowed=RoadLeft;
    for(int i=1;i<5;++i)mission.observeExitArrow(arrow,i,true);
    require(!mission.exitArrowTurnAuthorized(),"Ground arrow triggers before parking");
    ai.detections={box(LABEL_FLOW)};
    for(int i=1;i<=8;++i){ai.sequence=i;mission.observeSigns(ai,true);}
    mission.markStart();
    require(std::string(mission.name())=="LAP1_FLOW_THEN_SIGNED_FORK"&&mission.routeDirection()==0,
        "Starting FLOW or n skips lap one or masks signed Y decisions");
    traffic.selectedBranch=true;traffic.guideDirection=0;traffic.state="FORK_STRAIGHT";mission.observe(ai,traffic,false);
    traffic.selectedBranch=false;traffic.state="LANE";mission.observe(ai,traffic,true);
    require(std::string(mission.name())=="LAP1_FLOW_THEN_SIGNED_FORK"&&!mission.parkingSearch(),
        "Straight parking bypass is mistaken for completion of the signed Y fork");
    ai.detections={box(LABEL_PARK)};
    {auto switched=mission;
     ai.sequence=9;switched.observeSigns(ai,true);switched.observeSigns(ai,true);
     ai.sequence=10;switched.observeSigns(ai,false);
     require(!switched.parkingZone(),"Duplicate/stale P bypasses confirmation");
     for(int i=11;i<=13;++i){ai.sequence=i;switched.observeSigns(ai,true);}
     require(switched.parkingZone()&&switched.routeDirection()==-1,"Switched PARK blocked by missing Y completion");
     auto missing=ai;missing.sequence=14;missing.detections.clear();switched.observeSigns(missing,true);
     require(switched.parkingApproach()&&switched.routeDirection()==-1,"Confirmed entrance forgotten when P leaves view");}
    traffic.selectedBranch=true;traffic.guideDirection=-1;traffic.state="FORK_LEFT";mission.observe(ai,traffic,false);
    traffic.selectedBranch=false;traffic.guideDirection=0;traffic.state="LANE";mission.observe(ai,traffic,false);
    require(std::string(mission.name())=="LAP1_RETURN_PRESS_N_AT_START","Signed Y completion does not enter lap-one return");
    ai.detections={box(LABEL_FLOW)};
    for(int i=12;i<=16;++i){ai.sequence=i;mission.observeSigns(ai,true);}
    require(std::string(mission.name())=="LAP1_RETURN_PRESS_N_AT_START","Repeated FLOW prematurely advances the lap");
    {auto manual=mission;manual.markStart();require(std::string(manual.name())=="LAP2_WAIT_PARK_SIGN"&&manual.routeDirection()==0,
        "Manual lap landmark no longer permits the second-lap approach");}
    ai.detections={box(LABEL_PARK)};
    ai.sequence=17;mission.observeSigns(ai,true);mission.observeSigns(ai,true);ai.sequence=18;mission.observeSigns(ai,false);
    require(mission.routeDirection()==0,"Repeated/stale PARK triggers the entrance turn");
    for(int i=19;i<=21;++i){ai.sequence=i;mission.observeSigns(ai,true);}
    require(std::string(mission.name())=="LAP2_LEFT_TO_PARKING"&&mission.routeDirection()==-1,
        "Second-lap switched PARK cannot arm the entrance without a manual key");
    {RaceManager entrance(config);RacePlan p;RaceSnapshot s;
     for(int i=1;i<=5;++i){s.sequence=i;s.captured=i*.07;
         p=entrance.update(s,s.captured,.07,true,binary,track,{},false,true,mission.routeDirection(),true,mission.parkingApproach(),.3);}
     require(raceDirectionSeeking(p)&&p.guideDirection==-1&&raceDirectionServo(p,1500)==1100&&!p.stop,
         "Confirmed parking sign cannot demand full configured turn before a road split is visible");}
    traffic.selectedBranch=true;traffic.guideDirection=-1;traffic.state="FORK_LEFT";mission.observe(ai,traffic,false);
    traffic.selectedBranch=false;traffic.state="LANE";mission.observe(ai,traffic,true);require(mission.parkingSearch(),"Mission cannot reach visible bay");
    mission.parkingStarted();mission.parkingFinished();
    mission.observeExitArrow(arrow,10,true);mission.observeExitArrow(arrow,10,true);mission.observeExitArrow(arrow,11,false);
    require(!mission.exitArrowTurnAuthorized(),"Duplicate/stale arrow triggers turn");
    mission.observeExitArrow(arrow,12,true);mission.observeExitArrow(arrow,13,true);mission.observeExitArrow(arrow,14,true);
    require(mission.exitArrowTurnAuthorized(),"Parking exit cannot authorize proactive left search");
    {RaceManager manager(config);RaceSnapshot snapshot;RacePlan p;
     for(int i=1;i<=5;++i){snapshot.sequence=i;snapshot.captured=i*.07;p=manager.update(snapshot,snapshot.captured,.07,true,binary,track,{},false,true,-1,true,mission.exitArrowTurnAuthorized());}
     require(raceDirectionSeeking(p)&&p.guideDirection==-1&&raceDirectionServo(p,1500)<1500,"Exit arrow waits for a visible split");}
    traffic.selectedBranch=true;traffic.state="FORK_LEFT";mission.observe(ai,traffic,false);traffic.selectedBranch=false;traffic.state="LANE";mission.observe(ai,traffic,false);
    for(int i=20;i<25;++i)mission.observeExitArrow(arrow,i,true);
    require(std::string(mission.name())=="LAP2_RETURN_PRESS_N_TO_FINISH"&&mission.routeDirection()==0&&!mission.exitArrowTurnAuthorized(),
        "Parking exit starts an extra outer lap or reuses the exit arrow");
    ai.detections={box(LABEL_PARK),box(LABEL_FLOW)};
    for(int i=30;i<=34;++i){ai.sequence=i;mission.observeSigns(ai,true);}
    require(!mission.parkingSearch()&&mission.routeDirection()==0,"Completed parking can be retriggered");
    mission.markStart();require(mission.complete(),"Final manual landmark cannot finish after lap-two parking");
    RaceMission lab(true);lab.start();lab.markStart();require(lab.complete(),"Laboratory single loop behavior changed");
    std::cout<<"PASS Beijing lap1 FLOW and signed Y; lap2 switched PARK; no FLOW lap skip; full entrance guidance; parking exit then finish\n";
    std::cout<<"PASS exit-only ground arrow; duplicate/stale rejection; no-road left seek; outside-exit isolation\n";

    // Strong signed guidance must hand over gradually, then permit correction
    // in either direction once the target-road pilot owns the steering.
    {RacePlan p;p.fresh=true;p.guideDirection=-1;p.guidePwm=320;p.selectedBranch=true;
     p.laneBlend=0;require(raceDirectionServo(p,1400)==1180,"Same-direction handoff drops guidance");
     require(raceDirectionServo(p,1600)==1600,"Initial handoff cancels countersteer");
     p.laneBlend=.5;require(raceDirectionServo(p,1400)==1290,"Same-direction handoff is not blended");
     p.laneBlend=1;require(raceDirectionServo(p,1600)==1600,"Left commitment blocks rightward centering correction");
     p.guideDirection=1;p.laneBlend=0;require(raceDirectionServo(p,1400)==1400,"Right guidance cancels countersteer");
     p.laneBlend=1;require(raceDirectionServo(p,1400)==1400,"Right commitment blocks leftward centering correction");}
    cv::Mat split=binary.clone();cv::rectangle(split,{260,0,120,300},{0},-1);
    for(int sign:{LABEL_YLEFT,LABEL_YRIGHT}) {
        RaceMission flowMission;flowMission.start();RaceManager signedManager(config);RaceSnapshot s;RacePlan p;
        s.detections={box(LABEL_FLOW),box(sign)};
        for(int i=1;i<=8;++i){s.sequence=i;s.captured=i*.07;flowMission.observeSigns(s,true);
            p=signedManager.update(s,s.captured,.07,true,split,track,{},false,true,flowMission.routeDirection(),true);}
        const int direction=sign==LABEL_YLEFT?-1:1;
        require(p.selectedBranch&&p.guideDirection==direction,"FLOW or mission masks the explicit Y side");
    }
    {auto weak=config;weak.minBranchRows=1000;
     const auto locked=raceBranchPath(split,track,-1,weak,{},true);
     const auto row=std::find_if(locked.right.begin(),locked.right.end(),[](auto p){return p.x>=198&&p.x<=202;});
     require(row!=locked.right.end()&&row->y<260,"Committed left road replaced by closer right road when distant split is unconfirmed");}
    auto curve=[&](int farCenter){cv::Mat b=cv::Mat::zeros(480,640,CV_8UC1);
        const std::vector<cv::Point> polygon{{80,479},{560,479},{std::min(639,farCenter+100),0},{std::max(0,farCenter-100),0}};
        cv::fillPoly(b,std::vector<std::vector<cv::Point>>{polygon},{255});return b;};
    for(int direction:{-1,1}) {
        RaceManager manager(config);RaceSnapshot snapshot;RacePlan p;int i=0;
        snapshot.detections={box(direction<0?LABEL_YLEFT:LABEL_YRIGHT)};
        auto step=[&](const cv::Mat &b){snapshot.sequence=++i;snapshot.captured=i*.07;
            return manager.update(snapshot,snapshot.captured,.07,true,b,track,{},false,true,0,true);};
        for(int j=0;j<12;++j){p=step(curve(direction<0?490:150));
            require(!p.selectedBranch&&p.guidePwm<=400,"Y guidance exceeds servo range or accepts opposite road");
            if(j==config.confirmFrames)require(p.guidePwm>=360,"Initial Y steering remains weak");}
        for(int j=0;j<11;++j)p=step(binary);
        require(p.selectedBranch,"Centered road after Y turn cannot be acquired");
        for(int j=0;j<5;++j)p=step(binary);
        require(p.selectedBranch&&p.laneBlend==1,"Y handoff remains biased too long");
        require(raceDirectionServo(p,direction<0?1700:1300)==(direction<0?1700:1300),"Y handoff prevents countersteer");
        p=step(curve(direction<0?490:150));
        require(!p.selectedBranch&&p.guidePwm==0&&p.guideDirection==0&&p.alignRoad,"Target loss restarts Y bias instead of road alignment");
    }
    for(int direction:{-1,1}) {
        RaceManager manager(config);RaceSnapshot s;RacePlan p;
        s.detections={box(direction<0?LABEL_YLEFT:LABEL_YRIGHT)};
        cv::Mat missing=cv::Mat::zeros(480,640,CV_8UC1);
        for(int i=1;i<=35;++i){s.sequence=i;s.captured=i*.07;
            p=manager.update(s,s.captured,.07,true,missing,track,{},false,true,0,true);}
        require(!p.stop&&!raceDirectionSeeking(p)&&p.guidePwm==0&&p.alignRoad,"Missing road leaves a permanent sign steering bias");
    }
    {auto alignment=[](float offset,float slope){std::vector<PointX> centers;
        for(int row=476;row>=100;row-=4)centers.emplace_back(row,int(320+offset+(476-row)*slope));
        return raceLaneAlignment(centers);};
     auto a=alignment(0,0);require(a.valid&&a.aligned&&a.servo==1500,"Straight alignment changes neutral steering");
     for(int direction:{-1,1}){
        a=alignment(-25*direction,.5f*direction);
        require(a.valid&&!a.aligned&&direction*(int(a.servo)-1500)>0,"Near-road heading cannot countersteer against lateral offset");
        a=alignment(60*direction,0);require(!a.aligned&&direction*(int(a.servo)-1500)>0,"Lateral alignment sign reversed");
     }
     require(!raceLaneAlignment({}).valid,"Missing geometry fabricates an alignment measurement");}
    std::cout<<"PASS finite Y pulse; immediate countersteer; target-loss road recovery; mirrored near-road heading correction\n";
    for(int direction:{-1,1}) {
        RaceManager manager(config);RaceSnapshot s;RacePlan p;int i=0;
        auto step=[&](int route,bool confirmed){s.sequence=++i;s.captured=i*.07;
            return manager.update(s,s.captured,.07,true,binary,track,{},false,true,route,true,confirmed);};
        s.detections={box(direction<0?LABEL_YLEFT:LABEL_YRIGHT)};
        for(int j=0;j<24;++j)p=step(0,false);
        require(p.selectedBranch,"Preemption case did not acquire Y road");
        s.detections.clear();p=step(-1,true);
        require(raceDirectionSeeking(p)&&p.guideDirection==-1&&p.guidePwm>=360&&!p.stop,
            "Confirmed parking entrance cannot preempt pending same/opposite Y task");
    }
    std::cout<<"PASS switched PARK preempts unfinished same/opposite Y; finite stronger initial pulse\n";
    {
        for(float speed:{.10f,.15f,.20f}) {
            RaceYForkProgress progress;RaceForkFeedback road{true,true,20,35,speed};
            while(progress.pulsePhase<1.4f)progress.update(.01,true,true,road);
            require(progress.commandDistance>=.275&&progress.commandDistance<.285,"Y pulse distance depends on slow fork speed");
            require(!progress.complete(),"Visible sign releases fork cap");
        }
        RaceYForkProgress stationary;for(int i=0;i<300;++i)stationary.update(.01,true,false,{true,true,0,0,0});
        require(stationary.pulsePhase>=1.4&&!stationary.complete(),"Stationary car completes fork or leaves full lock indefinitely");
        RaceYForkProgress reused;for(int i=0;i<100;++i)reused.update(.07,false,false,{true,true,0,0,.2});
        require(!reused.complete(),"Repeated AI frame confirms fork exit");
        for(int direction:{-1,1})for(bool selected:{false,true}) {
            RaceManager manager(config);RaceMission mission;mission.start();RaceSnapshot s;RacePlan p;
            cv::Mat absent=cv::Mat::zeros(480,640,CV_8UC1);int sequence=0;bool completed=false;
            auto step=[&](bool sign,bool ready,float offset=25,float heading=35) {
                s.sequence=++sequence;s.captured=sequence*.07;s.detections=sign?std::vector<PredictResult>{box(direction<0?LABEL_YLEFT:LABEL_YRIGHT)}:std::vector<PredictResult>{};
                p=manager.update(s,s.captured,.07,true,selected?split:absent,track,{},false,true,0,true,false,.20,
                    {true,ready,offset,heading,.20});mission.observe(s,p,false);
            };
            for(int j=0;j<45;++j){step(true,true);if(j>config.confirmFrames)
                require(!p.completedDirection&&p.speedCap<=.201,"Visible sign prematurely ends task");}
            for(int j=0;j<15;++j){step(false,false);require(!p.completedDirection,"No road feedback completes fork");}
            for(int j=0;j<15;++j){step(false,true,100,90);require(!p.completedDirection,"Poor tracking completes fork");}
            for(int j=0;j<20;++j){step(false,true);if(p.completedDirection){completed=true;break;}}
            require(completed&&p.completedDirection==direction&&p.speedCap>.20&&!p.stop&&p.guidePwm==0,"Curved recovered road retains fork speed or bias");
            require(std::string(mission.name())=="LAP1_RETURN_PRESS_N_AT_START","Recovered fork completion never reaches mission state");
            step(false,true);require(p.speedCap>.20&&p.completedDirection==0,"Fork cap reapplied or completion event repeated");
        }
        std::cout<<"PASS speed-aware finite Y pulse; fresh curved-road exit from seek/select; fork cap and mission release\n";
    }
    {
        cv::Mat straight=cv::Mat::zeros(480,640,CV_8UC1);
        cv::rectangle(straight,{220,0,240,480},{255},-1);
        auto mouth=[&](int row,bool right=false,bool bend=false){
            auto result=straight.clone();
            for(int y=0;y<row;++y){
                int shift=std::min(160,(row-y)*3);
                result.row(y).setTo(0);
                int l=right?220:220-shift,r=right?459+shift:459;
                if(bend)r-=shift;
                result.row(y).colRange(l,r+1).setTo(255);
            }
            return result;
        };
        auto nearMouth=mouth(340),farMouth=mouth(240),rightMouth=mouth(340,true),curve=mouth(340,false,true);
        Track entranceTrack;entranceTrack.handle(straight);
        auto edges=[&](const cv::Mat &im){return raceParkingEntranceEdge(raceBranchPath(im,entranceTrack,-1,config,{},true));};
        require(!edges(straight).opening&&!edges(farMouth).opening&&!edges(rightMouth).opening&&!edges(curve).opening,
            "Straight/right opening/distant opening/constant-width curve triggers left entry");
        const auto estimate=edges(nearMouth);
        require(estimate.opening&&estimate.openingRow>290,"Near connected left opening not found");
        bool shifted=false;for(size_t k=0;k<estimate.path.left.size();++k)if(std::abs(estimate.path.left[k].x-300)<=2){
            const float center=(estimate.path.left[k].y+estimate.path.right[k].y)*.5f;
            require(center<240&&center>estimate.path.left[k].y+60,"Offset does not follow left boundary into road");shifted=true;
        }
        require(shifted,"No middle row in entrance estimate");
        RaceTaskPilot edgePilot(pilotConfig);
        for(int j=0;j<5;++j)edgePilot.update(estimate.path.left,estimate.path.right,.07,.30);
        require(edgePilot.ready&&edgePilot.servo<1450,
            "Estimated left-edge path never produces left steering through the real pilot");
        RaceMission mission;mission.start();RaceSnapshot s;RaceManager manager(config);RacePlan p;int sequence=0;
        auto step=[&](const cv::Mat &im,bool fresh=true){
            if(fresh){s.sequence=++sequence;s.captured=sequence*.07;}
            s.detections={box(LABEL_PARK,500,2,110,80)};mission.observeSigns(s,true);
            return manager.update(s,s.captured,.07,true,im,entranceTrack,{},false,true,mission.routeDirection(),true,
                mission.parkingEntrancePending(),.30,{},mission.parkingEntrancePending(),mission.parkingEntranceNear());
        };
        for(int j=0;j<15;++j)p=step(straight);
        require(mission.parkingEntranceNear()&&std::string(p.state)=="PARKING_ENTRY_APPROACH"&&!p.stop&&p.guidePwm==0,
            "Large near PARK triggers turn before road opening");
        for(int j=0;j<5;++j)p=step(farMouth);
        require(!p.selectedBranch&&!p.guideDirection,"Far left opening triggers early turn");
        p=step(nearMouth);for(int j=0;j<5;++j)p=step(nearMouth,false);
        require(!p.selectedBranch,"Repeated AI frame confirms entry opening");
        for(int j=0;j<4;++j)p=step(nearMouth);
        require(p.selectedBranch&&p.parkingEdgeEstimated&&!p.alignRoad&&p.guidePwm==0&&p.guideDirection==-1&&!p.stop,
            "Visible entrance not handed to edge path controller");
        require(raceDirectionServo(p,1700)==1700,"Edge controller cannot countersteer");
        auto stale=s;stale.failed=true;
        p=manager.update(stale,s.captured,.07,true,nearMouth,entranceTrack,{},false,true,-1,true,true,.30,{},true,true);
        require(p.stop,"Entrance AI failure does not hold motion");
        p=step(nearMouth);require(!p.stop&&p.selectedBranch,"Fresh entrance data remains permanently faulted after hold");
        cv::Mat absent=cv::Mat::zeros(480,640,CV_8UC1);p=step(absent);
        require(!p.stop&&!p.selectedBranch&&p.guidePwm==0&&p.alignRoad,"Lost edge restarts full-lock turn or stops");
        for(int j=0;j<100;++j){p=step(straight);if(p.completedDirection)break;}
        require(p.completedDirection==-1&&!p.stop,"Aligned entrance cannot complete");
        require(std::abs(mission.speedLimit(.5,.30)-.30)<1e-5&&std::abs(mission.speedLimit(.2,.30)-.2)<1e-5,"Parking cap changed");
        std::cout<<"PASS parking entrance edge offset; large sign alone rejected; distant/right/curve isolation; continuous near opening; countersteer and recovery; unchanged area cap\n";

        // Reproduce the field sequence: acquire entrance, wait at its gate,
        // then see bay-shaped splits forever. Gate release must hand off to
        // bay search instead of demanding zero split rows / far-road alignment.
        {
            RaceMission m;m.start();RaceManager gates(config);RaceSnapshot snap;
            RacePlan result;int n=0;
            auto tick=[&](std::vector<PredictResult> detections,bool visibleBay=false){
                snap.sequence=++n;snap.captured=n*.07;snap.detections=std::move(detections);
                m.observeSigns(snap,true);
                result=gates.update(snap,snap.captured,.07,true,nearMouth,entranceTrack,{},true,true,
                    m.routeDirection(),true,m.parkingEntrancePending(),.30,{},m.parkingEntrancePending(),true);
                m.observeGate(result);
                const bool scan=m.canScanParkingBays(result);
                m.observe(snap,result,scan&&visibleBay);
                return scan;
            };
            for(int j=0;j<8;++j)tick({box(LABEL_PARK,500,2,110,80)});
            require(result.selectedBranch&&m.parkingEntrancePending(),"Gate search case has not acquired entrance");
            for(int j=0;j<6;++j){
                const bool scan=tick({box(LABEL_GATE,100,350,440,70)},true);
                require(result.gateWaiting&&!scan&&!m.parkingSearch(),"Closed entrance gate starts bay search");
            }
            bool finished=false;
            for(int j=0;j<26&&!finished;++j){
                const bool scan=tick({box(LABEL_GATE,500,120,40,210)},true);
                if(result.gateOpened){
                    require(result.completedDirection==-1&&!result.selectedBranch&&result.guidePwm==0&&result.alignRoad,
                        "Gate release leaves entrance steering committed");
                    require(scan&&m.parkingSearch()&&!m.parkingEntrancePending(),"Gate release fails to scan visible bay in same frame");
                    finished=true;
                }
            }
            require(finished,"Entrance gate never hands off to bay search");
            m.parkingStarted();
            require(m.parkingRun(),"Automatic bay plan cannot arm after entrance handoff");
            for(int j=0;j<10;++j){tick({});require(!result.completedDirection&&!result.selectedBranch,
                "Entrance turn restarts after gate handoff");}
            RaceMission approachOnly;approachOnly.start();
            for(int j=1;j<=3;++j){snap.sequence=j;snap.detections={box(LABEL_PARK)};approachOnly.observeSigns(snap,true);}
            RacePlan unrelated;unrelated.fresh=true;unrelated.gateOpened=true;unrelated.state="LANE";
            approachOnly.observeGate(unrelated);approachOnly.observe(snap,unrelated,true);
            require(approachOnly.parkingEntrancePending()&&!approachOnly.canScanParkingBays(unrelated),
                "Unrelated gate opening skips unstarted entrance turn");
            std::cout<<"PASS entrance acquired gate release starts bay scan and maneuver; closed gate and unstarted turn excluded; no repeated left turn\n";
        }
    }
    {
        RaceEntranceProgress stationary;for(int i=0;i<100;++i)stationary.approachStep(.07,true,true,false,0);
        require(!stationary.fallbackReady(true),"Stationary PARK wait triggers blind turn");
        for(bool visible:{false,true}){
            RaceManager manager(config);RaceSnapshot s;RacePlan p;bool started=false;float firstDistance=0;
            cv::Mat absent=cv::Mat::zeros(480,640,CV_8UC1);
            for(int i=1;i<=120;++i){
                s.sequence=i;s.captured=i*.07;s.detections=visible?std::vector<PredictResult>{box(LABEL_PARK)}:std::vector<PredictResult>{};
                p=manager.update(s,s.captured,.07,true,absent,track,{},false,true,-1,true,true,.30,
                    {true,true,0,0,.30},true,true);
                if(p.parkingEntryFallback&&!started){started=true;firstDistance=p.parkingApproachCommandM;
                    require(p.guideDirection==-1&&p.guidePwm>=360,"Missing edge fallback does not actively turn left");}
                require(!p.stop,"Missing entrance edge stops instead of recovery");
            }
            require(started&&firstDistance>=(visible?.55f:.35f)&&firstDistance<(visible?.59f:.39f),"Approach delay absent or unbounded");
            require(p.guidePwm==0&&p.alignRoad,"Parking fallback leaves full lock indefinitely");
        }
        std::cout<<"PASS parking entrance bounded fallback; command approach delay; missing edge progresses; visible sign delay; pulse ends; stationary isolation\n";
    }
    {TrackviewConfig c;RaceSpeedSettings settings(c,.15f);RaceTaskPilot p(c);
     for(int i=0;i<8;++i)settings.adjust('+');
     for(int i=0;i<4;++i){settings.adjust(']');settings.adjust('.');settings.adjust('k');}
     for(int i=0;i<5;++i){p.setCurveSpeed(settings.curve);p.setDegradedSpeed(settings.degraded);}
     require(std::abs(settings.cruise-.5f)<1e-5&&p.curveSpeed()>.29f&&p.degradedSpeed()>.29f&&settings.fork>.34f,
         "Mission speed settings are erased or remain at initial limits");
     settings.adjust('[');settings.adjust(',');settings.adjust('j');settings.adjust('-');
     require(std::abs(settings.cruise-.45f)<1e-5&&std::abs(settings.fork-.30f)<1e-5,"Decrease keys fail");
     require(settings.adjust('a')==nullptr,"Speed keys consume the start command");
     for(int i=0;i<250;++i)settings.adjust('+');require(settings.cruise==c.maxSpeed,"Speed exceeds configured maximum");
     for(int i=0;i<250;++i)settings.adjust('-');require(settings.cruise==c.minSpeed,"Speed drops below configured minimum");}
    std::cout<<"PASS Y bounded guidance and fast centered-road handoff; no full-lock restart; mission speed keys and limits\n";
    {RaceManager manager(config);RaceSnapshot snapshot;RacePlan p;double t=0;uint64_t sequence=0;
     auto step=[&](const cv::Mat &b){t+=.07;snapshot.sequence=++sequence;snapshot.captured=t;
         return manager.update(snapshot,t,.07,true,b,track,{},false,true,-1,true,true);};
     for(int i=0;i<5;++i)p=step(binary);
     require(raceDirectionSeeking(p)&&raceDirectionServo(p,1500)<=1200,"No-road left guidance is still weak");
     for(int i=0;i<2;++i){p=step(split);require(!p.selectedBranch,"One glimpse prematurely takes over steering");}
     p=step(split);require(p.selectedBranch&&p.laneBlend==0&&raceDirectionServo(p,1500)<1200,"Initial target-road takeover snaps steering straight");
     float previousBlend=0;
     for(int i=0;i<12;++i){p=step(split);require(p.selectedBranch&&p.laneBlend>=previousBlend,"Target handoff is not continuous");previousBlend=p.laneBlend;}
     require(p.laneBlend==1,"Target-road pilot never gets full steering control");
     for(int center:{200,230,260,290,320})p=step(curve(center));
     for(int i=0;i<16;++i)p=step(binary);
     require(!p.selectedBranch&&std::string(p.state)=="LANE"&&!p.stop,"Aligned turn cannot finish and rejoin normal lane");}
    {RaceManager manager(config);RaceSnapshot snapshot;RacePlan p;
     for(int i=1;i<=16;++i){snapshot.sequence=i;snapshot.captured=i*.07;
        p=manager.update(snapshot,snapshot.captured,.07,true,curve(490),track,{},false,true,-1,true,true);
        require(!p.selectedBranch,"Visible opposite corridor was accepted as the left target");}
     require(raceDirectionSeeking(p)&&!p.stop,"Opposite-road rejection stops instead of keeping signed turn");}
    {RaceManager manager(config);RaceSnapshot snapshot;RacePlan p;int i=0;
     for(int center:{320,200,260,320}){snapshot.sequence=++i;snapshot.captured=i*.07;
        p=manager.update(snapshot,snapshot.captured,.07,true,curve(center),track,{},false,true,-1,true,true);}
     require(p.selectedBranch,"Target road centering during acquisition is mistaken for a missing left road");}
    {RaceManager manager(config);RaceSnapshot snapshot;RacePlan p;
     for(int i=1;i<=8;++i){snapshot.sequence=i;snapshot.captured=i*.07;
        p=manager.update(snapshot,snapshot.captured,.07,true,split,track,{},false,true,1,true);}
     require(p.selectedBranch&&p.guideDirection==1,"Right fork cannot acquire its mirrored target road");
     const auto row=std::find_if(p.path.left.begin(),p.path.left.end(),[](auto q){return q.x>=198&&q.x<=202;});
     require(row!=p.path.left.end()&&row->y>=380,"Right target path is on the left branch");}
    {RaceManager manager(config);RaceSnapshot snapshot;RacePlan p;
     for(int i=1;i<=8;++i){snapshot.sequence=i;snapshot.captured=i*.07;
        p=manager.update(snapshot,snapshot.captured,.07,true,split,track,{},false,true,-1,false,true);}
     require(p.selectedBranch,"Left target not acquired before disappearance check");
     snapshot.sequence=9;snapshot.captured=.63;
     p=manager.update(snapshot,.63,.07,true,curve(490),track,{},false,true,-1,false,true);
     require(!p.stop&&!p.selectedBranch&&p.guideDirection==0&&p.alignRoad,"Lost acquired road restarts signed turn or latches stop");
     p=manager.update(snapshot,1.2,.07,true,binary,track);
     require(p.stop&&!raceDirectionSeeking(p),"Strong guidance ignores stale AI stop");}
    {RaceManager manager(config);RaceSnapshot snapshot;RacePlan p;
     for(int i=1;i<=8;++i){snapshot.sequence=i;snapshot.captured=i*.07;
        p=manager.update(snapshot,snapshot.captured,.07,true,split,track,{},false,true,2,true);}
     require(p.selectedBranch&&p.guideDirection==0&&p.laneBlend==1,"Straight branch unexpectedly receives turning bias");}
    std::cout<<"PASS strong signed turn; continuous target acquisition; gradual steering handoff; opposite-road rejection; mirrored right turn; aligned exit\n";

    RaceParkingDimensions dimensions;RaceParkingSpace space(dimensions);auto route=racePlanTrackableParking(space,{0,-.4f,0});require(route.valid,"No metric entry route");
    RaceParkingControl parking(space);RaceParkingObservation observation{route.entry.front(),1,1,1,true};require(parking.arm(route,observation,1,true),"Cannot arm parking model");
    double time=1,dwell=-1,reverse=-1;RaceParkingCommand command;
    for(int i=0;i<4000&&!parking.done();++i){time+=.08;observation.captured=time;++observation.sequence;command=parking.update(observation,time,.08,true);
        require(!parking.failed(),"Parking model faults after native migration");
        if(std::string(command.state)=="PARK_DWELL"&&dwell<0)dwell=time;
        if(command.speed<0&&reverse<0)reverse=time;
        observation.pose=raceParkingAdvance(observation.pose,command.speed*.08,command.curvature,dimensions);}
    require(parking.done()&&reverse-dwell>=1&&reverse-dwell<1.5,"Entry/dwell/reverse no longer completes");
    RaceParkingBaySearch search;search.reject({-.22,.3},{-.22,.7});RaceParkingMouthCandidate failed;failed.nearGround={-.22,.3};failed.farGround={-.22,.7};require(!search.accepts(failed),"Failed bay not skipped");
    RaceParkingMouthCandidate next=failed;next.nearGround={-.22,.7};next.farGround={-.22,1.1};require(search.accepts(next),"Adjacent bay incorrectly excluded");
    std::cout<<"PASS metric parking entry/1s dwell/reverse and next-bay selection unchanged\n";
    {RaceParkingControl c(space);RaceParkingObservation o{route.entry.front(),1,1,1,true};
     require(c.arm(route,o,1,true),"Cannot arm resumable hold test");
     for(int i=1;i<=20;++i){o.captured=1+i*.08;++o.sequence;o.valid=false;
         const auto hold=c.update(o,o.captured,.08,true);require(!c.failed()&&hold.speed==0,"Missing pose permanently faults or moves");}
     o.captured+=.08;++o.sequence;o.valid=true;
     require(c.update(o,o.captured,.08,true).speed>0&&!c.failed(),"Valid stationary recovery after long hold cannot resume");
     const auto stop=c.update(o,o.captured+.08,.08,false);require(stop.speed==0,"Recovery bypassed traffic stop");
     c.abort();require(c.failed(),"Recovery bypassed explicit operator stop");}
    std::cout<<"PASS parking data hold remains resumable; traffic and operator stop preserved\n";
    parkingAreaChecks(config,ground,binary,track);

    RaceSnapshot snapshot;snapshot.sequence=1;snapshot.captured=1;snapshot.image=image;snapshot.detections={box(LABEL_CONE)};
    RaceVirtualBoundary virtualBoundary(navigation);RaceObstaclePreviewPlan plan;
    for(int i=1;i<8;++i){snapshot.sequence=i;snapshot.captured=i*.07;plan=virtualBoundary.update(snapshot,snapshot.captured,binary,track,config,true,.1);}
    std::cout<<"cone state="<<plan.state<<" reason="<<plan.reason<<" rows="<<plan.centers.size()<<'\n';
    require(plan.hasTarget&&plan.candidate&&!plan.centers.empty(),"Native cone path missing");
    require(plan.centers.front().x>250&&plan.visibleRight.front().y>320,"Cone route still uses 320x240 bounds");
    const auto compact=raceCompactView(image,binary,snapshot,snapshot.captured,config,false,&plan);
    require(compact.size()==cv::Size(640,300),"Native decision image not safely reduced for display only");
    auto smallBox=box(LABEL_CONE,600,400,30,40);require(raceDetectionBoxValid(smallBox),"Detector rejects native right half");
    std::cout<<"PASS native cone corridor, full-width detections and display-only thumbnails\n";
    {
        RaceVirtualBoundary recovery(navigation);RaceSnapshot s;s.image=image;
        RaceObstaclePreviewPlan p;int sequence=0;double clock=0;
        auto step=[&](const cv::Mat &road,const Track &edges,std::vector<PredictResult> detections) {
            s.sequence=++sequence;s.captured=clock+=.07;s.detections=std::move(detections);
            return recovery.update(s,clock,road,edges,config,true,.3);
        };
        p=step(binary,track,{box(LABEL_CONE,180,250,70,70)});
        require(p.state=="VIRTUAL_RECOVER"&&!p.stop&&!p.candidate&&p.direction==1,"Initial cone confirmation still stops or chooses the obstacle side");
        auto decision=recovery.decision(p,true);
        require(decision.follow&&!decision.stop&&decision.authorizeCone,"Recovery without a planned corridor is rejected by the decision layer");
        RaceTaskPilot missing(pilotConfig);missing.update({}, {},.07,.3);
        require(missing.speedForDirection(.3,.07,.3)>0&&raceVirtualRecoveryServo(p,missing,1500,.07)>1500,
                "No-lane recovery cannot issue forward speed and a rightward steering command");
        Track noEdges;cv::Mat noRoad=cv::Mat::zeros(480,640,CV_8UC1);
        for(int i=0;i<5;++i) {
            p=step(noRoad,noEdges,{box(LABEL_CONE,180,250,70,70)});
            require(!p.stop&&recovery.decision(p,true).follow,"Lost road edges still stop an active bypass");
        }
        bool resumed=false;
        for(int i=0;i<80;++i) {
            p=step(noRoad,noEdges,{});
            require(!p.stop&&!recovery.decision(p,true).stop,"Missing cone/road turns recovery into a stop");
            if(!p.hasTarget){resumed=true;break;}
        }
        require(resumed,"Geometry recovery never hands back to forward lane recovery");
        // An oversized obstacle leaves no constrained centre path. Continue
        // steering instead of presenting a fake valid corridor or zero speed.
        p=step(binary,track,{box(LABEL_CONE,90,200,460,210)});
        require(!p.stop&&recovery.decision(p,true).follow,"Envelope failure stops the vehicle");
        for(int i=0;i<4;++i)p=step(binary,track,{box(LABEL_CONE,90,200,460,210)});
        require(!p.candidate&&p.state=="VIRTUAL_RECOVER"&&!recovery.decision(p,true).stop,"No-clear-path recovery is not explicit or still requests a stop");
        p=recovery.update(s,clock+1,noRoad,noEdges,config,true,.3);
        require(recovery.decision(p,true).stop,"Geometry recovery disabled stale-data stopping");
        p=step(binary,track,{box(LABEL_PERSON)});
        require(p.stop&&recovery.decision(p,true).stop,"Geometry recovery overrides a person stop");
        RaceVirtualBoundary mirrored(navigation);s.sequence=1;s.captured=1;s.detections={box(LABEL_CONE,390,250,70,70)};
        p=mirrored.update(s,1,noRoad,noEdges,config,true,.3);
        require(p.direction==-1&&raceVirtualRecoveryServo(p,missing,1500,.07)<1500,"Leftward recovery is not mirrored");
        RaceVirtualBoundary invalid(navigation);s.sequence=2;s.captured=2;s.image.release();s.detections.clear();
        p=invalid.update(s,2,noRoad,noEdges,config,true,.3);
        require(invalid.decision(p,true).stop,"Missing matched imagery is mistaken for recoverable road geometry");
        std::cout<<"PASS virtual geometry recovery keeps moving; confirmation/no-road/envelope/lost-target; mirrored steering; stale/person stops preserved\n";
    }
    fs::remove_all(temp);return 0;
}catch(const std::exception &e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}}
