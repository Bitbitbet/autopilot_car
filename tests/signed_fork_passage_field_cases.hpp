#pragma once

static void signedForkPassageRun(const fs::path &dir) {
    RaceConfig cfg;cfg.load((dir/"race-config.json").string());
    TrackviewConfig vehicle;vehicle.load((dir/"trackview-config.json").string(),true);
    RaceGroundCalibration ground;ground.load((dir/"ground-calibration.json").string(),false);
    const auto half=ground.roadWidth/2;
    cv::Point2f src[4],dst[4]={{-half,ground.nearDistance},{half,ground.nearDistance},
                               {-half,ground.farDistance},{half,ground.farDistance}};
    for(size_t i=0;i<4;++i)src[i]={ground.imagePoints[i][0],ground.imagePoints[i][1]};
    Camera camera(cv::getPerspectiveTransform(src,dst));camera.blackRoad=vehicle.roadSurface=="black";
    struct Restore {Camera *saved;~Restore(){Track::camera=saved;}} restore{Track::camera};
    Track::camera=&camera;

    const auto window=dir/"signed-fork-wait";
    nlohmann::json event;std::ifstream(window/"event.json")>>event;
    const int stop=event.at("trigger_frame");
    require(event.at("event")=="SIGNED_FORK_WAIT_TARGET_ROAD"&&event.at("direction")==-1,
        "Missing recorded first-lap left-fork stop");
    std::map<int,nlohmann::json> rows;std::string line;
    std::ifstream input(dir/"parking-control.jsonl");
    while(std::getline(input,line)){
        const auto r=nlohmann::json::parse(line);const int f=r.at("frame");
        if(f>stop)break;
        if(f>=stop-45)rows.emplace(f,r);
    }
    require(rows.size()==46&&rows.at(stop).at("command_speed")==0&&
        rows.at(stop).at("traffic_reason")=="SIGNED_FORK_WAIT_TARGET_ROAD","Missing original hold or approach history");

    const cv::Mat blank=cv::Mat::zeros(480,640,CV_8UC1);
    std::array<cv::Mat,4> images;
    for(int i=0;i<4;++i) {
        images[i]=cv::imread((window/("frame-"+std::to_string(stop-3+i)+".png")).string());
        require(images[i].size()==cv::Size(640,480),"Missing original fork frame");
    }
    enum class Scene {Field,Mirror,FixedPlate,JumpingPlate,ConflictingSigns,WrongSteer,Stationary,Stale,MissingRoad,StillSplit};
    auto replay=[&](Scene scene) {
        const bool mirror=scene==Scene::Mirror;
        const int direction=mirror?1:-1;
        const bool fieldCheck=scene==Scene::Field||mirror;
        RaceManager manager(cfg);RaceTaskPilot pilot(vehicle);RacePlan plan;RaceSnapshot snap;
        double time=0;int handoff=0,completed=0;float oldStopSpeed=0,outputSpeed=0;uint16_t outputPwm=PWMSERVOMID;
        bool held=false;
        // Before the saved window only the logged sign boxes, near-lane
        // feedback and applied commands exist. Empty geometry cannot grant
        // a handoff. Beyond the old stop repeat the last picture with fresh
        // synthetic timestamps and feed back the newly calculated commands.
        for(int f=stop-44;f<=stop+22;++f) {
            const bool recorded=f<=stop;
            const auto &r=rows.at(std::min(f,stop));
            const auto &previous=rows.at(std::min(f-1,stop));
            const float dt=recorded?r.at("control_dt_s").get<float>():.07f;
            time=recorded?r.at("time").get<double>():time+dt;
            snap.sequence=recorded?r.at("ai_sequence").get<uint64_t>():snap.sequence+1;
            snap.captured=time-(recorded?r.at("ai_age_s").get<double>():.07);
            if(scene==Scene::Stale&&f==stop-4)snap.captured=time-cfg.maxDetectionAge-.1;
            snap.detections.clear();
            if(recorded)for(const auto &d:r.at("ai_detections")) {
                if(!d.at("accepted").get<bool>())continue;
                const auto &b=d.at("box");int x=b[0],w=b[2],type=d.at("type");
                if(mirror){x=640-x-w;if(type==LABEL_YLEFT)type=LABEL_YRIGHT;}
                if(type==LABEL_YLEFT&&scene==Scene::FixedPlate)x=64;
                if(type==LABEL_YLEFT&&scene==Scene::JumpingPlate)x=x+w/2<320?64:640-32-w;
                auto det=box(type,x,b[1],w,b[3]);det.score=d.at("score");snap.detections.push_back(det);
                if(scene==Scene::ConflictingSigns&&type==LABEL_YLEFT) {
                    det.type=LABEL_YRIGHT;det.x=640-x-w;snap.detections.push_back(det);
                }
            }
            cv::Mat image,binary=blank;
            if(f>=stop-3&&scene!=Scene::MissingRoad) {
                if(scene==Scene::StillSplit)binary=signedForkScene(-1,true);
                else{
                    const auto &saved=images[std::min(f,stop)-(stop-3)];
                    if(mirror)cv::flip(saved,image,1);else image=saved;
                    binary=camera.binarize(image);
                }
            }
            Track track;track.rowCutUp=vehicle.rowCutUp;track.rowCutBottom=vehicle.rowCutBottom;track.handle(binary);
            float applied=recorded?previous.at("command_speed").get<float>():outputSpeed;
            uint16_t pwm=recorded?previous.at("pwm").get<uint16_t>():outputPwm;
            if(recorded&&mirror)pwm=2*PWMSERVOMID-pwm;
            if(scene==Scene::WrongSteer)pwm=2*PWMSERVOMID-pwm;
            if(scene==Scene::Stationary)applied=0;
            const bool useImageFeedback=f>stop-3;
            const bool ready=useImageFeedback?pilot.ready&&pilot.alignment.valid&&!pilot.lost:
                previous.at("lane_ready").get<bool>()&&previous.at("lane_alignment_valid").get<bool>();
            const float sign=mirror?-1.f:1.f;
            const float offset=useImageFeedback?pilot.alignment.offset:sign*previous.at("lane_alignment_offset_px").get<float>();
            const float heading=useImageFeedback?pilot.alignment.heading:sign*previous.at("lane_alignment_heading_px").get<float>();
            plan=manager.update(snap,time,dt,true,binary,track,image,true,true,0,true,false,
                r.at("fork_setting"),{true,ready,offset,heading,applied,pwm});
            const bool passed=std::string_view(plan.reason)=="SIGNED_FORK_PASSED_SIGN_ROAD_HANDOFF";
            if(passed&&!handoff)handoff=f;
            held|=std::string_view(plan.reason)=="SIGNED_FORK_WAIT_TARGET_ROAD";
            if(f>=stop-3) {
                pilot.setCurveSpeed(r.at("curve_setting"));pilot.setDegradedSpeed(r.at("degraded_setting"));
                pilot.update(plan.selectedBranch?plan.path.left:track.pointsEdgeLeft,
                             plan.selectedBranch?plan.path.right:track.pointsEdgeRight,dt,applied);
                if(scene==Scene::Field&&recorded)require(pilot.ready&&pilot.alignment.valid&&
                    std::abs(pilot.alignment.offset-r.at("lane_alignment_offset_px").get<float>())<1.f&&
                    std::abs(pilot.alignment.heading-r.at("lane_alignment_heading_px").get<float>())<1.f,
                    "Recomputed field alignment differs from the recorded camera");
                if(fieldCheck)require(!plan.stop&&!plan.selectedBranch&&pilot.ready&&!pilot.lost,
                    "Recorded clear road is still stopped or overridden by a synthetic branch");
                outputSpeed=plan.stop?0:pilot.speedFor(std::min(plan.speedCap,r.at("cruise_setting").get<float>()),dt);
                outputPwm=raceDirectionServo(plan,plan.alignRoad&&pilot.alignment.valid?pilot.alignment.servo:pilot.servo);
                if(f==stop){oldStopSpeed=outputSpeed;
                    if(fieldCheck)require(plan.guidePwm==0&&oldStopSpeed>0&&outputPwm!=PWMSERVOMID,
                        "No positive, lane-steered command at the original stop");}
            }
            if(plan.completedDirection){
                require(fieldCheck&&plan.completedDirection==direction&&!plan.stop,"Unsafe or opposite fork completion");
                completed=f;break;
            }
        }
        if(fieldCheck)require(!held&&handoff>=stop-3&&handoff<=stop&&completed>=stop&&oldStopSpeed>0,
            "Field approach did not continue under the current-road pilot and finish the signed fork");
        else {
            require(!handoff&&!completed,"Invalid sign passage, motion, or geometry accepted");
            if(scene!=Scene::Stationary&&scene!=Scene::StillSplit&&scene!=Scene::ConflictingSigns)
                require(held,"Lost target traveled beyond the existing stop budget");
        }
        return std::make_pair(handoff,completed);
    };
    auto left=replay(Scene::Field);replay(Scene::Mirror);replay(Scene::FixedPlate);replay(Scene::JumpingPlate);
    replay(Scene::ConflictingSigns);replay(Scene::WrongSteer);replay(Scene::Stationary);replay(Scene::Stale);
    replay(Scene::MissingRoad);replay(Scene::StillSplit);
    std::cout<<"PASS signed fork passage "<<dir.filename().string()<<" original_stop="<<stop
        <<" field_window_handoff="<<left.first<<" frozen_image_completion="<<left.second
        <<"; recorded sign/motion history and four corrected images, modeled continuation; failed evidence remains guarded\n";
}

static void signedForkPassageFieldChecks(const fs::path &root,const fs::path &latestDir) {
    signedForkPassageRun(root/"vehicle_evidence/mission-1793783401337");
    signedForkPassageRun(latestDir);
}
