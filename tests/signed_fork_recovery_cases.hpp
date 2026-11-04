#pragma once

// Synthetic scene, not an image from the field run. The wide and narrow arms
// deliberately make a nearest-centre selector choose the arm opposite the sign.
static cv::Mat signedForkScene(int direction,bool completeDivider) {
    cv::Mat bin=cv::Mat::zeros(480,640,CV_8UC1);
    cv::rectangle(bin,{96,0,448,480},cv::Scalar(255),cv::FILLED);
    const int top=completeDivider?0:12;
    cv::rectangle(bin,{342,top,38,320-top},cv::Scalar(0),cv::FILLED);
    if(direction<0)cv::flip(bin,bin,1);
    return bin;
}

static cv::Mat signedForkSingleRoad(int direction,int bendStart=415) {
    cv::Mat bin=cv::Mat::zeros(480,640,CV_8UC1);
    for(int y=0;y<bin.rows;++y){
        const int center=320+direction*std::clamp((bendStart-y)/2,0,136);
        bin.row(y).colRange(center-78,center+79).setTo(255);
    }
    return bin;
}

static float signedForkAheadCenter(const RacePath &p) {
    float x=0;int count=0;
    for(size_t i=0;i<p.left.size();++i)if(p.left[i].x>=140&&p.left[i].x<=240){
        x+=(p.left[i].y+p.right[i].y)*.5f;++count;
    }
    require(count>=6,"Synthetic signed arm has no continuous visible section");
    return x/count;
}

struct SignedForkFixture {
    RaceConfig cfg;
    RaceManager manager{cfg};
    RaceSnapshot s;
    RacePlan plan;
    int direction;
    double now=0;
    explicit SignedForkFixture(int d):direction(d){}
    RacePlan step(const cv::Mat &bin,float dt,float applied,bool sign,bool usable,float offset=0,float heading=0) {
        Track track;track.handle(bin);
        ++s.sequence;s.captured=now+=dt;
        s.detections=sign?std::vector<PredictResult>{box(direction<0?LABEL_YLEFT:LABEL_YRIGHT,48,12,64,48)}:
                          std::vector<PredictResult>{};
        return plan=manager.update(s,now,dt,true,bin,track,{},false,true,0,true,false,.25f,
            {true,usable,offset,heading,applied});
    }
    void acquire(const cv::Mat &fork) {
        for(int i=0;i<32;++i)step(fork,.07f,.25f,true,true);
        require(plan.selectedBranch&&plan.guideDirection==direction&&plan.forkPulsePhase>=1.4f&&!plan.stop,
            "Synthetic signed fork did not establish its target before loss");
    }
};

static void signedForkLeftFieldChecks(const fs::path &root) {
    const auto dir=root/"vehicle_evidence/parking-20261104-064134/mission-1793774313016";
    RaceConfig cfg;cfg.load((dir/"race-config.json").string());
    TrackviewConfig vehicle;vehicle.load((dir/"trackview-config.json").string(),true);
    RaceGroundCalibration ground;ground.load((dir/"ground-calibration.json").string(),false);
    cv::Point2f corners[4],world[4]={{-ground.roadWidth/2,ground.nearDistance},{ground.roadWidth/2,ground.nearDistance},
        {-ground.roadWidth/2,ground.farDistance},{ground.roadWidth/2,ground.farDistance}};
    for(int i=0;i<4;++i)corners[i]={ground.imagePoints[i][0],ground.imagePoints[i][1]};
    Camera camera(cv::getPerspectiveTransform(corners,world));camera.blackRoad=vehicle.roadSurface=="black";
    require(camera.blackRoad&&vehicle.binary<0,"Incorrect field surface or binarization for left fork");
    struct RestoreCamera {Camera *saved;~RestoreCamera(){Track::camera=saved;}} restore{Track::camera};
    Track::camera=&camera;
    std::map<int,nlohmann::json> recorded;
    std::ifstream input(dir/"parking-control.jsonl");std::string line;
    while(std::getline(input,line)) {
        auto row=nlohmann::json::parse(line);int f=row.at("frame");
        if((f>=536&&f<=565)||f==709)recorded.emplace(f,std::move(row));
    }
    require(recorded.size()==31,"Missing recorded steering and road feedback before frame 565");
    std::map<int,cv::Mat> images;
    for(int f:{562,563,564,565,709}){
        auto path=f==709?dir/"sample-709.png":dir/"signed-fork-wait"/("frame-"+std::to_string(f)+".png");
        images[f]=cv::imread(path.string());require(images[f].size()==cv::Size(640,480),"Missing original corrected left-fork image");
        // Saved frames are already lens-corrected. Reapply only the production
        // ground mapping, black-road binarization and native track/pilot.
        const auto binary=camera.binarize(images[f]);Track native;native.handle(binary);
        RaceTaskPilot pilot(vehicle);pilot.update(native.pointsEdgeLeft,native.pointsEdgeRight,.07f,0);
        const auto &r=recorded.at(f);
        require(pilot.ready&&pilot.alignment.valid&&
            std::abs(pilot.alignment.offset-r.at("lane_alignment_offset_px").get<float>())<1.f&&
            std::abs(pilot.alignment.heading-r.at("lane_alignment_heading_px").get<float>())<1.f,
            "Cannot reproduce recorded near-road alignment from the original corrected frame");
        const int row=native.pointsEdgeLeft.front().x;
        const auto hood=raceRoadBlocks(binary,row,0,639,cfg,{});
        const auto ahead=raceRoadBlocks(binary,row-128,0,639,cfg,{});
        require(hood.size()==2&&hood[0].second<220&&hood[1].first>410&&ahead.size()==1&&
            (ahead[0].first>2||ahead[0].second<637),"Expected separated areas beside the bonnet and a real road edge ahead");
        require(raceBranchPath(binary,native,0,cfg).singleRoadAhead,
            "The car bonnet is counted as a continuing fork ahead of the vehicle");
    }
    // Removing every road edge ahead leaves the same apparent arms beside
    // the bonnet. The image alone then supplies no usable single corridor.
    auto noEdges=camera.binarize(images.at(565));
    noEdges.rowRange(0,390).setTo(255);
    Track edgeless;edgeless.handle(noEdges);
    require(!raceBranchPath(noEdges,edgeless,0,cfg).singleRoadAhead,
        "Bonnet and featureless full-width surface alone create a native lane handoff");
    // Only frames 562..565 were saved during this turn. Before 562 we replay
    // measured pilot feedback/sign boxes/timing on empty binary geometry. Once
    // the old run stops, use a frozen image and explicitly simulated commands.
    enum class Scenario { Field,Mirrored,WrongSide,Stationary,MissingRoad };
    auto simulate=[&](Scenario kind){
        const bool mirror=kind==Scenario::Mirrored;
        const float feedbackSign=(mirror||kind==Scenario::WrongSide)?-1.f:1.f;
        const int direction=mirror?1:-1;
        RaceManager manager(cfg);RaceSnapshot snap;RacePlan plan;double time=1;
        const cv::Mat blank=cv::Mat::zeros(480,640,CV_8UC1);
        bool completed=false,held=false;int handoffFrame=0,doneFrame=0;
        for(int f=537;f<=591;++f){
            const bool field=f<=565;
            const auto &r=recorded.at(std::min(f,565)),&previous=recorded.at(std::min(f-1,565));
            const float dt=field?r.at("control_dt_s").get<float>():.07f;
            float applied=field?previous.at("command_speed").get<float>():(plan.stop?0.f:.25f);
            if(kind==Scenario::Stationary)applied=0;
            time+=dt;
            snap.sequence=field?r.at("ai_sequence").get<uint64_t>():snap.sequence+1;
            snap.captured=field?time-r.at("ai_age_s").get<double>():time-.07;
            snap.detections.clear();
            if(field)for(const auto &obj:r.at("ai_detections")){
                const auto b=obj.at("box");int type=obj.at("type");
                if(mirror&&type==LABEL_YLEFT)type=LABEL_YRIGHT;
                auto d=box(type,mirror?640-b[0].get<int>()-b[2].get<int>():b[0].get<int>(),b[1],b[2],b[3]);
                d.score=obj.at("score");snap.detections.push_back(d);
            }
            cv::Mat image,binary=blank;
            if(f>=562&&kind!=Scenario::MissingRoad){
                if(mirror)cv::flip(images.at(std::min(f,565)),image,1);
                else image=images.at(std::min(f,565));
                binary=camera.binarize(image);
            }
            Track track;track.handle(binary);
            plan=manager.update(snap,time,dt,true,binary,track,image,true,true,0,true,false,.25f,
                {true,previous.at("lane_alignment_valid").get<bool>(),
                 feedbackSign*previous.at("lane_alignment_offset_px").get<float>(),
                 feedbackSign*previous.at("lane_alignment_heading_px").get<float>(),applied});
            held|=std::string_view(plan.reason)=="SIGNED_FORK_WAIT_TARGET_ROAD";
            if(std::string_view(plan.reason)=="SIGNED_FORK_NEAR_ROAD_HANDOFF"){
                if(!handoffFrame)handoffFrame=f;
                require(!plan.selectedBranch&&!plan.stop&&plan.alignRoad&&plan.guidePwm==0,
                    "Near-road handoff overrides the normal lane pilot or repeats sign steering");
            }
            if(plan.completedDirection){
                require(plan.completedDirection==direction,"Field fork handed off the opposite direction");
                completed=true;doneFrame=f;break;
            }
        }
        if(kind==Scenario::Field||mirror) {
            require(!held&&handoffFrame>=562&&handoffFrame<=565&&completed&&doneFrame>=565,
                "Visible selected-side road still stops or fails to rejoin the native lane");
        } else {
            require(!handoffFrame&&!completed,"Fork hands off without signed approach progress or real road evidence");
            if(kind==Scenario::MissingRoad)require(held,"Missing actual road does not retain the blind-travel hold");
        }
        return std::make_pair(handoffFrame,doneFrame);
    };
    const auto left=simulate(Scenario::Field),right=simulate(Scenario::Mirrored);
    simulate(Scenario::WrongSide);simulate(Scenario::Stationary);simulate(Scenario::MissingRoad);
    std::cout<<"PASS field left fork handoff: 562..565 and 709 calibrated near road; recorded command and offset history; "
        <<"near-road takeover "<<left.first<<" mirrored "<<right.first
        <<"; frozen-road completion "<<left.second<<", mirrored "<<right.second
        <<"; stationary, wrong-side and absent-road rejection\n";
}

static void signedForkRecoveryChecks(const fs::path &root) {
    const cv::Mat empty=cv::Mat::zeros(480,640,CV_8UC1);
    const auto centered=signedForkSingleRoad(0);
    auto waiting=[](const RacePlan &p){return p.stop&&std::string_view(p.reason)=="SIGNED_FORK_WAIT_TARGET_ROAD";};
    for(int dir:{-1,1}) {
        const auto full=signedForkScene(dir,true),clipped=signedForkScene(dir,false),single=signedForkSingleRoad(dir);
        Track t;t.handle(clipped);RaceConfig cfg;
        const auto signedPath=raceBranchPath(clipped,t,dir,cfg,{},true),neutral=raceBranchPath(clipped,t,2,cfg);
        require(signedPath.splitRows==0&&dir*(signedForkAheadCenter(signedPath)-signedForkAheadCenter(neutral))>80,
            "The clipped fork stimulus must distinguish the requested arm from nearest-centre selection");
        SignedForkFixture f(dir);f.acquire(full);
        for(int i=0;i<4;++i){
            const auto p=f.step(clipped,.07f,.25f,false,true);
            require(p.selectedBranch&&!p.stop&&dir*(signedForkAheadCenter(p.path)-320)>65,
                "Loss of the far divider silently switches the signed branch to the other visible arm");
        }
        auto stillSplit=f;
        for(int i=0;i<15;++i) {
            const auto p=stillSplit.step(clipped,.07f,.25f,false,true);
            require(p.selectedBranch&&!p.completedDirection&&!p.stop,
                "Visible adjacent fork arms count as a completed turn because their divider ends before the top");
        }
        const auto p=f.step(single,.07f,.25f,false,true);
        require(p.selectedBranch&&!p.stop&&dir*(signedForkAheadCenter(p.path)-320)>55,
            "The connected single road continuing the selected arm cannot take over");
        for(int i=0;i<12&&!waiting(f.plan);++i)f.step(empty,.07f,.25f,false,false);
        require(waiting(f.plan),"Loss of the known side arm does not reach the hold");
        for(int i=0;i<14;++i) {
            const auto plan=f.step(centered,.07f,0,false,true);
            require(waiting(plan)&&!plan.completedDirection,
                "An unrelated straight lane unlocks the hold after loss of an offset signed arm");
        }
    }
    std::cout<<"PASS signed fork branch continuity: mirrored clipped dividers and the resulting single road\n";

    const auto log=root/"vehicle_evidence/parking-20261102-102527/mission-1793614554935/mission-state.jsonl";
    std::ifstream in(log);require(bool(in),"Missing right-fork field telemetry");
    struct Row {int frame;float dt,previousSpeed;bool hadGeometry;};
    std::vector<Row> loss;float previousSpeed=0;std::string line;
    while(std::getline(in,line)){
        const auto r=nlohmann::json::parse(line);const int frame=r.at("frame");
        if(frame>=457&&frame<=603){
            require(!r.at("lane_alignment_valid").get<bool>(),"Unexpected valid road in field loss interval");
            loss.push_back({frame,r.at("control_dt_s").get<float>(),previousSpeed,
                r.at("traffic_reason").get<std::string>()!="TARGET_ROAD_LOST_CURRENT_LANE_RECOVERY"});
        }
        previousSpeed=r.at("speed_command");
    }
    require(loss.size()==147&&loss.front().frame==457&&loss.back().frame==603,
        "Incomplete 457..603 right-fork command history");
    // Only dt, command speed and the validity bit are recorded inputs. The two
    // branch images below represent coarse geometry present/absent respectively:
    // no camera frames exist to reconstruct the trajectory after a new decision.
    const auto full=signedForkScene(1,true),same=signedForkSingleRoad(1),opposite=signedForkSingleRoad(-1);
    // This right arm is already locally straight under the car. Its far end
    // identifies the lost arm while its near pixels can satisfy alignedTime.
    const auto alignedRight=signedForkSingleRoad(1,238);
    int fieldHold=0;
    for(bool slow:{false,true}){
        SignedForkFixture f(1);f.acquire(full);
        float distance=0,movingTime=0;bool held=false;
        for(const auto &r:loss){
            const float speed=slow?.035f:r.previousSpeed;
            const auto p=f.step(r.hadGeometry?full:empty,r.dt,speed,false,false);
            distance+=speed*r.dt;movingTime+=r.dt;
            require(!p.completedDirection,"Blind fork falsely reports completion");
            require(distance<=.1201f+speed*r.dt&&movingTime<=.9001f+r.dt,
                "Signed fork keeps driving beyond its blind travel or moving-time limit");
            if(waiting(p)){
                require(slow?movingTime>=.80f:distance>=.095f,"Signed fork never allows a brief dropout");
                held=true;if(!slow)fieldHold=r.frame;break;
            }
        }
        require(held,"Sustained lack of usable road never stops the ordinary right fork");
        // Once the new policy stops, old field commands are no longer inputs.
        // The following are stationary, explicitly synthetic observations.
        auto tick=[&](const cv::Mat &bin,bool valid,float heading=0){return f.step(bin,.07f,0,false,valid,0,heading);};
        for(int i=0;i<25;++i)require(waiting(tick(empty,false)),"Waiting at rest releases the lost road");
        require(waiting(tick(same,true,22)),"One stationary glimpse restarts the car");
        for(int i=0;i<8;++i)require(waiting(tick(opposite,true,-22)),"Opposite branch clears target-road hold");
        bool resumed=false;
        for(int i=0;i<12;++i){
            const auto p=tick(alignedRight,true);
            require(!p.completedDirection,"Recovery at rest fabricates completion");
            if(!p.stop){resumed=true;break;}
        }
        require(resumed,"A sustained same-side target cannot resume from zero applied speed");
        for(int i=0;i<25;++i) {
            const auto p=tick(alignedRight,true);
            require(!p.stop&&!p.completedDirection,"Geometric alignment completes the fork while stationary after reacquisition");
        }
        bool completed=false;
        for(int i=1;i<=25;++i) {
            const auto p=f.step(alignedRight,.07f,.20f,false,true);
            if(p.completedDirection){
                require(i*.07f>=.60f&&p.completedDirection==1&&!p.stop,"Fork handoff precedes sustained forward road tracking");
                completed=true;break;
            }
        }
        require(completed,"Resumed forward tracking cannot finish the reacquired road");
    }
    SignedForkFixture shortGap(1);shortGap.acquire(full);
    for(int i=0;i<2;++i)require(!shortGap.step(empty,.07f,.25f,false,false).stop,
        "Two missing frames immediately stop right fork");
    for(int i=0;i<4;++i){
        const auto p=shortGap.step(full,.07f,.25f,false,true);
        require(!p.stop&&!p.completedDirection&&p.selectedBranch,"Short dropout does not recover to selected right road");
    }
    std::cout<<"PASS recorded right fork loss: observed timing/speed and absent valid lane from frame 457; "
        <<"bounded hold at "<<fieldHold<<"; synthetic stationary reacquisition, slow timeout and brief dropout (no trajectory replay)\n";
    signedForkLeftFieldChecks(root);
}
