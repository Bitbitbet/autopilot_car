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
int main(int argc,char **argv){try{
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
    RaceGroundArrow arrow;arrow.allowed=RoadLeft;
    for(int i=1;i<5;++i)mission.observeExitArrow(arrow,i,true);
    require(!mission.exitArrowTurnAuthorized(),"Ground arrow triggers before parking");
    ai.detections={box(LABEL_PARK)};
    for(int i=1;i<=3;++i){ai.sequence=i;mission.observeSigns(ai,true);}
    traffic.selectedBranch=true;traffic.state="FORK_LEFT";mission.observe(ai,traffic,false);
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
    mission.markStart();for(int i=20;i<25;++i)mission.observeExitArrow(arrow,i,true);
    require(mission.routeDirection()==2&&!mission.exitArrowTurnAuthorized(),"Second lap reuses parking exit arrow");
    std::cout<<"PASS exit-only ground arrow; duplicate/stale rejection; no-road left seek; second-lap isolation\n";

    RaceParkingDimensions dimensions;RaceParkingSpace space(dimensions);auto route=racePlanTrackableParking(space,{0,-.4f,0});require(route.valid,"No metric entry route");
    RaceParkingControl parking(space);RaceParkingObservation observation{route.entry.front(),1,1,1,true};require(parking.arm(route,observation,1,true),"Cannot arm parking model");
    double time=1,dwell=-1,reverse=-1;RaceParkingCommand command;
    for(int i=0;i<4000&&!parking.done();++i){time+=.08;observation.captured=time;++observation.sequence;command=parking.update(observation,time,.08,true);
        require(!parking.failed(),"Parking model faults after native migration");
        if(std::string(command.state)=="PARK_DWELL"&&dwell<0)dwell=time;
        if(command.speed<0&&reverse<0)reverse=time;
        observation.pose=raceParkingAdvance(observation.pose,command.speed*.08,command.curvature,dimensions);}
    require(parking.done()&&reverse-dwell>=3,"Entry/dwell/reverse no longer completes");
    RaceParkingBaySearch search;search.reject({-.22,.3},{-.22,.7});RaceParkingMouthCandidate failed;failed.nearGround={-.22,.3};failed.farGround={-.22,.7};require(!search.accepts(failed),"Failed bay not skipped");
    RaceParkingMouthCandidate next=failed;next.nearGround={-.22,.7};next.farGround={-.22,1.1};require(search.accepts(next),"Adjacent bay incorrectly excluded");
    std::cout<<"PASS metric parking entry/3s dwell/reverse and next-bay selection unchanged\n";

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
    fs::remove_all(temp);return 0;
}catch(const std::exception &e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}}
