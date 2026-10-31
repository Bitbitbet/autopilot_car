#include "race_mission.hpp"
#include "race_parking_actuator.hpp"
#include <iostream>
static void require(bool ok,const char *why){if(!ok)throw std::runtime_error(why);}
int main(){try{
    RaceMission mission;RaceSnapshot ai;RacePlan plan;plan.fresh=true;plan.state="LANE";
    auto frame=[&](bool branch,bool mouth=false){++ai.sequence;plan.selectedBranch=branch;plan.state=branch?"FORK_LEFT":"LANE";mission.observe(ai,plan,mouth);};
    mission.markStart();require(!mission.complete()&&!mission.running(),"Start mark bypasses Ready");
    mission.start();require(mission.routeDirection()==2,"Start FLOW must continue straight before parking sign");
    ai.detections={PredictResult{}};ai.detections[0].type=LABEL_FLOW;ai.detections[0].score=.95f;
    for(int i=0;i<4;++i){++ai.sequence;mission.observeSigns(ai,true);}
    require(!mission.parkingZone()&&mission.routeDirection()==2,"Start FLOW skipped parking task");
    ai.detections[0].type=LABEL_PARK;
    for(int i=0;i<3;++i){++ai.sequence;mission.observeSigns(ai,true);}
    require(mission.routeDirection()==-1&&mission.parkingApproach(),"Confirmed parking sign does not immediately request left");
    ai.detections.clear();
    ++ai.sequence;plan.state="FORK_SEEK_LEFT";plan.guideDirection=-1;mission.observe(ai,plan,true);
    require(mission.routeDirection()==-1&&!mission.parkingSearch(),"direction guidance falsely completes entrance turn");
    plan.guideDirection=0;
    frame(false,true);require(!mission.parkingSearch(),"Bay visibility skips entrance turn");
    frame(true);++ai.sequence;plan.selectedBranch=false;plan.state="GATE_WAIT";plan.stop=true;mission.observe(ai,plan,false);
    plan.stop=false;frame(false);require(mission.parkingZone(),"Gate wait loses completed turn handoff");
    frame(false,true);require(mission.parkingSearch(),"Visible bay does not stop for planning");
    mission.nextParkingBay();require(mission.parkingApproach()&&!mission.parkingSearch()&&mission.routeDirection()==0,
        "Failed parking plan must resume aisle search without repeating entrance turn");
    frame(false,false);require(!mission.parkingSearch(),"No new visible bay immediately retries failed planning");
    frame(false,true);require(mission.parkingSearch(),"New available bay cannot restart parking planning");
    mission.parkingStarted();mission.markStart();require(mission.parkingRun(),"Manual lap mark skips active parking");
    mission.parkingFinished();require(mission.routeDirection()==-1,"Parking exit left missing");
    frame(true);frame(false);require(std::string(mission.name()).find("LAP1_RETURN")==0,"Parking exit not completed");
    ai.detections={PredictResult{}};ai.detections[0].type=LABEL_FLOW;ai.detections[0].score=.95f;
    for(int i=0;i<3;++i){++ai.sequence;mission.observeSigns(ai,true);}
    require(mission.routeDirection()==2,"Post-parking FLOW must follow straight continuation");
    ai.detections.clear();
    frame(true);frame(false);require(!mission.complete(),"Second route automatically claims finish without landmark");
    mission.markStart();require(mission.complete(),"Manual finish not latched");
    RaceMission lab(true);lab.start();require(lab.routeDirection()==0&&!lab.parkingZone(),"White lab inherits competition route");lab.markStart();require(lab.complete(),"Lab single loop does not finish");
    RaceMission stopped;stopped.start();stopped.stop();stopped.start();require(stopped.failed(),"Interrupted mission blindly resumes");
    RaceParkingActuator preview(false);preview.carControl(.1,1100);preview.pauseMotion();require(!preview.hasFault()&&preview.resumeMotion(preview.motionEpoch()),"Preview actuator fails without serial");
    Track track;cv::Mat bin=cv::Mat::zeros(240,320,CV_8UC1);cv::rectangle(bin,{40,0,240,240},{255},-1);track.handle(bin);
    const std::vector<cv::Point> straight{{160,55},{190,100},{172,100},{172,145},{148,145},{148,100},{130,100}};
    const std::vector<cv::Point> left{{95,105},{140,75},{140,93},{200,93},{200,117},{140,117},{140,135}};
    for(bool black:{false,true})for(int direction:{RoadLeft,RoadStraight,RoadRight}) {
        cv::Mat image(240,320,CV_8UC3,cv::Scalar(120,120,120));cv::rectangle(image,{40,0,240,240},black?cv::Scalar(20,20,20):cv::Scalar(230,230,230),-1);
        auto points=direction==RoadStraight?straight:left;if(direction==RoadRight)for(auto &p:points)p.x=320-p.x;
        cv::fillPoly(image,std::vector<std::vector<cv::Point>>{points},black?cv::Scalar(255,255,255):cv::Scalar(0,0,0));
        const auto arrow=raceGroundArrow(image,track,black);std::cout<<"surface="<<black<<" wanted="<<direction<<" found="<<arrow.allowed<<'\n';
        require(arrow.allowed==direction,"Ground arrow white/black contrast or direction incorrect");
    }
    require(!raceGroundArrow(cv::Mat(240,320,CV_8UC3,cv::Scalar(255,255,255)),track,true).allowed,"Blank road invents arrow");
    for(int direction:{-1,2}) {
        RaceConfig cfg;RaceManager manager(cfg);Track splitTrack;auto split=bin.clone();cv::rectangle(split,{95,0,16,140},{0},-1);splitTrack.handle(split);
        RaceSnapshot snap;RacePlan selected;
        for(int i=1;i<=8;++i){snap.sequence=i;snap.captured=i*.07;snap.image=cv::Mat(240,320,CV_8UC3,cv::Scalar(0,0,0));
            selected=manager.update(snap,snap.captured,.07f,true,split,splitTrack,snap.image,false,true,direction);}
        require(selected.selectedBranch&&!selected.stop,"Mission direction did not select connected branch");
        if(direction==2)require(selected.path.left.back().y>=110,"Straight continuation picked left branch");
        else require(selected.path.right.back().y<100,"Parking branch did not select left");
    }
    {
        RaceConfig cfg;RaceManager manager(cfg);Track roadTrack;roadTrack.handle(bin);
        RaceSnapshot snap;RacePlan selected;
        for(int i=1;i<=85;++i){snap.sequence=i;snap.captured=i*.07;
            selected=manager.update(snap,snap.captured,.07f,true,bin,roadTrack,{},false,true,-1,true,true);}
        require(!selected.stop&&selected.guideDirection==-1&&selected.guidePwm>=180,
                "Parking-sign mission waits for visible left road or stops on seek timeout");
    }
    {
        RaceConfig cfg;cfg.stopAfterCross=false;RaceManager manager(cfg);Track roadTrack;roadTrack.handle(bin);
        RaceSnapshot snap;RacePlan p;double now=0;
        auto step=[&](int label){now+=.1;snap.sequence++;snap.captured=now;snap.detections.clear();
            PredictResult d{};d.type=label;d.score=.95f;d.x=100;d.y=150;d.width=30;d.height=40;
            if(label>=0)snap.detections.push_back(d);
            return manager.update(snap,now,.1f,true,bin,roadTrack,{},false,true,-1,true,true);};
        for(int i=0;i<5;++i)p=step(-1);
        for(int i=0;i<4;++i)p=step(LABEL_STOP);
        require(p.stop&&std::string(p.state)=="SIGNAL_WAIT","Mission STOP during sign seek must wait for FLOW");
        for(int i=0;i<30;++i)p=step(LABEL_FLOW);
        require(!p.stop,"Mission STOP changed to FLOW does not resume automatically");
    }
    std::cout<<"PASS two-lap parking handoffs, explicit lap/finish, lab isolation, stop latch, no-motor preview and ground arrow shape tests; field validation pending\n";
}catch(const std::exception &e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}}
