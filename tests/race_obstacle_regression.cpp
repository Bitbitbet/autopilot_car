#include "race_obstacle.hpp"
#include "race_ui.hpp"
#include <filesystem>
#include <iostream>

static void check(bool ok,const char *reason) { if(!ok)throw std::runtime_error(reason); }
static PredictResult obstacle(int x,int width=30,int y=100,int height=60,int type=LABEL_CONE) {
    return {type,"",.95f,x,y,width,height};
}
static cv::Mat road() {
    cv::Mat bin=cv::Mat::zeros(240,320,CV_8UC1);cv::rectangle(bin,{40,0,240,240},{255},-1);return bin;
}
struct Fixture {
    RaceConfig race;
    RaceNavigationConfig cfg;
    RaceObstaclePreview preview;
    cv::Mat base=road(),bin;
    Track track;
    RaceSnapshot snapshot;
    uint64_t sequence=0;
    double time=0;
    explicit Fixture(RaceNavigationConfig config={}):cfg(config),preview(cfg) {}
    RaceObstaclePreviewPlan step(std::vector<PredictResult> ds={},bool next=true,double dt=.07) {
        time+=dt;
        if(next) {
            bin=base.clone();
            for(const auto &d:ds)if(raceDetectionBoxValid(d))cv::rectangle(bin,{d.x,d.y,d.width,d.height},{0},-1);
            track.handle(bin);cv::Mat rgb;cv::cvtColor(bin,rgb,cv::COLOR_GRAY2BGR);
            snapshot={++sequence,time,std::move(ds),false,{},rgb};
        }
        return preview.update(snapshot,time,bin,track,race);
    }
    RaceObstaclePreviewPlan repeat(std::vector<PredictResult> ds,int n=3) { RaceObstaclePreviewPlan p;while(n--)p=step(ds);return p; }
};
static void verifyGeometry(const RaceObstaclePreviewPlan &p,const RaceNavigationConfig &cfg) {
    check(p.candidate&&!p.stop&&!p.centers.empty(),"no preview route");
    check(p.centers.size()==p.left.size()&&p.left.size()==p.right.size(),"path edge size mismatch");
    for(size_t i=0;i<p.centers.size();++i) {
        check(p.left[i].y>=40&&p.right[i].y<=279,"proposed vehicle corridor left road");
        if(i)check(std::abs(p.centers[i].y-p.centers[i-1].y)<=
            (p.centers[i-1].x-p.centers[i].x)*cfg.maxLateralSlope+1e-4,"route has an excessive lateral step");
        for(const auto &o:p.obstacles) {
            const int pad=std::max(3,o.width/4);
            if(p.centers[i].x<o.y-pad||p.centers[i].x>o.br().y+pad)continue;
            check(p.right[i].y<=o.x||p.left[i].y>=o.br().x,"proposed corridor intersects obstacle");
        }
    }
}
int main(int argc,char **argv) {
    try {
        Fixture left;auto p=left.step({obstacle(60)});
        check(!p.candidate&&p.state=="OBS_CONFIRMING","single result confirmed obstacle");
        left.step({},false);left.step({},false);
        check(!left.step({},false).candidate,"old snapshot counted repeatedly");
        p=left.repeat({obstacle(60)},2);verifyGeometry(p,left.cfg);
        check(p.direction==1&&p.state=="OBS_BYPASS_RIGHT_PREVIEW","left obstacle route must pass right");
        const auto imageBefore=left.snapshot.image.clone(),binBefore=left.bin.clone();
        const auto rightPath=p;
        auto metadata=raceObstacleJson(p,left.cfg);
        check(metadata["preview_only"].get<bool>()&&!metadata["drives_vehicle"].get<bool>()&&
              !metadata["physical_clearance_verified"].get<bool>()&&metadata["units"]=="image_pixels","preview metadata claims physical clearance/actuation");
        auto image=raceCompactView(imageBefore,left.bin,left.snapshot,left.time,left.race,false,&p);
        const auto plain=raceCompactView(imageBefore,left.bin,left.snapshot,left.time,left.race,false);
        check(cv::norm(image,plain)>0,"matched preview path was not drawn");
        auto wrong=p;wrong.sequence++;
        check(cv::norm(plain,raceCompactView(imageBefore,left.bin,left.snapshot,left.time,left.race,false,&wrong))==0,
            "route from another AI frame was drawn");
        const auto staleImage=raceCompactView(imageBefore,left.bin,left.snapshot,left.time+1,left.race,false,&p);
        const auto stalePlain=raceCompactView(imageBefore,left.bin,left.snapshot,left.time+1,left.race,false);
        check(cv::norm(staleImage(cv::Rect(320,0,320,240)),stalePlain(cv::Rect(320,0,320,240)))==0,"stale route drawn over image");
        check(cv::norm(imageBefore,left.snapshot.image)==0&&cv::norm(binBefore,left.bin)==0,"planner/rendering mutated input image");
        Fixture right;p=right.repeat({obstacle(230,30,100,60,LABEL_PERSON)});verifyGeometry(p,right.cfg);
        check(p.direction==-1,"right person route must pass left");
        Fixture clipped;cv::rectangle(clipped.base,{40,0,280,240},{255},-1);
        p=clipped.repeat({obstacle(60)});
        check(p.candidate&&p.stop&&p.visibilityLimited&&p.clippedRows>0&&
              p.state=="OBS_VISIBLE_PASS_RIGHT_PREVIEW","clipped road did not yield a limited preview with stop retained");
        const auto limited=raceObstacleJson(p,clipped.cfg);
        check(!limited["physical_clearance_verified"].get<bool>()&&limited["would_stop"].get<bool>()&&
            limited["width_reference"]=="visible_image_corridor","partial field of view claimed complete clearance");
        for(size_t i=0;i<p.centers.size();++i) {
            check(p.left[i].y>=p.visibleLeft[i].y&&p.right[i].y<=p.visibleRight[i].y,"limited path left visible corridor");
            check(p.rightBoundarySeen[i]==0,"clipped right edge invented an observed boundary");
        }
        Fixture center;p=center.repeat({obstacle(145,30)});
        check(p.stop&&!p.candidate&&p.state=="OBS_NO_CLEAR_PATH","blocked center invented a passable gap");
        Fixture two;p=two.repeat({obstacle(100,25),obstacle(195,25)});
        check(p.stop&&!p.candidate,"multiple obstacles not jointly checked");
        p=left.repeat({obstacle(60),obstacle(180,50)},1);
        check(p.stop&&!p.candidate,"new unconfirmed obstacle ignored while planning around another target");
        p=left.step({});check(p.stop&&!p.candidate&&p.state=="OBS_LOST_BEFORE_NEAR","missing far target declared passed");
        p=left.repeat({obstacle(60)},3);verifyGeometry(p,left.cfg);
        p=left.step({},false,.5);check(p.stop&&!p.fresh&&p.state=="OBS_AI_STALE","stale AI preserved route");
        p=left.step({obstacle(60)});check(!p.candidate,"post-stale observation reused old confirmation count");
        Fixture gap;gap.repeat({obstacle(60)},2);p=gap.step({obstacle(60)},true,.35);
        check(!p.candidate,"capture gap did not reset confirmation");
        Fixture laneGap;laneGap.repeat({obstacle(60)},2);
        laneGap.base=cv::Mat::zeros(240,320,CV_8UC1);laneGap.step({obstacle(60)});
        laneGap.base=road();check(!laneGap.step({obstacle(60)}).candidate,"lane-unavailable observation retained confirmation");
        RaceNavigationConfig slow;slow.rowStep=1;slow.maxLateralSlope=.1f;
        Fixture slowSlope(slow);p=slowSlope.repeat({obstacle(60)});verifyGeometry(p,slowSlope.cfg);
        for(size_t i=1;i<p.centers.size();++i)check(p.centers[i].y==p.centers[i-1].y,"small slope bound rounded up");
        Fixture offRoad;p=offRoad.repeat({obstacle(5,20)});
        check(!p.hasTarget&&!p.candidate&&p.state=="OBS_NO_TARGET","off-road cone treated as in-lane obstacle");
        Fixture near;p=near.repeat({obstacle(60,30,170,69)});
        check(p.stop&&p.state=="OBS_NEAR_REGION_UNSEEN","partially unseen near obstacle allowed a path");
        Fixture empty;empty.step({});empty.snapshot.image.release();empty.snapshot.sequence++;
        empty.snapshot.captured=empty.time+.01;p=empty.preview.update(empty.snapshot,empty.time+.01,empty.bin,empty.track,empty.race);
        check(p.stop&&p.state=="OBS_NO_MATCHED_IMAGE","missing matched image accepted");
        Fixture fork;cv::rectangle(fork.base,{145,0,30,190},{0},-1);
        p=fork.repeat({obstacle(60)});check(p.stop&&p.state=="OBS_BRANCH_AMBIGUOUS","bypass chose an unconfigured branch");
        Fixture low;auto lowScore=obstacle(60);lowScore.score=.6f;
        check(!low.repeat({lowScore}).hasTarget,"other-class low scores leaked into bypass");
        RaceNavigationConfig disabled;disabled.obstacleMode="off";Fixture off(disabled);
        check(off.repeat({obstacle(60)}).state=="OBS_PREVIEW_OFF","preview disable ignored");
        bool rejected=false;try { RaceNavigationConfig c;c.obstacleMode="drive";c.validate(); }catch(const std::invalid_argument&) { rejected=true; }
        check(rejected,"uncalibrated pixel preview enabled vehicle control");
        // 新规划输出不改变现有实跑的障碍停车行为。
        RaceManager manager(left.race);Track straight;auto straightBin=road();straight.handle(straightBin);RacePlan control;
        for(int i=1;i<=5;++i) {
            RaceSnapshot s{uint64_t(i),i*.07,{obstacle(60)},false,{},imageBefore};
            control=manager.update(s,i*.07,.07f,true,straightBin,straight);
        }
        check(control.stop&&std::string(control.state)=="OBSTACLE_WAIT","preview planning changed real drive obstacle stop");
        if(argc>=2&&argc<=4) {
            const std::filesystem::path out=argv[1];std::filesystem::create_directories(out);
            auto mismatched=rightPath;mismatched.sequence=right.snapshot.sequence+100;
            check(saveRaceSample(out,right.snapshot,right.time,right.race,1,&mismatched,right.cfg),"sample save failed");
            check(!std::filesystem::is_regular_file(out/"ai-sample-1/navigation.json"),"mismatched navigation data saved with another image");
            // 使用匹配样本另存一份，导航数据须与图像绑定。
            const auto matching=right.preview.update(right.snapshot,right.time,right.bin,right.track,right.race);
            check(saveRaceSample(out,right.snapshot,right.time,right.race,2,&matching,right.cfg),"matched navigation sample save failed");
            check(std::filesystem::is_regular_file(out/"ai-sample-2/navigation.json"),"navigation metadata missing");
            std::ifstream saved(out/"ai-sample-2/navigation.json");nlohmann::json j;saved>>j;
            check(j["frame"]==right.snapshot.sequence&&j["preview_only"].get<bool>(),"saved navigation is not matched preview");
            // 保存整程序回放，最后添加道路中心阻断以验证停车响应仍由原策略控制。
            cv::VideoWriter video((out/"obstacle-scenario.avi").string(),cv::VideoWriter::fourcc('M','J','P','G'),15,{320,240});
            check(video.isOpened(),"cannot create obstacle replay");std::ofstream labels(out/"obstacle-scenario.jsonl");
            for(int frame=1;frame<=90;++frame) {
                const auto d=frame<55?obstacle(60):obstacle(145);auto bin=road();
                if(frame>12)cv::rectangle(bin,{d.x,d.y,d.width,d.height},{0},-1);
                cv::Mat rgb;cv::cvtColor(bin,rgb,cv::COLOR_GRAY2BGR);video.write(rgb);
                auto ds=nlohmann::json::array();
                if(frame>12)ds.push_back({{"type",d.type},{"score",d.score},{"x",d.x},{"y",d.y},{"width",d.width},{"height",d.height}});
                labels<<nlohmann::json({{"frame",frame},{"detections",ds}}).dump()<<'\n';
            }
        } else check(argc==1,"usage: race_obstacle_regression [output [saved-field-review [navigation.json]]]");
        if(argc==3||argc==4) {
            const std::filesystem::path review=argv[2],out=argv[1];int samples=0,accepted=0,candidates=0;
            RaceNavigationConfig navigation;if(argc==4)navigation.load(argv[3],true);
            for(const auto &run:std::filesystem::directory_iterator(review))if(run.is_directory()) {
                for(int index=1;index<=3;++index) {
                    const auto sample=run.path()/("ai-sample-"+std::to_string(index));
                    if(!std::filesystem::is_regular_file(sample/"camera.png"))continue;
                    const auto camera=cv::imread((sample/"camera.png").string());check(!camera.empty(),"field sample missing");
                    std::ifstream input(sample/"detections.json");nlohmann::json recorded;input>>recorded;
                    RaceSnapshot snap;snap.image=camera;
                    for(const auto &v:recorded.at("detections")) {
                        PredictResult d{};d.type=v.at("type");d.score=v.at("score");
                        d.x=v.at("x");d.y=v.at("y");d.width=v.at("width");d.height=v.at("height");snap.detections.push_back(d);
                    }
                    cv::Mat gray,binary;cv::cvtColor(camera,gray,cv::COLOR_BGR2GRAY);
                    cv::threshold(gray,binary,0,255,cv::THRESH_BINARY|cv::THRESH_OTSU);Track lane;lane.handle(binary);
                    RaceConfig cfg;RaceObstaclePreview planner(navigation);RaceObstaclePreviewPlan result;
                    for(int seq=1;seq<=3;++seq) { snap.sequence=seq;snap.captured=seq*.07;result=planner.update(snap,snap.captured,binary,lane,cfg); }
                    const bool detected=std::any_of(snap.detections.begin(),snap.detections.end(),[&](const auto &d) {
                        return d.type==LABEL_CONE&&reviewRaceDetection(d,camera,cfg).accepted;
                    });
                    if(detected) {
                        ++accepted;check(result.hasTarget&&result.visibilityLimited&&result.stop,"field clipping lost target or cleared stop");
                        check(result.state!="OBS_LANE_CLIPPED","field sample still rejected solely for image boundary clipping");
                    } else check(result.state=="OBS_NO_TARGET","missing model candidate was fabricated");
                    ++samples;candidates+=result.candidate;
                    saveRaceSample(out/"field-review"/run.path().filename(),snap,snap.captured,cfg,index,&result,navigation);
                    std::cout<<"field "<<run.path().filename()<<'/'<<index<<" state="<<result.state<<" clipped_rows="<<result.clippedRows<<" candidate="<<result.candidate<<'\n';
                }
            }
            check(samples==6&&accepted==4,"expected six saved field samples including four accepted cones");
            check(candidates>=1,"field visibility regression rejected every local corridor");
            std::cout<<"PASS saved field images="<<samples<<" accepted_cones="<<accepted<<" visible_candidates="<<candidates<<'\n';
        }
        std::cout<<"PASS obstacle preview: left/right, joint collision checks, blocked road, object identity, matched/stale images, confirmation, gaps, off-road targets and unchanged drive stop\n";
    }catch(const std::exception &e) { std::cerr<<"FAIL "<<e.what()<<'\n';return 1; }
}
