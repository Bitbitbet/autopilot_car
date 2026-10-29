#include "race_bypass_control.hpp"
#include "race_ground_obstacle.hpp"
#include "race_navigation_worker.hpp"
#include "trackview_pilot.hpp"
#include <filesystem>
#include <iostream>
#include <random>

static void check(bool ok,const char *why){if(!ok)throw std::runtime_error(why);}
static RaceGroundCalibration calibration() {
    RaceGroundCalibration c;c.lensFingerprint="synthetic-only";
    c.imagePoints={{{16,160},{304,160},{104.615385f,73.846154f},{215.384615f,73.846154f}}};return c;
}
static cv::Point2f image(cv::Point2f ground){return {160+160*ground.x/ground.y,20+35/ground.y};}
// 独立生成的平面纹理场景，不连接硬件；左右绕行、丢失、空白负例共用。
struct Scene {
    RaceGroundCalibration cfg=calibration();
    RaceGroundMap map{cfg};
    RaceConfig race;
    std::vector<std::array<float,4>> texture;
    uint64_t seq=0;
    Scene() {
        std::mt19937 rng(42);std::uniform_real_distribution<float> ux(-.5f,.5f),uz(.05f,2.f),shade(15,105),size(.005,.014);
        for(int i=0;i<1600;++i){const float x=ux(rng);if(x<-.25f||x>.235f)texture.push_back({x,uz(rng),shade(rng),size(rng)});}
    }
    // roadScale 用于模拟"四点标定的米制尺度整体偏大/偏小"：路仍按45cm画，
    // 但画面里的路面宽度按比例变化，与现场把45cm换算成51~53cm的现象一致。
    RaceSnapshot frame(float moved=0,float shift=0,float coneX=-.13f,bool textured=true,bool cone=true,
                       float roadScale=1.f) {
        cv::Mat rgb(240,320,CV_8UC3,cv::Scalar(25,25,25));
        // 先画地面纹理再画路面：路面应覆盖与它重叠的深色纹理块，
        // 否则 roadScale>1 时纹理压到路沿上，会人为制造宽度抖动。
        if(textured)for(const auto &p:texture) {
            const float z=p[1]-moved;if(z<.1f||z>1.4f)continue;
            std::vector<cv::Point> corners;
            for(const auto &offset:std::vector<cv::Point2f>{{0,0},{p[3],0},{p[3],p[3]},{0,p[3]}}) {
                const auto q=image({p[0]+offset.x-shift,z+offset.y});corners.emplace_back(std::lround(q.x*256),std::lround(q.y*256));
            }
            cv::fillConvexPoly(rgb,corners,{p[2],p[2],p[2]},cv::LINE_AA,8);
        }
        for(int y=21;y<240;++y) {
            const auto ground=map.toGround({160.f,float(y)});
            if(!std::isfinite(ground.y)||ground.y<=0)continue;
            const auto l=image({-.225f*roadScale-shift,ground.y}),r=image({.225f*roadScale-shift,ground.y});
            const int left=std::clamp(int(std::ceil(l.x)),0,319),right=std::clamp(int(std::floor(r.x)),0,319);
            if(left<=right)cv::line(rgb,{left,y},{right,y},{225,225,225});
        }
        std::vector<PredictResult> detections;
        const float z=.45f-moved;
        if(cone&&z>.11f) {
            const auto l=image({coneX-.035f-shift,z}),r=image({coneX+.035f-shift,z});
            const auto box=cv::Rect(int(l.x),int(l.y)-35,std::max(1,int(r.x-l.x)),35)&cv::Rect(0,0,320,240);
            if(box.area()>0){cv::rectangle(rgb,box,{0,70,240},-1);detections.push_back({LABEL_CONE,"cone",.99f,box.x,box.y,box.width,box.height});}
        }
        ++seq;return {seq,seq*.07,std::move(detections),false,{},rgb};
    }
};
static RaceObstaclePreviewPlan plan(RaceGroundObstacle &planner,const RaceSnapshot &s,const RaceConfig &cfg) {
    cv::Mat gray,bin;cv::cvtColor(s.image,gray,cv::COLOR_BGR2GRAY);cv::threshold(gray,bin,150,255,cv::THRESH_BINARY);
    Track track;track.rowCutBottom=20;track.handle(bin);return planner.update(s,s.captured,bin,track,cfg);
}
static void acquisitionOnly(const std::filesystem::path &evidence={}) {
    Scene scene;RaceGroundObstacle planner(scene.cfg);RaceObstaclePreviewPlan result;
    auto step=[&](bool visible,bool committed,bool automatic,bool jitter=false) {
        auto snap=scene.frame();if(!visible)snap.detections.clear();
        if(jitter)for(auto &d:snap.detections)d.height+=10;
        cv::Mat gray,bin;cv::cvtColor(snap.image,gray,cv::COLOR_BGR2GRAY);cv::threshold(gray,bin,150,255,cv::THRESH_BINARY);
        Track track;track.rowCutBottom=20;track.handle(bin);
        return planner.update(snap,snap.captured,bin,track,scene.race,committed,automatic);
    };
    for(int i=0;i<6;++i)result=step(true,false,false);
    check(result.candidate,"manual acquisition did not find initial cone route");
    result=step(false,false,false);check(!result.hasTarget&&result.trackedTargets==0,"manual dropout retained old confirmed cone");
    for(int i=0;i<6;++i)result=step(true,false,false);
    check(result.candidate&&result.trackedTargets==1,"manual reacquisition remained blocked after cone reappeared");
    for(int i=0;i<6;++i)result=step(true,false,true,true);
    check(result.candidate&&result.trackedTargets==1,"AUTO acquisition bbox jitter left a duplicate old target");
    result=step(false,true,true);
    check(result.candidate&&result.predictedTargets==1&&result.trackedTargets>=1,"committed bypass lost bounded mapped cone prediction");
    if(!evidence.empty()) {
        RaceGroundCalibration cfg;cfg.load((evidence/"ground-calibration.json").string(),false);
        RaceGroundObstacle field(cfg);RaceConfig race;const auto camera=cv::imread((evidence/"ai-sample-1/camera.png").string());
        check(!camera.empty(),"missing saved field sample");cv::Mat gray,bin;cv::cvtColor(camera,gray,cv::COLOR_BGR2GRAY);
        cv::threshold(gray,bin,0,255,cv::THRESH_BINARY|cv::THRESH_OTSU);Track track;track.rowCutBottom=20;track.handle(bin);
        std::ifstream input(evidence/"detections.jsonl");std::string line;int frames=0,candidates=0,lost=0,maxTracks=0;
        while(frames<100&&std::getline(input,line)) {
            const auto j=nlohmann::json::parse(line);RaceSnapshot s;s.sequence=j.at("frame");s.captured=j.at("captured");s.image=camera;
            for(const auto &v:j.at("detections"))s.detections.push_back({v.at("type"),"",v.at("score"),v.at("x"),v.at("y"),v.at("width"),v.at("height")});
            const auto p=field.update(s,s.captured,bin,track,race,false,false);
            ++frames;candidates+=p.candidate;lost+=p.state=="BYPASS_TARGET_LOST_BEFORE_CLEARANCE";maxTracks=std::max(maxTracks,p.trackedTargets);
        }
        check(frames==100&&lost==0&&candidates>0&&maxTracks<=1,"saved stationary scene/detection sequence remained stuck in old target history");
        std::cout<<"FIELD stationary saved-image replay: frames="<<frames<<" candidates="<<candidates<<" lost_waits="<<lost<<" max_tracks="<<maxTracks<<'\n';
    }
    std::cout<<"PASS manual reacquisition, pre-motion duplicate cleanup, committed bounded prediction\n";
}
static void steeringOnly(const std::filesystem::path &evidence={}) {
    RaceNavigationConfig cfg;RaceObstaclePreviewPlan p;p.metric=p.fresh=p.motionValid=p.candidate=p.hasTarget=true;p.stop=false;p.direction=1;
    p.centers={{200,160},{100,180}};p.obstacles={{60,100,30,60}};
    for(int i=0;i<=26;++i)p.groundCenters.push_back({std::min(.05f,i*.025f*.25f),i*.025f});
    auto right=raceGroundSteering(p,cfg,.1f,1500,.07f);
    check(right.valid&&right.requested>1650&&right.requested<=PWMSERVOMAX&&right.pwm<=1542,"metric right lookahead missing or slew exceeded");
    auto mirrored=p;for(auto &v:mirrored.groundCenters)v[0]=-v[0];
    auto left=raceGroundSteering(mirrored,cfg,.1f,1500,.07f);
    check(left.valid&&left.requested<1350&&left.requested>=PWMSERVOMIN,"left lookahead did not mirror steering");
    auto stale=p;stale.fresh=false;check(!raceGroundSteering(stale,cfg,.1f,1500,.07f).valid,"stale path commanded steering");
    RaceBypassControl controller;RaceConfig race;RaceSnapshot ai;ai.image=cv::Mat(240,320,CV_8UC3,cv::Scalar(255,255,255));
    ai.detections={{LABEL_CONE,"cone",.99f,60,100,30,60}};RaceBypassDecision decision;
    for(int i=1;i<=3;++i){ai.sequence=p.sequence=i;ai.captured=p.captured=i*.07;decision=controller.update(p,ai,race,ai.captured,true);}
    check(decision.follow,"failed to prime committed bypass");
    auto unavailable=p;unavailable.candidate=false;unavailable.stop=true;unavailable.centers.clear();unavailable.state="BYPASS_NO_CLEAR_PATH";
    ai.sequence=unavailable.sequence=4;ai.captured=unavailable.captured=.28;
    decision=controller.update(unavailable,ai,race,.28,true);
    check(decision.stop&&!decision.follow&&std::string(decision.state)=="BYPASS_REPLAN_WAIT","geometry gap did not stop for replan");
    for(int i=5;i<=7;++i) {
        ai.sequence=p.sequence=i;ai.captured=p.captured=i*.07;decision=controller.update(p,ai,race,ai.captured,true);
        if(i<7)check(decision.stop&&!decision.follow,"replanned route resumed without three fresh observations");
    }
    check(decision.follow&&!decision.stop,"valid replanned route did not resume");
    ai.failed=true;decision=controller.update(p,ai,race,ai.captured,true);
    check(decision.stop&&!decision.follow,"AI failure did not stop immediately");
    decision=controller.update(p,ai,race,ai.captured+.5,true);
    check(decision.stop&&std::string(decision.state)=="BYPASS_FAULT","persistent AI failure did not latch stop");
    if(!evidence.empty()) {
        cfg.load((evidence/"navigation-config.json").string(),true);
        check(std::abs(cfg.bypassWheelbase-.20f)<1e-5,"old navigation config did not retain measured wheelbase default");
        std::ifstream input(evidence/"navigation.jsonl");std::string line;nlohmann::json last;
        while(std::getline(input,line)){auto row=nlohmann::json::parse(line);if(row["state"]=="BYPASS_NO_CLEAR_PATH")break;if(row["candidate"].get<bool>())last=std::move(row);}
        check(!last.is_null(),"no pre-stop field route found");
        RaceObstaclePreviewPlan actual=p;actual.groundCenters=last.at("ground_center").get<std::vector<std::array<float,2>>>();
        auto command=raceGroundSteering(actual,cfg,.1f,1500,.07f);
        check(command.valid&&command.requested>1650,"field pre-stop route still understeered");
        std::cout<<"FIELD route="<<last.at("frame")<<" target="<<command.targetX<<","<<command.targetZ
            <<" requested_pwm="<<command.requested<<" first_slew_pwm="<<command.pwm<<" wheelbase="<<cfg.bypassWheelbase<<'\n';
    }
    std::cout<<"PASS metric lookahead left/right, original servo range, steering slew, stopped replanning, fresh reconfirmation and AI-failure latch\n";
}
static void recoveryOnly() {
    Scene scene;RaceGroundMotion motion(scene.cfg);
    motion.update(scene.frame());auto odo=motion.update(scene.frame(.005f));
    check(odo.valid,"cannot prime textured odometry");const int epoch=odo.epoch;
    auto dropout=scene.frame(.01f);dropout.image.setTo(cv::Scalar(230,230,230));
    odo=motion.update(dropout);check(!odo.valid&&odo.epoch==epoch,"single dropout destroyed coordinate frame");
    odo=motion.update(scene.frame(.015f));
    check(odo.valid&&odo.epoch==epoch&&odo.pose.z>.011f&&odo.pose.z<.019f,
          "recovered odometry lost travel across failed frame");
    std::cout<<"bridged_z="<<odo.pose.z<<" features="<<odo.features<<" matches="<<odo.matches<<" reference_age="<<odo.referenceAge<<'\n';
    motion.setMetricScale(1.1f);odo=motion.update(scene.frame(.015f));
    check(!odo.valid&&odo.epoch>epoch&&std::abs(odo.pose.z)<1e-6f,"scale change retained old metric pose");
    RaceBypassControl control;RaceConfig cfg;RaceObstaclePreviewPlan p;
    p.metric=p.fresh=p.motionValid=p.candidate=p.hasTarget=true;p.stop=false;p.direction=1;
    p.centers={{200,160},{100,180}};p.obstacles={{60,100,30,60}};
    RaceSnapshot ai;ai.image=cv::Mat(240,320,CV_8UC3,cv::Scalar(255,255,255));
    ai.detections={{LABEL_CONE,"cone",.99f,60,100,30,60}};RaceBypassDecision d;int frame=0;
    auto step=[&](){ai.sequence=p.sequence=++frame;ai.captured=p.captured=frame*.07;return control.update(p,ai,cfg,ai.captured,true);};
    for(int i=0;i<3;++i)d=step();
    check(d.follow,"cannot prime recovery control");
    p.motionValid=false;d=step();check(d.stop&&!d.follow,"invalid odometry still drove");
    p.motionValid=true;
    for(int i=0;i<3;++i){d=step();check(i==2?d.follow:d.stop,"transient recovery skipped fresh reconfirmation");}
    ++p.motionEpoch;d=step();check(d.stop&&control.engaged(),"reset released bypass ownership");
    ai.detections.clear();p.hasTarget=false;p.passed=true;
    d=step();check(d.involved&&d.stop&&!d.follow&&std::string(d.reason)=="REACQUIRE_VISIBLE_CONE","reset with missing cone resumed lane control");
    control.update(p,ai,cfg,ai.captured,false);d=step();check(d.stop&&d.involved,"manual/auto bypassed reacquisition");
    ai.detections={{LABEL_CONE,"cone",.99f,60,100,30,60}};p.hasTarget=true;p.passed=false;
    for(int i=0;i<3;++i){d=step();check(i==2?d.follow:d.stop,"visible target did not require three fresh routes");}
    // 重定位过程中重复改变epoch也应计数，不能无限重试。
    for(int i=0;i<4;++i){++p.motionEpoch;d=step();}
    check(d.stop&&std::string(d.state)=="BYPASS_FAULT","repeated resets escaped fault counter");
    Scene wide;RaceGroundObstacle planner(wide.cfg);RaceObstaclePreviewPlan route;
    for(int i=0;i<8;++i){auto s=wide.frame(0,0,-.13f,true,true,.90f);route=plan(planner,s,wide.race);}
    check(route.candidate&&route.laneScale>1.10f&&route.laneScale<1.15f,"DS field-scale correction above 1.10 regressed");
    std::cout<<"PASS last-good-keyframe bridge, scale reset, zero-speed transient/reacquisition, persistent ownership, reset budget, DS scale correction\n";
}
static void continuityOnly() {
    const auto cfg=calibration();RaceGroundMap map(cfg);RaceNavigationConfig nav;
    // 已量车宽/车长/轴距的闭环几何检查。PWM映射为假设值，不代表实车轮角标定。
    for(int side:{1,-1}) {
        RaceGroundPose pose;uint16_t pwm=1500;float maximum=0;
        for(int frame=0;frame<100;++frame) {
            const auto cone=pose.local({-.13f*side,.45f});
            const float c=std::cos(pose.angle),s=std::sin(pose.angle);
            RaceObstaclePreviewPlan p;p.metric=p.fresh=p.motionValid=p.candidate=true;p.stop=false;
            // 世界道路 x=0，在当前车头坐标系中拟合成同样的直线。
            const bool ok=RaceGroundObstacle::solveRoute(cfg,map,-pose.x/c,s/c,.45f/c,
                {{cone.x,cone.y,.045f}},side,p);
            if(!ok)std::cout<<"closed loop stop side="<<side<<" frame="<<frame<<" pose="<<pose.x<<","<<pose.z<<","<<pose.angle
                <<" reason="<<p.reason<<" blocked="<<p.routeBlockedZ<<'\n';
            check(ok,"closed-loop route lost before passage/centering");
            auto steering=raceGroundSteering(p,nav,.15f,pwm,.07f);check(steering.valid,"closed loop steering invalid");pwm=steering.pwm;
            const float delta=(float(pwm)-1500)/400*float(CV_PI)/4;
            const float k=std::tan(delta)/nav.bypassWheelbase,ds=.0105f,frontOffset=.25f;
            // 后轴自行车模型，转换回车头参考点；不使用指令冒充实车里程。
            const float theta=-pose.angle;
            pose.x+=ds*(std::sin(theta)+frontOffset*k*std::cos(theta));
            pose.z+=ds*(std::cos(theta)-frontOffset*k*std::sin(theta));pose.angle-=ds*k;
            maximum=std::max(maximum,side*pose.x);
        }
        check(pose.z>.95f&&maximum>.045f&&std::abs(pose.x)<.04f,"closed loop did not pass and return");
        std::cout<<"closed_loop side="<<side<<" forward="<<pose.z<<" max_offset="<<maximum<<" final_offset="<<pose.x<<'\n';
    }
    RaceObstaclePreviewPlan blocked;
    check(!RaceGroundObstacle::solveRoute(cfg,map,0,0,.45f,{{0,.2f,.06f}},1,blocked)&&blocked.routeBlockedZ>=0,
          "blocked lane lost collision rejection/diagnostics");
    // 两轮现场故障链：有里程时交替无路与漏检，停止重算超过0.4s仍可重确认恢复。
    RaceBypassControl control;RaceConfig race;RaceSnapshot ai;
    ai.image=cv::Mat(240,320,CV_8UC3,cv::Scalar(255,255,255));ai.detections={{LABEL_CONE,"cone",.99f,60,100,30,60}};
    RaceObstaclePreviewPlan p;p.metric=p.fresh=p.motionValid=p.candidate=p.hasTarget=true;p.stop=false;
    p.centers={{200,160},{100,180}};p.obstacles={{60,100,30,60}};
    auto step=[&](){++ai.sequence;p.sequence=ai.sequence;p.captured=ai.captured=ai.sequence*.07;return control.update(p,ai,race,ai.captured,true);};
    RaceBypassDecision d;for(int i=0;i<3;++i)d=step();check(d.follow,"failed to prime continuity control");
    p.candidate=false;p.stop=true;
    for(int i=0;i<400;++i){p.state=i%2?"BYPASS_NO_CLEAR_PATH":"BYPASS_TARGET_LOST_BEFORE_CLEARANCE";d=step();
        check(d.stop&&!d.follow&&std::string(d.state)!="BYPASS_FAULT","geometry/dropout mix latched fault while motion valid");}
    p.candidate=true;p.stop=false;
    for(int i=0;i<3;++i){d=step();check(i==2?d.follow:d.stop,"replan did not require three fresh routes");}
    d=step();check(d.follow,"paused time counted as execution timeout");
    p.motionValid=false;for(int i=0;i<9;++i)d=step();
    check(d.stop&&std::string(d.state)=="BYPASS_FAULT","persistent visual failure escaped latch");
    // 可预测的短暂漏检仍须重新规划，超时未重识别则停车。
    Scene scene;RaceGroundObstacle planner(cfg);RaceObstaclePreviewPlan route;
    for(int i=0;i<8;++i)route=plan(planner,scene.frame(),scene.race);
    for(int i=0;i<3;++i){auto snap=scene.frame();snap.detections.clear();route=plan(planner,snap,scene.race);
        if(!route.candidate)std::cout<<"dropout state="<<route.state<<" reason="<<route.reason<<'\n';
        check(route.candidate&&route.predictedTargets==1,"bounded mapped-cone dropout broke route");}
    for(int i=0;i<4;++i){auto snap=scene.frame();snap.detections.clear();route=plan(planner,snap,scene.race);}
    check(route.stop&&!route.candidate,"expired cone prediction kept driving");
    std::cout<<"PASS early offset, mirrored closed-loop passage/return, footprint rejection, mixed replan/dropout recovery, bounded cone prediction, visual failure latch\n";
}
static void coneGuardOnly(const std::filesystem::path &root) {
    RaceConfig cfg;cv::Mat bin=cv::Mat::zeros(240,320,CV_8UC1);cv::rectangle(bin,{40,0,240,240},{255},-1);
    Track track;track.handle(bin);
    for(const auto &name:{"1793179683297-10089","1793179761808-13674"}) {
        RaceManager manager(cfg);std::ifstream input(root/name/"detections.jsonl");std::string line;
        uint64_t first=0;int held=0;
        while(std::getline(input,line)) {
            const auto j=nlohmann::json::parse(line);RaceSnapshot s;s.sequence=j.at("frame");s.captured=j.at("captured");
            for(const auto &d:j.at("detections"))s.detections.push_back({d.at("type"),"",d.at("score"),d.at("x"),d.at("y"),d.at("width"),d.at("height")});
            const auto p=manager.update(s,s.captured,.067f,true,bin,track);
            if(p.stop&&std::string(p.reason)=="CONE_ROUTE_OR_RESTART_REQUIRED"&&!first)first=s.sequence;
            if(first&&s.sequence>=first&&s.sequence<=330){check(p.stop,"field detector dropout released near-cone wait");++held;}
        }
        check(first&&held>20,"field cone did not trigger and retain wait");
        if(std::string(name)=="1793179683297-10089")check(first<=288,"near cone still waited for three reappearances");
        std::cout<<"field cone_guard run="<<name<<" first_stop_ai="<<first<<" retained_observations="<<held<<'\n';
    }
    RaceManager manager(cfg);RaceSnapshot s;RacePlan p;int sequence=0;
    auto step=[&](std::vector<PredictResult> ds,bool automatic=true,bool authorized=false) {
        s={uint64_t(++sequence),sequence*.07,std::move(ds),false,{},{}};
        return manager.update(s,s.captured,.07f,automatic,bin,track,{},authorized);
    };
    const PredictResult cone{LABEL_CONE,"cone",.95f,100,110,40,40};
    p=step({cone});check(p.stop,"single accepted near cone did not stop immediately");
    for(int i=0;i<30;++i)check(step({}).stop,"empty detections timed out cone safety obligation");
    step({},false);check(step({}).stop,"manual/auto silently cleared cone obligation");
    check(!step({cone},true,true).stop,"verified bypass route could not discharge cone stop");
    for(int i=0;i<4;++i)p=step({{LABEL_PERSON,"person",.95f,180,120,35,50}},true,true);
    check(p.stop,"cone authorization overrode person stop");
    for(int i=0;i<15;++i)p=step({},true,true);
    check(!p.stop,"person-only ordinary clear-scene release was broken");
    RaceObstaclePreviewPlan route;route.sequence=1;route.captured=1;route.metric=route.fresh=route.motionValid=route.candidate=true;
    route.stop=false;route.centers={{200,160}};route.groundCenters={{0,0},{.05f,.3f}};
    check(!raceBypassArmReady(route,1,cfg),"bypass armed without an observed cone");
    route.hasTarget=true;route.confirmedTargets=1;
    check(raceBypassArmReady(route,1,cfg),"confirmed fresh metric route cannot arm bypass");
    check(!raceBypassArmReady(route,1.3,cfg),"stale route armed bypass");
    route.passed=true;check(!raceBypassArmReady(route,1,cfg),"old passed cone armed a new bypass");
    std::cout<<"PASS actual missed-cone detection replay, immediate stop, persistent wait, authorized bypass handoff and person priority\n";
}
static void referenceBankOnly() {
    Scene scene;RaceGroundMotion motion(scene.cfg);
    auto full=scene.frame(0,0,-.13f,true,false);
    full.image.setTo(cv::Scalar(210,210,210));
    for(int y=65;y<=185;y+=20)for(int x=25;x<=295;x+=30)
        cv::rectangle(full.image,{x,y,6,6},{15,15,15},-1);
    motion.update(full);
    auto s=full;
    auto feed=[&](const cv::Mat &image) {++s.sequence;s.captured+=.07;s.image=image;return motion.update(s,true);};
    auto result=feed(full.image);check(result.valid,"cannot initialize textured reference bank");const int epoch=result.epoch;
    // 已知平面被交替遮挡：最新参考可见的区域随后消失，较早的已定位图仍有共同地面。
    auto upper=full.image.clone();cv::rectangle(upper,{0,100,320,140},{230,230,230},-1);
    for(int i=0;i<3;++i)result=feed(upper);
    if(!result.valid)std::cout<<"reduced reference reason="<<result.reason<<" features="<<result.features
        <<" matches="<<result.matches<<" inliers="<<result.inliers<<'\n';
    check(result.valid,"could not establish reduced-texture latest reference");
    auto lower=full.image.clone();cv::rectangle(lower,{0,0,320,100},{230,230,230},-1);
    result=feed(lower);
    std::cout<<"backup valid="<<result.valid<<" used="<<result.backupReferenceUsed<<" attempts="<<result.backupAttempts
        <<" inliers="<<result.inliers<<" x="<<result.pose.x<<" z="<<result.pose.z<<" reason="<<result.reason<<'\n';
    check(result.valid&&result.backupReferenceUsed&&result.epoch==epoch&&std::hypot(result.pose.x,result.pose.z)<.003f,
          "earlier verified ground reference failed or introduced coordinate jump");
    auto blank=cv::Mat(240,320,CV_8UC3,cv::Scalar(230,230,230));result=feed(blank);
    check(!result.valid&&result.coordinatesHeld&&result.epoch==epoch,"blank scene invented motion from backup bank");
    motion.setMetricScale(1.1f);result=feed(full.image);
    check(!result.valid&&!result.backupReferenceUsed&&result.epoch>epoch,"scale reset retained mixed-coordinate references");
    std::cout<<"PASS bounded verified reference fallback, unchanged epoch, blank rejection and scale reset\n";
}
static void turnFieldOnly(const std::filesystem::path &run) {
    RaceGroundCalibration ground;ground.load((run/"ground-calibration.json").string(),false);
    auto read=[](const std::filesystem::path &path) {
        std::ifstream input(path);check(bool(input),"missing current turn edges");std::string line;std::getline(input,line);
        std::pair<std::vector<PointX>,std::vector<PointX>> edges;
        while(std::getline(input,line)){std::replace(line.begin(),line.end(),',',' ');std::istringstream row(line);std::string side;int y,x;
            if(row>>side>>y>>x)(side=="left"?edges.first:edges.second).emplace_back(y,x);}
        return edges;
    };
    for(bool mirror:{false,true}) {
        AutoPilot original,seeded;const int count=seeded.seedCalibratedWidths(raceGroundLaneWidthProfile(ground));
        check(count>=30,"current calibration has insufficient usable width rows");
        auto edges=[&](int frame) {
            auto e=read(run/"event-1"/(std::to_string(frame)+"-edges.csv"));
            if(mirror){auto l=e.first;e.first.clear();for(auto p:e.second)e.first.emplace_back(p.x,319-p.y);
                e.second.clear();for(auto p:l)e.second.emplace_back(p.x,319-p.y);}
            return e;
        };
        auto start=edges(184);
        for(int i=0;i<15;++i){original.update(start.first,start.second,.067f);seeded.update(start.first,start.second,.067f);}
        int oldInvalid=0,newInvalid=0;
        for(int frame=184;frame<=198;++frame) {
            const auto e=edges(frame);original.update(e.first,e.second,.067f,.2f);seeded.update(e.first,e.second,.067f,.2f);
            oldInvalid+=!original.valid;newInvalid+=!seeded.valid;
            std::cout<<"turn frame="<<frame<<" mirror="<<mirror<<" old="<<original.pathIssue<<" old_center="<<original.center
                <<" new="<<seeded.pathIssue<<" new_center="<<seeded.center<<" single="<<seeded.singlePoints<<" pwm="<<seeded.servo<<'\n';
        }
        check(oldInvalid>0&&newInvalid==0&&seeded.ready&&seeded.singlePoints>0,"current turn still loses path after calibrated cold start");
        check(mirror?seeded.servo>1500:seeded.servo<1500,"current turn steering did not mirror");
        std::vector<PointX> l,r;for(int y=219;y>=35;y-=2){l.emplace_back(y,0);r.emplace_back(y,319);}
        seeded.update(l,r,.067f);check(!seeded.valid,"calibrated width reference invented a path on fully clipped white image");
    }
    std::cout<<"PASS current turn edge sequence, mirrored steering, calibrated width cold start and full-clipping rejection; not physical turn validation\n";
}
static void motionPairsOnly(const std::filesystem::path &root) {
    for(const auto &name:{"1793177164704-15969","1793177215454-18402"}) {
        const auto run=root/name;
        const auto first=run/"bypass-stop-evidence"/(std::string(name)=="1793177164704-15969"?"ai-sample-5":"ai-sample-1");
        const auto j=nlohmann::json::parse(std::ifstream(first/"navigation.json"));
        RaceGroundCalibration cfg;cfg.load((run/"ground-calibration.json").string(),false);
        auto a=cv::imread((first/"camera.png").string()),b=cv::imread((run/"ai-sample-1/camera.png").string());
        check(!a.empty()&&!b.empty(),"missing matched stop/reference samples");
        {
            RaceGroundMotion motion(cfg);motion.setMetricScale(j.at("lane_scale"));
            RaceSnapshot s{1,.07,{},false,{},a};motion.update(s);
            ++s.sequence;s.captured=.14;motion.update(s,true);
            ++s.sequence;s.captured=.49;s.image=b;const auto result=motion.update(s,true);
            std::cout<<"field_pair run="<<name<<" features="<<result.features<<" matches="<<result.matches
                <<" inliers="<<result.inliers<<" scale="<<result.fitScale<<" residual="<<result.residual<<" reason="<<result.reason
                <<" valid="<<result.valid<<" x="<<result.pose.x<<" z="<<result.pose.z<<" angle="<<result.pose.angle<<'\n';
        }
    }
    std::cout<<"DONE saved-image pair geometry probe; pair time is diagnostic normalization, not field video timing\n";
}
static void nearConeOnly(const std::filesystem::path &field={}) {
    Scene scene;RaceGroundObstacle planner(scene.cfg);RaceBypassControl control;
    RaceObstaclePreviewPlan route;RaceBypassDecision command;
    auto step=[&](float moved,bool visible,bool blackout) {
        auto snap=scene.frame(moved,.06f);if(!visible)snap.detections.clear();
        if(blackout)snap.image.setTo(cv::Scalar(230,230,230));
        route=plan(planner,snap,scene.race);command=control.update(route,snap,scene.race,snap.captured,true);
        return snap;
    };
    for(int i=0;i<8;++i)step(.18f,true,false);
    check(command.follow&&route.trackedTargets==1,"cannot prime near-cone map/control");
    const int epoch=route.motionEpoch;
    for(int i=0;i<4;++i) {
        step(.185f+.004f*i,false,true);
        check(command.stop&&!command.follow&&!route.motionValid&&route.motionCoordinatesHeld&&route.motionEpoch==epoch&&route.trackedTargets==1,
              "three-frame dropout erased committed cone map or drove without motion");
    }
    for(int i=0;i<3;++i) {
        step(.21f,false,false);
        if(!route.candidate)std::cout<<"near recovery state="<<route.state<<" reason="<<route.reason<<" motion="<<route.motionReason<<'\n';
        check(route.motionValid&&route.motionEpoch==epoch&&route.trackedTargets==1&&route.candidate,
              "lost cone detection prevented verified ground-reference recovery");
        check(i==2?command.follow:command.stop,"ground-reference recovery skipped three fresh route confirmations");
    }
    std::cout<<"near_cone held_epoch="<<epoch<<" restored_z="<<route.motionZ<<" tracked="<<route.trackedTargets<<" resumed_without_cone_detection="<<command.follow<<'\n';
    // 不能接回地面参考时必须停车并最终锁停，绝不能自动遗忘障碍后转回巡线。
    for(int i=0;i<15;++i)step(.21f,false,true);
    check(command.stop&&!command.follow&&std::string(command.state)=="BYPASS_FAULT"&&route.motionEpoch==epoch&&route.trackedTargets==1,
          "persistent visual outage lost map/stop ownership");
    // 只有车身固定纹理时，不得声明地面零运动有效。
    Scene body;RaceGroundMotion bodyMotion(body.cfg);auto sample=body.frame(0,0,-.13f,false,false);
    sample.image.setTo(cv::Scalar(230,230,230));
    for(int y=198;y<238;y+=8)for(int x=55;x<280;x+=8)if((x/8+y/8)%2)cv::rectangle(sample.image,{x,y,5,5},{15,15,15},-1);
    bodyMotion.update(sample);++sample.sequence;sample.captured+=.07;
    const auto noGround=bodyMotion.update(sample);
    check(!noGround.valid&&noGround.features==0,"vehicle-fixed body texture was used as ground evidence");
    // 锁定后小幅路宽波动：按更窄观测规划，不能把47cm观测当作真实道路扩宽。
    for(float widthScale:{.95f,1.05f,.85f}) {
        Scene road;RaceGroundObstacle measured(road.cfg);RaceObstaclePreviewPlan p;
        for(int i=0;i<8;++i)p=plan(measured,road.frame(),road.race);
        const float locked=p.laneScale;
        for(int i=0;i<4;++i)p=plan(measured,road.frame(0,0,-.13f,true,true,widthScale),road.race);
        std::cout<<"lane scale="<<widthScale<<" measured="<<p.laneWidthMeasured<<" used="<<p.laneWidthUsed
            <<" rows="<<p.laneRows<<" rejected="<<p.laneRejectedRows<<" state="<<p.state<<'\n';
        check(std::abs(p.laneScale-locked)<1e-6f,"automatic lane-width jitter changed metric scale");
        if(widthScale<.9f)check(p.stop&&!p.candidate,"genuinely narrow lane was accepted");
        else check(p.candidate&&p.laneWidthUsed<=road.cfg.roadWidth&&
                   (widthScale>=1.f||p.laneWidthUsed<.435f),"small lane variation failed or invented lateral clearance");
    }
    // 故障日志必须把候选、拟合尺度、空间跨度和早退原因区分开。
    std::vector<cv::Point2f> a,b;
    for(int i=0;i<5;++i)for(int j=0;j<5;++j){a.emplace_back(-.2f+.1f*i,.2f+.1f*j);b.push_back(a.back()+cv::Point2f(0,-.055f));}
    RaceGroundPose pose;int inliers=0;float residual=0;RaceGroundFitDiagnostics info;
    check(!RaceGroundMotion::fit(a,b,pose,inliers,residual,&info)&&std::string(info.reason)=="GROUND_STEP_TOO_LARGE",
          "regular-frame excessive displacement was not rejected/reported");
    check(RaceGroundMotion::fit(a,b,pose,inliers,residual,&info,.35f)&&std::abs(pose.z-.055f)<.001f,
          "bridged true-image displacement kept single-frame step limit");
    auto scaled=b;for(auto &p:scaled)p*=1.2f;pose={};
    check(!RaceGroundMotion::fit(a,scaled,pose,inliers,residual,&info,.35f),"scale change became a valid ground bridge");
    if(!field.empty()) {
        RaceGroundCalibration actual;actual.load((field/"ground-calibration.json").string(),false);
        RaceGroundMotion stationary(actual);auto image=cv::imread((field/"ai-sample-1/camera.png").string());
        check(!image.empty(),"missing saved final-run image");RaceSnapshot s{1,.07,{},false,{},image};
        stationary.update(s);++s.sequence;s.captured+=.07;const auto result=stationary.update(s);
        std::cout<<"saved_stationary frame=631 features="<<result.features<<" matches="<<result.matches<<" inliers="<<result.inliers
            <<" reason="<<result.reason<<" valid="<<result.valid<<'\n';
        check(result.valid&&std::abs(result.pose.z)<.002f,"saved stationary view could not retain ground evidence after body mask");
    }
    check(raceObstacleJson(route,{}).contains("motion_fit_scale")&&raceObstacleJson(route,{}).contains("lane_width_observed_m"),
          "new failure diagnostics absent from navigation JSON");
    std::cout<<"PASS committed near-cone four-frame outage/verified recovery without redetection, bounded stop, body exclusion, conservative width handling and fit diagnostics\n";
}
static void fieldGeometryOnly(const std::filesystem::path &run) {
    RaceGroundCalibration cfg;cfg.load((run/"ground-calibration.json").string(),false);RaceGroundMap map(cfg);
    std::ifstream navigation(run/"navigation.jsonl"),detections(run/"detections.jsonl");std::string line;
    nlohmann::json first,stopped;
    while(std::getline(navigation,line)) {
        auto j=nlohmann::json::parse(line);const auto frame=j.at("frame").get<int>();
        if(frame==414)first=j;
        if(frame==453)stopped=j;
    }
    check(!first.is_null()&&!stopped.is_null(),"missing second field run geometry");
    for(const auto &record:{first,stopped}) {
        detections.clear();detections.seekg(0);nlohmann::json found;
        while(std::getline(detections,line)){auto j=nlohmann::json::parse(line);if(j.at("frame")==record.at("frame")){found=j;break;}}
        check(!found.is_null(),"missing matching field detections");map.setMetricScale(record.at("lane_scale"));
        std::vector<std::array<float,3>> cones;
        for(const auto &d:found.at("detections"))if(d.at("type").get<int>()==LABEL_CONE&&d.at("accepted").get<bool>()) {
            const float x=d.at("x"),y=d.at("y"),w=d.at("width"),h=d.at("height");
            const auto l=map.toGround({x,y+h}),r=map.toGround({x+w,y+h}),c=(l+r)*.5f;
            cones.push_back({c.x,c.y,std::max(.025f,(r.x-l.x)*.5f)+.01f});
        }
        RaceObstaclePreviewPlan p;p.metric=p.fresh=p.motionValid=p.candidate=true;p.stop=false;
        const bool ok=RaceGroundObstacle::solveRoute(cfg,map,record.at("lane_offset_m"),record.at("lane_heading"),
            record.at("lane_width_measured_m"),cones,1,p);
        if(record.at("frame")==414) {
            check(ok,"field starting geometry has no route");
            const auto old=record.at("ground_center").get<std::vector<std::array<float,2>>>();
            check(p.groundCenters.at(8)[0]>old.at(8)[0]+.006f,"field approach did not establish earlier offset");
            std::cout<<"field frame=414 old_20cm_x="<<old.at(8)[0]<<" new_20cm_x="<<p.groundCenters.at(8)[0]<<'\n';
        }
        std::cout<<"field frame="<<record.at("frame")<<" candidate="<<ok<<" reason="<<p.reason<<" blocked_z="<<p.routeBlockedZ
            <<" right_gap="<<p.rightGap<<" cone_z="<<p.coneZ<<'\n';
    }
    std::cout<<"PASS matched field geometry diagnostics and earlier offset; not a moving-image replay\n";
}
// 现场 2026-10-28 11:36 那轮的时序：导航年龄 0.14s 时 0.8s 恢复窗口刚起步，
// 下一帧年龄 0.21s（导航成果晚到，fresh 变为 false），旧实现把同一轮恢复从 0.8s
// 缩回 0.4s，于是约 0.42s 后就锁停。这里把它固化成回归：
// 1) 计划晚到不能让任何一路恢复提前结束；2) 各原因窗口独立；3) 同一轮总时长有上限。
static void timingOnly() {
    RaceConfig cfg;
    const double dt=.07;
    auto make=[&](RaceObstaclePreviewPlan &p,RaceSnapshot &ai) {
        p={};p.metric=p.fresh=p.motionValid=p.candidate=p.hasTarget=true;p.stop=false;p.direction=1;
        p.centers={{200,160},{100,180}};p.obstacles={{60,100,30,60}};p.motionEpoch=1;
        p.groundCenters={{0.10f,0.20f},{0.10f,0.45f}};
        ai.sequence=0;ai.captured=0;ai.image=cv::Mat(240,320,CV_8UC3,cv::Scalar(255,255,255));
        ai.detections={{LABEL_CONE,"cone",.99f,60,100,30,60}};ai.failed=false;
    };
    auto run=[&](RaceBypassControl &ctl,RaceObstaclePreviewPlan &p,RaceSnapshot &ai,double &now,
                 double planAge,bool motionValid,bool held,long epoch) {
        now+=dt;p.sequence=++ai.sequence;ai.captured=now;
        p.captured=now-planAge;p.fresh=true;p.motionValid=motionValid;
        p.motionCoordinatesHeld=held;p.motionEpoch=epoch;
        return ctl.update(p,ai,cfg,now,true);
    };
    {   // A. 计划晚到（age 0.21 > 0.18）：0.56s 内只能停车等待，不能锁停
        RaceBypassControl ctl;RaceObstaclePreviewPlan p;RaceSnapshot ai;make(p,ai);double now=0;
        RaceBypassDecision d;
        for(int i=0;i<3;++i)d=run(ctl,p,ai,now,.05,true,false,1);
        check(d.follow,"timing: cannot prime bypass");
        for(int i=0;i<8;++i) {
            d=run(ctl,p,ai,now,.21,true,false,1);
            check(d.stop&&!d.follow,"timing: stale plan still commanded motion");
            check(std::string(d.state)!="BYPASS_FAULT","timing: stale plan latched before its own window");
        }
        for(int i=0;i<3;++i)d=run(ctl,p,ai,now,.05,true,false,1);
        check(d.follow,"timing: did not resume after plan became fresh again");
    }
    {   // B. 保留参考图验证：即使 fresh 变 false，窗口仍是 0.8s
        RaceBypassControl ctl;RaceObstaclePreviewPlan p;RaceSnapshot ai;make(p,ai);double now=0;
        RaceBypassDecision d;
        for(int i=0;i<3;++i)d=run(ctl,p,ai,now,.05,true,false,1);
        check(d.follow,"timing: cannot prime bypass (B)");
        bool latched=false;double latchedAt=0;
        const double tHeld=now;
        for(int i=0;i<13;++i) {
            d=run(ctl,p,ai,now,.21,false,true,1);
            check(d.stop&&!d.follow,"timing: held-reference path commanded motion");
            if(std::string(d.state)=="BYPASS_FAULT"){latched=true;latchedAt=now-tHeld;break;}
        }
        check(latched,"timing: held-reference window never expired");
        check(latchedAt>.72,"timing: held-reference window shortened by late plan");
    }
    {   // C. 真正的里程失效（无保留坐标、计划新鲜）：约 0.5s 后锁停
        RaceBypassControl ctl;RaceObstaclePreviewPlan p;RaceSnapshot ai;make(p,ai);double now=0;
        RaceBypassDecision d;
        for(int i=0;i<3;++i)d=run(ctl,p,ai,now,.05,true,false,1);
        check(d.follow,"timing: cannot prime bypass (C)");
        bool latched=false;double latchedAt=0;
        const double tMotion=now;
        for(int i=0;i<12;++i) {
            d=run(ctl,p,ai,now,.05,false,false,1);
            if(std::string(d.state)=="BYPASS_FAULT"){latched=true;latchedAt=now-tMotion;break;}
            check(d.stop&&!d.follow,"timing: lost motion path commanded motion");
        }
        check(latched&&latchedAt>=.45&&latchedAt<=.65,"timing: motion-loss window not ~0.5s");
    }
    {   // D. 原因反复切换不能让停车无限延长：总轮时长上限 1.2s
        RaceBypassControl ctl;RaceObstaclePreviewPlan p;RaceSnapshot ai;make(p,ai);double now=0;
        RaceBypassDecision d;
        for(int i=0;i<3;++i)d=run(ctl,p,ai,now,.05,true,false,1);
        check(d.follow,"timing: cannot prime bypass (D)");
        bool latched=false;
        for(int i=0;i<24;++i) {
            const bool flip=(i%2)==0;
            d=run(ctl,p,ai,now,flip?.21:.05,flip,flip?false:true,1);
            if(std::string(d.state)=="BYPASS_FAULT"){latched=true;break;}
        }
        check(latched,"timing: alternating causes kept the car stopped forever");
    }
    std::cout<<"PASS recovery timing decoupled from plan freshness, per-cause windows, bounded episode\n";
}
// 现场 1793190973677 的第二层故障：承诺绕行后里程失效，参考帧被冻结（早退不更新
// captured/previous），gap 涨到超过 horizon 后每帧只能报 REFERENCE_TOO_OLD，
// 540 帧无法恢复、只能锁停。要求连续无效 3 帧后重锚，画面正常时能重新测出位移。
static void reanchorOnly() {
    {
        Scene strictScene;RaceGroundMotion strict(calibration());strict.update(strictScene.frame());
        auto held=strict.update(strictScene.frame(.005f),true,false);
        check(held.valid,"parking strict odometry cannot initialize");const auto origin=held.pose;
        for(int i=0;i<12;++i) {
            auto blank=strictScene.frame(.010f);blank.image.setTo(cv::Scalar(120,120,120));
            held=strict.update(blank,true,false);
            check(!held.valid&&held.coordinatesHeld&&held.reason!="REFERENCE_REANCHORED"&&
                std::hypot(held.pose.x-origin.x,held.pose.z-origin.z)<1e-6,
                "strict parking odometry silently reanchored or advanced through missing motion");
        }
    }
    Scene scene;RaceGroundMotion motion(calibration());
    motion.update(scene.frame());
    auto odo=motion.update(scene.frame(.005f),true);
    check(odo.valid,"reanchor: cannot prime existing odometry");
    for(int i=0;i<4;++i) {
        auto snap=scene.frame(.010f);
        snap.image.setTo(cv::Scalar(120,120,120));      // 无特征帧：拟合必然失败
        odo=motion.update(snap,true);
        check(!odo.valid,"reanchor: blank frame reported motion");
    }
    for(int i=0;i<3;++i) odo=motion.update(scene.frame(.020f+.005f*i),true);
    check(odo.valid,"reanchor: motion never recovered after frozen reference (REFERENCE_TOO_OLD deadlock)");
    std::cout<<"reanchor recovered z="<<odo.pose.z<<" reason="<<odo.reason<<'\n';
    std::cout<<"PASS frozen-reference reanchor, held coordinates preserved\n";
}
int main(int argc,char **argv) {
    try {
        if(argc>=2&&std::string(argv[1])=="--timing-only") {timingOnly();return 0;}
        if(argc>=2&&std::string(argv[1])=="--reanchor-only") {reanchorOnly();return 0;}
        if(argc>=2&&std::string(argv[1])=="--recovery-only") {recoveryOnly();return 0;}
        if(argc>=2&&std::string(argv[1])=="--continuity-only") {continuityOnly();return 0;}
        if(argc>=3&&std::string(argv[1])=="--cone-guard-only") {coneGuardOnly(argv[2]);return 0;}
        if(argc>=2&&std::string(argv[1])=="--reference-bank-only") {referenceBankOnly();return 0;}
        if(argc>=3&&std::string(argv[1])=="--turn-field-only") {turnFieldOnly(argv[2]);return 0;}
        if(argc>=3&&std::string(argv[1])=="--motion-pairs-only") {motionPairsOnly(argv[2]);return 0;}
        if(argc>=2&&std::string(argv[1])=="--near-cone-only") {nearConeOnly(argc>=3?std::filesystem::path(argv[2]):std::filesystem::path{});return 0;}
        if(argc>=3&&std::string(argv[1])=="--field-geometry-only") {fieldGeometryOnly(argv[2]);return 0;}
        if(argc>=2&&std::string(argv[1])=="--acquisition-only") {acquisitionOnly(argc>=3?std::filesystem::path(argv[2]):std::filesystem::path{});return 0;}
        if(argc>=2&&std::string(argv[1])=="--steering-only") {steeringOnly(argc>=3?std::filesystem::path(argv[2]):std::filesystem::path{});return 0;}
        const auto cfg=calibration();RaceGroundMap mapping(cfg);
        for(const auto &p:std::vector<cv::Point2f>{{0,.25f},{-.1f,.4f},{.12f,.6f}})
            check(cv::norm(mapping.toGround(image(p))-p)<1e-5,"metric mapping round trip failed");
        bool rejected=false;try {auto bad=cfg;std::swap(bad.imagePoints[0],bad.imagePoints[3]);RaceGroundMap invalid(bad);}catch(...){rejected=true;}
        check(rejected,"misordered ground calibration accepted");
        rejected=false;try {auto bad=cfg;bad.vehicleLength=NAN;bad.validate();}catch(...){rejected=true;}
        check(rejected,"invalid measured length accepted");
        auto rolled=cfg;const float roll=.02f;
        for(auto &p:rolled.imagePoints){const auto q=cv::Point2f(p[0]-160,p[1]-120);p={160+std::cos(roll)*q.x-std::sin(roll)*q.y,120+std::sin(roll)*q.x+std::cos(roll)*q.y};}
        RaceGroundMap rolledMap(rolled);RaceObstaclePreviewPlan rolledPath;std::vector<cv::Point2f> groundPath;
        for(int i=0;i<=26;++i)groundPath.emplace_back(.05f*i/26.f,i*.025f);
        check(raceGroundProjectPath(rolledMap,rolled,groundPath,rolledPath),"rolled camera route projection rejected");
        for(const auto &p:rolledPath.centers){const auto q=rolledMap.toGround({p.y,float(p.x)});check(std::abs(q.x-.05f*q.y/.65f)<1e-5,"rolled camera target shifted off ground route");}
        check(raceBypassServo(1900,1500,.1f)==1560&&raceBypassServo(1100,1500,.1f)==1440,"bypass steering slew exceeded");
        check(raceBypassServo(1900,1900,.1f)==1900&&raceBypassServo(1100,1100,.1f)==1100,"bypass steering range exceeded");
        std::vector<cv::Point2f> a,b;
        const float angle=.01f,c=std::cos(angle),s=std::sin(angle);
        for(int i=0;i<5;++i)for(int j=0;j<5;++j){cv::Point2f p(-.2f+i*.1f,.2f+j*.1f);a.push_back(p);b.emplace_back(c*p.x-s*p.y-.002f,s*p.x+c*p.y-.007f);}
        RaceGroundPose pose;int inliers=0;float residual=0;
        check(RaceGroundMotion::fit(a,b,pose,inliers,residual)&&inliers==25&&pose.z>.006f&&pose.angle<0,"metric rigid visual motion fit failed");
        auto wrong=b;for(auto &p:wrong)p*=1.2f;check(!RaceGroundMotion::fit(a,wrong,pose,inliers,residual),"scale change accepted as ground motion");
        Scene scene;RaceGroundMotion motion(cfg);auto snap=scene.frame();check(!motion.update(snap).valid,"first frame claimed motion");
        snap=scene.frame();auto odo=motion.update(snap);check(odo.valid&&std::abs(odo.pose.z)<.002f,"stationary textured image could not verify zero motion");
        snap=scene.frame(.005f);odo=motion.update(snap);std::cout<<"motion inliers="<<odo.inliers<<" z="<<odo.pose.z<<" residual="<<odo.residual<<'\n';
        check(odo.valid&&odo.pose.z>.002f&&odo.pose.z<.008f,"rendered forward motion not recovered");
        // 旧断言把"纯白路面 + 深色背景"当成无特征，但里程掩膜已按另一轮改动放宽到
        // 路旁 z<=1.0（允许路侧纹理参与），路沿本身就是合法的静止特征，所以原断言
        // 与现行设计冲突、在未改动的仓库上也是红的。这里拆成两条更强的检查：
        // 1) 真正整幅纯色（连路沿都没有）不得凭空产生位移；
        // 2) 只有路沿时可估计，但结果必须与真实 5mm 位移一致，不能乱给。
        {
            RaceGroundMotion noTexture(cfg);
            RaceSnapshot flat;flat.image=cv::Mat(240,320,CV_8UC3,cv::Scalar(90,90,90));
            flat.captured=0;flat.sequence=1;flat.failed=false;
            noTexture.update(flat);
            flat.captured=.07;flat.sequence=2;
            check(!noTexture.update(flat).valid,"featureless ground invented visual travel");
            Scene blank;RaceGroundMotion edgesOnly(cfg);
            edgesOnly.update(blank.frame(0,0,-.13f,false,false));
            const auto edgeOdo=edgesOnly.update(blank.frame(.005f,0,-.13f,false,false));
            if(edgeOdo.valid)
                check(std::abs(edgeOdo.pose.z-.005f)<.01f,"edge-only motion disagreed with true 5 mm translation");
            std::cout<<"edge-only valid="<<edgeOdo.valid<<" z="<<edgeOdo.pose.z
                     <<" inliers="<<edgeOdo.inliers<<" reason="<<edgeOdo.reason<<'\n';
        }
        RaceConfig race;RaceBypassControl control;RaceObstaclePreviewPlan p;
        p.metric=p.fresh=p.motionValid=p.hasTarget=p.candidate=true;p.stop=false;p.direction=1;
        p.centers={{200,160},{100,175}};p.obstacles={{60,100,30,60}};
        RaceSnapshot ai;ai.image=cv::Mat(240,320,CV_8UC3,cv::Scalar(255,255,255));ai.detections={{LABEL_CONE,"cone",.99f,60,100,30,60}};
        RaceBypassDecision decision;
        for(int i=1;i<=3;++i){ai.sequence=p.sequence=i;ai.captured=p.captured=i*.07;decision=control.update(p,ai,race,ai.captured,true);if(i<3)check(decision.stop&&!decision.follow,"route not confirmed before takeover");}
        check(decision.follow&&decision.authorizeCone&&!decision.stop,"calibrated route could not authorize cone bypass");
        ai.sequence=4;ai.captured=.28;p.captured=.28;p.sequence=4;p.motionValid=false;
        check(control.update(p,ai,race,.28,true).stop,"motion failure did not stop bypass");
        p.motionValid=true;control.update(p,ai,race,.28,false);
        check(control.update(p,ai,race,.28,true).stop,"manual/auto cleared latched bypass fault");
        // 原停车策略只对已授权锥桶解除；行人、红牌、施工等义务不由绕行覆盖。
        Track track;cv::Mat bin=cv::Mat::zeros(240,320,CV_8UC1);cv::rectangle(bin,{40,0,240,240},{255},-1);track.handle(bin);
        RaceManager manager(race);RacePlan command;
        for(int i=1;i<=6;++i){ai.sequence=i;ai.captured=i*.07;command=manager.update(ai,ai.captured,.07,true,bin,track,ai.image,true);}
        check(!command.stop,"authorized cone still blocked by original fallback");
        ai.detections.push_back({LABEL_PERSON,"person",.99f,180,100,35,60});
        for(int i=7;i<=10;++i){ai.sequence=i;ai.captured=i*.07;command=manager.update(ai,ai.captured,.07,true,bin,track,ai.image,true);}
        check(command.stop&&std::string(command.state)=="OBSTACLE_WAIT","person priority bypassed");
        for(int label:{LABEL_STOP,LABEL_BUSY,LABEL_PARK,LABEL_GATE}) {
            auto policy=race;policy.stopAfterCross=false;RaceManager priority(policy);
            ai.detections={{label,"",.99f,100,145,60,15},{LABEL_CONE,"cone",.99f,60,100,30,60}};
            for(int i=1;i<=6;++i){ai.sequence=i;ai.captured=i*.07;command=priority.update(ai,ai.captured,.07,true,bin,track,ai.image,true);}
            check(command.stop,"cone authorization bypassed stop/construction/parking/gate priority");
        }
        // 像素建议、过期路线和最新观测中的新增锥桶都不能解锁实跑。
        auto original=p;ai.detections={{LABEL_CONE,"cone",.99f,60,100,30,60}};
        ai.sequence=p.sequence=10;ai.captured=p.captured=.7;p.metric=false;
        RaceBypassControl pixels;check(pixels.update(p,ai,race,.7,true).stop,"pixel route authorized real bypass");
        p=original;p.sequence=10;p.captured=.2;RaceBypassControl stale;
        check(stale.update(p,ai,race,.7,true).stop,"stale metric route authorized real bypass");
        p.captured=.7;p.state="BYPASS_NO_TARGET";p.hasTarget=false;p.obstacles.clear();RaceBypassControl newest;
        check(newest.update(p,ai,race,.7,true).stop,"newest cone ignored while worker still had empty scene");
        Scene field;RaceGroundObstacle planner(cfg);RaceObstaclePreviewPlan route;
        for(int i=0;i<6;++i){snap=field.frame();route=plan(planner,snap,field.race);}
        std::cout<<"initial route="<<route.state<<" reason="<<route.reason<<" points="<<route.centers.size()<<" inliers="<<route.motionInliers<<'\n';
        check(route.candidate&&route.metric&&!route.stop&&route.direction==1,"left cone did not produce grounded right route");
        for(size_t i=0;i<route.groundCenters.size();++i) {
            const auto v=route.groundCenters[i];
            const float slope=i?(v[0]-route.groundCenters[i-1][0])/(v[1]-route.groundCenters[i-1][1]):0;
            const float norm=std::hypot(1.f,slope);const cv::Point2f forward(slope/norm,1/norm),normal(forward.y,-forward.x);
            for(float along=-cfg.vehicleLength;along<.001f;along+=.01f)for(float across=-cfg.vehicleWidth/2;across<=cfg.vehicleWidth/2+.001f;across+=.01f) {
                const auto point=cv::Point2f(v[0],v[1])+forward*along+normal*across;
                check(std::abs(point.x)<.225f-cfg.clearance+.002f,"rotated 31cm body left road");
                check(cv::norm(point-cv::Point2f(-.13f,.45f))>=.035f+cfg.clearance-.002f,"rotated swept body collided with cone");
            }
        }
        Scene right;RaceGroundObstacle other(cfg);for(int i=0;i<6;++i){snap=right.frame(0,0,.13f);route=plan(other,snap,right.race);}
        check(route.candidate&&route.direction==-1,"right cone did not produce grounded left route");
        Scene blocked;RaceGroundObstacle center(cfg);for(int i=0;i<6;++i){snap=blocked.frame(0,0,0);route=plan(center,snap,blocked.race);}
        check(route.stop&&!route.candidate,"center cone with insufficient footprint gap was bypassed");
        // 现场现象：四点标定的米制尺度整体偏大，45cm路面被换算成51~53cm。
        // 必须按实测路宽在有界范围内锁定修正后继续规划，而不是整轮停车。
        Scene wide;RaceGroundObstacle widePlanner(cfg);
        for(int i=0;i<6;++i){snap=wide.frame(0,0,-.13f,true,true,1.16f);route=plan(widePlanner,snap,wide.race);}
        std::cout<<"inflated-scale route="<<route.state<<" reason="<<route.reason<<" raw="<<route.laneWidthRaw
                 <<" scale="<<route.laneScale<<'\n';
        check(route.candidate&&!route.stop&&std::abs(route.laneScale-.862f)<.05f,
            "inflated four-point scale did not lock a bounded metric correction");
        // 明显不是同一条45cm直道（宽度超出允许修正范围）时仍必须停车。
        Scene outOfRange;RaceGroundObstacle farPlanner(cfg);
        for(int i=0;i<6;++i){snap=outOfRange.frame(0,0,-.13f,true,true,1.5f);route=plan(farPlanner,snap,outOfRange.race);}
        check(route.stop&&!route.candidate,"out-of-range lane width still produced a bypass route");
        Scene moving;RaceGroundObstacle movingPlanner(cfg);bool observedPass=false,observedReturn=false;int eligible=0;
        std::ofstream trace;
        cv::VideoWriter video;std::ofstream labels;
        if(argc>=2){
            const std::filesystem::path out=argv[1];std::filesystem::create_directories(out);trace.open(out/"dynamic-route.jsonl");
            video.open((out/"bypass-scenario.avi").string(),cv::VideoWriter::fourcc('M','J','P','G'),15,{320,240});
            check(video.isOpened(),"cannot create moving bypass replay");labels.open(out/"bypass-scenario.jsonl");
        }
        for(int frame=0;frame<190;++frame) {
            const float progress=std::max(0,frame-6)*.007f;
            const float shift=.07f*std::min(progress/.20f,1.f)*std::clamp((1.2f-progress)/.2f,0.f,1.f);
            snap=moving.frame(progress,shift);route=plan(movingPlanner,snap,moving.race);
            if(video.isOpened()) {
                video.write(snap.image);auto ds=nlohmann::json::array();
                for(const auto &d:snap.detections)ds.push_back({{"type",d.type},{"score",d.score},{"x",d.x},{"y",d.y},{"width",d.width},{"height",d.height}});
                labels<<nlohmann::json({{"frame",snap.sequence},{"detections",ds}}).dump()<<'\n';
            }
            if(trace){auto j=raceObstacleJson(route,{});j["truth_forward_m"]=progress;j["truth_lateral_m"]=shift;trace<<j.dump()<<'\n';}
            if(frame>=5&&progress<.98f)check(route.candidate&&!route.stop,("moving route lost: "+route.state+" "+route.reason).c_str());
            if(route.candidate)++eligible;
            if(route.passed){check(progress>.8f,"rear passage declared before 31cm body could clear");observedPass=true;}
            if(route.state=="BYPASS_RETURN")observedReturn=true;
        }
        check(eligible>100&&observedPass&&observedReturn,"dynamic mapped cone never reached verified rear passage/return");
        std::cout<<"dynamic eligible="<<eligible<<" rear_passed="<<observedPass<<" return="<<observedReturn<<'\n';
        RaceNavigationConfig shadowConfig;RaceNavigationWorker background(shadowConfig,race,150,10,20,{},RaceNavigationWorker::Process{},cfg);
        Scene backgroundScene;
        for(int i=1;i<=4;++i) {
            auto fresh=backgroundScene.frame();check(background.submit(fresh,fresh.captured),"metric worker submit failed");
            for(int wait=0;wait<400&&background.completedCount()<i;++wait)std::this_thread::sleep_for(std::chrono::milliseconds(5));
            check(background.completedCount()==i,"metric worker did not complete bounded job");
        }
        const auto published=background.view(.29);background.finish();
        check(published.ai.sequence==4&&published.plan.sequence==4&&published.plan.metric&&
            std::abs(published.plan.groundVehicleWidth-.18f)<1e-5,"metric worker lost calibrated branch or matching image");
        check(published.plan.candidate||(!published.plan.fresh&&published.plan.stop),"metric worker neither published route nor expired safely");
        if(argc>=2) {
            const std::filesystem::path out=argv[1];std::filesystem::create_directories(out);
            auto replayGround=cfg;
            if(argc>=3)replayGround.lensFingerprint=RaceGroundCalibration::fingerprint(argv[2]);
            std::ofstream(out/"synthetic-ground.json")<<nlohmann::json(replayGround).dump(2);
            cv::imwrite((out/"synthetic-road.png").string(),snap.image);
        }
        std::cout<<"PASS calibrated mapping, visual ground motion, route authority, task priority, left/right/blocked geometry, moving rear passage\n";return 0;
    }catch(const std::exception &e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}
}
