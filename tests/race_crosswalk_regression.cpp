#include "race_manager.hpp"
#include "race_ui.hpp"
#include <chrono>
#include <thread>
#include <filesystem>
#include <iostream>
#include <stdexcept>

static void require(bool value,const char *why) { if(!value)throw std::runtime_error(why); }
static cv::Mat road() {
    cv::Mat image(240,320,CV_8UC3,cv::Scalar(130,55,20));
    cv::rectangle(image,{30,0,260,240},{190,195,200},-1);return image;
}
static cv::Mat crossing(int top=90,int height=28,int width=12,int pitch=24,int bars=9) {
    auto image=road();
    for(int i=0;i<bars;++i)cv::rectangle(image,{45+i*pitch,top,width,height},{30,32,32},-1);
    return image;
}
static PredictResult detection(int type,float score=.95f) {
    return {type,"",score,100,120,30,40};
}
struct Scenario {
    RaceConfig config;
    RaceManager manager;
    RaceSnapshot snapshot;
    Track track;
    cv::Mat bin=cv::Mat(240,320,CV_8UC1,cv::Scalar(255));
    double time=0;
    uint64_t sequence=0;
    explicit Scenario(RaceConfig cfg={}):config(cfg),manager(config) {
        track.handle(bin);step(road()); // 首次切入自动模式会清除确认计数。
    }
    RacePlan step(const cv::Mat &frame,std::vector<PredictResult> ds={},bool newFrame=true,float dt=.07f) {
        time+=dt;
        if(newFrame) {
            snapshot={++sequence,time,std::move(ds),false,{},frame,{},
                config.verifyCrosswalkGeometry?detectCrosswalkGeometry(frame):CrosswalkGeometry{}};
        }
        return manager.update(snapshot,time,dt,true,bin,track);
    }
    RacePlan repeat(const cv::Mat &frame,std::vector<PredictResult> ds={},int n=3) {
        RacePlan p;while(n--)p=step(frame,ds);return p;
    }
};
static void integratedChecks() {
    RaceConfig cfg;
    const auto near=crossing(170,45,17,25),far=crossing();
    RaceSnapshot s{1,1,{},false,{},near,{},detectCrosswalkGeometry(near)};
    auto low=detection(LABEL_CROSS,.18207407f);
    low.x=s.crosswalk.box.x;low.y=s.crosswalk.box.y;low.width=s.crosswalk.box.width;low.height=s.crosswalk.box.height;
    const float original=low.score;
    require(reviewRaceDetection(low,near,cfg,&s.crosswalk).accepted,"low cross + matched stripes rejected");
    require(low.score==original,"geometry must not inflate model score");
    require(!reviewRaceDetection(low,road(),cfg).accepted,"low cross on plain road accepted");
    require(!reviewRaceDetection(low,cv::Mat{},cfg,&s.crosswalk).accepted,"missing matched image accepted");
    auto displaced=low;displaced.x=0;displaced.y=0;displaced.width=30;displaced.height=30;
    require(!reviewRaceDetection(displaced,near,cfg,&s.crosswalk).accepted,"unrelated cross box accepted");
    auto under=low;under.score=.09f;
    require(!reviewRaceDetection(under,near,cfg,&s.crosswalk).accepted,"below cross floor accepted");
    for(int type:{LABEL_FLOW,LABEL_LIMIT,LABEL_CONE,LABEL_BUSY,LABEL_STOP}) {
        auto other=low;other.type=type;
        require(!reviewRaceDetection(other,near,cfg,&s.crosswalk).accepted,"cross floor leaked to another class");
    }
    const auto fallbackJson=raceDetectionJson(s,1.1,cfg);
    require(fallbackJson["detections"].empty()&&fallbackJson["crosswalk_geometry"]["model_score"].is_null()&&
            fallbackJson["crosswalk_geometry"]["fallback_used"].get<bool>(),"geometry misreported as model confidence");
    s.detections={low};const auto checkedJson=raceDetectionJson(s,1.1,cfg);
    require(checkedJson["detections"][0]["checked_cross"].get<bool>()&&
            std::abs(checkedJson["detections"][0]["score"].get<float>()-original)<1e-6&&
            !checkedJson["crosswalk_geometry"]["fallback_used"].get<bool>(),"checked model metadata mismatch");
    require(!racePreviewImage(s,1.1,cfg).empty(),"checked cross preview failed");
    s.detections.clear();require(!raceCompactView(near,cv::Mat{},s,1.1,cfg,false).empty(),"fallback compact view failed");

    Scenario noRed;
    auto p=noRed.repeat(near,{},8);
    require(!p.stop&&std::string(p.state)=="LANE","cross alone must not invent a stop signal");
    Scenario signal;
    p=signal.repeat(far,{detection(LABEL_STOP)});
    require(std::string(p.state)=="CROSS_APPROACH"&&!p.stop,"red must approach cross before stopping");
    p=signal.step(near); // 一帧近处证据不能按三帧计数。
    // 先前远处证据连续有效时，近处位置可以更新；另建场景测试孤立近帧。
    Scenario one;
    one.repeat(road(),{detection(LABEL_STOP)});
    one.step(near);
    one.step(near,{},false);one.step(near,{},false);
    p=one.repeat(road(),{},6);
    require(std::string(p.state)=="CROSS_APPROACH"&&!p.stop,"repeated old snapshot faked a passed crossing");
    signal.repeat(near,{},3);
    p=signal.repeat(road(),{},6);
    require(p.stop&&std::string(p.state)=="SIGNAL_WAIT","confirmed near crossing + clear interval did not stop");
    require(signal.repeat(road(),{detection(LABEL_FLOW,.18f)},20).stop,"low flow score released stop");
    p=signal.repeat(road(),{detection(LABEL_FLOW)},3);
    require(!p.stop&&std::string(p.state)=="LANE","normal confirmed green cannot release");
    Scenario tooFar;
    tooFar.repeat(far,{detection(LABEL_STOP)});
    p=tooFar.repeat(road(),{},7);
    require(std::string(p.state)=="CROSS_APPROACH"&&!p.stop,"far crossing disappearance faked a pass");
    p=tooFar.repeat(road(),{},90);
    require(p.stop&&std::string(p.reason)=="CROSS_PASS_NOT_VERIFIED","unverified crossing timeout removed");
    Scenario oversized;
    auto largeCross=low;largeCross.x=20;largeCross.y=70;largeCross.width=260;largeCross.height=160;
    oversized.repeat(far,{detection(LABEL_STOP),largeCross});p=oversized.repeat(road(),{},6);
    require(std::string(p.state)=="CROSS_APPROACH","oversized low-score model box faked a near crossing");
    Scenario stale;
    stale.repeat(near,{detection(LABEL_STOP)});
    p=stale.step(road(),{},false,.45f);
    require(p.stop&&!p.fresh,"stale AI cannot prove crossing clearance");
    cfg.verifyCrosswalkGeometry=false;
    require(!reviewRaceDetection(low,near,cfg,&s.crosswalk).accepted&&!raceCrosswalkFallback(s,cfg),"geometry disable ignored");
    Scenario disabled(cfg);
    disabled.repeat(near,{detection(LABEL_STOP)});p=disabled.repeat(road(),{},6);
    require(std::string(p.state)=="CROSS_APPROACH","disabled geometry still altered planner");

    RacePerception worker([](const cv::Mat &){return std::vector<PredictResult>{};});
    worker.submit(near,77,1.0);
    RaceSnapshot result;
    for(int i=0;i<200;++i) { result=worker.snapshot();if(result.sequence==77)break;std::this_thread::sleep_for(std::chrono::milliseconds(5)); }
    require(result.sequence==77&&result.crosswalk.accepted&&cv::norm(result.image,near)==0,"AI worker did not attach matched-frame geometry");
}
int main(int argc,char **argv) {
    try {
        int cases=0;
        for(const auto &image:std::vector<cv::Mat>{crossing(),crossing(20,9,7,14),crossing(160,45,17,25)}) {
            const auto found=detectCrosswalkGeometry(image);
            require(found.accepted&&found.stripes>=6,"synthetic crosswalk missed");++cases;
        }
        auto image=crossing();cv::Mat dim;image.convertTo(dim,CV_8UC3,.65);
        require(detectCrosswalkGeometry(dim).accepted,"moderately dim crosswalk missed");++cases;
        for(double angle:{-4.,4.}) {
            cv::Mat rotated;cv::warpAffine(image,rotated,cv::getRotationMatrix2D({160,120},angle,1),image.size(),cv::INTER_LINEAR,cv::BORDER_REPLICATE);
            require(detectCrosswalkGeometry(rotated).accepted,"mildly rotated crosswalk missed");++cases;
        }
        std::vector<cv::Mat> negatives={road(),crossing(90,28,12,24,4)};
        auto divider=road();cv::rectangle(divider,{145,0,30,170},{20,20,20},-1);negatives.push_back(divider);
        auto horizontal=road();for(int y=60;y<180;y+=20)cv::rectangle(horizontal,{70,y,180,8},{25,25,25},-1);negatives.push_back(horizontal);
        auto checker=road();for(int y=70;y<150;y+=10)for(int x=70;x<250;x+=10) {
            if(((x+y)/10)%2==0)cv::rectangle(checker,{x,y,10,10},{25,25,25},-1);
        }
        negatives.push_back(checker);
        auto colored=crossing();for(int i=0;i<9;++i)cv::rectangle(colored,{45+i*24,90,12,28},{20,30,180},-1);negatives.push_back(colored);
        auto blue=cv::Mat(240,320,CV_8UC3,cv::Scalar(130,55,20));for(int i=0;i<9;++i) {
            cv::rectangle(blue,{45+i*24,90,12,28},{20,20,20},-1);
        }
        negatives.push_back(blue);
        auto irregular=road();for(int i=0;i<9;++i)cv::rectangle(irregular,{35+i*27,70+(i%3)*20,9,20},{20,20,20},-1);negatives.push_back(irregular);
        for(const auto &negative:negatives) { require(!detectCrosswalkGeometry(negative).accepted,"ordinary road/texture/color falsely accepted");++cases; }
        require(!detectCrosswalkGeometry(cv::Mat{}).accepted,"empty image accepted");++cases;
        integratedChecks();
        if(argc==2||argc==3) {
            namespace fs=std::filesystem;
            const fs::path saved=argv[1];
            for(int i=1;i<=4;++i) {
                const auto frame=cv::imread((saved/("ai-sample-"+std::to_string(i))/"camera.png").string());
                require(!frame.empty(),"saved crosswalk sample missing");
                const auto result=detectCrosswalkGeometry(frame);
                std::cout<<"sample="<<i<<" accepted="<<result.accepted<<" stripes="<<result.stripes<<" box="<<result.box<<" contrast="<<result.contrast<<'\n';
                require(result.accepted==(i<4),"saved crosswalk/empty-road regression");++cases;
                cv::Mat gray,bin;cv::cvtColor(frame,gray,cv::COLOR_BGR2GRAY);
                cv::threshold(gray,bin,0,255,cv::THRESH_BINARY|cv::THRESH_OTSU);
                Track track;track.handle(bin);RaceConfig cfg;RaceManager manager(cfg);RacePlan plan;
                for(int seq=1;seq<=5;++seq) {
                    RaceSnapshot observation{uint64_t(seq),seq*.07,{},false,{},frame,{},result};
                    plan=manager.update(observation,seq*.07,.07f,true,bin,track,frame);
                }
                require(!plan.stop&&std::string(plan.state)=="LANE","saved stripe geometry falsely triggered fork/stop");
            }
        } else require(argc==1,"usage: race_crosswalk_regression [saved-run [reviewed-negative-paths.json]]");
        if(argc==3) {
            std::ifstream input(argv[2]);nlohmann::json paths;input>>paths;
            require(paths.is_array()&&!paths.empty(),"invalid reviewed negative image list");
            for(const auto &path:paths) {
                const auto frame=cv::imread(path.get<std::string>());
                require(!frame.empty(),"reviewed negative image missing");
                require(!detectCrosswalkGeometry(frame).accepted,"saved plain road/sign/hood falsely identified as crosswalk");++cases;
            }
            std::cout<<"PASS reviewed saved negatives="<<paths.size()<<'\n';
        }
        std::cout<<"PASS crosswalk geometry cases="<<cases<<" + review/metadata/freshness/planner/worker checks\n";
    }catch(const std::exception &e) { std::cerr<<"FAIL "<<e.what()<<'\n';return 1; }
}
