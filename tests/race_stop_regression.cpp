#include "race_manager.hpp"
#include "race_preview.hpp"
#include <opencv2/imgcodecs.hpp>
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <filesystem>

void check(bool value,const char *message) { if(!value)throw std::runtime_error(message); }
PredictResult detection(int label,float score,int y=120,int height=40,int x=100,int width=80) {
    PredictResult d{};d.type=label;d.score=score;d.x=x;d.y=y;d.width=width;d.height=height;return d;
}
struct StopFixture {
    RaceConfig cfg;
    RaceManager manager;
    Track track;cv::Mat bin=cv::Mat::zeros(240,320,CV_8UC1);double time=0;uint64_t sequence=0;
    RaceSnapshot snap;
    explicit StopFixture(bool enabled=true):cfg([&]{RaceConfig c;c.stopAfterCross=false;c.verifyStopCandidates=enabled;return c;}()),manager(cfg) {
        cv::rectangle(bin,{40,0,241,240},cv::Scalar(255),-1);track.handle(bin);step({});
    }
    RacePlan step(std::vector<PredictResult> ds,const cv::Mat &image={},bool newFrame=true,bool active=true) {
        time+=.07;
        if(newFrame)snap={++sequence,time,std::move(ds),false,{},image};
        return manager.update(snap,time,.07f,active,bin,track);
    }
};
int main(int argc,char **argv) {
    try {
        check(argc==3,"need saved-image evidence root and output directory");
        namespace fs=std::filesystem;
        fs::path evidence=argv[1],out=argv[2];fs::create_directories(out);
        auto image=cv::imread((evidence/"sample-4/camera.png").string());
        check(!image.empty(),"saved low-score image missing");
        std::ifstream input(evidence/"sample-4/candidates.json");nlohmann::json sample;input>>sample;
        const auto v=sample.at("candidates").at(0);
        auto stop=detection(LABEL_STOP,v.at("score"),v.at("y"),v.at("height"),v.at("x"),v.at("width"));
        RaceConfig cfg;cfg.stopAfterCross=false;
        auto accepted=reviewRaceDetection(stop,image,cfg);
        check(stop.score<cfg.confidence&&accepted.accepted&&accepted.checkedStop,"saved low score must require appearance acceptance");
        check(!reviewRaceDetection(stop,cv::Mat{},cfg).accepted,"missing matched original cannot accept low score");
        auto low=stop;low.score=.34f;check(!reviewRaceDetection(low,image,cfg).accepted,"below candidate floor must fail");
        auto wrong=stop;wrong.x=-1;check(!reviewRaceDetection(wrong,image,cfg).accepted,"invalid boxes must fail");
        auto green=cv::imread((evidence/"sample-1/camera.png").string());
        auto white=cv::imread((evidence/"sample-2/camera.png").string());
        check(!reviewRaceDetection(stop,green,cfg).accepted&&!reviewRaceDetection(stop,white,cfg).accepted,
              "green and white cannot validate a low-score stop candidate");
        auto flow=stop;flow.type=LABEL_FLOW;flow.score=.64f;
        check(!reviewRaceDetection(flow,image,cfg).accepted,"shape verification cannot lower flow threshold");
        auto limit=stop;limit.type=LABEL_LIMIT;
        check(!reviewRaceDetection(limit,image,cfg).accepted,"other classes retain original threshold");
        auto json=raceDetectionJson(RaceSnapshot{1,1,{stop},false,{},image},1.1,cfg);
        check(json["detections"][0]["accepted"]==true&&json["detections"][0]["passes_threshold"]==false,
              "metadata must distinguish model threshold and verified acceptance");
        check(std::fabs(json["detections"][0]["score"].get<float>()-stop.score)<1e-6,"never inflate model score");
        check(raceSignRegions(RaceSnapshot{1,1,{stop},false,{},image},cfg).size()==1,"verified low stop must mask only its sign region");

        StopFixture live;
        check(!live.step({stop},image).stop&&!live.step({stop},image).stop,"one or two observations cannot stop");
        check(live.step({stop},image).stop,"three near verified observations must enter wait");
        for(int i=0;i<20;++i)check(live.step({}).stop,"red disappearing cannot release waiting car");
        for(int i=0;i<5;++i)check(live.step({flow},green).stop,"low-score green cannot release");
        flow.score=.95f;
        check(live.step({flow},green).stop&&live.step({flow},green).stop,"flow still needs three observations");
        auto released=live.step({flow},green);
        check(!released.stop&&std::string(released.state)=="LANE","high-confidence confirmed flow releases normally");

        StopFixture repeated;repeated.step({stop},image);
        check(!repeated.step({},image,false).stop&&!repeated.step({},image,false).stop,"same AI frame cannot count multiple times");
        repeated.step({});check(!repeated.step({stop},image).stop,"missing observation must reset consecutive confirmation");
        StopFixture disabled(false);
        for(int i=0;i<5;++i)check(!disabled.step({stop},image).stop,"disabled fallback must preserve original threshold");
        auto far=detection(LABEL_STOP,.40f,19,115,0,116);
        auto side=cv::imread((evidence/"sample-5/camera.png").string());StopFixture distant;
        check(reviewRaceDetection(far,side,cfg).accepted,"side low candidate should pass appearance");
        for(int i=0;i<5;++i)check(!distant.step({far},side).stop,"visible but distant sign must not bypass trigger row");
        auto stale=live.snap;stale.captured=live.time-1;
        check(live.manager.update(stale,live.time,.07,true,live.bin,live.track).stop,"stale recognition must still stop motion");

        // 固定输入回放覆盖两种模型分数交替、等待、低分绿牌拒绝及高分绿牌放行。
        cv::VideoWriter video((out/"stop-scenario.avi").string(),cv::VideoWriter::fourcc('M','J','P','G'),15,{320,240});
        check(video.isOpened(),"cannot create integrated replay video");
        std::ofstream labels(out/"stop-scenario.jsonl");
        cv::Mat road(240,320,CV_8UC3,cv::Scalar(20,20,20));cv::rectangle(road,{40,0,241,240},{230,230,230},-1);
        auto marked=road.clone();cv::circle(marked,{250,85},31,{35,35,200},-1);
        cv::rectangle(marked,{225,80,50,10},{225,225,225},-1);
        auto synthetic=detection(LABEL_STOP,.45f,53,64,218,64);
        check(reviewRaceDetection(synthetic,marked,cfg).accepted,"synthetic replay stop must pass appearance");
        // 回放场景通过触发行配置单独降低到110以验证远处牌面，不修改正式145默认。
        for(int n=1;n<=140;++n) {
            auto ds=nlohmann::json::array();const bool showing=n>=25&&n<=65;
            video.write(showing?marked:road);
            if(showing)ds.push_back({{"type",LABEL_STOP},{"score",n%2?.45:.70},{"x",218},{"y",53},{"width",64},{"height",64}});
            if(n>=90)ds.push_back({{"type",LABEL_FLOW},{"score",n<115?.50:.95},{"x",220},{"y",100},{"width",40},{"height",50}});
            labels<<nlohmann::json({{"frame",n},{"detections",ds}}).dump()<<'\n';
        }
        std::ofstream(out/"stop-race.json")<<nlohmann::json({{"stopAfterCross",false},{"triggerRow",110}}).dump(2);
        std::cout<<"PASS integrated low-stop acceptance, real negative images, freshness, consecutive counts, trigger position, flow threshold and replay fixture\n";
    }catch(const std::exception &e) { std::cerr<<"FAIL "<<e.what()<<'\n';return 1; }
}
