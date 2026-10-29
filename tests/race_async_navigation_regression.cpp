#include "race_navigation_worker.hpp"
#include "race_sample_writer.hpp"
#include "race_fork_evidence.hpp"
#include <future>
#include <iostream>
#include <unistd.h>

static void require(bool ok,const char *why) { if(!ok)throw std::runtime_error(why); }
static bool waitFor(const std::function<bool()> &done) {
    for(int i=0;i<400;++i) { if(done())return true;std::this_thread::sleep_for(std::chrono::milliseconds(5)); }return false;
}
struct Release {
    std::promise<void> promise;bool released=false;
    void open() { if(!released) { released=true;promise.set_value(); } }
    ~Release() { open(); }
};
struct OpenOnExit { Release &release;~OpenOnExit(){release.open();} };
static RaceSnapshot frame(int sequence,double captured=10.) {
    RaceSnapshot s;s.sequence=sequence;s.captured=captured;
    s.image=cv::Mat(240,320,CV_8UC3,cv::Scalar(sequence%255,40,50));return s;
}
static RaceObstaclePreviewPlan proposed(const RaceSnapshot &s) {
    RaceObstaclePreviewPlan p;p.sequence=s.sequence;p.captured=s.captured;
    p.fresh=p.hasTarget=p.candidate=true;p.stop=false;p.state="OBS_BYPASS_RIGHT_PREVIEW";
    p.centers={{219,160},{217,160}};p.left={{219,100},{217,100}};p.right={{219,220},{217,220}};return p;
}
int main(int argc,char **argv) {
    try {
        const auto out=argc==2?std::filesystem::path(argv[1]):std::filesystem::temp_directory_path()/
            ("race-async-"+std::to_string(getpid())+"-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(out);RaceNavigationConfig cfg;RaceConfig race;
        std::atomic_int started{0};Release release;auto permit=release.promise.get_future().share();
        RaceNavigationWorker worker(cfg,race,-1,10,20,out/"queued-navigation.jsonl",[&](const RaceSnapshot &s,double) {
            ++started;permit.wait();return proposed(s);
        });
        OpenOnExit releaseGuard{release};
        worker.submit(frame(1),10.);
        const bool firstStarted=waitFor([&]{return started.load()>0;});
        if(!firstStarted) { release.open();throw std::runtime_error("navigation worker did not start"); }
        for(int i=2;i<=80;++i)require(worker.submit(frame(i),10.),"fresh pending frame not accepted");
        const bool duplicate=worker.submit(frame(80),10.);
        const auto before=worker.view(10.1);const int pendingDrops=worker.droppedCount();
        release.open();worker.finish();
        require(!duplicate&&before.ai.sequence==0&&pendingDrops>=78,"navigation submission waited for blocked planner or queued every frame");
        auto view=worker.view(10.1);
        require(view.ai.sequence==80&&view.plan.sequence==80&&view.ai.image.at<cv::Vec3b>(0,0)[0]==80,
            "navigation result not bound to latest retained source image");
        require(worker.completedCount()==2&&view.plan.candidate,"navigation queue lost latest job or processed stale backlog");
        auto stale=worker.view(10.5);
        require(!stale.plan.candidate&&stale.plan.stop&&stale.plan.centers.empty()&&stale.plan.state=="OBS_AI_STALE","expired navigation still offered a route");
        {
            std::ifstream input(out/"queued-navigation.jsonl");std::string line;int count=0;
            while(std::getline(input,line)) { auto j=nlohmann::json::parse(line);++count;require(j["preview_only"].get<bool>()&&!j["drives_vehicle"].get<bool>(),"worker output claimed vehicle control"); }
            require(count==2,"worker output not flushed at shutdown");
        }
        RaceNavigationWorker failed(cfg,race,-1,10,20,{},[](const RaceSnapshot&,double)->RaceObstaclePreviewPlan {throw std::runtime_error("injected planner failure");});
        failed.submit(frame(81),10.);require(waitFor([&]{return failed.completedCount()==1;}),"failed worker not published");
        require(failed.view(10.1).plan.state=="OBS_PLANNER_FAILED"&&failed.view(10.1).plan.stop,"planner exception did not yield stopped shadow result");failed.finish();
        RaceConfig shortAge;shortAge.maxDetectionAge=.05f;
        RaceNavigationWorker delayed(cfg,shortAge,-1,10,20,{},[](const RaceSnapshot &s,double) {
            std::this_thread::sleep_for(std::chrono::milliseconds(70));return proposed(s);
        });
        delayed.submit(frame(82),10.);require(waitFor([&]{return delayed.completedCount()==1;}),"delayed planner did not finish");
        require(!delayed.view(10.01).plan.candidate&&delayed.view(10.01).plan.stop&&
            delayed.view(10.01).plan.state=="OBS_AI_STALE","queue/compute latency did not expire route at publication");delayed.finish();

        RaceNavigationWorker actual(cfg,race,-1,10,20,out/"actual-navigation.jsonl");
        for(int i=1;i<=3;++i) {
            auto s=frame(i,1.+i*.07);s.image.setTo(cv::Scalar(0,0,0));cv::rectangle(s.image,{40,0,240,240},{255,255,255},-1);
            cv::rectangle(s.image,{60,100,30,60},{0,0,0},-1);s.detections={{LABEL_CONE,"cone",.95f,60,100,30,60}};
            actual.submit(s,s.captured);require(waitFor([&]{return actual.completedCount()==i;}),"actual planner did not complete matched frame");
        }
        view=actual.view(1.22);require(view.plan.candidate&&view.plan.state=="OBS_BYPASS_RIGHT_PREVIEW","background real planner missed route");actual.finish();

        std::atomic_int saving{0},badBinding{0};Release saveRelease;auto savePermit=saveRelease.promise.get_future().share();
        RaceSampleWriter writer([&](const RaceSampleWriter::Job &job) {
            ++saving;savePermit.wait();if(job.ai.sequence!=uint64_t(job.index)||job.ai.image.at<cv::Vec3b>(0,0)[0]!=job.index)++badBinding;return true;
        });
        OpenOnExit saveGuard{saveRelease};
        auto job=[&](int i) { return RaceSampleWriter::Job{out,frame(i),10.1,race,i,false,{},cfg}; };
        require(writer.submit(job(1)),"sample not accepted");
        if(!waitFor([&]{return saving.load()>0;})) { saveRelease.open();throw std::runtime_error("sample writer did not start"); }
        const bool second=writer.submit(job(2)),third=writer.submit(job(3)),overflow=writer.submit(job(4));
        saveRelease.open();writer.finish();
        require(second&&third&&!overflow&&writer.savedCount()==3&&writer.droppedCount()==1&&!badBinding,
            "sample queue blocked, grew unbounded, or mixed images");
        RaceSampleWriter disk;
        RaceSampleWriter::Job diskJob{out/"disk",view.ai,1.22,race,1,true,view.plan,cfg};
        diskJob.context={{"race_frame",82},{"ai_frame",view.ai.sequence},{"reason","WAIT_LEFT_SPLIT"}};
        diskJob.laneImage=cv::Mat(240,320,CV_8UC3,cv::Scalar(71,72,73));
        diskJob.laneBinary=cv::Mat(240,320,CV_8UC1,cv::Scalar(255));
        require(disk.submit(diskJob),"disk sample not queued");disk.finish();
        require(disk.savedCount()==1&&disk.failedCount()==0,"queued sample was not written");
        const auto saved=cv::imread((out/"disk/ai-sample-1/camera.png").string());require(cv::norm(saved,view.ai.image)==0,"queued image was changed");
        std::ifstream detFile(out/"disk/ai-sample-1/detections.json"),navFile(out/"disk/ai-sample-1/navigation.json");
        nlohmann::json det,nav;detFile>>det;navFile>>nav;
        require(det["frame"]==nav["frame"]&&det["captured"]==nav["captured"],"queued navigation saved with wrong AI image");
        std::ifstream contextFile(out/"disk/ai-sample-1/context.json");nlohmann::json context;contextFile>>context;
        require(context["race_frame"]==82&&context["ai_frame"]==view.ai.sequence,"fork context lost frame binding");
        const auto laneSaved=cv::imread((out/"disk/ai-sample-1/lane-camera.png").string());
        require(cv::norm(laneSaved,diskJob.laneImage)==0,"control image was confused with AI image");
        RaceForkEvidence evidence;
        require(!evidence.due("LANE","CRUISE",0,1),"ordinary lane floods fork evidence");
        require(evidence.due("FORK_ARMED","WAIT_LEFT_SPLIT",1,2),"fork entry not captured");
        evidence.queued("FORK_ARMED","WAIT_LEFT_SPLIT",1,2);
        require(!evidence.due("FORK_ARMED","WAIT_LEFT_SPLIT",1.1,3),"fork capture is not rate limited");
        require(evidence.due("FORK_ARMED","WAIT_LEFT_SPLIT",1.6,4),"fork periodic image missing");
        require(evidence.due("REVIEW_STOP","DIRECTION_MEMORY_EXPIRED",1.7,4),"fault transition capture suppressed");
        evidence.queued("REVIEW_STOP","DIRECTION_MEMORY_EXPIRED",1.7,4);
        require(!evidence.due("REVIEW_STOP","DIRECTION_MEMORY_EXPIRED",5,9),"latched stop floods evidence");
        RaceForkEvidence bounded;
        for(int i=0;i<64;++i)bounded.queued("FORK_ARMED","WAIT_LEFT_SPLIT",i,i+1);
        require(!bounded.due("FORK_ARMED","WAIT_LEFT_SPLIT",100,100),"fork evidence has no run budget");
        RaceSampleWriter badDisk([](const RaceSampleWriter::Job&)->bool {throw std::runtime_error("injected disk error");});
        badDisk.submit(job(1));badDisk.finish();require(badDisk.failedCount()==1,"sample exception escaped worker");
        std::cout<<"PASS background route queue, immutable matched frames, expiry, failures, bounded sample queue and disk persistence\n";
    }catch(const std::exception &e) { std::cerr<<"FAIL "<<e.what()<<'\n';return 1; }
}
