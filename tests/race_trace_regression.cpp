#include "race_trace.hpp"
#include <future>
static void check(bool ok,const char *why) { if(!ok) throw std::runtime_error(why); }
int main(int argc,char **argv) {
    try {
        if(argc!=2) return 2;
        auto directory=std::filesystem::path(argv[1]);
        RaceConfig cfg; Track track;
        cv::Mat bin(240,320,CV_8UC1,cv::Scalar(255)); track.handle(bin);
        cv::Mat rgb(240,320,CV_8UC3,cv::Scalar(40,50,60));
        RaceSnapshot ai{2,.1,{},false,{},rgb.clone()};
        RacePlan lane; lane.fresh=true; lane.state="LANE";
        RacePlan fork=lane; fork.state="FORK_WAIT"; fork.reason="NO_CONFIRMED_DIRECTION"; fork.stop=true;
        {
            RaceTrace trace(directory,cfg);
            for(int n=1;n<=7;++n) trace.observe(n,n*.07,rgb,bin,track,ai,n==4?fork:lane);
            rgb.setTo(cv::Scalar(200,200,200)); bin.setTo(0); ai.image.setTo(0);
            trace.finish();
            check(trace.savedCount()==1&&trace.failedCount()==0,"save one complete episode");
        }
        std::ifstream file(directory/"fork-trace-1/trace.json"); nlohmann::json meta; file>>meta;
        check(meta["trigger_frame"]==4&&meta["frames"].size()==7,"three before, trigger, three after");
        check(meta["frames"][3]["frame"]==4&&meta["frames"][3]["ai"]["frame"]==2,"different frame ids preserved");
        auto saved=cv::imread((directory/"fork-trace-1/4-camera.png").string());
        check(saved.at<cv::Vec3b>(0,0)[0]==40,"capture image cloned before overlays or reuse");
        saved=cv::imread((directory/"fork-trace-1/ai-2.png").string());
        check(saved.at<cv::Vec3b>(0,0)[0]==40,"AI image preserved independently");
        rgb.setTo(cv::Scalar(40,50,60)); bin.setTo(255); track.handle(bin);
        std::promise<void> entered,release;
        auto gate=release.get_future().share();
        RaceTrace slow(directory,cfg,[&](const RaceTrace::Episode&) {
            static bool first=true;
            if(first) { first=false; entered.set_value(); gate.wait(); }
        });
        for(int n=1;n<=4;++n) slow.observe(n,n*.07,rgb,bin,track,ai,n==1?fork:lane);
        check(entered.get_future().wait_for(std::chrono::seconds(3))==std::future_status::ready,"writer started");
        const auto start=std::chrono::steady_clock::now();
        for(int n=10;n<100;++n) slow.observe(n,n*.07,rgb,bin,track,ai,n%2?fork:lane);
        const auto elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        // 先释放写线程，避免断言失败时测试本身在析构中等待。
        release.set_value(); slow.finish();
        check(elapsed<2.0,"slow writer must not block capture loop");
        check(slow.droppedCount()>0,"bounded queue reports overflow");
        check(slow.savedCount()+slow.droppedCount()<=12,"bounded episode count");
        int partialFrames=0;
        {
            RaceTrace partial(directory,cfg,[&](const RaceTrace::Episode &e) { partialFrames=int(e.frames.size()); });
            partial.observe(1,.1,rgb,bin,track,ai,fork); partial.finish();
        }
        check(partialFrames==1,"shutdown flushes partial episode");
        {
            RaceTrace failure(directory,cfg,[](const RaceTrace::Episode&) { throw std::runtime_error("simulated disk fault"); });
            failure.observe(1,.1,rgb,bin,track,ai,fork); failure.finish();
            check(failure.failedCount()==1,"disk fault remains diagnostic only");
        }
        std::cout<<"PASS fork trace matching, before/after, immutable images, slow writer, bounded queue, partial exit and disk fault\n";
    } catch(const std::exception &e) { std::cerr<<"FAIL "<<e.what()<<'\n'; return 1; }
}
