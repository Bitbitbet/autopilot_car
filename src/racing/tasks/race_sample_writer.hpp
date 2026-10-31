#pragma once
#include "race_preview.hpp"
#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <iostream>
#include <thread>

// 图像编码和磁盘写入由专用线程处理，允许低速实跑时按 s 取证。
class RaceSampleWriter {
public:
    struct Job {
        std::filesystem::path directory;
        RaceSnapshot ai;
        double time=0;
        RaceConfig race;
        int index=0;
        bool hasNavigation=false;
        RaceObstaclePreviewPlan navigation;
        RaceNavigationConfig navigationConfig;
        nlohmann::json context;
        cv::Mat laneImage,laneBinary;
    };
    using Sink=std::function<bool(const Job&)>;
private:
    std::mutex mutex;
    std::condition_variable wake;
    std::deque<Job> pending;
    Sink sink;
    bool stopping=false;
    std::atomic_int saved{0},failed{0},dropped{0};
    std::thread worker;
    void run() noexcept {
        for(;;) {
            Job job;
            {
                std::unique_lock lock(mutex);wake.wait(lock,[&]{return stopping||!pending.empty();});
                if(pending.empty())return;
                job=std::move(pending.front());pending.pop_front();
            }
            try {
                const bool ok=sink?sink(job):saveRaceSample(job.directory,job.ai,job.time,job.race,job.index,
                    job.hasNavigation?&job.navigation:nullptr,job.navigationConfig);
                if(!ok)throw std::runtime_error("Sample writer could not save matched image");
                if(!sink&&!job.context.empty()) {
                    const auto path=job.directory/("ai-sample-"+std::to_string(job.index));
                    std::ofstream metadata(path/"context.json");metadata<<job.context.dump(2);
                    if(!metadata)throw std::runtime_error("Cannot save sample context");
                    if(!job.laneImage.empty()&&!cv::imwrite((path/"lane-camera.png").string(),job.laneImage))
                        throw std::runtime_error("Cannot save current lane image");
                    if(!job.laneBinary.empty()&&!cv::imwrite((path/"lane-binary.png").string(),job.laneBinary))
                        throw std::runtime_error("Cannot save current lane binary");
                }
                ++saved;std::cout<<"[Sample] Saved ai-sample-"<<job.index<<'\n';
            }catch(const std::exception &e) { ++failed;std::cerr<<"[Sample] "<<e.what()<<'\n'; }
            catch(...) { ++failed;std::cerr<<"[Sample] Unexpected save failure\n"; }
        }
    }
public:
    explicit RaceSampleWriter(Sink fn={}):sink(std::move(fn)),worker([this]{run();}) {}
    ~RaceSampleWriter() { finish(); }
    bool submit(Job job) {
        if(job.directory.empty()||job.ai.image.empty()||job.index<=0)return false;
        {
            std::lock_guard lock(mutex);
            if(stopping||pending.size()>=2) { ++dropped;return false; }
            pending.push_back(std::move(job));
        }
        wake.notify_one();return true;
    }
    void finish() {
        if(!worker.joinable())return;
        { std::lock_guard lock(mutex);stopping=true; }
        wake.notify_one();worker.join();
    }
    int savedCount() const { return saved.load(); }
    int failedCount() const { return failed.load(); }
    int droppedCount() const { return dropped.load(); }
};
