#pragma once
#include "race_obstacle.hpp"
#include "race_ground_obstacle.hpp"
#include "../predeal/include/road_binary.hpp"
#include <atomic>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <functional>
#include <mutex>
#include <optional>
#include <thread>

// 默认只作旁路观测；显式绕行实验使用独立米制规划器。队列有界，不阻塞串口控制。
class RaceNavigationWorker {
public:
    struct View { RaceSnapshot ai;RaceObstaclePreviewPlan plan; };
    using Process=std::function<RaceObstaclePreviewPlan(const RaceSnapshot&,double)>;
private:
    struct Job { RaceSnapshot ai;double time,queuedWall; };
    RaceNavigationConfig config;
    RaceConfig race;
    int threshold,rowUp,rowBottom;
    bool blackRoad;
    std::filesystem::path logPath;
    Process process;
    std::optional<RaceGroundCalibration> ground;
    std::mutex mutex;
    std::condition_variable wake;
    std::optional<Job> pending;
    View latest;
    uint64_t submitted=0;
    bool stopping=false;
    std::atomic_int dropped{0},completed{0};
    std::atomic_int groundControlState{0}; // 0 MANUAL，1 AUTO待启动，2 已进入绕行
    std::thread worker;

    void run() noexcept {
        RaceObstaclePreview planner(config);Track track;track.rowCutUp=rowUp;track.rowCutBottom=rowBottom;
        std::optional<RaceGroundObstacle> calibrated;if(ground)calibrated.emplace(*ground);
        std::ofstream output;
        if(!logPath.empty())output.open(logPath);
        for(;;) {
            Job job;
            {
                std::unique_lock lock(mutex);wake.wait(lock,[&]{return stopping||pending.has_value();});
                if(!pending)return;
                job=std::move(*pending);pending.reset();
            }
            RaceObstaclePreviewPlan plan;
            const auto started=raceTime();
            try {
                if(process)plan=process(job.ai,job.time);
                else {
                    cv::Mat binary=roadBinary(job.ai.image,threshold,blackRoad);
                    const int control=groundControlState.load();
                    track.handle(binary);plan=calibrated?calibrated->update(job.ai,job.time,binary,track,race,control==2,control!=0):
                        planner.update(job.ai,job.time,binary,track,race);
                }
                plan.sequence=job.ai.sequence;plan.captured=job.ai.captured;
                const double ageAtPublish=job.time-job.ai.captured+raceTime()-job.queuedWall;
                if(job.ai.failed||ageAtPublish<0||ageAtPublish>race.maxDetectionAge) {
                    plan.fresh=false;plan.stop=true;plan.candidate=false;plan.state="OBS_AI_STALE";
                    plan.centers.clear();plan.left.clear();plan.right.clear();
                }
                if(output) {
                    auto row=raceObstacleJson(plan,config);row["worker_ms"]=(raceTime()-started)*1000;
                    row["age_at_publish_s"]=ageAtPublish;
                    row["dropped_pending_frames"]=dropped.load();output<<row.dump()<<'\n';output.flush();
                }
            }catch(const std::exception &e) {
                plan={};plan.sequence=job.ai.sequence;plan.captured=job.ai.captured;
                plan.state="OBS_PLANNER_FAILED";plan.reason=e.what();
            }catch(...) {
                plan={};plan.sequence=job.ai.sequence;plan.captured=job.ai.captured;
                plan.state="OBS_PLANNER_FAILED";plan.reason="Unknown navigation worker error";
            }
            { std::lock_guard lock(mutex);latest={std::move(job.ai),std::move(plan)}; }
            ++completed;
        }
    }
public:
    RaceNavigationWorker(const RaceNavigationConfig &cfg,const RaceConfig &rc,int binary,int up,int bottom,
                         std::filesystem::path log={},Process fn={},std::optional<RaceGroundCalibration> calibration={},bool darkRoad=false)
        :config(cfg),race(rc),threshold(binary),rowUp(up),rowBottom(bottom),blackRoad(darkRoad),logPath(std::move(log)),process(std::move(fn)),ground(std::move(calibration)) {
        config.validate();race.validate();
        if(binary< -1||binary>255||up<0||up>60||bottom<0||bottom>60)throw std::invalid_argument("Invalid navigation worker image settings");
        worker=std::thread([this]{run();});
    }
    ~RaceNavigationWorker() { finish(); }
    void setControlState(bool automatic,bool committed) {groundControlState.store(automatic?(committed?2:1):0);}
    // 输入是感知线程发布的只读快照，其图像用引用计数保留，不复制整张图或等待推理。
    bool submit(const RaceSnapshot &s,double now) {
        if(!s.sequence||s.image.type()!=CV_8UC3||s.image.size()!=cv::Size(320,240))return false;
        {
            std::lock_guard lock(mutex);
            if(stopping||s.sequence<=submitted)return false;
            submitted=s.sequence;if(pending)++dropped;pending=Job{s,now,raceTime()};
        }
        wake.notify_one();return true;
    }
    View view(double now) {
        std::lock_guard lock(mutex);auto out=latest;
        const bool fresh=!out.ai.failed&&out.ai.sequence&&std::isfinite(now)&&now>=out.ai.captured&&now-out.ai.captured<=race.maxDetectionAge;
        if(!fresh) {
            out.plan.fresh=false;out.plan.stop=true;out.plan.candidate=false;
            out.plan.centers.clear();out.plan.left.clear();out.plan.right.clear();
            out.plan.state=out.ai.sequence?"OBS_AI_STALE":"OBS_WORKER_WAIT";
        }
        return out;
    }
    void finish() {
        if(!worker.joinable())return;
        { std::lock_guard lock(mutex);stopping=true; }
        wake.notify_one();worker.join();
    }
    int droppedCount() const { return dropped.load(); }
    int completedCount() const { return completed.load(); }
};
