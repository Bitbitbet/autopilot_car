#pragma once
#include "race_manager.hpp"
#include "race_preview.hpp"
#include <atomic>
#include <deque>
#include <iostream>
#include <optional>
#include <set>

// 仅由 --no-control 入口启用。采集线程只复制小图，编码和文件写入在独立线程完成。
class RaceTrace {
public:
    struct Frame {
        uint64_t number=0;
        double time=0;
        cv::Mat camera,binary;
        RaceSnapshot ai;
        std::string state,reason;
        bool stop=false;
        int rawSplitRows=0,maskedSplitRows=0,rowCutUp=0,rowCutBottom=0;
        int colorFilteredSplitRows=0,colorRejectedComponents=0;
        std::vector<PointX> left,right;
        std::vector<cv::Rect> signRegions;
    };
    struct Episode { int index=0; uint64_t trigger=0; std::vector<Frame> frames; };
    using Sink=std::function<void(const Episode&)>;
private:
    std::filesystem::path directory;
    RaceConfig config;
    Sink sink;
    std::deque<Frame> recent;
    std::optional<Episode> collecting;
    int remaining=0,started=0;
    double lastTrigger=-100;
    std::string previous;
    std::mutex mutex;
    std::condition_variable wake;
    std::deque<Episode> queue;
    bool stopping=false;
    std::atomic_int saved{0},failed{0},dropped{0};
    std::thread worker;

    void enqueue() {
        if(!collecting) return;
        {
            std::lock_guard lock(mutex);
            if(queue.size()<2) queue.push_back(std::move(*collecting));
            else ++dropped;
        }
        collecting.reset(); wake.notify_one();
    }
    void write(const Episode &episode) {
        const auto path=directory/("fork-trace-"+std::to_string(episode.index));
        std::filesystem::create_directories(path);
        nlohmann::json frames=nlohmann::json::array();
        std::set<uint64_t> images;
        auto save=[&](const std::string &name,const cv::Mat &image) {
            if(!cv::imwrite((path/name).string(),image)) throw std::runtime_error("Cannot save fork trace image");
        };
        for(const auto &f:episode.frames) {
            const auto prefix=std::to_string(f.number);
            save(prefix+"-camera.png",f.camera); save(prefix+"-binary.png",f.binary);
            const auto aiName="ai-"+std::to_string(f.ai.sequence)+".png";
            if(!f.ai.image.empty()&&images.insert(f.ai.sequence).second) save(aiName,f.ai.image);
            nlohmann::json left=nlohmann::json::array(),right=left,regions=left;
            for(const auto &p:f.left) left.push_back({p.x,p.y});
            for(const auto &p:f.right) right.push_back({p.x,p.y});
            for(const auto &r:f.signRegions) regions.push_back({r.x,r.y,r.width,r.height});
            frames.push_back({{"frame",f.number},{"time",f.time},{"state",f.state},{"reason",f.reason},
                {"stop",f.stop},{"raw_split_rows",f.rawSplitRows},{"masked_split_rows",f.maskedSplitRows},
                {"color_filtered_split_rows",f.colorFilteredSplitRows},{"red_cap_components",f.colorRejectedComponents},
                {"row_cut_up",f.rowCutUp},{"row_cut_bottom",f.rowCutBottom},{"left",left},{"right",right},
                {"sign_regions",regions},{"camera",prefix+"-camera.png"},{"binary",prefix+"-binary.png"},
                {"ai_image",f.ai.image.empty()?"":aiName},
                {"ai",raceDetectionJson(f.ai,f.time,config.confidence)}});
        }
        std::ofstream file(path/"trace.json");
        file<<nlohmann::json({{"trigger_frame",episode.trigger},{"frames",frames},
            {"race_config",config},{"point_order","row,col"}}).dump(2);
        if(!file) throw std::runtime_error("Cannot save fork trace metadata");
        std::cout<<"[Trace] Saved "<<path<<'\n';
    }
    void run() noexcept {
        for(;;) {
            Episode job;
            {
                std::unique_lock lock(mutex);
                wake.wait(lock,[&] { return stopping||!queue.empty(); });
                if(queue.empty()) return;
                job=std::move(queue.front()); queue.pop_front();
            }
            try { if(sink) sink(job); else write(job); ++saved; }
            catch(const std::exception &e) { ++failed; std::cerr<<"[Trace] "<<e.what()<<'\n'; }
            catch(...) { ++failed; }
        }
    }
public:
    explicit RaceTrace(std::filesystem::path path,const RaceConfig &cfg,Sink output={})
        :directory(std::move(path)),config(cfg),sink(std::move(output)),worker([this] { run(); }) {}
    ~RaceTrace() { finish(); }
    RaceTrace(const RaceTrace&)=delete;
    RaceTrace &operator=(const RaceTrace&)=delete;
    void observe(uint64_t number,double time,const cv::Mat &image,const cv::Mat &bin,
                 const Track &track,const RaceSnapshot &snapshot,const RacePlan &plan) {
        if(started>=12&&!collecting) return;
        Frame f; f.number=number; f.time=time; f.camera=image.clone(); f.binary=bin.clone();
        f.ai=snapshot; f.ai.image=snapshot.image.clone();
        f.state=plan.state; f.reason=plan.reason; f.stop=plan.stop;
        f.rowCutUp=track.rowCutUp; f.rowCutBottom=track.rowCutBottom;
        f.left=track.pointsEdgeLeft; f.right=track.pointsEdgeRight;
        if(plan.fresh) f.signRegions=raceSignRegions(snapshot,config);
        f.rawSplitRows=raceBranchPath(bin,track,0,config).splitRows;
        f.maskedSplitRows=raceBranchPath(bin,track,0,config,f.signRegions).splitRows;
        f.colorFilteredSplitRows=plan.geometrySplitRows;
        f.colorRejectedComponents=plan.colorRejectedComponents;
        if(collecting) {
            collecting->frames.push_back(f);
            if(--remaining==0) enqueue();
        }
        if(f.state=="FORK_WAIT"&&previous!="FORK_WAIT"&&!collecting&&
           started<12&&time-lastTrigger>=.5) {
            collecting=Episode{++started,number,{recent.begin(),recent.end()}};
            collecting->frames.push_back(f); remaining=3; lastTrigger=time;
        }
        previous=f.state; recent.push_back(std::move(f));
        if(recent.size()>3) recent.pop_front();
    }
    // 采集线程结束后调用。退出时保留尚未凑齐后续帧的最后一次事件。
    void finish() {
        if(!worker.joinable()) return;
        enqueue();
        { std::lock_guard lock(mutex); stopping=true; }
        wake.notify_one(); worker.join();
    }
    int savedCount() const { return saved.load(); }
    int failedCount() const { return failed.load(); }
    int droppedCount() const { return dropped.load(); }
};
