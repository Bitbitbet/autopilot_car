#pragma once
#include <atomic>
#include <chrono>
#include <deque>
#include <mutex>
#include <opencv2/core.hpp>
#include <vector>

// 主线程独占图形界面操作，控制线程不等待远程绘图。
class TrackviewDisplay {
    std::mutex imageMutex, keyMutex;
    cv::Mat latestImage, latestBinary;
    uint64_t published=0,shown=0;
    long long lastPublishedMs=-1000;
    std::deque<int> keys;
    std::atomic_bool emergency{false};
    std::atomic<long long> progressMs{0};
    static long long clockMs() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }
public:
    std::atomic_bool finished{false}, failed{false};
    TrackviewDisplay() { progress(); }
    void progress() { progressMs.store(clockMs()); }
    long long ageMs() const { return clockMs()-progressMs.load(); }
    bool stalled(int timeoutMs=2000) const { return ageMs()>=timeoutMs; }
    // One producer, one GUI consumer. Avoid constructing/copying HUDs at camera FPS.
    bool frameDue(int intervalMs=100) const {return clockMs()-lastPublishedMs>=intervalMs;}
    void publish(const cv::Mat &image,const cv::Mat &binary) {
        // 在锁外复制图像；只保留最新帧，不积压队列，也不在持锁期间绘图。
        cv::Mat a=image.clone(), b=binary.clone();
        std::unique_lock lock(imageMutex,std::try_to_lock);
        if (!lock.owns_lock()) return;
        latestImage=std::move(a); latestBinary=std::move(b);++published;lastPublishedMs=clockMs();
    }
    bool snapshot(cv::Mat &image,cv::Mat &binary) {
        std::lock_guard lock(imageMutex);
        if (latestImage.empty()||shown==published) return false;
        // 快照通过引用计数保持有效，发布新图像不会使旧快照失效。
        image=latestImage; binary=latestBinary;shown=published;
        return true;
    }
    void post(int key) {
        if(key<0) return;
        if(key==' ') { emergency=true; return; }
        std::lock_guard lock(keyMutex);
        if(keys.size()>=32) { keys.clear(); emergency=true; }
        else keys.push_back(key);
    }
    std::vector<int> takeKeys() {
        std::lock_guard lock(keyMutex);
        std::vector<int> result(keys.begin(),keys.end()); keys.clear();
        if(emergency.exchange(false)) { result.clear(); result.push_back(' '); }
        return result;
    }
};
