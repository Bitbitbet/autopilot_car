#pragma once
#include <atomic>
#include <chrono>
#include <deque>
#include <mutex>
#include <opencv2/core.hpp>
#include <vector>

// Main thread alone owns HighGUI. Controller never waits for network rendering.
class TrackviewDisplay {
    std::mutex imageMutex, keyMutex;
    cv::Mat latestImage, latestBinary;
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
    void publish(const cv::Mat &image,const cv::Mat &binary) {
        // Clone outside the lock; no growing queue and no rendering under this lock.
        cv::Mat a=image.clone(), b=binary.clone();
        std::unique_lock lock(imageMutex,std::try_to_lock);
        if (!lock.owns_lock()) return;
        latestImage=std::move(a); latestBinary=std::move(b);
    }
    bool snapshot(cv::Mat &image,cv::Mat &binary) {
        std::lock_guard lock(imageMutex);
        if (latestImage.empty()) return false;
        // Immutable reference-counted snapshots remain valid across publications.
        image=latestImage; binary=latestBinary;
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
