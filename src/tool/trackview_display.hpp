#pragma once
// 主线程独占 HighGUI；控制线程只发布只读快照，不等待远程绘图。
#include <atomic>
#include <chrono>
#include <deque>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <vector>

using Clock = std::chrono::steady_clock;
using std::atomic;
using std::atomic_bool;
using std::deque;
using std::lock_guard;
using std::mutex;
using std::unique_lock;
using std::vector;
using cv::Mat;
using std::chrono::duration_cast;
using std::chrono::milliseconds;

// Main thread alone owns HighGUI. Controller never waits for network rendering.
class TrackviewDisplay {
    mutex imageMutex, keyMutex;
    Mat latestImage, latestBinary;
    deque<int> keys;
    atomic_bool emergency{false};
    atomic<long long> progressMs{0};
    static long long clockMs() {
        return duration_cast<milliseconds>(Clock::now().time_since_epoch())
            .count();
    }

  public:
    atomic_bool finished{false}, failed{false};
    TrackviewDisplay() { progress(); }
    void progress() { progressMs.store(clockMs()); }
    long long ageMs() const { return clockMs() - progressMs.load(); }
    bool stalled(int timeoutMs = 2000) const { return ageMs() >= timeoutMs; }
    void publish(const Mat &image, const Mat &binary) {
        // Clone outside the lock; no growing queue and no rendering under this
        // lock.
        Mat a = image.clone(), b = binary.clone();
        unique_lock lock(imageMutex, std::try_to_lock);
        if (!lock.owns_lock())
            return;
        latestImage = std::move(a);
        latestBinary = std::move(b);
    }
    bool snapshot(Mat &image, Mat &binary) {
        std::lock_guard lock(imageMutex);
        if (latestImage.empty())
            return false;
        // Immutable reference-counted snapshots remain valid across
        // publications.
        image = latestImage;
        binary = latestBinary;
        return true;
    }
    void post(int key) {
        if (key < 0)
            return;
        if (key == ' ') {
            emergency = true;
            return;
        }
        lock_guard lock(keyMutex);
        if (keys.size() >= 32) {
            keys.clear();
            emergency = true;
        } else
            keys.push_back(key);
    }
    vector<int> takeKeys() {
        lock_guard lock(keyMutex);
        vector<int> result(keys.begin(), keys.end());
        keys.clear();
        if (emergency.exchange(false)) {
            result.clear();
            result.push_back(' ');
        }
        return result;
    }
};
