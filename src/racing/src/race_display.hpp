#pragma once

#include <atomic>
#include <deque>
#include <mutex>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

/**
 * @brief HighGUI线程
 */
class RaceDisplay {
    std::mutex mutex;
    cv::Mat latest;
    bool fresh = false;
    std::deque<int> keys;

  public:
    std::atomic_bool stop{false};

    void publish(const cv::Mat &image) {
        std::lock_guard lock(mutex);
        latest = image;
        fresh = true;
    }

    /**
     * @brief 取窗口里按下的键，没有返回 -1
     */
    int takeKey() {
        std::lock_guard lock(mutex);
        if (keys.empty())
            return -1;
        const int key = keys.front();
        keys.pop_front();
        return key;
    }

    void run() {
        try {
            loop();
        } catch (...) {
        }
        try {
            cv::destroyAllWindows();
        } catch (...) {
        }
    }

  private:
    void loop() {
        while (!stop.load()) {
            cv::Mat image;
            {
                std::lock_guard lock(mutex);
                if (fresh) {
                    image = latest;
                    fresh = false;
                }
            }
            if (image.empty()) {
                cv::waitKey(5);
                continue;
            }
            cv::imshow("track_view", image);
            const int key = cv::waitKey(1);
            if (key >= 0) {
                std::lock_guard lock(mutex);
                keys.push_back(key);
            }
        }
    }
};
