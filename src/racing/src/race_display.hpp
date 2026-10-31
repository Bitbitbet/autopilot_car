#pragma once

#include <atomic>
#include <deque>
#include <mutex>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <thread>

/**
 * @brief 把 HighGUI 挪到单独线程
 *
 * 控制回路只往这里丢最新一帧（共享 Mat 头，零拷贝），绝不等 X11：
 * imshow/waitKey 在窗口服务器慢、窗口没被服务时能卡到秒级，
 * 放在主线程里会把 dt 拉长，进而误判 FRAME_GAP/丢线。
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
            // X11 出问题就只丢显示，别把控制进程带崩
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
