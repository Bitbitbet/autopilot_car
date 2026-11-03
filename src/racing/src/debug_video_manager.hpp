#include "camera.hpp"
#include <condition_variable>
#include <mutex>
#include <opencv2/opencv.hpp>

class DebugVideoManager {
  private:
    cv::VideoWriter writer;
    std::mutex mtx;
    std::condition_variable cond;
    cv::Mat latest;
    bool fresh = false;
    std::thread writer_thread;
    std::atomic_bool thread_stop;

    DebugVideoManager(cv::VideoWriter writer) : writer(writer) {}

    void writer_thread_main();

  public:
    static std::shared_ptr<DebugVideoManager> create(const Camera &camera);

    void publish(cv::Mat image);

    ~DebugVideoManager();
};