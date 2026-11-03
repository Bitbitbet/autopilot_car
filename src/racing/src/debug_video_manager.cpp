#include "debug_video_manager.hpp"

using cv::Mat;
using cv::Size;
using std::cerr;
using std::cout;
using std::endl;
using std::lock_guard;
using std::memory_order_relaxed;
using std::shared_ptr;
using std::thread;
using std::unique_lock;

void DebugVideoManager::writer_thread_main() {
    while (!thread_stop.load(memory_order_relaxed)) {
        Mat image;
        {
            unique_lock lock(mtx);
            cond.wait(lock, [this]() {
                return fresh || thread_stop.load(memory_order_relaxed);
            });
            if (!fresh) {
                break;
            }
            image = latest;
        }
        writer.write(image);

        fresh = false;
    }
}

shared_ptr<DebugVideoManager> DebugVideoManager::create(const Camera &camera) {
    const char *DEBUG_VIDEO_PATH = "/tmp/output.avi";
    cout << "Debug视频输出路径：" << DEBUG_VIDEO_PATH << "。" << endl;

    cv::VideoWriter debug_video(DEBUG_VIDEO_PATH,
                                cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30,
                                Size{camera.cols(), camera.rows()}, true);
    if (!debug_video.isOpened()) {
        cerr << "无法打开debug视频文件。" << endl;
        return nullptr;
    }

    auto dvm =
        shared_ptr<DebugVideoManager>(new DebugVideoManager(debug_video));
    dvm->thread_stop.store(false, memory_order_relaxed);
    auto raw = dvm.get();
    dvm->writer_thread = thread([raw]() { raw->writer_thread_main(); });

    return dvm;
}
void DebugVideoManager::publish(Mat image) {
    lock_guard lock(mtx);
    latest = image.clone();
    fresh = true;

    cond.notify_one();
}
DebugVideoManager::~DebugVideoManager() {
    thread_stop.store(true, memory_order_relaxed);
    cond.notify_one();
    writer_thread.join();
    writer.release();
}