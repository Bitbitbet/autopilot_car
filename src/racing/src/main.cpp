#include "camera.hpp"
#include "control.hpp"
#include "joystick.hpp"
#include "pilot.hpp"
#include "race_display.hpp"
#include "race_manager.hpp"
#include "race_perception.hpp"
#include "terminal_keys.hpp"
#include "track.hpp"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <thread>

using Clock = std::chrono::steady_clock;
using cv::Mat;
using cv::Size;
using cv::VideoWriter;
using std::atomic_bool;
using std::cerr;
using std::clamp;
using std::condition_variable;
using std::cout;
using std::endl;
using std::lock_guard;
using std::max;
using std::memory_order_relaxed;
using std::min;
using std::mutex;
using std::shared_ptr;
using std::string;
using std::thread;
using std::unique_lock;
using std::unique_ptr;

namespace {

atomic_bool signal_stop{false};
void onSignal(int) { signal_stop = true; }

double nowSeconds() {
    return std::chrono::duration<double>(Clock::now().time_since_epoch())
        .count();
}

enum class Mode { Manual, Auto };

void drawHud(Mat &image, const AutoPilot &pilot, Mode mode, bool panic,
             float cruise, float speed, uint16_t servo, const RacePlan &race) {
    for (const auto &p : pilot.centerEdge)
        cv::circle(image, cv::Point(p.x, p.y), 2, cv::Scalar(0, 0, 255), -1);
    int y = 22;
    auto line = [&](const std::string &s) {
        cv::putText(image, s, {8, y}, cv::FONT_HERSHEY_PLAIN, 1.2,
                    {0, 255, 255}, 1);
        y += 20;
    };
    char text[160];
    snprintf(text, sizeof text, "%s%s%s  speed=%.2f limit=%.2f pwm=%u",
             mode == Mode::Auto ? "AUTO" : "MANUAL", panic ? " STOP" : "",
             pilot.lost ? " LOST" : "", speed, cruise, servo);
    line(text);
    snprintf(text, sizeof text, "center=%d err=%.2f quality=%.2f curve=%.2f",
             pilot.center, pilot.error, pilot.confidence, pilot.severity);
    line(text);
    snprintf(text, sizeof text, "race=%s ai_age=%.2fs side=%d", race.state,
             race.detectionAge, race.branchSide);
    line(text);
    line(race.reason);
    line("a:auto  space:stop  +/-:limit  q:quit");
}
struct ParsedArgument {
    bool headless = false;
    bool do_exit = false;
    bool do_produce_debug_video = false;
};
ParsedArgument process_arguments(int argc, char **argv) {
    ParsedArgument arg;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--headless"))
            arg.headless = true;
        else if (!std::strcmp(argv[i], "--debug-video")) {
            arg.do_produce_debug_video = true;

        } else {
            cerr << "用法: racing [--headless] [--debug-video]\n";
            arg.do_exit = true;
        }
    }

    return arg;
}

class DebugVideoManager {
  private:
    VideoWriter writer;
    mutex mtx;
    condition_variable cond;
    Mat latest;
    bool fresh = false;
    thread writer_thread;
    atomic_bool thread_stop;

    DebugVideoManager(VideoWriter writer) : writer(writer) {}

    void writer_thread_main() {
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

  public:
    static shared_ptr<DebugVideoManager> create(const Camera &camera) {
        const char *DEBUG_VIDEO_PATH = "/tmp/output.avi";
        cout << "Debug视频输出路径：" << DEBUG_VIDEO_PATH << "。" << endl;

        cv::VideoWriter debug_video(
            DEBUG_VIDEO_PATH, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30,
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

    void publish(Mat image) {
        lock_guard lock(mtx);
        latest = image.clone();
        fresh = true;

        cond.notify_one();
    }

    ~DebugVideoManager() {
        thread_stop.store(true, memory_order_relaxed);
        cond.notify_one();
        writer_thread.join();
        writer.release();
    }
};

} // namespace

int main(int argc, char **argv) {
    const auto [headless, do_exit, do_produce_debug_video] =
        process_arguments(argc, argv);
    if (do_exit) {
        return 2;
    }

    signal(SIGINT, onSignal);
    signal(SIGTERM, onSignal);

    Camera camera;

    unique_ptr<RacePerception> perception;
    try {
        perception = createRacePerception(RaceConfig{});
    } catch (const std::exception &e) {
        cerr << "[Error] 初始化失败: " << e.what() << '\n';
        return 1;
    }

    // 台架上没有串口时只跑图像和决策，不驱动车轮。
    auto car = CarControl::create();
    if (!car) {
        cerr << "[Fatal] 打开串口失败" << endl;
        return 3;
    }

    atomic_bool joyLost{false};
    auto js = JoyStick::create([&joyLost] { joyLost = true; });
    if (!js)
        cout << "[Warn] 没有手柄。\n";

    AutoPilot pilot(camera.cols(), camera.rows());
    RaceManager race_manager;

    Mode mode = Mode::Manual;
    bool panic = false;
    float cruise = cfg::CRUISE_SPEED, jsSpeed = 0, speed = 0;
    uint16_t jsServo = PWMSERVOMID;
    TrackRows rows;
    double last = nowSeconds();
    string lastState;
    cout << "[Start] " << (headless ? "HEADLESS " : "") << camera.cols() << "x"
         << camera.rows()
         << " | 默认 MANUAL，按 a 启动（终端或图像窗口都行，q 退出）\n";

    TerminalKeys terminal; // 终端也能按键：headless 下同样可以操作

    // 窗口单独一个线程：控制回路只丢图，不等 X11
    RaceDisplay display;
    thread displayThread;
    if (!headless)
        displayThread = thread([&] { display.run(); });

    shared_ptr<DebugVideoManager> dvm;
    if (do_produce_debug_video) {
        dvm = DebugVideoManager::create(camera);
        if (!dvm) {
            cerr << "启动debug视频失败。" << endl;
            return 5;
        }
    }

    while (!signal_stop) {
        Mat image;
        if (!camera.read(image)) {
            cerr << "[Error] 取不到图像\n";
            break;
        }
        Mat bin = camera.binarize(image);
        track_identify_from_bin(bin, camera, rows);

        const double t = nowSeconds();
        float dt = float(t - last);
        last = t;

        perception->submit(image, uint64_t(t * 1000), t);
        const DetectionSnapshot observation = perception->snapshot();
        RacePlan plan =
            race_manager.update(observation, t, dt, mode == Mode::Auto);

        auto toggle = [&] {
            if (mode == Mode::Auto) {
                mode = Mode::Manual;
                jsSpeed = 0;
                pilot.resetControl();
                cout << "[Mode] MANUAL\n";
            } else if (pilot.valid && race_manager.aiReady(observation, t) &&
                       !joyLost && (!js || !js->isStopped()) &&
                       std::fabs(jsSpeed) <= 0.05f) {
                mode = Mode::Auto;
                panic = false;
                jsSpeed = 0;
                pilot.resetControl();
                cout << "[Mode] AUTO\n";
            } else if (!race_manager.aiReady(observation, t)) {
                cout << "[Wait] 还没有可用的检测结果\n";
            } else if (!pilot.valid) {
                cout << "[Wait] 路径未稳定: " << pilot.pathIssue << '\n';
            } else {
                cout << "[Wait] 先松开油门或重新连接手柄\n";
            }
        };

        int key = terminal.poll();
        if (key < 0)
            key = display.takeKey(); // 窗口线程收上来的键

        if (key == 27 || key == 'q' || key == 'Q')
            break;
        if (key == 'a' || key == 'A') {
            toggle();
        } else if (key == ' ') {
            panic = true;
            mode = Mode::Manual;
            jsSpeed = 0;
            pilot.resetControl();
            cout << "[Stop] 键盘停车\n";
        } else if (key == '+' || key == '=') {
            cruise = min(cfg::MAX_SPEED, cruise + 0.05f);
            cout << "[Set] limit=" << cruise << " m/s\n";
        } else if (key == '-' || key == '_') {
            cruise = max(cfg::MIN_SPEED, cruise - 0.05f);
            cout << "[Set] limit=" << cruise << " m/s\n";
        } else if (key == '[' || key == ']') {
            pilot.setCurveSpeed(pilot.curveSpeed() +
                                (key == ']' ? 0.05f : -0.05f));
            cout << "[Set] curve=" << pilot.curveSpeed() << " m/s\n";
        } else if (key == ',' || key == '.') {
            pilot.setDegradedSpeed(pilot.degradedSpeed() +
                                   (key == '.' ? 0.05f : -0.05f));
            cout << "[Set] quality=" << pilot.degradedSpeed() << " m/s\n";
        } else if (key == 'w' || key == 'W') {
            pilot.cycleWeight();
            cout << "[Set] weight=" << pilot.weightName() << '\n';
        }

        if (js) {
            float s = 0;
            uint16_t sv = PWMSERVOMID;
            const bool newCommand = js->takeCarControl(s, sv);
            if (newCommand) {
                jsSpeed = s;
                jsServo = sv;
            }
            if (js->takeSampleOnce())
                toggle();
            js->takeBuzzer();
            if (js->isStopped() || joyLost) {
                if (!panic)
                    cout << "[Stop] 手柄断开\n";
                panic = true;
                mode = Mode::Manual;
                jsSpeed = 0;
                pilot.resetControl();
            } else if (mode == Mode::Auto && newCommand &&
                       std::fabs(jsSpeed) > 0.05f) {
                mode = Mode::Manual;
                jsSpeed = 0;
                pilot.resetControl();
                cout << "[Mode] MANUAL（手柄介入）\n";
            }
        }

        pilot.update(rows, camera, dt, speed, plan.branchSide);
        if (mode == Mode::Auto && pilot.lost) {
            mode = Mode::Manual;
            panic = true;
            jsSpeed = 0;
            pilot.resetControl();
            cout << "[Stop] 丢线: " << pilot.pathIssue << " (dt=" << dt
                 << " s)；恢复后按 a\n";
        }

        speed = 0;
        uint16_t servo = PWMSERVOMID;
        if (!panic) {
            if (mode == Mode::Auto) {
                if (plan.stop) {
                    pilot.resetControl();
                } else {
                    speed = pilot.speedFor(min(cruise, plan.speedCap), dt);
                    servo = pilot.servo;
                }
            } else {
                speed = clamp(jsSpeed, -cfg::MAX_SPEED, cfg::MAX_SPEED);
                servo = jsServo;
            }
        }

        // 最终下发指令
        try {
            car->carControl(speed, servo);
        } catch (const std::exception &e) {
            cerr << "[Error] 串口: " << e.what() << '\n';
            break;
        }

        if (lastState != plan.state) {
            cout << "[Race] " << plan.state << " | " << plan.reason << '\n';
            lastState = plan.state;
        }

        if (!headless || do_produce_debug_video) {
            track_draw_result(rows, image);
            drawHud(image, pilot, mode, panic, cruise, speed, servo, plan);
            if (!headless)
                display.publish(image);
            if (do_produce_debug_video)
                dvm->publish(image);
        }
    }

    try {
        car->latch_stop();
    } catch (...) {
    }

    display.stop = true;
    if (!headless)
        displayThread.join();
    cout << "racing 正常退出。" << endl;
    return 0;
}
