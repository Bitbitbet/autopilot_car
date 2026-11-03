#include "camera.hpp"
#include "control.hpp"
#include "debug_video_manager.hpp"
#include "joystick.hpp"
#include "odometry.hpp"
#include "pilot.hpp"
#include "race_display.hpp"
#include "race_manager.hpp"
#include "race_perception.hpp"
#include "terminal_keys.hpp"
#include "timing.hpp"
#include "track.hpp"
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <thread>

using Clock = std::chrono::steady_clock;
using cv::Mat;
using cv::Rect;
using cv::Size;
using std::atomic_bool;
using std::cerr;
using std::clamp;
using std::cout;
using std::endl;
using std::max;
using std::memory_order_relaxed;
using std::min;
using std::shared_ptr;
using std::string;
using std::thread;
using std::unique_ptr;
using std::vector;

namespace {

atomic_bool signal_stop{false};
void onSignal(int) { signal_stop = true; }

enum class Mode { Manual, Auto };

void draw_hud(Mat &image, const AutoPilot &pilot, Mode mode, bool panic,
              float cruise, float speed, uint16_t servo, float dt,
              const RacePlan &plan, const GroundOdometry &odometry) {
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
    snprintf(text, sizeof text, "center=%d  quality=%.2f curve=%.2f, dt=%.2f",
             pilot.center, pilot.confidence, pilot.severity, dt);
    line(text);
    snprintf(text, sizeof text, "race=%s ai_age=%.2fs side=%d", plan.state,
             plan.detectionAge, plan.branchSide);
    line(text);
    const auto &pose = odometry.pose();
    snprintf(text, sizeof text, "odom x=%+.2f y=%+.2f th=%+.2f inl=%d", pose.x,
             pose.y, pose.theta, odometry.inliers());
    line(text);
    line(plan.reason_for_stage);
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

} // namespace

int main(int argc, char **argv) {
    const auto [headless, do_exit, do_produce_debug_video] =
        process_arguments(argc, argv);
    if (do_exit) {
        return 2;
    }

    signal(SIGINT, onSignal);
    signal(SIGTERM, onSignal);

    auto car = CarControl::create();
    if (!car) {
        cerr << "[Fatal] 打开串口失败" << endl;
        return 3;
    }

    Camera camera;

    unique_ptr<RacePerception> perception;
    try {
        perception = createRacePerception(RaceConfig{});
    } catch (const std::exception &e) {
        cerr << "[Error] 初始化失败: " << e.what() << '\n';
        return 1;
    }

    atomic_bool joyLost{false};
    auto js = JoyStick::create([&joyLost] { joyLost = true; });
    if (!js)
        cout << "[Warn] 没有手柄。\n";

    AutoPilot pilot(camera.cols(), camera.rows());
    RaceManager race_manager;
    GroundOdometry odometry;
    TerminalKeys terminal;
    RaceDisplay display;
    thread displayThread;
    if (!headless)
        displayThread = thread([&] { display.run(); });

    Mode mode = Mode::Manual;
    bool panic = false;
    float cruise = cfg::CRUISE_SPEED, jsSpeed = 0, speed = 0;
    uint16_t jsServo = PWMSERVOMID;
    TrackRows rows;
    string lastState;

    shared_ptr<DebugVideoManager> dvm;
    if (do_produce_debug_video) {
        dvm = DebugVideoManager::create(camera);
        if (!dvm) {
            cerr << "启动debug视频失败。" << endl;
            return 5;
        }
    }
    auto toggle = [&] {
        if (mode == Mode::Auto) {
            mode = Mode::Manual;
            jsSpeed = 0;
            pilot.resetControl();
            cout << "[Mode] 切换到 MANUAL" << endl;
            return;
        }
        // 准备切换到 AUTO
        if (!pilot.valid) {
            cout << "[Mode] 未检测到稳定的路径：" << pilot.path_issue
                 << "，无法切换到 AUTO。" << endl;
        }
        if (js && !joyLost.load(memory_order_relaxed) && fabs(jsSpeed) > 0.05) {
            cout << "[Mode] 手柄正在控制油门，无法切换到 AUTO" << endl;
        }
        mode = Mode::Auto;
        panic = false;
        jsSpeed = 0;
        pilot.resetControl();
        cout << "[Mode] 切换到 AUTO\n";
    };

    cout << "racing 启动，默认处于MANUAL模式，按a进入自动驾驶模式。按q退出。"
         << endl;

    LoopTiming timing;
    while (!signal_stop) {
        timing.begin();
        // ---- READ SECTION
        Mat image;
        if (!camera.read(image)) {
            cerr << "[Error] 取不到图像\n";
            break;
        }
        timing.lap("read");

        perception->submit(image);
        const auto observation = perception->snapshot();

        timing.lap(nullptr);
        Mat bin = camera.binarize(image);
        timing.lap("bin");

        track_identify_from_bin(bin, camera, rows);
        timing.lap("track");

        double dt = timing.last_loop() * 1000.0;

        // ---- ODOMETRY SECTION
        // 地面视觉里程计：用当前帧的地面纹理估车走了多少，检测框不参与。
        vector<Rect> blockers;
        for (const auto &d : observation.detections)
            blockers.emplace_back(d.x, d.y, d.width, d.height);
        odometry.update(image, camera, dt, blockers);
        timing.lap("odom");

        // ---- RACE SECTION
        const auto plan =
            race_manager.update(observation, dt, mode == Mode::Auto);

        // ---- KEY SECTION
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
            cout << "[Set] weight=" << weight_mode_name(pilot.get_weight_mode())
                 << '\n';
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

        // ---- PILOT SECIONT
        timing.lap(nullptr);
        pilot.update(rows, camera, dt, speed, plan.branchSide);
        if (mode == Mode::Auto && pilot.lost) {
            mode = Mode::Manual;
            panic = true;
            jsSpeed = 0;
            pilot.resetControl();
            cout << "[Stop] 丢线: " << pilot.path_issue << " (dt=" << dt
                 << " s)；恢复后按 a\n";
        }
        timing.lap("pilot");

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
            car->control(speed, servo);
        } catch (const std::exception &e) {
            cerr << "[Error] 串口: " << e.what() << '\n';
            break;
        }

        timing.lap("motor");
        if (lastState != plan.state) {
            cout << "[Race] " << plan.state << " | " << plan.reason_for_stage
                 << '\n';
            lastState = plan.state;
        }

        if (!headless || do_produce_debug_video) {
            track_draw_result(rows, image);
            draw_hud(image, pilot, mode, panic, cruise, speed, servo, dt, plan,
                     odometry);
            if (!headless)
                display.publish(image);
            if (do_produce_debug_video)
                dvm->publish(image);
        }
        timing.lap("draw");
        timing.end();
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
