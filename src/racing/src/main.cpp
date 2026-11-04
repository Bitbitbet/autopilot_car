#include "camera.hpp"
#include "control.hpp"
#include "debug_video_manager.hpp"
#include "detection.hpp"
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
#include <csignal>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <thread>

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

// ---- 目标点控制 ----
// 到点判据（米）
constexpr float ARRIVE_DIST = 0.10f;
// 往图像上投影时要求的最小前向距离（米）。不能太大：目标偏到侧面时，
// 画面内那一段可能已经很贴近车头了
constexpr float AIM_MIN_FORWARD = 0.05f;
// 认为“在画面内”的边界（像素）
constexpr int AIM_MARGIN = 8;

/**
 * @brief 画目标点和期望路径（车体系原点到目标点的直线，地面坐标）
 */
void draw_target(Mat &image, const Camera &camera, const cv::Point2f &target,
                 bool aiming, int aim_col) {
    const int cols = camera.cols(), rows = camera.rows();
    const auto project = [&](const cv::Point2f &p) {
        const cv::Point2d img = camera.ground_to_image(p);
        if (!std::isfinite(img.x) || !std::isfinite(img.y))
            return cv::Point(-1, -1);
        const cv::Point at(int(std::lround(img.x)), int(std::lround(img.y)));
        if (at.x < 0 || at.x >= cols || at.y < 0 || at.y >= rows)
            return cv::Point(-1, -1);
        return at;
    };
    cv::Point previous(-1, -1);
    for (int i = 0; i <= 24; ++i) {
        const cv::Point now = project(target * (float(i) / 24.f));
        if (now.x < 0) {
            previous = now;
            continue;
        }
        if (previous.x >= 0)
            cv::line(image, previous, now, {255, 200, 0}, 1);
        previous = now;
    }
    const cv::Point at = project(target);
    if (at.x >= 0)
        cv::circle(image, at, 6, {255, 0, 255}, 2);
    if (aiming && aim_col >= 0)
        cv::line(image, {aim_col, rows - 1}, {aim_col, rows - 50},
                 {255, 0, 255}, 1);
}

/**
 * @brief 把车体系里的目标点投到图像列
 *
 * 目标点可能落在画面外（一般不会太远）：沿“车 -> 目标”这条射线往回走，
 * 取一个仍在画面内的点作为临时中途目标。整条射线都不可见时返回 -1。
 */
int aim_column(const cv::Point2f &local, const Camera &camera) {
    const int cols = camera.cols(), rows = camera.rows();
    for (float s = 1.0f; s >= 0.15f; s -= 0.05f) {
        const cv::Point2f p = local * s;
        if (p.y < AIM_MIN_FORWARD)
            continue;
        const cv::Point2d img = camera.ground_to_image(p);
        if (!std::isfinite(img.x) || !std::isfinite(img.y))
            continue;
        const int col = int(std::lround(img.x)), row = int(std::lround(img.y));
        if (col >= AIM_MARGIN && col < cols - AIM_MARGIN && row >= AIM_MARGIN &&
            row < rows - AIM_MARGIN)
            return col;
    }
    return -1;
}

void draw_hud(Mat &image, const AutoPilot &pilot, Mode mode, bool panic,
              float cruise, float speed, uint16_t servo, float dt,
              const RacePlan &plan, const GroundOdometry &odometry,
              const cv::Point2f &target, bool aiming) {
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
    snprintf(text, sizeof text, "odom x=%+.2f y=%+.2f th=%+.2f inl=%d trk=%d",
             pose.x, pose.y, pose.theta, odometry.inliers(),
             odometry.tracked());
    line(text);
    if (aiming)
        snprintf(text, sizeof text, "tgt x=%+.2f y=%+.2f", target.x, target.y);
    else
        snprintf(text, sizeof text, "tgt -");
    line(text);
    line(plan.reason_for_stage);
    line("a:auto  space:stop q:quit");
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

    vector<cv::Point2d> targets;
    size_t target_index = 0;
    int target_seq = 0;
    bool aiming = false;

    LoopTiming timing;
    while (!signal_stop) {
        double dt = timing.last_loop() / 1000000.0;

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
        Mat grey = camera.grey();
        Mat bin = camera.bin();
        timing.lap("bin");

        track_identify_from_bin(bin, camera, rows);
        timing.lap("track");

        // ---- RACE SECTION
        const auto plan = race_manager.update(
            observation, dt, mode == Mode::Auto, camera, aiming);

        // ---- TARGET / ODOMETRY SECTION
        // 目标点只在需要时才跟踪：锚定那一刻把里程计清零，于是“世界系”就是
        // 锚定时的车体系，targets 里每个点都是这个系里的坐标，用 pose.local()
        // 就能得到它们相对小车的当前位置（走过头就是 y 变负）。
        if (plan.targetSeq != target_seq) {
            // 新的一串目标：整串抄下来，锚定一次（位姿清零）后按顺序走。
            odometry.reset();
            target_seq = plan.targetSeq;
            targets = plan.targets;
            target_index = 0;
            aiming = !targets.empty();
        }
        if (aiming) {
            // 挖掉非地面物品的点
            vector<Rect> blockers;
            for (const auto &d : observation.detections)
                if (!is_ground_label(d.type))
                    blockers.emplace_back(d.rect.x, d.rect.y, d.rect.width,
                                          d.rect.height);
            odometry.update(grey, camera, dt, blockers);
        }
        timing.lap("odom");

        if (lastState != plan.state) {
            cout << "[Plan] " << plan.state << " | " << plan.reason_for_stage
                 << endl;
            lastState = plan.state;
        }

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

        // ---- PILOT SECTION
        timing.lap(nullptr);
        cv::Point2f target_local{};
        if (aiming) {
            target_local = odometry.pose().local(targets[target_index]);
            if (hypot(target_local.x, target_local.y) <= ARRIVE_DIST) {
                if (++target_index < targets.size()) {
                    cout << "[Target] 第 " << target_index << " 个到点\n";
                } else {
                    aiming = false;
                    cout << "[Target] 全部到点\n";
                }
            }
        }
        int aim_col = -1;
        if (aiming) {
            aim_col = aim_column(target_local, camera);
            // 目标在画面外时先朝那一侧满舵，等它进画面再按图像列对准。
            if (aim_col < 0)
                aim_col = target_local.x >= 0 ? camera.cols() - AIM_MARGIN
                                              : AIM_MARGIN;
            pilot.aim_at(aim_col, dt, speed);
        } else {
            pilot.update(rows, camera, dt, speed, plan.branchSide);
        }
        if (mode == Mode::Auto && pilot.lost) {
            mode = Mode::Manual;
            panic = true;
            jsSpeed = 0;
            pilot.resetControl();
            cout << "[Stop] 丢线: " << pilot.path_issue << " (dt=" << dt
                 << " s)；恢复后按 a\n";
        }
        timing.lap("pilot");

        // ---- MOTOR SECTION
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

        try {
            car->control(speed, servo);
        } catch (const std::exception &e) {
            cerr << "[Error] 串口: " << e.what() << '\n';
            break;
        }

        // ---- OUTPUT/DEBUG SECTION
        timing.lap(nullptr);
        if (!headless || do_produce_debug_video) {
            track_draw_result(rows, image);
            if (aiming)
                draw_target(image, camera, target_local, aiming, aim_col);
            draw_hud(image, pilot, mode, panic, cruise, speed, servo, dt, plan,
                     odometry, target_local, aiming);
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

    if (!headless) {
        display.stop = true;
        displayThread.join();
    }

    cout << "racing 正常退出。" << endl;
    return 0;
}
