/**
 * 比赛入口：一个循环里完成巡线、AI 检测和模式切换，组织和 icar 一样直白。
 * 图像、标定和识别口径都是 640x480，和 track_view 同一套 Camera / Track。
 * 检测结果只用来改运行模式（限速、停车、选路），不做离线记录。
 */
#include "camera.hpp"
#include "control.hpp"
#include "joystick.hpp"
#include "race_manager.hpp"
#include "race_perception.hpp"
#include "track.hpp"
#include "trackview_pilot.hpp"
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <opencv2/opencv.hpp>

using Clock = std::chrono::steady_clock;
using std::cerr;
using std::cout;
using std::greater;
using std::map;
using std::pair;

namespace {
std::atomic_bool stopping{false};
void onSignal(int) { stopping = true; }
double nowSeconds() {
    return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}
enum class Mode { Manual, Auto };
} // namespace

static void drawHud(cv::Mat &image, const AutoPilot &pilot, Mode mode, bool panic,
                    float cruise, float speed, uint16_t servo, const RacePlan &race) {
    for (const auto &p : pilot.centerEdge)
        cv::circle(image, cv::Point(p.y, p.x), 2, cv::Scalar(0, 0, 255), -1);
    int y = 22;
    auto line = [&](const std::string &s) {
        cv::putText(image, s, {8, y}, cv::FONT_HERSHEY_PLAIN, 1.2, {0, 255, 255}, 1);
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
    snprintf(text, sizeof text, "race=%s  ai_age=%.2fs", race.state, race.detectionAge);
    line(text);
    line(race.reason);
    line("a:auto  space:stop  +/-:limit  q:quit");
}

int main(int argc, char **argv) {
    bool headless = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--headless"))
            headless = true;
        else {
            cerr << "用法: trackview_race [--headless]\n";
            return 2;
        }
    }
    std::signal(SIGINT, onSignal);
    std::signal(SIGTERM, onSignal);

    // 标定和模型都在构造函数里读，失败会抛异常。
    std::unique_ptr<Camera> camera;
    std::unique_ptr<RacePerception> perception;
    try {
        camera = std::make_unique<Camera>();
        perception = createRacePerception(RaceConfig{});
    } catch (const std::exception &e) {
        cerr << "[Error] 初始化失败: " << e.what() << '\n';
        return 1;
    }

    // 台架上没有串口时只跑图像和决策，不驱动车轮。
    auto car = CarControl::create();
    if (!car)
        cout << "[Warn] 没有串口控制，本次不驱动车轮。\n";

    std::atomic_bool joyLost{false};
    auto js = JoyStick::create([&joyLost] { joyLost = true; });
    if (!js)
        cout << "[Warn] 没有手柄。\n";

    AutoPilot pilot(camera->cols(), camera->rows());
    RaceManager raceManager{RaceConfig{}};

    Mode mode = Mode::Manual;
    bool panic = false;
    float cruise = cfg::kCruiseSpeed, jsSpeed = 0, speed = 0;
    uint16_t jsServo = PWMSERVOMID;
    map<int, pair<int, int>, greater<int>> rows;
    double last = nowSeconds();
    std::string lastState;
    cout << "[Start] " << (headless ? "HEADLESS " : "") << camera->cols() << "x"
         << camera->rows() << " | 默认 MANUAL，按 a 启动。\n";

    while (!stopping) {
        cv::Mat image;
        if (!camera->read(image)) {
            cerr << "[Error] 取不到图像\n";
            break;
        }
        cv::Mat bin = camera->binarize(image);
        track_identify_from_bin(bin, *camera, rows);

        const double t = nowSeconds();
        float dt = float(t - last);
        last = t;
        if (dt <= 0 || dt > cfg::kMaxFrameGap)
            dt = cfg::kReferenceDt;

        perception->submit(image, uint64_t(t * 1000), t);
        const RaceSnapshot observation = perception->snapshot();
        RacePlan race = raceManager.update(observation, t, dt, mode == Mode::Auto, bin,
                                           rows, image);

        auto toggle = [&] {
            if (mode == Mode::Auto) {
                mode = Mode::Manual;
                jsSpeed = 0;
                pilot.resetControl();
                cout << "[Mode] MANUAL\n";
            } else if (pilot.ready && raceManager.aiReady(observation, t) && !joyLost &&
                       (!js || !js->isStopped()) && std::fabs(jsSpeed) <= 0.05f) {
                mode = Mode::Auto;
                panic = false;
                jsSpeed = 0;
                pilot.resetControl();
                cout << "[Mode] AUTO\n";
            } else if (!raceManager.aiReady(observation, t)) {
                cout << "[Wait] 还没有可用的检测结果\n";
            } else if (!pilot.ready) {
                cout << "[Wait] 路径未稳定: " << pilot.pathIssue << '\n';
            } else {
                cout << "[Wait] 先松开油门或重新连接手柄\n";
            }
        };

        int key = -1;
        if (!headless)
            key = cv::waitKey(1);
        if (key == 'a' || key == 'A') {
            toggle();
        } else if (key == ' ') {
            panic = true;
            mode = Mode::Manual;
            jsSpeed = 0;
            pilot.resetControl();
            cout << "[Stop] 键盘停车\n";
        } else if (key == '+' || key == '=') {
            cruise = std::min(cfg::kMaxSpeed, cruise + 0.05f);
            cout << "[Set] limit=" << cruise << " m/s\n";
        } else if (key == '-' || key == '_') {
            cruise = std::max(cfg::kMinSpeed, cruise - 0.05f);
            cout << "[Set] limit=" << cruise << " m/s\n";
        } else if (key == '[' || key == ']') {
            pilot.setCurveSpeed(pilot.curveSpeed() + (key == ']' ? 0.05f : -0.05f));
            cout << "[Set] curve=" << pilot.curveSpeed() << " m/s\n";
        } else if (key == ',' || key == '.') {
            pilot.setDegradedSpeed(pilot.degradedSpeed() + (key == '.' ? 0.05f : -0.05f));
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
            } else if (mode == Mode::Auto && newCommand && std::fabs(jsSpeed) > 0.05f) {
                mode = Mode::Manual;
                jsSpeed = 0;
                pilot.resetControl();
                cout << "[Mode] MANUAL（手柄介入）\n";
            }
        }

        pilot.update(race.selectedBranch ? race.path.rows : rows, dt, speed);
        if (mode == Mode::Auto && pilot.lost) {
            mode = Mode::Manual;
            panic = true;
            jsSpeed = 0;
            pilot.resetControl();
            cout << "[Stop] 丢线: " << pilot.pathIssue << "；恢复后按 a\n";
        }

        speed = 0;
        uint16_t servo = PWMSERVOMID;
        if (!panic && !stopping) {
            if (mode == Mode::Auto) {
                if (race.stop) {
                    pilot.resetControl();
                } else {
                    speed = pilot.speedFor(std::min(cruise, race.speedCap), dt);
                    servo = pilot.servo;
                }
            } else {
                speed = std::clamp(jsSpeed, -cfg::kMaxSpeed, cfg::kMaxSpeed);
                servo = jsServo;
            }
        }
        if (car) {
            // 串口写失败是致命故障，直接停住并发后退出。
            try {
                car->carControl(speed, servo);
                if (race.buzzer && !panic && mode == Mode::Auto)
                    car->buzzerSound(Buzzer::ding);
            } catch (const std::exception &e) {
                cerr << "[Error] 串口: " << e.what() << '\n';
                break;
            }
        }
        if (lastState != race.state) {
            cout << "[Race] " << race.state << " | " << race.reason << '\n';
            lastState = race.state;
        }

        if (!headless) {
            track_draw_result(rows, image);
            drawHud(image, pilot, mode, panic, cruise, speed, servo, race);
            cv::imshow("track_view", image);
            if (key == 27 || key == 'q' || key == 'Q')
                break;
        }
    }

    if (car) {
        try {
            car->latch_stop();
        } catch (...) {
        }
    }
    if (!headless)
        cv::destroyAllWindows();
    cout << "[Stop] 程序退出\n";
    return 0;
}
