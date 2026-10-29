#include "camera.hpp"
#include "control.hpp"
#include "joystick.hpp"
#include "tools.hpp"
#include "track.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <deque>
#include <iomanip>
#include <iostream>
#include <map>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using cv::Mat;
using std::atomic;
using std::atomic_bool;
using std::cerr;
using std::cout;
using std::deque;
using std::greater;
using std::jthread;
using std::map;
using std::memory_order_relaxed;
using std::mutex;
using std::pair;
using std::ref;
using std::string;
using std::vector;
using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::this_thread::sleep_for;

using Clock = std::chrono::steady_clock;
#include "trackview_pilot.hpp"
#include "trackview_display.hpp"
namespace {
std::atomic_bool stopping{false};
static_assert(std::atomic_bool::is_always_lock_free);
void onSignal(int) { stopping.store(true); }
} // namespace

enum class Mode { Manual, Auto };

// Declared after the driving thread: inhibit motion before unwinding/joining
// it.
struct StopOnExit {
    CarControl &car;
    ~StopOnExit() {
        stopping = true;
        try {
            car.latch_stop();
        } catch (...) {
        }
    }
};

static void drawHud(Mat &image, const AutoPilot &pilot, Mode mode, bool panic,
                    float cruise, float speed, uint16_t servo, double fps,
                    bool displaySlow) {
    for (const auto &p : pilot.centerEdge)
        cv::circle(image, cv::Point(p.y, p.x), 1, cv::Scalar(0, 0, 255), -1);
    int y = 18;
    auto line = [&](const string &s) {
        cv::putText(image, s, {5, y}, cv::FONT_HERSHEY_PLAIN, 0.9,
                    {0, 255, 255}, 1);
        y += 18;
    };
    auto fmt = [](float v) {
        std::ostringstream s;
        s << std::fixed << std::setprecision(2) << v;
        return s.str();
    };
    line(string(mode == Mode::Auto ? "AUTO" : "MANUAL") +
         (panic ? " STOP LATCHED" : "") + (pilot.lost ? " LOST" : ""));
    line("speed=" + fmt(speed) + " limit=" + fmt(cruise) +
         " pwm=" + std::to_string(servo));
    line("turn=" + fmt(pilot.curveSpeed()) +
         " qualityLimit=" + fmt(pilot.degradedSpeed()));
    line(string(pilot.valid ? "center=" : "last center=") +
         std::to_string(pilot.center) + " err=" + fmt(pilot.error));
    line("dual=" + std::to_string(pilot.dualPoints) +
         " single=" + std::to_string(pilot.singlePoints) +
         " reject=" + std::to_string(pilot.rejectedPoints));
    line("quality=" + fmt(pilot.confidence) + " curve=" + fmt(pilot.severity) +
         " h=" + fmt(pilot.curveHeading));
    line(string(pilot.valid
                    ? (pilot.degraded
                           ? "DEGRADED"
                           : (pilot.singlePoints > 0 ? "MIXED" : "DUAL"))
                    : "NO TRUSTED PATH") +
         (pilot.ready ? " READY" : " WAIT"));
    line("target=" + fmt(pilot.speedTarget) + " " +
         (mode == Mode::Auto ? pilot.speedReason : "MANUAL"));
    if (!pilot.valid)
        line(string(pilot.pathIssue) +
             " width=" + std::to_string(pilot.learnedWidthRows()) +
             " miss=" + std::to_string(pilot.missingWidthPoints));
    if (displaySlow)
        line("DISPLAY DELAY: control independent");
    line("fps=" + std::to_string(int(fps)) + " weight=" + pilot.weightName());
    line("a:auto  SPACE:stop  +/-:limit  ESC:quit");
    line("[/]:turn  ,/.:quality  w:weight");
}

void driving_thread(Camera &camera, CarControl &car, AutoPilot &pilot,
                    JoyStick &js, TrackviewDisplay &display,
                    atomic_bool &joyLost) {
    float cruise = cfg::kCruiseSpeed, jsSpeed = 0, speed = 0;
    auto start = Clock::now(), last = start;
    uint16_t jsServo = PWMSERVOMID;
    auto mode = Mode::Manual;
    bool panic = false;

    map<int, pair<int, int>, greater<int>> rows;
    try {
        while (!stopping.load()) {
            if (car.hasFault())
                throw std::runtime_error(
                    "Serial fault or command watchdog timeout");
            Mat image;
            if (!camera.read(image)) {
                car.resetVelocity();
                throw std::runtime_error("Camera frame unavailable");
            }
            Mat bin = camera.binarize(image);
            track_identify_from_bin(bin, camera, rows);
            const auto now = Clock::now();
            auto dt = duration<double>(now - last).count();
            last = now;
            if (dt <= 0)
                dt = cfg::kReferenceDt;
            pilot.update(rows, dt, speed);
            track_draw_result(rows, image);
            auto toggle = [&]() {
                if (mode == Mode::Auto) {
                    mode = Mode::Manual;
                    jsSpeed = 0;
                    pilot.resetControl();
                    cout << "[Mode] MANUAL" << endl;
                } else if (pilot.ready && !joyLost && !js.isStopped() &&
                           fabs(jsSpeed) <= 0.05f) {
                    mode = Mode::Auto;
                    panic = false;
                    jsSpeed = 0;
                    pilot.resetControl();
                    cout << "[Mode] AUTO\n";
                } else {
                    if (!pilot.ready)
                        cout << "[Wait] Lane " << pilot.pathIssue
                             << "; width rows=" << pilot.learnedWidthRows()
                             << " missing width=" << pilot.missingWidthPoints
                             << "; wait for stable trusted path.\n";
                    else
                        cout << "[Wait] Release gamepad throttle "
                                "or reconnect gamepad.\n";
                }
            };
            string stopReason;
            for (const int key : display.takeKeys()) {
                if (key == ' ') {
                    // Stop = the write below this loop (speed 0) plus panic,
                    // which only a successful toggle() clears.
                    panic = true;
                    mode = Mode::Manual;
                    jsSpeed = 0;
                    pilot.resetControl();
                    stopReason = "keyboard stop";
                } else if (key == 'a' || key == 'A')
                    toggle();
                else if (key == '+' || key == '=') {
                    cruise = std::min(cfg::kMaxSpeed, cruise + 0.05f);
                    cout << "[Set] limit=" << cruise << " m/s\n";
                } else if (key == '-' || key == '_') {
                    cruise = std::max(cfg::kMinSpeed, cruise - 0.05f);
                    cout << "[Set] limit=" << cruise << " m/s\n";
                } else if (key == '[' || key == ']') {
                    pilot.setCurveSpeed(pilot.curveSpeed() +
                                        (key == ']' ? 0.05f : -0.05f));
                    cout << "[Set] curveSpeed=" << pilot.curveSpeed()
                         << " m/s\n";
                } else if (key == ',' || key == '.') {
                    pilot.setDegradedSpeed(pilot.degradedSpeed() +
                                           (key == '.' ? 0.05f : -0.05f));
                    cout << "[Set] degradedSpeed=" << pilot.degradedSpeed()
                         << " m/s\n";
                } else if (key == 'w' || key == 'W') {
                    pilot.cycleWeight();
                    cout << "[Set] weight=" << pilot.weightName() << '\n';
                }
            }
            // X11 rendering is telemetry, not the control-loop
            // heartbeat. Keep command watchdog/vision loss independent
            // of remote UI delay.
            const auto displayAge = display.ageMs();
            float s = 0;
            uint16_t sv = PWMSERVOMID;
            const bool newCommand = js.takeCarControl(s, sv);
            if (newCommand) {
                jsSpeed = s;
                jsServo = sv;
            }
            if (js.takeSampleOnce())
                toggle();
            js.takeBuzzer();
            if (js.isStopped() || joyLost) {
                if (!panic)
                    stopReason = "gamepad disconnected; stop latched";
                panic = true;
                mode = Mode::Manual;
                jsSpeed = 0;
                pilot.resetControl();
            } else if (mode == Mode::Auto && newCommand &&
                       std::fabs(jsSpeed) > 0.05f) {
                mode = Mode::Manual;
                pilot.resetControl();
                cout << "[Mode] MANUAL (gamepad takeover)\n";
            }

            if (mode == Mode::Auto && pilot.lost) {
                mode = Mode::Manual;
                panic = true;
                jsSpeed = 0;
                pilot.resetControl();
                stopReason = string("trusted lane lost; stop latched, "
                                    "press a after recovery; path=") +
                             pilot.pathIssue;
            }
            speed = 0;
            uint16_t servo = PWMSERVOMID;
            if (!panic && !stopping) {
                if (mode == Mode::Auto) {
                    speed = pilot.speedFor(cruise, dt);
                    servo = pilot.servo;
                } else {
                    speed =
                        std::clamp(jsSpeed, -cfg::kMaxSpeed, cfg::kMaxSpeed);
                    servo = jsServo;
                }
            }
            if (car.hasFault())
                throw std::runtime_error(
                    "Serial command timeout during processing");
            car.carControl(speed, servo);

            drawHud(image, pilot, mode, panic, cruise, speed, servo, 1 / dt,
                    displayAge >= 2000);
            if (!stopReason.empty())
                cerr << "[Stop] " << stopReason << '\n';
            display.publish(image, bin);
        }
        car.resetVelocity();
        cout << "[Stop] "
             << (stopping.load() ? "signal or UI stop" : "normal exit") << '\n';
    } catch (const std::exception &e) {
        try {
            car.latch_stop();
        } catch (...) {
        }

        if (stopping.load() && !display.failed.load()) {
            cout << "[Stop] operator requested exit\n";
        } else {
            cerr << "[Error] " << e.what() << endl;
            display.failed = true;
        }
    } catch (...) {
        try {
            car.latch_stop();
        } catch (...) {
        }

        cerr << "[Error] unexpected driving thread failure." << endl;
        display.failed = true;
    }
    display.finished = true;
}
int main() {
    signal(SIGINT, onSignal);
    signal(SIGTERM, onSignal);
    signal(SIGHUP, onSignal);
    auto car = CarControl::create();
    if (!car) {
        cerr << "无法加载串口控制。" << endl;
        return 1;
    }

    atomic_bool joyLost{false}; // Must outlive gamepad callback.
    auto js = JoyStick::create(
        [&joyLost] { joyLost.store(true, memory_order_relaxed); });
    if (!js) {
        cerr << "Gamepad not connected." << endl;
        return 2;
    }

    Camera camera;

    AutoPilot pilot(camera.cols(), camera.rows());

    try {
        TrackviewDisplay display;
        cout << "trackview | frame=" << camera.cols() << "x" << camera.rows()
             << " | limit=" << cfg::kCruiseSpeed << " m/s (max "
             << cfg::kMaxSpeed << ") turn=" << pilot.curveSpeed()
             << " quality=" << pilot.degradedSpeed() << endl;

        jthread driving(driving_thread, ref(camera), ref(*car), ref(pilot),
                        ref(*js), ref(display), ref(joyLost));

        StopOnExit stopBeforeJoin{*car};

        auto lastRender = Clock::now() - milliseconds(100);
        while (!stopping.load() && !display.finished.load()) {
            // GUI stays on the main thread; only immutable latest snapshots
            // cross threads.
            if (car->hasFault()) {
                display.failed = true;
                break;
            }
            const auto rendered = Clock::now();
            Mat image, bin;
            if (rendered - lastRender >= milliseconds(100)) {
                if (!display.snapshot(image, bin)) {
                    sleep_for(milliseconds(5));
                    display.progress();
                    continue;
                }
                cv::imshow("track_view", image);
                cv::imshow("binarized", bin);
                lastRender = Clock::now();
            }
            const int key = cv::waitKey(10);
            display.progress();
            if (key == 27)
                break;
            display.post(key);
            if (key == ' ' && car) {
                // The worker also sees this key and stops itself; writing 0
                // here covers a worker blocked in frame capture.
                car->resetVelocity();
            }
        }
        stopping = true;
        try {
            car->latch_stop();
        } catch (...) {
        }
        driving.join();
        cout << "[Stop] trackview finished.\n";
        return display.failed.load() ? 1 : 0;
    } catch (const std::exception &e) {
        try {
            car->resetVelocity();
        } catch (...) {
        }

        cerr << "[Error] " << e.what() << '\n';
        return 1;
    } catch (...) {
        try {
            car->resetVelocity();
        } catch (...) {
        }

        cerr << "[Error] unexpected exception\n";
        return 1;
    }
}
