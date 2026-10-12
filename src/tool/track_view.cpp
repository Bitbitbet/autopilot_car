/**
 * @file track_view.cpp
 * @brief 赛道识别 + 自动舵机 的台架/实车测试工具
 *
 * 与主程序的关系：
 *  - 复用 Track / Predeal / CarControl / JoyStick 这些库；
 *  - **不复用** Params / Center / Motion（它们的职责混在一起），
 *    而是把需要的“边缘→中心线→加权控制中心→PD 舵机”逻辑复制到这里，
 *    顺手修掉了原实现里的几个问题（见 AutoPilot 注释）。
 *
 * 串口写者仲裁（重要）：
 *  CarControl::writeBuffer() 没有锁，多线程同时写会把帧字节插花。
 *  因此本工具**不启动** startJoystickControlThread()，
 *  摇杆只被主循环读取，串口也只由主循环写 —— 全程单写者。
 *
 *  模式仲裁规则：
 *   1) 默认 MANUAL，必须显式按键/摇杆 X 才进入 AUTO（避免一上电就窜车）；
 *   2) 任何时刻把油门拉过阈值(>0.05m/s) → 立刻交还 MANUAL（人手优先）；
 *   3) 摇杆掉线/被拔出 → 强制 MANUAL + 急停；
 *   4) 视觉丢线超过 kLostFrames 帧 → AUTO
 * 下速度归零、舵机回正，恢复后自动继续； 5) 空格 = 急停（切 MANUAL 且速度
 * 0），按 a / X 解除。
 */

#include "control.hpp"
#include "joystick.hpp"
#include "predeal.hpp"
#include "tools.hpp"
#include "track.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

using std::cerr;
using std::cout;
using std::endl;
using std::make_shared;
using std::shared_ptr;
using std::string;
using std::vector;

using namespace cv;

/* ==========================================================================
 * 可调参数（本工具自用）
 * ========================================================================== */
namespace cfg {
constexpr int kRowCutUp = 10;     // 图像顶部切行（前瞻距离）
constexpr int kRowCutBottom = 20; // 图像底部切行（车头盲区）

constexpr size_t kMinEdgePoints = 5;       // 少于此点数视为该侧没有边缘
constexpr float kDefaultLaneWidth = 150.f; // 单边补线用的默认车道宽(px)
constexpr float kLaneWidthAlpha = 0.10f;   // 实测车道宽的 EMA 系数
constexpr int kMinLaneWidth = 30;          // 合理的车道宽下限(px)
constexpr int kMaxLaneWidth = 300;         // 合理的车道宽上限(px)

constexpr float kCruiseSpeed = 0.25f; // AUTO 默认巡航速度(m/s)
constexpr float kSpeedStep = 0.05f;   // +/- 每次调整量
constexpr float kSpeedMin = 0.05f;
constexpr float kSpeedMax = 1.00f;

constexpr float kRunP1 = 2.2f;   // 直线比例系数（=config.json runP1）
constexpr float kRunP2 = 0.007f; // 动态 P 变化系数（=runP2）
constexpr float kTurnD = 3.5f;   // 微分系数（=turnD）
constexpr float kMaxErrorStep = IMAGE_COL / 10.f; // 单帧偏差限幅，防猛打方向

constexpr int kLostFrames = 20;            // 连续无边缘帧数 → 判定丢线
constexpr float kTakeoverThrottle = 0.05f; // 油门超过该值 → 人手接管
} // namespace cfg

/* ==========================================================================
 * 控制中心权重策略（可在运行时用 'w' 切换，方便台架对比）
 * ========================================================================== */
enum class WeightMode {
    FarFull = 0,  // 远场满权重、近处线性衰减（与原 Center::fitting 等价）
    NearDominant, // 近处主导（车头附近更可信）
    Uniform,      // 等权
};

/* ==========================================================================
 * 舵机换算：先在浮点域钳位再转 uint16_t
 * （原 Motion::poseControl 是 (uint16_t)(1500 + pwmDiff) 之后才比较，
 *   负值会环绕成一个巨大的数，被当成“左打死”而不是“右打死”）
 * ========================================================================== */
inline uint16_t clampServo(float pwm) {
    const float lo = static_cast<float>(PWMSERVOMIN);
    const float hi = static_cast<float>(PWMSERVOMAX);
    return static_cast<uint16_t>(std::lround(std::clamp(pwm, lo, hi)));
}

/* ==========================================================================
 * 自动舵机：边缘点集 → 中心线 → 加权控制中心 → PD → 舵机 PWM
 * ========================================================================== */
struct AutoPilot {
    // ---- 输出 ----
    vector<PointX> centerEdge;    // 中心线点集（x=行, y=列），也用于绘制
    int center = IMAGE_COL / 2;   // 加权控制中心（列号）
    uint16_t servo = PWMSERVOMID; // PD 输出的舵机 PWM
    bool lost = false;            // 是否判定为丢线

    // ---- 状态 ----
    float laneWidthEma = cfg::kDefaultLaneWidth; // 实测车道宽 EMA
    float errorLast = 0.f; // 上一帧偏差（原实现用函数内 static）
    int lostFrames = 0;
    WeightMode weightMode = WeightMode::FarFull;

    void cycleWeight() {
        weightMode =
            static_cast<WeightMode>((static_cast<int>(weightMode) + 1) % 3);
    }

    const char *weightName() const {
        switch (weightMode) {
        case WeightMode::FarFull:
            return "far-full";
        case WeightMode::NearDominant:
            return "near-dominant";
        case WeightMode::Uniform:
        default:
            return "uniform";
        }
    }

    /**
     * @brief 单点权重：行号越小越远(图像上方)，越大越近(车头)
     */
    float weightOf(int row) const {
        switch (weightMode) {
        case WeightMode::FarFull:
            return row < static_cast<int>(IMAGE_ROW) / 2
                       ? IMAGE_ROW / 2.f
                       : static_cast<float>(IMAGE_ROW - row);
        case WeightMode::NearDominant:
            return row < static_cast<int>(IMAGE_ROW) / 2
                       ? static_cast<float>(row)
                       : IMAGE_ROW / 2.f;
        case WeightMode::Uniform:
        default:
            return 1.f;
        }
    }

    /**
     * @brief 主更新：拟合中心线并算出舵机输出
     */
    void update(const vector<PointX> &left, const vector<PointX> &right) {
        const bool leftOk = left.size() >= cfg::kMinEdgePoints;
        const bool rightOk = right.size() >= cfg::kMinEdgePoints;

        if (!leftOk && !rightOk) { // ---- 两侧都没有：丢线 ----
            lostFrames = std::min(lostFrames + 1, cfg::kLostFrames * 10);
            lost = lostFrames >= cfg::kLostFrames;
            centerEdge.clear();
            center = IMAGE_COL / 2; // 下面 PD 会把舵机平滑拉回中位
        } else {                    // ---- 至少一侧可用 ----
            lostFrames = 0;
            lost = false;

            vector<PointX> raw; // 未平滑的中心点
            if (leftOk && rightOk) {
                // 双边缘：按行号双指针配对取中点。
                // Track::handle() 每行都是左右成对 append 的，行号天然一致；
                // 这里仍用配对而不是同下标，是为了防住以后有人只改单侧点集。
                size_t i = 0, j = 0;
                while (i < left.size() && j < right.size()) {
                    if (left[i].x == right[j].x) {
                        raw.emplace_back(left[i].x,
                                         (left[i].y + right[j].y) / 2);
                        const int w = right[j].y - left[i].y;
                        if (w > cfg::kMinLaneWidth && w < cfg::kMaxLaneWidth)
                            laneWidthEma +=
                                cfg::kLaneWidthAlpha * (w - laneWidthEma);
                        ++i;
                        ++j;
                    } else if (left[i].x > right[j].x) {
                        ++i; // 行号大 = 更靠车头，先往前走等另一侧追上
                    } else {
                        ++j;
                    }
                }
            } else {
                // 单边缘：用**实测**车道宽的一半偏移到中心，
                // 原实现是“假设对侧边缘在图像边界上”，弯道里误差很大。
                const vector<PointX> &edge = leftOk ? left : right;
                const float off = laneWidthEma / 2.f;
                const int sign =
                    leftOk ? +1 : -1; // 左边缘→中心在右；右边缘→中心在左
                raw.reserve(edge.size());
                for (const auto &p : edge)
                    raw.emplace_back(
                        p.x, static_cast<int>(std::lround(p.y + sign * off)));
            }

            // 3 点滑动平均：原实现拿 4 个控制点做贝塞尔，端点处容易过冲
            centerEdge.clear();
            centerEdge.reserve(raw.size());
            if (raw.size() < 3) {
                centerEdge = raw;
            } else {
                for (size_t i = 0; i < raw.size(); ++i) {
                    const int y0 = raw[i > 0 ? i - 1 : i].y;
                    const int y1 = raw[i].y;
                    const int y2 = raw[i + 1 < raw.size() ? i + 1 : i].y;
                    centerEdge.emplace_back(raw[i].x, (y0 + y1 + y2) / 3);
                }
            }
        }

        // ---- 加权控制中心（虚拟“正前方”点权重 1，点数少时平滑回正）----
        float num = IMAGE_COL / 2.f;
        float den = 1.f;
        for (const auto &p : centerEdge) {
            const float w = weightOf(p.x);
            num += w * p.y;
            den += w;
        }
        center = static_cast<int>(std::lround(num / den));
        center = std::clamp(center, 0, static_cast<int>(IMAGE_COL));

        // ---- PD → 舵机（移植自 Motion::poseControl，三处修正）----
        //  1) 不再写回 config.turnP（原实现每帧覆盖配置项）
        //  2) 不再用函数内 static 保存 errorLast
        //  3) 浮点钳位后再转 uint16_t
        float error = center - IMAGE_COL / 2.f;
        if (std::fabs(error - errorLast) > cfg::kMaxErrorStep)
            error = error > errorLast ? errorLast + cfg::kMaxErrorStep
                                      : errorLast - cfg::kMaxErrorStep;
        const float turnP = cfg::kRunP1 + std::fabs(error) * cfg::kRunP2;
        const float pwmDiff = error * turnP + (error - errorLast) * cfg::kTurnD;
        errorLast = error;
        servo = clampServo(static_cast<float>(PWMSERVOMID) + pwmDiff);
    }
};

/* ==========================================================================
 * 运行模式
 * ========================================================================== */
enum class Mode { Manual, Auto };

static const char *modeName(Mode m) {
    return m == Mode::Auto ? "AUTO" : "MANUAL";
}

/* ==========================================================================
 * 屏幕 HUD
 * ========================================================================== */
static void drawHud(Mat &img, const AutoPilot &pilot, Mode mode, bool panic,
                    bool joyOk, bool joyLost, float cruise, float speed,
                    uint16_t servo, double fps) {
    // 中心线点集
    for (const auto &p : pilot.centerEdge)
        circle(img, Point(p.y, p.x), 1, Scalar(0, 0, 255), -1); // 红点

    // 控制中心：底部一条红条
    rectangle(img, Rect(pilot.center, IMAGE_ROW - 20, 10, 20),
              Scalar(0, 0, 255), cv::FILLED);

    const Scalar warn = panic ? Scalar(0, 0, 255) : Scalar(0, 255, 0);
    int y = 20;
    auto line = [&](const string &s, Scalar color) {
        putText(img, s, Point(10, y), FONT_HERSHEY_PLAIN, 1.0, color, 1);
        y += 18;
    };

    line(string("[") + modeName(mode) + (panic ? " PANIC]" : "]") +
             (pilot.lost ? "  << TRACK LOST >>" : ""),
         warn);
    line("speed=" + std::to_string(speed).substr(0, 4) + " (cruise " +
             std::to_string(cruise).substr(0, 4) + ")",
         Scalar(0, 255, 255));
    line("servo=" + std::to_string(servo), Scalar(0, 255, 255));
    line("center=" + std::to_string(pilot.center) +
             " err=" + std::to_string(pilot.center - (int)IMAGE_COL / 2),
         Scalar(0, 255, 255));
    line("laneW=" + std::to_string((int)std::lround(pilot.laneWidthEma)) +
             " weight=" + pilot.weightName(),
         Scalar(0, 255, 255));
    line(joyOk ? (joyLost ? "joystick: LOST" : "joystick: ok")
               : "joystick: none",
         joyLost ? Scalar(0, 0, 255) : Scalar(0, 255, 255));
    line("fps=" + std::to_string((int)std::lround(fps)), Scalar(0, 255, 0));
    line("a:auto/manual  space:stop  +/-:speed  w:weight  ESC:quit",
         Scalar(200, 200, 200));
}

/* ==========================================================================
 * 信号：只置标志，真正的停车在主循环里做
 * ========================================================================== */
namespace {
std::atomic_bool g_stop{false};
void onSignal(int) { g_stop.store(true, std::memory_order_relaxed); }
} // namespace

int main(int argc, char **argv) {
    std::signal(SIGINT, onSignal);
    std::signal(SIGTERM, onSignal);

    try {
        // ---------------- 车辆通信 ----------------
        auto car = CarControl::create();
        if (!car) {
            cerr << "Failed to initialize uart!" << endl;
            return 1;
        }
        car->resetVelocity();

        // ---------------- 摇杆（只读，不起写串口线程）----------------
        // joyLost 必须声明在 js 之前：js 析构时其接收线程可能回调本标志
        std::atomic_bool joyLost{false};
        shared_ptr<JoyStick> js =
            JoyStick::create([&joyLost] { joyLost = true; });
        if (!js) {
            cerr << "[Warn]: joystick unavailable, keyboard-only control."
                 << endl;
        }

        // ---------------- 图像源 ----------------
        const string source = (argc > 1) ? argv[1] : "/dev/video0";
        VideoCapture capture;
        capture.open(source, cv::CAP_V4L2);
        if (!capture.isOpened())
            capture.open(source); // 视频文件 / 其它后端
        if (!capture.isOpened()) {
            cerr << "Failed to open video source: " << source << endl;
            return 2;
        }
        capture.set(CAP_PROP_FRAME_WIDTH, IMAGE_COL);
        capture.set(CAP_PROP_FRAME_HEIGHT, IMAGE_ROW);

        Track track;
        track.rowCutUp = cfg::kRowCutUp;
        track.rowCutBottom = cfg::kRowCutBottom;
        shared_ptr<Predeal> predeal = make_shared<Predeal>(); // 默认大津法

        // ---------------- 运行状态 ----------------
        AutoPilot pilot;
        Mode mode = Mode::Manual; // 默认手动，必须显式接管
        bool panic = false;
        float cruise = cfg::kCruiseSpeed;
        float jsSpeed = 0.f; // 摇杆最近一次指令（手动模式下持续使用）
        uint16_t jsServo = PWMSERVOMID;

        cout << "==== track_view: track following + auto steering ====" << endl;
        cout << "source: " << source << endl;
        cout << "keys  : ESC quit | a auto/manual | space stop | +/- speed | "
                "w weight mode"
             << endl;
        cout << "pad   : X button = toggle auto/manual, throttle pulls > "
             << cfg::kTakeoverThrottle << " m/s = human takeover" << endl;
        cout << "start in MANUAL; press 'a' to engage auto steering." << endl;

        auto timeLast = std::chrono::steady_clock::now();
        double fps = 0.0;

        while (!g_stop.load()) {
            Mat img;
            if (!capture.read(img) || img.empty()) {
                cerr << "Camera frame unavailable; stopping." << endl;
                break;
            }
            predeal->correct(img);
            Mat bin = predeal->binarize(img);

            track.handle(bin);                 // [1] 边缘识别
            pilot.update(track.pointsEdgeLeft, // [2] 中心线 + 舵机
                         track.pointsEdgeRight);

            // ---------- 摇杆输入：只有这里读，只有这里写串口 ----------
            bool joyStopped = false;
            if (js) {
                float s = jsSpeed;
                uint16_t sv = jsServo;
                if (js->takeCarControl(s,
                                       sv)) { // 有新指令（未知按键会带 0 速）
                    jsSpeed = s;
                    jsServo = sv;
                }
                if (js->takeSampleOnce()) { // X 键：切换
                                            // 自动/手动（并解除急停）
                    mode = (mode == Mode::Auto) ? Mode::Manual : Mode::Auto;
                    panic = false;
                    car->buzzerSound(Buzzer::ding);
                }
                js->takeBuzzer(); // 丢弃，避免标志堆积

                joyStopped = js->isStopped() || joyLost.load();

                // 人手优先：油门拉过阈值就交还手动
                if (mode == Mode::Auto &&
                    std::fabs(jsSpeed) > cfg::kTakeoverThrottle) {
                    mode = Mode::Manual;
                    car->buzzerSound(Buzzer::warning);
                }
            }

            // ---------- 仲裁输出 ----------
            float speed = 0.f;
            uint16_t servo = PWMSERVOMID;
            if (joyStopped) { // 摇杆掉线：停车并回正（但保留键盘可用）
                speed = 0.f;
                servo = PWMSERVOMID;
                mode = Mode::Manual;
            } else if (panic) {
                speed = 0.f;
                servo = PWMSERVOMID;
            } else if (mode == Mode::Manual) {
                speed = jsSpeed;
                servo = jsServo;
            } else { // Auto
                speed = pilot.lost ? 0.f : cruise;
                servo = pilot.lost ? PWMSERVOMID : pilot.servo;
            }
            car->carControl(speed, servo); // 单写者：唯一的串口出口

            // ---------- 显示 ----------
            track.drawImage(img); // 绿=左边缘 黄=右边缘
            drawHud(img, pilot, mode, panic, js != nullptr, joyLost.load(),
                    cruise, speed, servo, fps);
            imshow("track_view", img);
            imshow("binarized", bin);

            // ---------- 按键（放在显示之后取，键盘响应更跟手）----------
            const int key = waitKey(1);
            if (key == 27) { // ESC
                break;
            } else if (key == 'a' || key == 'A') {
                mode = (mode == Mode::Auto) ? Mode::Manual : Mode::Auto;
                panic = false;
            } else if (key == ' ') {
                panic = true;
                mode = Mode::Manual;
            } else if (key == '+' || key == '=') {
                cruise = std::min(cfg::kSpeedMax, cruise + cfg::kSpeedStep);
            } else if (key == '-' || key == '_') {
                cruise = std::max(cfg::kSpeedMin, cruise - cfg::kSpeedStep);
            } else if (key == 'w' || key == 'W') {
                pilot.cycleWeight();
            }

            const auto now = std::chrono::steady_clock::now();
            const double dt =
                std::chrono::duration<double>(now - timeLast).count();
            timeLast = now;
            if (dt > 0)
                fps = fps * 0.9 + (1.0 / dt) * 0.1; // 平滑
        }

        car->resetVelocity(); // 退出前一定停车 + 舵机归中
        cout << "track_view exit." << endl;
        return 0;
    } catch (const std::exception &e) {
        cerr << "[Error]: " << e.what() << endl;
        return 1;
    } catch (...) {
        cerr << "[Error]: Unexpected track_view failure." << endl;
        return 1;
    }
}
