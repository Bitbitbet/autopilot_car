/**
 * @file track_view.cpp
 * @brief 赛道识别 + 自动舵机 的台架/实车测试工具（多线程版）
 *
 * 线程分工（串口严格单写者）：
 *   - 识别/UI 线程（主线程）：相机 → Predeal → Track → AutoPilot 解算，
 *     并负责 imshow / waitKey / HUD。不直接写串口。
 *   - 摇杆线程（joystickLoop）：只把摇杆事件翻译成"手动指令"提交给仲裁模块。
 *   - 串口线程（serialLoop）：只做一件事 —— 从仲裁模块取最新结果写小车。
 *
 * 为什么识别+UI 放在主线程：OpenCV highgui 要求创建窗口、imshow、waitKey
 * 在同一个线程里做，拆开会不定期崩窗口或按键无响应。识别逻辑本身很轻，
 * 真正需要解耦的是"串口刷新"和"摇杆响应"，它们各自独立成线程。
 *
 * 仲裁模块（ControlArbiter）—— 唯一决定"当前该执行谁"的地方：
 *   1) 默认 MANUAL，必须显式按 a / 摇杆 X 才进入 AUTO（避免一上电就窜车）；
 *   2) 手动油门拉过 cfg::kTakeoverThrottle → 人手优先，立刻手动接管；
 *   3) 摇杆掉线/被拔出 → 强制停车 + 舵机回正；
 *   4) 视觉丢线 → AUTO 下速度归零、舵机回正，恢复后自动继续；
 *   5) 空格 = 急停（切 MANUAL 且速度 0），按 a / X 解除。
 *   串口线程和 HUD 都只是 arbiter.take() 的调用者，仲裁规则只有一份。
 *
 * 串口写者仲裁（重要）：
 *   CarControl::writeBuffer() 没有锁，多线程同时写会把帧字节插花。
 *   因此蜂鸣器也走仲裁模块、由串口线程代发，全程只有一个线程碰串口。
 */

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
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/opencv.hpp>

using std::cerr;
using std::cout;
using std::endl;
using std::memory_order_relaxed;
using std::shared_ptr;
using std::string;
using std::vector;

using namespace cv;

/* ==========================================================================
 * 信号：只置标志，真正的停车由各线程收尾
 * ========================================================================== */
namespace {
std::atomic_bool g_stop{false};
void onSignal(int) { g_stop.store(true, memory_order_relaxed); }
} // namespace

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

constexpr int kSerialPeriodMs = 20; // 串口刷新周期(ms)，兼作MCU心跳
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
 *   负值会环绕成一个巨大的数，被当成"左打死"而不是"右打死"）
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
                // 原实现是"假设对侧边缘在图像边界上"，弯道里误差很大。
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

        // ---- 加权控制中心（虚拟"正前方"点权重 1，点数少时平滑回正）----
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
        servo = clampServo(PWMSERVOMID + pwmDiff);
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
 * 仲裁模块：汇总"自动识别"与"摇杆手动"两路输入，输出唯一控制指令
 *   - 识别线程 publishAuto()  ：每帧提交丢线标志 + 舵机
 *   - 摇杆线程 publishManual()：提交速度 + 舵机 + 掉线状态
 *   - 串口线程 take()         ：唯一读取点，全部仲裁规则都在这里
 *   （HUD 也调用 take()，它只查询不改变状态，所以是 const）
 * ========================================================================== */
class ControlArbiter {
  public:
    struct Command {
        float speed = 0.f;            // 车速(m/s)
        uint16_t servo = PWMSERVOMID; // 舵机PWM
    };

    // ---------------- 生产者：自动识别模块 ----------------
    void publishAuto(bool lost, uint16_t servo) {
        std::lock_guard<std::mutex> lk(mtx);
        autoLost = lost;
        autoServo = servo;
        hasAuto = true;
    }

    // ---------------- 生产者：摇杆线程 ----------------
    void publishManual(float speed, uint16_t servo, bool stopped) {
        std::lock_guard<std::mutex> lk(mtx);
        manualSpeed = speed;
        manualServo = servo;
        hasManual = true;
        joyStopped = stopped;
    }
    void setJoyStopped(bool stopped) {
        std::lock_guard<std::mutex> lk(mtx);
        joyStopped = stopped;
    }

    // ---------------- 模式/急停/巡航速度（键盘与摇杆共用）----------------
    void setMode(Mode m) {
        std::lock_guard<std::mutex> lk(mtx);
        mode = m;
    }
    Mode getMode() const {
        std::lock_guard<std::mutex> lk(mtx);
        return mode;
    }
    void toggleMode() {
        std::lock_guard<std::mutex> lk(mtx);
        mode = (mode == Mode::Auto) ? Mode::Manual : Mode::Auto;
        panic = false;
    }
    void setPanic(bool on) {
        std::lock_guard<std::mutex> lk(mtx);
        panic = on;
    }
    bool isPanic() const {
        std::lock_guard<std::mutex> lk(mtx);
        return panic;
    }
    void setCruise(float v) {
        std::lock_guard<std::mutex> lk(mtx);
        cruise = v;
    }
    float getCruise() const {
        std::lock_guard<std::mutex> lk(mtx);
        return cruise;
    }
    bool isJoyStopped() const {
        std::lock_guard<std::mutex> lk(mtx);
        return joyStopped;
    }

    // ---------------- 蜂鸣器：由串口线程代发 ----------------
    void requestBuzzer(Buzzer sound) {
        std::lock_guard<std::mutex> lk(mtx);
        buzzer = sound;
        hasBuzzer = true;
    }
    bool takeBuzzer(Buzzer &out) {
        std::lock_guard<std::mutex> lk(mtx);
        if (!hasBuzzer)
            return false;
        out = buzzer;
        hasBuzzer = false;
        return true;
    }

    // ---------------- 消费者：串口线程 / HUD ----------------
    /** 仲裁规则都在这里；纯查询，可被多个线程同时调用 */
    Command take() const {
        std::lock_guard<std::mutex> lk(mtx);
        const Command stop{}; // 停 + 舵机中位
        if (joyStopped)       // 摇杆掉线：安全第一
            return stop;
        if (panic) // 急停
            return stop;
        if (mode == Mode::Manual || humanTakeoverLocked())
            return hasManual ? Command{manualSpeed, manualServo} : stop;
        return autoLost ? stop : Command{cruise, autoServo};
    }

    /** 手动油门是否已超过接管阈值（供 HUD 显示，不改变模式） */
    bool humanTakeover() const {
        std::lock_guard<std::mutex> lk(mtx);
        return humanTakeoverLocked();
    }

  private:
    bool humanTakeoverLocked() const {
        return hasManual && std::fabs(manualSpeed) > cfg::kTakeoverThrottle;
    }

    mutable std::mutex mtx;
    // 自动路
    uint16_t autoServo = PWMSERVOMID;
    bool autoLost = false;
    bool hasAuto = false;
    // 手动路
    float manualSpeed = 0.f;
    uint16_t manualServo = PWMSERVOMID;
    bool hasManual = false;
    bool joyStopped = false;
    // 仲裁状态
    Mode mode = Mode::Manual; // 默认手动，必须显式接管
    bool panic = false;
    float cruise = cfg::kCruiseSpeed;
    // 串口代发
    Buzzer buzzer = Buzzer::ding;
    bool hasBuzzer = false;
};

/* ==========================================================================
 * 串口线程：只做"从仲裁模块取最新结果 -> 写小车"
 * ========================================================================== */
void serialLoop(const shared_ptr<CarControl> &car, ControlArbiter &arbiter) {
    const auto period = std::chrono::milliseconds(cfg::kSerialPeriodMs);
    while (!g_stop.load()) {
        const auto cmd = arbiter.take();
        car->carControl(cmd.speed, cmd.servo); // 唯一的串口写入口
        Buzzer sound = Buzzer::ding;
        if (arbiter.takeBuzzer(sound))
            car->buzzerSound(sound);
        std::this_thread::sleep_for(period);
    }
    car->resetVelocity(); // 退出前停车 + 舵机归中
}

/* ==========================================================================
 * 摇杆线程：只生产"手动指令"，不碰串口
 * ========================================================================== */
void joystickLoop(const shared_ptr<JoyStick> &js, ControlArbiter &arbiter,
                  const std::atomic_bool &joyLost) {
    while (!g_stop.load()) {
        if (!js->waitEvent()) { // 最多等 50ms，用于检查退出/掉线
            arbiter.setJoyStopped(js->isStopped() || joyLost.load());
            continue;
        }

        float speed = 0.f;
        uint16_t servo = PWMSERVOMID;
        if (js->takeCarControl(speed, servo)) { // 有新指令（未知按键会带 0 速）
            arbiter.publishManual(speed, servo,
                                  js->isStopped() || joyLost.load());
            // 人手优先：AUTO 下把油门拉过阈值，立刻提示并交还手动
            if (std::fabs(speed) > cfg::kTakeoverThrottle &&
                arbiter.getMode() == Mode::Auto)
                arbiter.requestBuzzer(Buzzer::warning);
        }
        if (js->takeSampleOnce()) { // X 键：切换自动/手动（并解除急停）
            arbiter.toggleMode();
            arbiter.requestBuzzer(Buzzer::ding);
        }
        if (js->takeBuzzer())
            arbiter.requestBuzzer(Buzzer::ding);
    }
}

/* ==========================================================================
 * 屏幕 HUD（只在识别/UI 线程调用）
 * ========================================================================== */
static void drawHud(Mat &img, const AutoPilot &pilot, Mode mode, bool panic,
                    bool takeover, bool joyOk, bool joyLost, float cruise,
                    float speed, uint16_t servo, double fps) {
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

    line(string("[") + modeName(mode) + (takeover ? " +HUMAN" : "") +
             (panic ? " PANIC]" : "]") +
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
 * 识别/UI 线程入口
 * ========================================================================== */
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

        // ---------------- 摇杆（只读，由摇杆线程消费）----------------
        // joyLost 必须声明在 js 之前：js 析构时其接收线程可能回调本标志
        std::atomic_bool joyLost{false};
        shared_ptr<JoyStick> js =
            JoyStick::create([&joyLost] { joyLost = true; });
        if (!js) {
            cerr << "[Warn]: joystick unavailable, keyboard-only control."
                 << endl;
        }

        // ---------------- 图像源 ----------------
        const auto source = (argc > 1) ? argv[1] : "/dev/video0";
        Camera camera(source);

        Track track;
        track.rowCutUp = cfg::kRowCutUp;
        track.rowCutBottom = cfg::kRowCutBottom;

        // ---------------- 运行状态 ----------------
        ControlArbiter arbiter;
        AutoPilot pilot; // 只被识别/UI 线程访问，无需加锁

        cout << "==== track_view: track following + auto steering ====" << endl;
        cout << "source: " << source << endl;
        cout << "keys  : ESC quit | a auto/manual | space stop | +/- speed | "
                "w weight mode"
             << endl;
        cout << "pad   : X button = toggle auto/manual, throttle pulls > "
             << cfg::kTakeoverThrottle << " m/s = human takeover" << endl;
        cout << "start in MANUAL; press 'a' to engage auto steering." << endl;

        // ---------------- 启动串口线程 / 摇杆线程 ----------------
        car->buzzerSound(Buzzer::start); // 建线程前发，保持单写者
        std::thread serialThread([&] { serialLoop(car, arbiter); });
        std::thread joystickThread;
        if (js)
            joystickThread =
                std::thread([&] { joystickLoop(js, arbiter, joyLost); });

        auto timeLast = std::chrono::steady_clock::now();
        double fps = 0.0;

        // ---------------- 识别/UI 线程主循环 ----------------
        while (!g_stop.load()) {
            Mat img;
            if (!camera.read(img)) {
                cerr << "Camera frame unavailable; stopping." << endl;
                break;
            }
            Mat bin = camera.binarize(img);

            track.handle(bin);                 // [1] 边缘识别
            pilot.update(track.pointsEdgeLeft, // [2] 中心线 + 舵机
                         track.pointsEdgeRight);

            // [3] 识别结果交给仲裁模块（不直接写串口）
            arbiter.publishAuto(pilot.lost, pilot.servo);

            // [4] 按键：只改仲裁模块的状态
            const int key = waitKey(1);
            if (key == 27) { // ESC
                break;
            } else if (key == 'a' || key == 'A') {
                arbiter.toggleMode();
            } else if (key == ' ') {
                arbiter.setPanic(true);
                arbiter.setMode(Mode::Manual);
            } else if (key == '+' || key == '=') {
                arbiter.setCruise(std::min(
                    cfg::kSpeedMax, arbiter.getCruise() + cfg::kSpeedStep));
            } else if (key == '-' || key == '_') {
                arbiter.setCruise(std::max(
                    cfg::kSpeedMin, arbiter.getCruise() - cfg::kSpeedStep));
            } else if (key == 'w' || key == 'W') {
                pilot.cycleWeight();
            }

            // [5] 显示：[3] 之后取键更跟手；HUD 用 take() 查当前实际输出
            const auto cmd = arbiter.take();
            track.drawImage(img); // 绿=左边缘 黄=右边缘
            drawHud(img, pilot, arbiter.getMode(), arbiter.isPanic(),
                    arbiter.humanTakeover(), js != nullptr, joyLost.load(),
                    arbiter.getCruise(), cmd.speed, cmd.servo, fps);
            imshow("track_view", img);
            imshow("binarized", bin);

            const auto now = std::chrono::steady_clock::now();
            const double dt =
                std::chrono::duration<double>(now - timeLast).count();
            timeLast = now;
            if (dt > 0)
                fps = fps * 0.9 + (1.0 / dt) * 0.1; // 平滑
        }

        // ---------------- 收尾：停线程 -> 停车 ----------------
        g_stop.store(true, memory_order_relaxed);
        if (joystickThread.joinable())
            joystickThread.join();
        serialThread.join(); // 串口线程退出时会 resetVelocity()

        car->resetVelocity(); // 双保险：退出前一定停车 + 舵机归中
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
