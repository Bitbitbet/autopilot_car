#include "center.hpp"
#include "control.hpp"
#include "detection.hpp"
#include "fsm/busy.hpp"
#include "fsm/cross.hpp"
#include "fsm/fork.hpp"
#include "fsm/obstacle.hpp"
#include "fsm/park.hpp"
#include "fsm/slow.hpp"
#include "fsm/stop.hpp"
#include "fsm/yfork.hpp"
#include "latest_result.hpp"
#include "loop.hpp"
#include "motion.hpp"
#include "predeal.hpp"
#include "show.hpp"
#include "stop_signal.hpp"
#include <condition_variable>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <sys/types.h>
#include <sys/wait.h>

using namespace cv;
using std::atomic;
using std::cerr;
using std::condition_variable;
using std::cout;
using std::endl;
using std::lock_guard;
using std::make_shared;
using std::mutex;
using std::shared_ptr;
namespace this_thread = std::this_thread;
namespace chrono = std::chrono;

namespace {
auto &stopRequested = car_signal::requested;
}

class Icar {
  private:
    /**
     * @brief 状态机管理
     *
     */
    struct FsmFactory {
        shared_ptr<FsmBusy> busy;         // 施工区控制（原避障控制）
        shared_ptr<FsmObstacle> obstacle; // 避障控制(锥桶绕行 + 行人礼让)
        shared_ptr<FsmPark> park;         // 停车场控制
        shared_ptr<FsmYfork> yfork;       // 岔路口控制
        shared_ptr<FsmStop> stop;         // 停车区控制
        shared_ptr<FsmCross> cross;       // 斑马线停车控制
        shared_ptr<FsmFork> fork;         // 停车场岔路控制
        shared_ptr<FsmSlow> slow;         // 慢行区控制
    };

    FsmFactory fsmFactory;       // 状态机管理
    bool parkWasActive = false;  // 上一帧停车场是否激活（检测退出过渡）
    int yforkLock = 0;           // 停车场退出后锁定 yfork 的剩余帧数
    shared_ptr<Predeal> predeal; // 图像预处理类
    shared_ptr<Show> show;       // 初始化UI显示窗口
    shared_ptr<cv::VideoCapture> capture; // Opencv相机类
    shared_ptr<Detection> detection;      // 目标检测类
    shared_ptr<CarControl> control;       // UART通信类
    shared_ptr<Params> params;            // 车辆状态参数（FSM共享传递）
    shared_ptr<Loops> loops;              // 子线程循环
    shared_ptr<Center> center;            // 控制中心处理类
    shared_ptr<Motion> motion;            // 运动控制器

    // 全局共享数据链
    cv::Mat imgShare;
    mutex mtxImg;
    condition_variable cvImg;
    atomic<bool> readyImg{false};
    atomic<bool> shuttingDown{false};
    LatestResult<std::vector<PredictResult>> modelResults;

    /**
     * @brief 鼠标的事件回调函数
     *
     */
    static void callbackMouse(int event, int x, int y, int flags,
                              void *userdata) {
        Icar *self = static_cast<Icar *>(userdata);
        if (self)
            self->handleMouse(event, x, y, flags);
    }
    void handleMouse(int event, int x, int y, int flags) {
        double value;
        switch (event) {
        case cv::EVENT_MOUSEWHEEL: // 鼠标滑球
        {
            value = cv::getMouseWheelDelta(flags); // 获取滑球滚动值
            if (value > 0)
                show->index++;
            else if (value < 0)
                show->index--;

            if (show->index < 0)
                show->index = 0;
            if (show->index > show->frameMax)
                if (show->index > show->frameMax)
                    show->index = show->frameMax;
            break;
        }
        default:
            break;
        }
    }

    /**
     * @brief AI 模型推理
     *
     */
    void runModel() {
        try {
            std::unique_lock<std::mutex> lock(mtxImg);
            cvImg.wait_for(lock, chrono::milliseconds(50), [this] {
                return shuttingDown.load() || readyImg.load();
            });
            if (shuttingDown || !readyImg)
                return;
            cv::Mat img = imgShare.clone(); // 图像拷贝出来再释放锁
            readyImg = false;
            lock.unlock();

            // 启动AI推理
            detection->inference(img);
            modelResults.publish(detection->results);
        } catch (...) {
            shuttingDown = true;
            control->resetVelocity();
            modelResults.fail(std::current_exception());
        }
    }

    /**
     * @brief 有限状态机任务执行
     *
     */
    void runFsm(Mat &img) {
        if (params->mode == FsmMode::fork || params->mode == FsmMode::busy ||
            params->mode == FsmMode::slow) // 状态复位
            params->mode = FsmMode::normal;

        fsmFactory.stop->run(img); // 停车区识别与规划
        params->mode = fsmFactory.stop->getMode();
        fsmFactory.cross->run(img); // 斑马线停车识别与规划
        params->mode = fsmFactory.cross->getMode();
        if (params->mode == FsmMode::normal || params->mode == FsmMode::park ||
            params->mode == FsmMode::cross) // 停车场图像处理
        {
            fsmFactory.park->run(img);
            FsmMode mode = fsmFactory.park->getMode();
            if (mode != FsmMode::normal)
                params->mode = mode;
        }
        // 停车场退出后 100 帧内屏蔽 yfork 触发：park 从激活回到 NORMAL
        // 的过渡视为"刚退出"
        {
            bool parkActive = (fsmFactory.park->getMode() != FsmMode::normal);
            if (parkWasActive && !parkActive)
                yforkLock = 100; // 刚退出停车场，锁定 yfork
            parkWasActive = parkActive;
        }
        if (yforkLock > 0)
            yforkLock--;
        if (params->mode == FsmMode::normal) // 岔路识别与规划
        {
            fsmFactory.fork->run(img);
            params->mode = fsmFactory.fork->getMode();
        }
        fsmFactory.obstacle->run(img); // 避障

        fsmFactory.slow->run(img); // 慢行区识别与规划
        if (params->mode == FsmMode::normal)
            params->mode = fsmFactory.slow->getMode();

        fsmFactory.busy->run(img); // 施工区识别与规划
        if (params->mode == FsmMode::normal)
            params->mode = fsmFactory.busy->getMode();

        if (yforkLock ==
            0) // 停车场退出后100帧内锁定 yfork，防止刚出停车场误触发岔路
        {
            fsmFactory.yfork->run(img); // 岔路口识别与规划
            if (params->mode == FsmMode::normal)
                params->mode = fsmFactory.yfork->getMode();
        }

        if (params->mode != params->modeLast) {
            control->buzzerSound(Buzzer::ding); // 提示音效
            params->modeLast = params->mode;
        }
    }

  public:
    double fpsDisplay = 0.0;
    /**
     * @brief 参数初始化
     *
     */
    Icar() {
        control = CarControl::create();
        if (!control)
            throw std::runtime_error("Cannot open car serial port");
        control->resetVelocity();       // 在配置和模型初始化之前清除残留速度
        params = make_shared<Params>(); // 初始化参数
        center = make_shared<Center>(); // 控制中心处理类
        motion = make_shared<Motion>(); // 运动控制器
        predeal = make_shared<Predeal>(params->config.binary); // 图像预处理类
        detection =
            make_shared<Detection>(params->config.model,
                                   params->config.score); // AI模型初始化

        control->buzzerSound(Buzzer::ok); // 提示音效

        // 相机初始化
        // USB摄像头初始化
        if (params->config.debug)
            capture = make_shared<cv::VideoCapture>(
                params->config.video); // 打开本地视频
        else
            capture =
                make_shared<cv::VideoCapture>("/dev/video0"); // 打开摄像头
        if (!capture->isOpened()) {
            cerr << "[Error]: Can not open video device!" << endl;
            throw std::runtime_error("Cannot open video device");
        }
        capture->set(cv::CAP_PROP_FRAME_WIDTH, COLSCAMERA);  // 设置图像分辨率
        capture->set(cv::CAP_PROP_FRAME_HEIGHT, ROWSCAMERA); // 设置图像分辨率
        capture->set(cv::CAP_PROP_FPS, 30);                  // 设置帧率

        if (params->config.debug) {
            show = make_shared<Show>(4); // 调试UI初始化
            show->frameMax = capture->get(cv::CAP_PROP_FRAME_COUNT) - 1;
            cv::createTrackbar("Frame", "ICAR", &show->index, show->frameMax,
                               [](int, void *) {}); // 创建Opencv图像滑条控件
            cv::setMouseCallback("ICAR",
                                 this->callbackMouse); // 创建鼠标键盘快捷键事件
        }

        // FSM有限状态机初始化
        fsmFactory.busy = make_shared<FsmBusy>(params); // 施工区控制实例化
        fsmFactory.obstacle =
            make_shared<FsmObstacle>(params);           // 避障控制实例化
        fsmFactory.park = make_shared<FsmPark>(params); // 停车场控制实例化
        fsmFactory.stop = make_shared<FsmStop>(params); // 斑马线停车控制实例化
        fsmFactory.cross =
            make_shared<FsmCross>(params);              // 斑马线停车控制实例化
        fsmFactory.fork = make_shared<FsmFork>(params); // 停车场岔路控制实例化
        fsmFactory.slow = make_shared<FsmSlow>(params); // 慢行区控制实例化
        fsmFactory.yfork = make_shared<FsmYfork>(params); // 岔路口控制实例化

        // 启动AI推理子线程
        loops = make_shared<Loops>("LoopAI", 1.f / 30.f,
                                   std::bind(&Icar::runModel, this));
        loops->start(); // RL开始推理

        cout << "[OK]: Params initial succeed!" << endl;
    };
    ~Icar() {
        stop(); // 在等待推理线程结束之前发送停车指令
        {
            lock_guard<std::mutex> lock(mtxImg);
            shuttingDown = true;
        }
        cvImg.notify_all();
        if (loops)
            loops->shutdown();
    }

    void stop() noexcept {
        if (control)
            control->resetVelocity();
    }

    bool finished() const { return params->quit; }

    /**
     * @brief 程序主循环
     *
     */
    void mainLoop() {
        if (stopRequested) {
            params->quit = true;
            stop();
            return;
        }
        modelResults.take(params->results); // FSM与绘图只读取主线程的结果快照
        //[01] 视频源读取
        cv::Mat img;
        if (params->config.debug) {               // 综合显示调试UI窗口
            if (show->indexLast == show->index) { // 图像帧未更新
                if (control->keypress) {
                    control->buzzerSound(Buzzer::finish); // 祖传提示音效
                    cout << "-----> System Exit!!! <-----" << endl;
                    params->quit = true;
                    stop();
                    return;
                }
                show->show();         // 显示综合绘图
                control->sendHeart(); // 发送给服务器在线心跳
                usleep(10 * 1000);    // us延迟
                return;
            }

            capture->set(cv::CAP_PROP_POS_FRAMES, show->index); // 设置读取帧
            if (!capture->read(img) || img.empty()) {
                params->quit = true;
                stop();
                return;
            }
            show->indexLast = show->index;
        } else if (!capture->read(img) || img.empty()) {
            cerr << "[Error]: Camera frame unavailable; stopping." << endl;
            params->quit = true;
            stop();
            return;
        }

        //[02] 图像存储
        if (params->config.saveImg && !params->config.debug) // 存储原始图像
            savePicture(img);
        else if (params->config.saveImg && params->config.debug) // 存储调式图像
            show->save = true;

        //[03] 图像预处理
        cv::Mat imgBin;
        predeal->correct(img); // 图像矫正
        /*---------------子线程共享数据，避免浅拷贝-----------------*/
        {
            lock_guard<std::mutex> lock(mtxImg);
            imgShare = img.clone();
            readyImg = true;
        }
        cvImg.notify_one();
        /*-------------------------------------------------------*/
        imgBin = predeal->binarize(img); // 图像二值化

        //[04] 赛道识别
        params->track->handle(imgBin);
        if (params->config.debug) {
            show->setNewWindow(1, "Bin", imgBin);
            cv::Mat imgTrack = img.clone();
            params->track->drawImage(imgTrack); // 图像绘制赛道识别结果
            show->setNewWindow(2, "Track", imgTrack);
            if (params->config.saveIpm && params->config.saveImg) {
                cv::Mat imgIpm;
                ipm.homography(imgTrack, imgIpm);
                savePicture(imgIpm); // 保存图像
            }
        }

        //[05] 有限状态机任务执行
        params->ctrl.fitting = false;
        runFsm(imgBin);

        //[06] 控制中心计算
        center->fitting(params);

        if (params->quit || stopRequested) {
            params->quit = true;
            stop();
            return;
        }

        //[07] 车辆运动控制
        motion->poseControl(params);
        motion->speedControl(params);

        //[08] 综合显示调试UI窗口
        if (params->config.debug) {
            detection->drawBox(img, params->results);
            center->drawImage(params, img); // 图像绘制控制路径
            motion->drawImage(params, img); // 图像绘制速度
            show->setNewWindow(3, "Ctrl", img);

            // 特殊区域图像处理结果显示
            Mat imgRes =
                Mat::zeros(Size(COLSIMAGE, ROWSIMAGE), CV_8UC3); // 创建全黑图像
            fsmFactory.busy->show(imgRes);
            fsmFactory.park->show(imgRes);
            fsmFactory.stop->show(imgRes);
            fsmFactory.cross->show(imgRes);
            fsmFactory.fork->show(imgRes);
            fsmFactory.slow->show(imgRes);
            fsmFactory.yfork->show(imgRes);
            show->setNewWindow(4, "FSM", imgRes);
        } else // 实车控制
        {
            if (stopRequested || shuttingDown || control->exitBoot) {
                params->quit = true;
                stop();
                return;
            }
            control->carControl(params->ctrl.speed,
                                params->ctrl.servo); // 串口通信控制车辆
        }

        if (params->quit) // 停标触发退出：先舵机归中再退出进程
        {
            params->ctrl.servo = PWMSERVOMID;    // 舵机归中
            params->ctrl.speed = 0;              // 停车
            control->carControl(0, PWMSERVOMID); // 发送归中+停车指令
            cout << "-----> System Exit (servo centered)! <-----" << endl;
            params->quit = true;
            return;
        }
    }
};

int main() {
    car_signal::install();
    try {
        Icar icar;

        for (int i = 0; i < 100 && !stopRequested; ++i)
            this_thread::sleep_for(chrono::milliseconds(100));

        const chrono::milliseconds durations(1000 / 30); // 控制周期：30Fps
        while (!stopRequested && !icar.finished()) {
            auto timeStart = chrono::high_resolution_clock::now();
            icar.mainLoop(); // 系统主线程

            // 计算处理耗时
            auto timeEnd = chrono::high_resolution_clock::now();
            auto elapsed = chrono::duration_cast<chrono::milliseconds>(
                timeEnd - timeStart);
            // printf(">> FrameTime: %ldms | %.2ffps \n", elapsed.count(),
            // 1000.0 / elapsed.count());
            icar.fpsDisplay =
                elapsed.count() > 0 ? 1000.0 / elapsed.count() : 0.0;

            // 睡眠等待
            if (elapsed < durations)
                this_thread::sleep_for(durations - elapsed);
        }

        icar.stop();
        return 0;
    } catch (const std::exception &e) {
        cerr << "[Error]: " << e.what() << endl;
        return 1;
    } catch (...) {
        cerr << "[Error]: Unexpected failure." << endl;
        return 1;
    }
}
