#pragma once

#include "fsm/fsm.hpp"

/**
 * @brief 施工区控制
 *
 */
class FsmBusy : public FSMState {
  public:
    FsmBusy(std::shared_ptr<Params> par);
    ~FsmBusy();
    void run(Mat &img);
    void show(Mat &img);
    FsmMode getMode();
    bool slowing = false; // 减速使能

  private:
    /**
     * @brief 施工区执行步骤
     *
     */
    enum Step {
        NONE = 0, // 未知状态
        ENABLE,   // 施工区使能
        FORKIN,   // 进区岔路转向
        RUNNING,  // 区内巡线（限速/避障）
        FORKOUT,  // 出区岔路转向
    };

    Step step = Step::NONE;  // 施工区步骤
    bool enable = false;     // 场景检测使能标志
    int countRec = 0;        // AI场景识别计数器
    int countSes = 0;        // 场次计数器
    int timeout = 0;         // 超时计数器
    bool slowingIn = false;  // 区内低速标志（限速）
    PredictResult resultObs; // 避障目标锥桶

    void setStep(Step st);
    void replanTracking(bool left);
    bool findSymbols(std::vector<PredictResult> results, int label);
    void curtailTracking(bool left);
};