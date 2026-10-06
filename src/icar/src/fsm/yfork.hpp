#pragma once

#include "fsm/fsm.hpp"

/**
 * @brief 岔路口(Y型)控制
 *
 */
class FsmYfork : public FSMState {
  public:
    FsmYfork(std::shared_ptr<Params> par);
    ~FsmYfork();
    void run(Mat &img);
    void show(Mat &img);
    FsmMode getMode();

  private:
    /**
     * @brief 岔路口执行步骤
     *
     */
    enum Step {
        FORKWAIT = 0, // 监测LABEL_FORK是否到达图像底
        FORKTURN,     // fork到底确认
        NONE,         // 监测左/右岔路标牌
        TURN,         // 已确认分支，单边转向
        RUN,          // 原有：单边完成，恢复正常跟线，保持YFORK状态
        DONE          // 本次岔路动作结束，回到监测
    };

    Step step = Step::FORKWAIT; // 岔路口步骤
    int countFork = 0;          // fork框底边到图像底的连续帧计数
    int countLeft = 0;          // 左岔路标牌连续帧计数
    int countRight = 0;         // 右岔路标牌连续帧计数
    int countSes = 0;           // 计数经久未凑够帧数则清零
    int timeout = 0;            // 动作执行计数
    int lostFrames = 0;         // 岔路标牌连续消失帧数
    bool branchLeft = false;    // 分支方向

    void setStep(Step st);
    void reset();
    void runBranch(bool left); // 单边转向
    void cutRows();            // 切行
};