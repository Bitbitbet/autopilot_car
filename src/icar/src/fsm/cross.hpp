#pragma once

#include "fsm/fsm.hpp"

/**
 * @brief 斑马线停车控制
 *
 */
class FsmCross : public FSMState {
  public:
    FsmCross(std::shared_ptr<Params> par);
    ~FsmCross();
    void run(Mat &img);
    void show(Mat &img);
    FsmMode getMode();

  private:
    /**
     * @brief 场景状态
     *
     */
    enum Step {
        NONE = 0, // AI未识别
        ENABLE,   // 场景使能
        STOP      // 停车
    };

    Step step = Step::NONE; // 场景状态
    uint16_t countRec = 0;  // AI场景识别计数器
    uint16_t countSes = 0;  // 场次计数器
    int timeout = 0;        // 超时计数器
    int countCross = 0;     // 斑马线屏蔽计数器
    int countInit = 0;      // 起点屏蔽计数器
    int countStop = 0;      // 停标连续识别计数器（连续2帧确认）
    bool stopSeen = false;  // 曾看到STOP停标

    void setStep(Step st);
};
