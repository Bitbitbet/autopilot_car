#pragma once

#include "fsm/fsm.hpp"

/**
 * @brief 减速区识别与规划
 *
 */
class FsmSlow : public FSMState {
  public:
    FsmSlow(std::shared_ptr<Params> par);
    ~FsmSlow();
    void run(Mat &img);
    void show(Mat &img);
    FsmMode getMode();

  private:
    /**
     * @brief 场景状态
     *
     */
    enum Step {
        NONE = 0, // 未知状态
        ENABLE,   // 场景使能
    };
    Step step = Step::NONE; // 场景状态
    uint16_t countRec = 0;  // AI场景识别计数器
    uint16_t countSes = 0;  // 场次计数器
    uint16_t timeout = 0;   // 程序退出计数器

    void setStep(Step st);
};
