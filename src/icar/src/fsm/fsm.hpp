#pragma once

#include "params.hpp"

/**
 * @brief 有限状态机状态基类
 *
 */
class FSMState {
  public:
    FSMState(FsmMode mode, std::shared_ptr<Params> par)
        : fsmMode(mode), params(par) {}
    virtual ~FSMState() = default;
    virtual void run(Mat &img) = 0;
    virtual void show(Mat &img) = 0;
    virtual FsmMode getMode() { return FsmMode::normal; }
    FsmMode fsmMode;

  protected:
    std::shared_ptr<Params> params;
};