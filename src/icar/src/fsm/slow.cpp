#include "fsm/slow.hpp"

/**
 * @brief Construct a new Fsm Park
 *
 * @param par
 */
FsmSlow::FsmSlow(std::shared_ptr<Params> par) : FSMState(FsmMode::slow, par) {}

/**
 * @brief Destroy the Fsm Park
 *
 */
FsmSlow::~FsmSlow() {}

/**
 * @brief 检查状态切换
 *
 * @return FsmMode 切换后的状态
 */
FsmMode FsmSlow::getMode() {
    // 输出场景状态结果
    if (!params->config.slow || step == Step::NONE)
        return FsmMode::normal;

    return FsmMode::slow;
}

/**
 * @brief 运行FSM状态（循环主程序）
 *
 */
void FsmSlow::run(Mat &img) {
    if (!params->config.slow) // 该模式未启用
        return;

    switch (step) {
    case Step::NONE: // AI标志检测

        params->ctrl.slow = false; // 车辆减速标志
        for (int i = 0; i < params->results.size(); i++) {
            if (params->results[i].type == LABEL_LIMIT) // AI识别标志
            {
                if (params->results[i].height < 100 &&
                    params->results[i].width < 80) // 标志距离计算
                {
                    countRec++;
                    break;
                }
            }
        }
        if (countRec >= 2) {
            params->ctrl.slow =
                true; // 立马置减速标志：切换帧即生效，避免掉到动态速度
            setStep(Step::ENABLE);
        }

        if (countRec > 0) // 识别AI标志后开始场次计数
        {
            countSes++;
            if (countSes >= 5) {
                countRec = 0;
                countSes = 0;
            }
        }
        break;

    case Step::ENABLE: // 减速阶段
        timeout++;
        params->ctrl.slow = true; // 车辆减速标志
        for (int i = 0; i < params->results.size(); i++) {
            if (params->results[i].type == LABEL_LIMIT) // AI识别标志
            {
                if (params->results[i].height < 100 &&
                    params->results[i].width < 80 &&
                    (params->results[i].y + params->results[i].height) >
                        ROWSIMAGE * 0.2) // 标志距离计算
                {
                    timeout = 0;
                }
            }
            if (params->results[i].type == LABEL_UNLIMIT) // AI识别标志
            {
                if (params->results[i].height < 100 &&
                    params->results[i].width < 80 &&
                    (params->results[i].y + params->results[i].height) >
                        ROWSIMAGE * 0.2) // 标志距离计算
                {
                    countRec++;
                    break;
                }
            }
        }
        if (countRec >= 2 || timeout > 110) // 减速超时时长
            setStep(Step::NONE);

        if (countRec > 0) // 识别AI标志后开始场次计数
        {
            countSes++;
            if (countSes >= 5) {
                countRec = 0;
                countSes = 0;
            }
        }
        break;

    default:
        break;
    }
}

/**
 * @brief 图形化显示FSM数据
 *
 * @param img
 */
void FsmSlow::show(Mat &img) {
    if (params->mode != FsmMode::slow)
        return;

    putText(img, "[6] Slow", Point(COLSIMAGE / 2 - 50, 20),
            cv::FONT_HERSHEY_TRIPLEX, 0.5, cv::Scalar(0, 255, 0), 0.5);
}

/**
 * @brief 设置新状态
 *
 * @param step
 */
void FsmSlow::setStep(Step st) {
    step = st;
    countRec = 0; // AI场景识别计数器
    countSes = 0; // 场次计数器
    timeout = 0;  // 超时计数器
}