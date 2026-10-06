#include "fsm/yfork.hpp"

/**
 * @brief Construct a new Fsm yfork
 *
 * @param par
 */
FsmYfork::FsmYfork(std::shared_ptr<Params> par)
    : FSMState(FsmMode::yfork, par) {}

/**
 * @brief Destroy the Fsm Yfork
 *
 */
FsmYfork::~FsmYfork() {}

/**
 * @brief 状态切换
 *
 * @return FsmMode 切换后的状态
 */
FsmMode FsmYfork::getMode() {
    if (!params->config.yfork) // 该模式未启用
        return FsmMode::normal;
    if (step == Step::FORKWAIT) // 空闲
        return FsmMode::normal;
    return FsmMode::yfork; // 已进入fork处理
}

/**
 * @brief 运行FSM状态（循环主程序）
 *
 * 识别左/右岔路标牌：看见左标牌走左边，看见右标牌走右边
 */
void FsmYfork::run(Mat &img) {
    if (!params->config.yfork)
        return;

    switch (step) {
    case Step::FORKWAIT: // 切删左线单边
    {
        PredictResult resFork; // 拐角fork标志：选最高分
        resFork.score = 0;
        for (int i = 0; i < params->results.size(); i++) {
            if (params->results[i].type == LABEL_FORK &&
                params->results[i].width < 100 &&
                params->results[i].height < 120 &&
                params->results[i].score > resFork.score)
                resFork = params->results[i];
        }

        // 标志中心计数
        if (resFork.score > 0 &&
            (resFork.y + resFork.height / 2) > ROWSIMAGE * 0.28)
            countFork++;
        else
            countFork = 0;

        if (countFork > 5) // 连续几帧 → 进入删左线单边
            setStep(Step::FORKTURN);
        break;
    }

    case Step::FORKTURN: // 固定只持续3帧，然后切NONE
    {
        params->track->pointsEdgeLeft.clear();
        timeout++; // 单边帧计数
        if (timeout >= 1)
            setStep(Step::NONE);
        break;
    }

    case Step::NONE: // 监测左/右岔路标牌
    {
        // [1] 左右岔路标牌按置信度取最高
        float bestLeft = 0.0f, bestRight = 0.0f; // 本帧左/右标牌最高置信度
        for (int i = 0; i < params->results.size(); i++) {
            if (params->results[i].type == LABEL_YLEFT &&
                params->results[i].score > bestLeft)
                bestLeft = params->results[i].score;
            else if (params->results[i].type == LABEL_YRIGHT &&
                     params->results[i].score > bestRight)
                bestRight = params->results[i].score;
        }
        bool detectLeft = bestLeft > 0.0f &&
                          bestLeft >= bestRight; // 左分更高（或只有左）判左
        bool detectRight =
            bestRight > 0.0f && bestRight > bestLeft; // 右分更高判右

        // [2] 连续帧确认
        if (detectLeft) {
            countLeft++;
            countRight = 0;
            branchLeft = true;
        } else if (detectRight) {
            countRight++;
            countLeft = 0;
            branchLeft = false;
        }

        if (detectLeft)
            runBranch(true); // 左岔路:删左线,立即左拐
        else if (detectRight)
            runBranch(false); // 右岔路:删右线,立即右拐

        // [3] 防止计数长期累积
        if (countLeft > 0 || countRight > 0) {
            countSes++;
            if (countSes >= 6) {
                countSes = 0;
                countLeft = 0;
                countRight = 0;
            }
        }

        if (countLeft >= 2 ||
            countRight >= 2) // 识别1帧即进入岔路状态,维持单边动作
            setStep(Step::TURN);
        else if (++timeout >= 50) // 超时保护
            reset();
        break;
    }

    case Step::TURN: // 单边转向
    {
        timeout++;
        runBranch(branchLeft);

        // 判定本帧屏幕里还有没有左/右岔路标牌
        bool signSeen = false;
        for (int i = 0; i < params->results.size(); i++) {
            if (params->results[i].type == LABEL_YLEFT ||
                params->results[i].type == LABEL_YRIGHT) {
                signSeen = true;
                break;
            }
        }
        if (!signSeen)
            lostFrames++; // 标牌消失
        else
            lostFrames = 0; // 又出现了，重置

        if (lostFrames >= 5 ||
            timeout >=
                30) // 标牌消失持续5帧才恢复正常跟线,让单边动作充分执行不被打回中线
            setStep(Step::RUN);
        break;
    }

    case Step::RUN: // 恢复正常跟线
    {
        timeout++;         // 从0累计
        if (timeout >= 30) // 正常巡线30帧后退出yfork切正常
            setStep(Step::DONE);
        break;
    }

    case Step::DONE: {
        reset();
        break;
    }
    }
}

/**
 * @brief 图形化显示FSM数据
 *
 * @param img
 */
void FsmYfork::show(Mat &img) {
    if (step == Step::NONE || params->mode != FsmMode::yfork)
        return;

    putText(img, "[Yfork]" + std::string(branchLeft ? " L" : " R"),
            Point(COLSIMAGE / 2 - 50, 20), cv::FONT_HERSHEY_TRIPLEX, 0.5,
            cv::Scalar(0, 255, 0), 0.5);
}

/**
 * @brief 设置新状态
 *
 * @param st
 */
void FsmYfork::setStep(Step st) {
    step = st;
    countFork = 0;  // fork底边计数
    countLeft = 0;  // AI左岔路识别计数器
    countRight = 0; // AI右岔路识别计数器
    countSes = 0;   // 防累积计数器
    timeout = 0;    // 动作执行计数器（每次状态切换都归零）
}

/**
 * @brief 复位FSM状态
 *
 */
void FsmYfork::reset() {
    step = Step::FORKWAIT; // 回到空闲：等下一个拐角fork
    countFork = 0;
    countLeft = 0;
    countRight = 0;
    countSes = 0;
    timeout = 0;
    lostFrames = 0;
    branchLeft = false;
}

/**
 * @brief 岔路行驶动作（左/右岔路动作不同）
 *
 * 强制单边：删除对侧边缘
 *
 * @param left true：左岔路（走左边）| false：右岔路（走右边）
 */
void FsmYfork::runBranch(bool left) {
    if (left) // 左单边
        params->track->pointsEdgeLeft.clear();
    else // 右单边
        params->track->pointsEdgeRight.clear();
}
