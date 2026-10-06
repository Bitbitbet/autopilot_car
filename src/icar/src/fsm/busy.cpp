#include "fsm/busy.hpp"

using std::vector;

/**
 * @brief Construct a new Fsm Busy
 *
 * @param par
 */
FsmBusy::FsmBusy(std::shared_ptr<Params> par) : FSMState(FsmMode::busy, par) {}

/**
 * @brief Destroy the Fsm Busy
 *
 */
FsmBusy::~FsmBusy() {}

/**
 * @brief 检查状态切换
 *
 * @return FsmMode 切换后的状态
 */
FsmMode FsmBusy::getMode() {
    if (step == Step::NONE || !params->config.busy)
        return FsmMode::normal;
    else
        return FsmMode::busy;
}

/**
 * @brief 运行FSM状态（循环主程序）
 *
 */
void FsmBusy::run(Mat &img) {
    if (!params->config.busy) // 该模式未启用
        return;

    resultObs = PredictResult();

    switch (step) {
    case Step::NONE: // AI未识别：等待施工区标志(busy)
    {
        // 连续2帧确认识别到施工区标志
        bool busyNow = false;
        for (int i = 0; i < params->results.size(); i++) {
            if (params->results[i].type == LABEL_BUSY) // 施工区标志
            {
                busyNow = true;
                break;
            }
        }
        if (busyNow)
            countRec++;
        else
            countRec = 0;
        if (countRec >= 2)
            setStep(Step::ENABLE); // 确认进入施工区
        break;
    }

    case Step::ENABLE: // 已确认施工区
    {
        timeout++;
        // 搜索最底行的双箭头(离车最近)
        PredictResult choice;
        choice.score = 0;
        choice.y = 0;
        for (int i = 0; i < params->results.size(); i++) {
            if (params->results[i].type == LABEL_CHOICE) // 双向箭头
            {
                if (params->results[i].y > choice.y &&
                    params->results[i].width < 120 &&
                    params->results[i].height < 120)
                    choice = params->results[i]; // 取最底行
            }
        }
        if (choice.score > 0) // 检测到箭头
        {
            timeout = 0;
            if ((choice.y + choice.height / 2) > ROWSIMAGE * 0.33) // 箭头靠近
                countRec++;
            if (countRec > 1)
                setStep(Step::FORKIN); // 左拐进区
        }
        if (timeout > 120)       // 超时未等到箭头：错失，复位
            setStep(Step::NONE); // 复位
        break;
    }

    case Step::FORKIN: // 进区岔路转向（左拐）
    {
        timeout++;
        replanTracking(true);       // 车道线重绘（左转进入）
        if (timeout > 23)           // 转向超时
            setStep(Step::RUNNING); // 进入区内巡线
        break;
    }

    case Step::RUNNING: // 区内巡线
    {
        //[03] 出区检测：左转标志
        timeout++;
        PredictResult resLeft;
        resLeft.score = 0;
        for (int i = 0; i < params->results.size(); i++) // 搜索左转箭头
        {
            if (params->results[i].type == LABEL_LEFT &&
                params->results[i].width < 100 &&
                params->results[i].height < 120 &&
                params->results[i].score > resLeft.score)
                resLeft = params->results[i]; // 取置信度最高的左转箭头
        }
        if (resLeft.score > 0 &&
            (resLeft.y + resLeft.height / 2) > ROWSIMAGE * 0.3)
            countRec++; // 左转标志开到画面下
        if (countRec > 2)
            setStep(Step::FORKOUT); // 出区左拐
        break;
    }

    case Step::FORKOUT: // 出区岔路转向（左拐）
    {
        timeout++;
        replanTracking(true); // 车道线重绘

        // 搜索左转标志：转向期间仍看到则hold，防止过早切出
        bool leftSign = false;
        for (int i = 0; i < params->results.size(); i++) {
            if (params->results[i].type == LABEL_LEFT &&
                params->results[i].width < 100 &&
                params->results[i].height < 120) {
                leftSign = true;
                break;
            }
        }
        if (leftSign)
            timeout = 0; // 左转标志未丢失，继续转向

        if (timeout > 20) // 转向超时
        {
            setStep(Step::NONE); // 回到起始状态
        }
        break;
    }
    }
}

/**
 * @brief 图形化显示FSM数据
 *
 * @param img
 */
void FsmBusy::show(Mat &img) {
    if (step == Step::NONE || params->mode != FsmMode::busy)
        return;

    if (resultObs.x > 0 && resultObs.y > 0) {
        cv::Rect rect(resultObs.x, resultObs.y, resultObs.width,
                      resultObs.height);
        cv::rectangle(img, rect, cv::Scalar(0, 0, 255), 1);
    }

    putText(img, "[1] Busy", Point(COLSIMAGE / 2 - 50, 20),
            cv::FONT_HERSHEY_TRIPLEX, 0.5, cv::Scalar(0, 255, 0), 0.5);
}

/**
 * @brief 设置新状态
 *
 * @param st
 */
void FsmBusy::setStep(Step st) {
    step = st;
    countRec = 0; // AI场景识别计数器
    countSes = 0; // 场次计数器
    timeout = 0;  // 超时计数器
}

/**
 * @brief 车道线重绘（岔路转向）
 *
 * @param left true：左转 | false：右转
 */
void FsmBusy::replanTracking(bool left) {
    params->track->pointsEdgeLeft.clear();  // 清空原来数据
    params->track->pointsEdgeRight.clear(); // 清空原来数据

    if (left) {
        // 左车道线
        PointX startPoint = PointX(ROWSIMAGE - 10, 1); // 补线起点:固定左下角
        PointX endPoint = PointX(ROWSIMAGE / 3, 1);    // 补线终点
        PointX midPoint = PointX((startPoint.x + endPoint.x) * 0.3,
                                 (startPoint.y + endPoint.y) / 2); // 补线中点
        vector<PointX> repairPoints = {startPoint, midPoint, endPoint};
        vector<PointX> modifyEdge =
            Bezier(0.02, repairPoints); // 三阶贝塞尔曲线拟合
        params->track->pointsEdgeLeft = modifyEdge;
        // 右车道线
        startPoint =
            PointX(ROWSIMAGE - 10, COLSIMAGE * 0.8); // 补线起点:固定左下角
        endPoint = PointX(ROWSIMAGE / 3, 30);        // 补线终点
        midPoint = PointX((startPoint.x + endPoint.x) * 0.3,
                          (startPoint.y + endPoint.y) / 2); // 补线中点
        repairPoints = {startPoint, midPoint, endPoint};
        modifyEdge = Bezier(0.02, repairPoints); // 三阶贝塞尔曲线拟合
        params->track->pointsEdgeRight = modifyEdge;
    } else {
        // 左车道线
        PointX startPoint =
            PointX(ROWSIMAGE - 10, COLSIMAGE * 0.2); // 补线起点:固定左下角
        PointX endPoint = PointX(ROWSIMAGE / 3, COLSIMAGE - 1); // 补线终点
        PointX midPoint = PointX((startPoint.x + endPoint.x) * 0.3,
                                 (startPoint.y + endPoint.y) / 2); // 补线中点
        vector<PointX> repairPoints = {startPoint, midPoint, endPoint};
        vector<PointX> modifyEdge =
            Bezier(0.02, repairPoints); // 三阶贝塞尔曲线拟合
        params->track->pointsEdgeLeft = modifyEdge;
        // 右车道线
        startPoint =
            PointX(ROWSIMAGE - 10, COLSIMAGE - 1);       // 补线起点:固定左下角
        endPoint = PointX(ROWSIMAGE / 3, COLSIMAGE - 1); // 补线终点
        midPoint = PointX((startPoint.x + endPoint.x) * 0.3,
                          (startPoint.y + endPoint.y) / 2); // 补线中点
        repairPoints = {startPoint, midPoint, endPoint};
        modifyEdge = Bezier(0.02, repairPoints); // 三阶贝塞尔曲线拟合
        params->track->pointsEdgeRight = modifyEdge;
    }
}

/**
 * @brief 搜索AI标志
 *
 * @param label
 * @return true
 * @return false
 */
bool FsmBusy::findSymbols(vector<PredictResult> results, int label) {
    for (int i = 0; i < results.size(); i++) {
        if (results[i].type == label) // AI标志
            return true;
    }

    return false;
}

/**
 * @brief 缩减优化车道线（双车道→单车道）
 *
 * @param left
 */
void FsmBusy::curtailTracking(bool left) {
    if (left) // 向左侧缩进
    {
        if (params->track->pointsEdgeRight.size() >
            params->track->pointsEdgeLeft.size())
            params->track->pointsEdgeRight.resize(
                params->track->pointsEdgeLeft.size());

        for (int i = 0; i < params->track->pointsEdgeRight.size(); i++) {
            params->track->pointsEdgeRight[i].y =
                (params->track->pointsEdgeRight[i].y +
                 params->track->pointsEdgeLeft[i].y) /
                2;
        }
    } else // 向右侧缩进
    {
        if (params->track->pointsEdgeRight.size() <
            params->track->pointsEdgeLeft.size())
            params->track->pointsEdgeLeft.resize(
                params->track->pointsEdgeRight.size());

        for (int i = 0; i < params->track->pointsEdgeLeft.size(); i++) {
            params->track->pointsEdgeLeft[i].y =
                (params->track->pointsEdgeRight[i].y +
                 params->track->pointsEdgeLeft[i].y) /
                2;
        }
    }
}