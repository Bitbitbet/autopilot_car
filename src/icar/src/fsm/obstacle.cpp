#include "fsm/obstacle.hpp"

using std::vector;

/**
 * @brief Construct a new Fsm Obstacle
 *
 * @param par
 */
FsmObstacle::FsmObstacle(std::shared_ptr<Params> par)
    : FSMState(FsmMode::obstacle, par) {}

/**
 * @brief Destroy the Fsm Obstacle
 *
 */
FsmObstacle::~FsmObstacle() {}

/**
 * @brief 检查状态切换
 *
 * @return FsmMode 切换后的状态
 */
FsmMode FsmObstacle::getMode() {
    return FsmMode::
        normal; // 避障不占用独立状态：任何状态下看到障碍物都执行，相当于track
}

/**
 * @brief 运行FSM状态（循环主程序）
 *
 */
void FsmObstacle::run(Mat &img) {
    if (!params->config.obstacle) // 该模式未启用
        return;

    PredictResult target = findTarget(); // 本帧锥桶/行人目标（面积最大）

    switch (step) {
    case Step::NONE: // AI未识别：等待目标
    {
        if (target.score > 0) // 发现目标（锥桶/行人）
            countRec++;
        else
            countRec = 0;

        if (countRec >= 2)
            setStep(Step::ENABLE); // 连续2帧确认进入避障
        break;
    }

    case Step::ENABLE: // 避障使能（锥桶绕行 / 行人中线收缩）
    {
        timeout++;
        if (target.score > 0) {
            countSes = 0; // 目标还在，清丢失计数
            resultObs = target;
            planObstacle(
                target); // 边线重规划：锥桶贝塞尔绕行 / 行人对侧中线收缩
        } else {
            countSes++; // 目标绕开/丢失
            if (countSes >= 30) {
                setStep(Step::NONE); // 避障结束（近处长时间无框，认为已让过）
                return;
            }
        }

        if (timeout > 300) // 超时兜底
            setStep(Step::NONE);
        break;
    }
    }
}

/**
 * @brief 图形化显示FSM数据
 *
 * @param img
 */
void FsmObstacle::show(Mat &img) {
    if (step == Step::NONE) // 避障激活时绘制目标框（track级别，不依赖mode）
        return;

    if (resultObs.x > 0 && resultObs.y > 0) {
        cv::Rect rect(resultObs.x, resultObs.y, resultObs.width,
                      resultObs.height);
        cv::rectangle(img, rect, cv::Scalar(0, 0, 255), 1);
    }

    putText(img, "[Obs] bypass", Point(COLSIMAGE / 2 - 50, 20),
            cv::FONT_HERSHEY_TRIPLEX, 0.5, cv::Scalar(0, 255, 0), 0.5);
}

/**
 * @brief 设置新状态
 *
 * @param st
 */
void FsmObstacle::setStep(Step st) {
    step = st;
    countRec = 0;              // AI目标识别计数器
    countSes = 0;              // 让路/丢失计数器
    timeout = 0;               // 超时计数器
    lostFrames = 0;            // 目标丢失计数
    params->ctrl.stop = false; // 恢复行驶
}

/**
 * @brief 搜索锥桶/行人目标，取面积最大者
 *
 * @return PredictResult
 */
PredictResult FsmObstacle::findTarget() {
    PredictResult best = PredictResult(); // 全零(score=0 表示未找到)
    int areaMax = 0;
    for (int i = 0; i < params->results.size(); i++) {
        if ((params->results[i].type == LABEL_CONE ||
             params->results[i].type == LABEL_PERSON) &&
            (params->results[i].y + params->results[i].height) >
                ROWSIMAGE * 0.2 &&
            params->results[i].height > 20 &&
            params->results[i].height < 160 && // 放宽上限
            params->results[i].width > 20 && params->results[i].width < 140) {
            int area = params->results[i].width * params->results[i].height;
            if (area >= areaMax) {
                areaMax = area;
                best = params->results[i];
            }
        }
    }
    return best;
}

/**
 * @brief 锥桶贝塞尔绕行
 *
 * @param obs 避障目标
 */
void FsmObstacle::planObstacle(PredictResult obs) {
    // 障碍物方向判定（左/右）
    int row = 0, width = COLSIMAGE;
    for (size_t i = 0; i < params->track->pointsEdgeLeft.size(); i++) {
        int w = abs(obs.y - params->track->pointsEdgeLeft[i].x);
        if (w < 2) {
            row = i;
            break;
        }
        if (w < width) {
            width = w;
            row = i;
        }
    }
    if (row > (int)params->track->pointsEdgeRight.size() - 1)
        row = params->track->pointsEdgeRight.size() - 1;

    // 路径重规划
    int disLeft = obs.x - params->track->pointsEdgeLeft[row].y;
    int disRight = params->track->pointsEdgeRight[row].y - (obs.x + obs.width);
    if (obs.x + obs.width > params->track->pointsEdgeLeft[row].y &&
        params->track->pointsEdgeRight[row].y > obs.x &&
        abs(disLeft) <= abs(disRight)) //[1] 障碍物靠左
    {
        if (obs.type == LABEL_PERSON) // 行人避障
            curtailTracking(false);   // 缩减优化车道线（双车道→单车道）
        else {
            vector<PointX> points(4); // 三阶贝塞尔曲线
            points[0] = params->track->pointsEdgeLeft[row / 2];
            points[1] = {ROWSIMAGE,
                         obs.x + obs.width * 1}; // 清障列延伸到画面底(车头)
            points[2] = {(obs.y + obs.height + obs.y) / 2,
                         obs.x + obs.width * 1};
            if (obs.y >
                params->track
                    ->pointsEdgeLeft[params->track->pointsEdgeLeft.size() - 1]
                    .x)
                points[3] =
                    params->track
                        ->pointsEdgeLeft[params->track->pointsEdgeLeft.size() -
                                         1];
            else
                points[3] = {obs.y, obs.x + obs.width * 1};

            params->track->pointsEdgeLeft.resize((size_t)row /
                                                 2);      // 删除错误路线
            vector<PointX> repair = Bezier(0.01, points); // 重新规划车道线
            for (int i = 0; i < repair.size(); i++)
                params->track->pointsEdgeLeft.push_back(repair[i]);
        }
    } else if (obs.x + obs.width > params->track->pointsEdgeLeft[row].y &&
               params->track->pointsEdgeRight[row].y > obs.x &&
               abs(disLeft) > abs(disRight)) //[2] 障碍物靠右
    {
        if (obs.type == LABEL_PERSON) // 行人避障
            curtailTracking(true);    // 缩减优化车道线（双车道→单车道）
        else {
            vector<PointX> points(4); // 三阶贝塞尔曲线
            points[0] = params->track->pointsEdgeRight[row / 2];
            points[1] = {ROWSIMAGE,
                         obs.x - obs.width * 2}; // 清障列延伸到画面底(车头
            points[2] = {(obs.y + obs.height + obs.y) / 2,
                         obs.x - obs.width * 2};
            if (obs.y >
                params->track
                    ->pointsEdgeRight[params->track->pointsEdgeRight.size() - 1]
                    .x)
                points[3] = params->track->pointsEdgeRight
                                [params->track->pointsEdgeRight.size() - 1];
            else
                points[3] = {obs.y, obs.x - obs.width * 2};

            params->track->pointsEdgeRight.resize((size_t)row /
                                                  2);     // 删除错误路线
            vector<PointX> repair = Bezier(0.01, points); // 重新规划车道线
            for (int i = 0; i < repair.size(); i++)
                params->track->pointsEdgeRight.push_back(repair[i]);
        }
    }

    // 车道线切除顶行，避免弯道权重过大
    params->track->pointsEdgeLeft.resize(
        (size_t)(params->track->pointsEdgeLeft.size() * 0.8));
    params->track->pointsEdgeRight.resize(
        (size_t)(params->track->pointsEdgeRight.size() * 0.8));
}

/**
 * @brief 缩减优化车道线（双车道→单车道）
 *
 * @param left
 */
void FsmObstacle::curtailTracking(bool left) {
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