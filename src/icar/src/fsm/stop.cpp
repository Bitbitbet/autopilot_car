#include "fsm/stop.hpp"
using std::vector;

/**
 * @brief Construct a new Fsm Park
 *
 * @param par
 */
FsmStop::FsmStop(std::shared_ptr<Params> par) : FSMState(FsmMode::stop, par) {}

/**
 * @brief Destroy the Fsm Park
 *
 */
FsmStop::~FsmStop() {}

/**
 * @brief 检查状态切换
 *
 * @return FsmMode 切换后的状态
 */
FsmMode FsmStop::getMode() {
    // 输出场景状态结果
    if (step == Step::NONE || !params->config.stop)
        return FsmMode::normal;
    else
        return FsmMode::stop;
}

/**
 * @brief 运行FSM状态（循环主程序）
 *
 */
void FsmStop::run(Mat &img) {
    if (!params->config.stop) // 该模式未启用
        return;

    polyRoad.clear();
    polyCar.clear();

    switch (step) {
    case Step::NONE: // AI未识别
    {
        for (int i = 0; i < params->results.size(); i++) {
            if (params->results[i].type == LABEL_GATE) // 障碍物：道闸
            {
                countRec++;
                break;
            }
        }

        if (countRec > 2)
            setStep(Step::ENABLE); // 设置新状态

        if (countRec > 0) // 识别AI标志后开始场次计数
        {
            countSes++;
            if (countSes > 4)
                setStep(Step::NONE);
        }
        break;
    }

    case Step::ENABLE: // 场景使能
    {
        countSes++; // 场次计数器
        timeout++;
        for (int i = 0; i < params->results.size(); i++) {
            if (params->results[i].type == LABEL_GATE) // 障碍物：道闸
            {
                countSes = 0;
                timeout = 0;
                if ((params->results[i].y + params->results[i].height) >
                    ROWSIMAGE * 0.3) // 停车距离计算
                {
                    countRec++;
                    break;
                }
                break;
            }
        }
        if (countRec > 2)
            setStep(Step::STOP); // 设置新状态
        if (countSes >= 10 || timeout > 50)
            setStep(Step::NONE); // 设置新状态
        break;
    }

    case Step::STOP: // 停车
    {
        params->ctrl.stop = true; // 停车标志
        countSes++;               // 场次计数器
        for (int i = 0; i < params->results.size(); i++) {
            if (params->results[i].type == LABEL_GATE) // 障碍物：道闸
            {
                countSes = 0;
                break;
            }
        }
        if (countSes >= 20) {
            setStep(Step::NONE); // 设置新状态
            params->ctrl.stop = false;
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
void FsmStop::show(Mat &img) {
    if (params->mode != FsmMode::stop)
        return;

    putText(img, "[7] Stop", Point(COLSIMAGE / 2 - 50, 20),
            cv::FONT_HERSHEY_TRIPLEX, 0.5, cv::Scalar(0, 255, 0), 0.5);

    switch (step) {
    case Step::ENABLE: // 场景使能
        putText(img, "[7] STOP - ENABLE", Point(100, 50),
                cv::FONT_HERSHEY_TRIPLEX, 0.5, cv::Scalar(0, 0, 255), 0.5);
        break;

    case Step::STOP: // 停车
        putText(img, "[7] STOP - STOPING", Point(100, 50),
                cv::FONT_HERSHEY_TRIPLEX, 0.5, cv::Scalar(0, 0, 255), 0.5);
        break;
    }

    if (polyRoad.size() > 0 && polyCar.size() > 0) {
        drawPolygon(img, polyRoad, false, true); // 绘制赛道多边形
        drawPolygon(img, polyCar, false, true);  // 绘制车辆多边形

        putText(img, "Overlap:" + double2String(overlap, 2),
                Point(100, ROWSIMAGE - 60), FONT_HERSHEY_TRIPLEX, 0.5,
                Scalar(0, 0, 255), 0.5);
    }
}

/**
 * @brief 计算多边形重合度
 *
 * @param track
 * @param result
 */
double FsmStop::getRoadCarPloy(PredictResult result) {
    if (params->track->pointsEdgeLeft.size() < ROWSIMAGE / 5 &&
        params->track->pointsEdgeRight.size() < ROWSIMAGE / 5)
        return 0.0;

    // 添加赛道多边形轮廓
    PointX ppoliy;
    if (params->track->pointsEdgeLeft.size() < 5 &&
        params->track->pointsEdgeRight.size() > 5) // 右单边
    {
        // [1]左上点
        ppoliy.x =
            params->track
                ->pointsEdgeRight[params->track->pointsEdgeRight.size() - 1]
                .x;
        ppoliy.y = 1;
        polyRoad.push_back(cv::Point(ppoliy.y, ppoliy.x)); // 1
        // [2]左下点
        ppoliy.x = params->track->pointsEdgeRight[0].x;
        ppoliy.y = 1;
        polyRoad.push_back(cv::Point(ppoliy.y, ppoliy.x)); // 2
    } else {
        // [1]左上点
        ppoliy = params->track
                     ->pointsEdgeLeft[params->track->pointsEdgeLeft.size() - 1];
        polyRoad.push_back(cv::Point(ppoliy.y, ppoliy.x)); // 1
        ppoliy =
            params->track
                ->pointsEdgeLeft[params->track->pointsEdgeLeft.size() * 0.8];
        polyRoad.push_back(cv::Point(ppoliy.y, ppoliy.x)); // 1
        ppoliy =
            params->track
                ->pointsEdgeLeft[params->track->pointsEdgeLeft.size() * 0.6];
        polyRoad.push_back(cv::Point(ppoliy.y, ppoliy.x)); // 1
        ppoliy =
            params->track
                ->pointsEdgeLeft[params->track->pointsEdgeLeft.size() * 0.4];
        polyRoad.push_back(cv::Point(ppoliy.y, ppoliy.x)); // 1
        ppoliy =
            params->track
                ->pointsEdgeLeft[params->track->pointsEdgeLeft.size() * 0.2];
        polyRoad.push_back(cv::Point(ppoliy.y, ppoliy.x)); // 1
        // [2]左下点
        ppoliy = params->track->pointsEdgeLeft[0];
        polyRoad.push_back(cv::Point(ppoliy.y, ppoliy.x)); // 2
    }

    if (params->track->pointsEdgeLeft.size() > 5 &&
        params->track->pointsEdgeRight.size() < 5) // 左单边
    {
        //[3] 右下点
        ppoliy.x = params->track->pointsEdgeLeft[0].x;
        ppoliy.y = COLSIMAGE - 1;

        //[4] 右上点
        ppoliy.x =
            params->track
                ->pointsEdgeLeft[params->track->pointsEdgeLeft.size() - 1]
                .x;
        ppoliy.y = COLSIMAGE - 1;
    } else {
        //[3] 右下点
        ppoliy = params->track->pointsEdgeRight[0];
        polyRoad.push_back(cv::Point(ppoliy.y, ppoliy.x));
        ppoliy =
            params->track
                ->pointsEdgeRight[params->track->pointsEdgeRight.size() * 0.2];
        polyRoad.push_back(cv::Point(ppoliy.y, ppoliy.x));
        ppoliy =
            params->track
                ->pointsEdgeRight[params->track->pointsEdgeRight.size() * 0.4];
        polyRoad.push_back(cv::Point(ppoliy.y, ppoliy.x));
        ppoliy =
            params->track
                ->pointsEdgeRight[params->track->pointsEdgeRight.size() * 0.6];
        polyRoad.push_back(cv::Point(ppoliy.y, ppoliy.x));
        ppoliy =
            params->track
                ->pointsEdgeRight[params->track->pointsEdgeRight.size() * 0.8];
        polyRoad.push_back(cv::Point(ppoliy.y, ppoliy.x));
        //[4] 右上点
        ppoliy =
            params->track
                ->pointsEdgeRight[params->track->pointsEdgeRight.size() - 1];
        polyRoad.push_back(cv::Point(ppoliy.y, ppoliy.x));
    }

    // 添加车辆多边形轮廓
    polyCar.push_back(cv::Point(result.x, result.y));
    polyCar.push_back(cv::Point(result.x, result.y + result.height));
    polyCar.push_back(
        cv::Point(result.x + result.width, result.y + result.height));
    polyCar.push_back(cv::Point(result.x + result.width, result.y));

    return getOverlapArea(polyRoad, polyCar); // 计算多边形重合度
}

/**
 * @brief 计算两个多边形图形的面积重叠度
 *
 * @param polyA 多边形顶点
 * @param polyB
 * @return double [0,1]
 */
double FsmStop::getOverlapArea(const std::vector<cv::Point> &polyA,
                               const std::vector<cv::Point> &polyB) {
    // 创建两个多边形的掩模
    cv::Mat maskA = cv::Mat::zeros(500, 500, CV_8UC1);
    cv::Mat maskB = cv::Mat::zeros(500, 500, CV_8UC1); // 智能车

    // 填充多边形
    cv::fillConvexPoly(maskA, polyA, cv::Scalar(255));
    cv::fillConvexPoly(maskB, polyB, cv::Scalar(255));

    // 计算交集
    cv::Mat intersection;
    cv::bitwise_and(maskA, maskB, intersection);

    // 计算交集的面积
    double areaB = cv::countNonZero(maskB);
    double areaAnd = cv::countNonZero(intersection);

    // 计算重合度
    double overlap = (areaB > 0) ? (areaAnd / areaB) : 0;

    return overlap;
}

/**
 * @brief 绘制多边形
 *
 * @param img
 * @param poly 多边形点集
 * @param fill  填充
 * @param close 封闭图形
 */
void FsmStop::drawPolygon(Mat img, vector<Point> poly, bool fill, bool close) {
    vector<vector<Point>> contours;
    contours.push_back(poly);

    if (fill)
        fillPoly(img, contours, Scalar(255, 255, 255), 8);
    else
        polylines(img, poly, close, Scalar(255, 255, 255), 2, 8);
}

/**
 * @brief 设置新状态
 *
 * @param step
 */
void FsmStop::setStep(Step st) {
    step = st;
    countRec = 0; // AI场景识别计数器
    countSes = 0; // 场次计数器
    timeout = 0;  // 超时计数器
}