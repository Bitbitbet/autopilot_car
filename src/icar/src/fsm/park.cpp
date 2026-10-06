#include "fsm/park.hpp"

using std::string;
using std::to_string;
using std::vector;

/**
 * @brief Construct a new Fsm Park
 *
 * @param par
 */
FsmPark::FsmPark(std::shared_ptr<Params> par) : FSMState(FsmMode::park, par) {}

/**
 * @brief Destroy the Fsm Park
 *
 */
FsmPark::~FsmPark() {}

/**
 * @brief 检查状态切换
 *
 * @return FsmMode 切换后的状态
 */
FsmMode FsmPark::getMode() {
    if (step == Step::NONE || !params->config.park) {
        params->ctrl.parking = false;
        return FsmMode::normal;
    } else {
        params->ctrl.parking = true;
        return FsmMode::park;
    }
}

/**
 * @brief 运行FSM状态（循环主程序）
 *
 */
void FsmPark::run(Mat &img) {
    if (!params->config.park) // 该模式未启用
        return;

    stopping = false; // 停车等待标志
    speedUp++;
    if (speedUp > 1000) // 出库缓加速计时
        speedUp = 1000;

    switch (step) {
    case Step::NONE: // AI未识别
    {
        for (int i = 0; i < params->results.size(); i++) {
            if (params->results[i].type == LABEL_PARK &&
                params->results[i].height < 100 &&
                params->results[i].width < 90 &&
                params->results[i].x > COLSIMAGE / 2) // AI标志：停车场
                countRes++;
        }

        if (countRes >= 3)
            setStep(Step::ENABLE); // 设置停车场新步骤

        if (countRes > 0) // 识别AI标志后开始场次计数
        {
            countSes++;
            if (countSes >= 5) {
                countSes = 0;
                countRes = 0;
            }
        }
        break;
    }

    case Step::ENABLE: // 停车场使能
    {
        countSes++;
        if (!params->ctrl.stop) // 停车等待
            timeout++;
        if (timeout > 80) // 超时退出停车状态
            reset();      // 停车场数据复位

        if (findSymbols(params->results, LABEL_PARK)) // 搜索AI标志：停车场
            countSes = 0;

        if (countSes >
            25) // 停车场Park标志丢失后，倒计时15场图像开始左转向: 25场
            setStep(Step::FORKIN); // 设置停车场新步骤

        // 检测入库AI标志
        PredictResult fork;
        fork.score = 0;
        fork.y = 0;
        for (int i = 0; i < params->results.size(); i++) {
            if (params->results[i].type == LABEL_CHOICE) // AI标志：左转直行
            {
                if (params->results[i].y > fork.y &&
                    params->results[i].width < 120 &&
                    params->results[i].height < 120) {
                    fork = params->results[i]; // 搜索最底行的岔路标志
                }
            }
        }
        if (fork.score > 0) // 检测到AI标志
        {
            countSes = 0; // 定时入库关闭 | 仅依靠AI标志转向
            if ((fork.y + fork.height / 2) > ROWSIMAGE * 0.3)
                countRes++;
            if (countRes > 1)
                setStep(Step::FORKIN); // 设置停车场新步骤
        }

        // 退出停车动作
        if (findSymbols(params->results, LABEL_LEFT)) // 搜索AI
            countFlow++;                              // 直行计数器
        if (countFlow > 5)
            setStep(Step::NONE); // 退出停车

        break;
    }

    case Step::FORKIN: // 入库岔路转向
    {
        timeout++;
        replanTracking();                             // 车道线重绘（岔路左转）
        if (findSymbols(params->results, LABEL_GATE)) // 搜索AI标志：道闸
            countRes++;
        if (countRes > 2 || timeout > 30) // 转向超时：
        {
            spots.reset(); // 车位信息复位
            countOut = 0;
            setStep(Step::TRACKIN); // 设置停车场新步骤
        }

        if (timeout > 20) {
            if (findSymbols(params->results,
                            LABEL_FORK)) // 搜索AI标志：岔路箭头
                countRes++;
        }
        break;
    }

    case Step::TRACKIN: // 入库巡线中
    {
        timeout++;  // 超时计数
        countSes++; // 图像场次计数

        //[01] 道闸检测
        for (int i = 0; i < params->results.size(); i++) {
            if (params->results[i].type == LABEL_GATE &&
                params->results[i].width > 100 &&
                params->results[i].height < 130) // 搜索AI标志：岔路箭头
            {
                if ((params->results[i].y + params->results[i].height) >
                    ROWSIMAGE * 0.4) // 停车距离计算
                {
                    stopping = true; // 停车等待标志
                    break;
                }

                // 出停车场检测
                if ((params->results[i].y + params->results[i].height) >
                        ROWSIMAGE * 0.25 &&
                    spots.checked) // 道闸距离估算
                {
                    countOut++;
                    if (countOut > 4)
                        setStep(Step::TRACKOUT); // 设置停车场新步骤

                    break;
                }
                timeout = 0; // 超时计数
                countRes = 0;
            }
        }

        //[02] 控制中心重规划
        spots.forks = findParkStation(params->results); // 搜索停车位坐标
        PredictResult resLeft;
        resLeft.score = 0;
        for (int i = 0; i < params->results.size(); i++) // 搜索左转箭头
        {
            if (params->results[i].type == LABEL_LEFT &&
                params->results[i].width < 100 &&
                params->results[i].height < 120 &&
                params->results[i].score > resLeft.score)
                resLeft = params->results[i];
        }
        if (spots.forks.size() > 0 || resLeft.score > 0) {
            PredictResult direction;
            if (resLeft.score > 0)
                direction = resLeft;
            else
                direction = spots.forks[0];

            PointX start =
                PointX(ROWSIMAGE - 20, COLSIMAGE / 2); // 补线起点：车头
            if (countIn < 50)                          // 起点矫正
            {
                if (params->track->pointsEdgeLeft.size() > ROWSIMAGE / 5 &&
                    params->track->pointsEdgeRight.size() > ROWSIMAGE / 5) {
                    start = {(params->track->pointsEdgeLeft[0].x +
                              params->track->pointsEdgeRight[0].x) /
                                 2,
                             (params->track->pointsEdgeLeft[0].y +
                              params->track->pointsEdgeRight[0].y) /
                                 2};
                }
                countIn++;
            }
            PointX end =
                PointX(direction.y,
                       direction.x + direction.width / 2); // 补线终点：AI标志
            PointX mid = PointX((start.x + end.x) / 2,
                                (start.y + end.y) / 2); // 补线中点
            vector<PointX> repairPoints = {start, mid, end};
            params->ctrl.centerEdge =
                Bezier(0.02, repairPoints); // 三阶贝塞尔曲线拟合
            params->ctrl.fitting = true;    // 控制中心拟合标志
            timeout = 0;                    // 超时计数
            countRes = 0;

            // 出停车场检测
            if (resLeft.score > 0 &&
                (resLeft.y + resLeft.height / 2) > ROWSIMAGE * 0.2)
                countOut++;
            if (countOut > 3)
                setStep(Step::TRACKOUT); // 设置停车场新步骤
        }

        //[03] 停车位使能（按配置目标车库，不检查车壳占位）
        if (spots.forks.size() == 2) // 当AI图像同时检测到两个岔路箭头时
        {
            findParkCars(
                params->results); // 透视变换（供调试显示，不做车壳占位判定）

            // 目标车库N使能，其余视为占用（spotEnable[i]=false）
            int target = params->config.parkSpot; // 1-4号车库
            if (target < 1)
                target = 1;
            if (target > 4)
                target = 4;
            for (int i = 0; i < 4; i++)
                spots.spotEnable[i] = (i == target - 1);

            // 驶入车位编号确认
            if ((spots.forks[0].y + spots.forks[0].height / 2) >
                    ROWSIMAGE * 0.5 ||
                (spots.forks[1].y + spots.forks[1].height / 2) >
                    ROWSIMAGE * 0.5)
                spots.countRes++;
            else
                spots.countRes = 0;
            if (spots.countRes > 3) // 开始确认驶入车位号
            {
                spots.checked = true;
                spots.countRes = 0;
            }
        }

        //[04] 入库检测
        if (spots.checked && params->config.spot) // 车位停车使能
        {
            if (spots.spotEnable[0] || spots.spotEnable[3]) // 1/4号车位（近）
            {
                if (spots.forks.size() > 0) {
                    if ((spots.forks[0].y + spots.forks[0].height / 2) >
                        ROWSIMAGE * spotUp) // 入库距离判定
                    {
                        spots.times++;
                        if (spots.times > 1) {
                            spots.times = 0;
                            setStep(Step::ENTER);       // 设置停车场新步骤
                            pointsEdgeLeftPast.clear(); // 清空入库路径
                            pointsEdgeRightPast.clear();
                        }
                    }
                }
            } else if (spots.spotEnable[1] ||
                       spots.spotEnable[2]) // 2/3号车位（远）
            {
                if (spots.forks.size() == 2) {
                    if ((spots.forks[1].y + spots.forks[1].height / 2) >
                        ROWSIMAGE * spotDown) // 入库距离判定
                    {
                        spots.times++;
                        if (spots.times > 1) {
                            spots.times = 0;
                            setStep(Step::ENTER);       // 设置停车场新步骤
                            pointsEdgeLeftPast.clear(); // 清空入库路径
                            pointsEdgeRightPast.clear();
                        }
                    }
                }
            }
        }

        // 出库状态切换
        if (countSes > 100) {
            if (timeout > 20) // 超时
                countRes++;   // 出库计数器
            if (countRes > 3)
                setStep(Step::FORKOUT); // 设置停车场新步骤
        }
        break;
    }

    case Step::ENTER: // 驶入停车位
    {
        timeout++;
        if (timeout < 21) // 入库转向
        {
            if (spots.spotEnable[0] || spots.spotEnable[1]) // 1/4号车位（近）
                replanTracking(true);                       // 入库车道线重绘
            else if (spots.spotEnable[2] ||
                     spots.spotEnable[3]) // 2/3号车位（远）
                replanTracking(false);    // 入库车道线重绘
        } else {
            // Track重新捕获正常车道线
            int height = 0;
            if (params->track->pointsEdgeLeft.size() > COLSIMAGE / 2 &&
                params->track->pointsEdgeRight.size() > COLSIMAGE / 2) {
                for (int i = 1; i <= 10; i++) {
                    height += params->track
                                  ->pointsEdgeLeft
                                      [params->track->pointsEdgeLeft.size() - i]
                                  .x;
                    height +=
                        params->track
                            ->pointsEdgeRight
                                [params->track->pointsEdgeRight.size() - i]
                            .x;
                }
                height = height / 20;
            }

            if (height > ROWSIMAGE * 0.3) // 入库结束
            {
                spots.countRes++;
                if (spots.countRes > 2)
                    setStep(Step::PARKING); // 停车完成
            }

            // 入库直行
            params->track->pointsEdgeLeft.clear();  // 清空数据
            params->track->pointsEdgeRight.clear(); // 清空数据
            // 左车道线
            PointX startPoint = PointX(
                ROWSIMAGE - 30, COLSIMAGE * 0.2); // 入库补线起点:固定左下角
            PointX endPoint = PointX(50, COLSIMAGE * 0.35); // 入库补线终点
            PointX midPoint =
                PointX((startPoint.x + endPoint.x) * 0.5,
                       (startPoint.y + endPoint.y) * 0.5); // 入库补线中点
            vector<PointX> repairPoints = {startPoint, midPoint, endPoint};
            params->track->pointsEdgeLeft =
                Bezier(0.02, repairPoints); // 三阶贝塞尔曲线拟合

            // 右车道
            startPoint = PointX(ROWSIMAGE - 30,
                                COLSIMAGE * 0.8);    // 入库补线起点:固定左下角
            endPoint = PointX(50, COLSIMAGE * 0.65); // 入库补线终点
            midPoint =
                PointX((startPoint.x + endPoint.x) * 0.5,
                       (startPoint.y + endPoint.y) * 0.5); // 入库补线中点
            repairPoints = {startPoint, midPoint, endPoint};
            params->track->pointsEdgeRight =
                Bezier(0.02, repairPoints); // 三阶贝塞尔曲线拟合
        }

        pointsEdgeLeftPast.push_back(
            params->track->pointsEdgeLeft); // 记录入库路径
        pointsEdgeRightPast.push_back(params->track->pointsEdgeRight);

        if (timeout > 15)           // 转向超时
            setStep(Step::PARKING); // 停车完成
        break;
    }

    case Step::PARKING: // 停车
    {
        params->ctrl.stop = true; // 停车标志
        timeout++;
        if (timeout > 25)        // 停车计时
            setStep(Step::EXIT); // 出库

        break;
    }

    case Step::EXIT: // 出库
    {
        params->ctrl.stop = false; // 停车标志
        params->ctrl.back = true;  // 倒车

        if (pointsEdgeLeftPast.size() < 1 || pointsEdgeRightPast.size() < 1)
            setStep(Step::TRACKOUT); // 停车完成
        else {
            // 出库轨迹复现
            params->track->pointsEdgeLeft =
                pointsEdgeLeftPast[pointsEdgeLeftPast.size() - 1];
            params->track->pointsEdgeRight =
                pointsEdgeRightPast[pointsEdgeRightPast.size() - 1];
            pointsEdgeLeftPast.pop_back(); // 删除最后一组路径
            pointsEdgeRightPast.pop_back();
        }

        break;
    }

    case Step::TRACKOUT: // 出库巡线
    {
        timeout++; // 超时计数
        //[01] 道闸检测
        for (int i = 0; i < params->results.size(); i++) {
            if (params->results[i].type == LABEL_GATE &&
                params->results[i].width > 100 &&
                params->results[i].height < 130) // 搜索AI标志：岔路箭头
            {
                timeout = 0; // 超时计数
                countRes = 0;
                if ((params->results[i].y + params->results[i].height) >
                    ROWSIMAGE * 0.4) // 停车距离计算
                {
                    stopping = true; // 停车等待标志
                    break;
                }
            }
        }

        //[02] 控制中心重规划
        spots.forks = findParkStation(params->results); // 搜索停车位坐标
        PredictResult resLeft;
        resLeft.score = 0;
        for (int i = 0; i < params->results.size(); i++) // 搜索左转箭头
        {
            if (params->results[i].type == LABEL_LEFT &&
                params->results[i].width < 100 &&
                params->results[i].height < 120 &&
                params->results[i].score > resLeft.score)
                resLeft = params->results[i];
        }

        if (spots.forks.size() > 0 || resLeft.score > 0) {
            PredictResult direction;
            if (resLeft.score > 0)
                direction = resLeft;
            else
                direction = spots.forks[0];
            PointX start =
                PointX(ROWSIMAGE - 20, COLSIMAGE / 2); // 补线起点：车头
            PointX end =
                PointX(direction.y,
                       direction.x + direction.width / 2); // 补线终点：AI标志
            PointX mid = PointX((start.x + end.x) / 2,
                                (start.y + end.y) / 2); // 补线中点
            vector<PointX> repairPoints = {start, mid, end};
            params->ctrl.centerEdge =
                Bezier(0.02, repairPoints); // 三阶贝塞尔曲线拟合
            params->ctrl.fitting = true;    // 控制中心拟合标志
            if (spots.forks.size() > 0)
                timeout = 0; // 定时入库关闭 | 仅依靠AI标志转向

            if (resLeft.score > 0 &&
                (resLeft.y + resLeft.height / 2) > ROWSIMAGE * 0.35) {
                countRes++;
            }
        }

        // 出库状态切换
        if (timeout > 45)
            countRes++; // 出库计数器
        if (countRes > 2)
            setStep(Step::FORKOUT); // 设置停车场新步骤

        break;
    }

    case Step::FORKOUT: // 出库岔路转向
    {
        speedUp = 0;
        timeout++;
        replanTracking(); // 车道线重绘（岔路左转）

        // 搜索左转标志
        bool leftSign = false;
        for (int i = 0; i < params->results.size(); i++) // 搜索左转箭头
        {
            if (params->results[i].type == LABEL_LEFT &&
                params->results[i].width < 100 &&
                params->results[i].height < 120) {
                leftSign = true;
                break;
            }
        }
        if (leftSign) // 标志未丢失
            countRes = 0;
        else
            countRes++;

        if (timeout > 20 || countRes > 2) // 转向超时
        {
            params->ctrl.countAcc = 0;
            setStep(Step::NONE); // 设置停车场新步骤
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
void FsmPark::show(Mat &img) {
    if (params->mode != FsmMode::park)
        return;

    putText(img, "[5] Park", Point(COLSIMAGE / 2 - 50, 20),
            cv::FONT_HERSHEY_TRIPLEX, 0.5, cv::Scalar(0, 255, 0), 0.5);

    // 绘制边缘点
    for (int i = 0; i < params->track->pointsEdgeLeft.size(); i++) {
        circle(img,
               Point(params->track->pointsEdgeLeft[i].y,
                     params->track->pointsEdgeLeft[i].x),
               2, Scalar(0, 255, 0), -1); // 绿色点
    }
    for (int i = 0; i < params->track->pointsEdgeRight.size(); i++) {
        circle(img,
               Point(params->track->pointsEdgeRight[i].y,
                     params->track->pointsEdgeRight[i].x),
               2, Scalar(0, 255, 255), -1); // 黄色点
    }

    // 绘制中心点集
    if (params->ctrl.fitting) {
        for (int i = 0; i < params->ctrl.centerEdge.size(); i++)
            circle(img,
                   Point(params->ctrl.centerEdge[i].y,
                         params->ctrl.centerEdge[i].x),
                   1, Scalar(0, 0, 255), -1);
    }

    string str = "Enable";
    switch (step) {
    case Step::FORKIN:
        str = "Forkin";
        break;
    case Step::TRACKIN: {
        str = "Trackin";
        for (int i = 0; i < spots.forks.size(); i++) // 绘制停车位箭头标志
        {
            putText(img, to_string(i + 1),
                    Point(spots.forks[i].x + spots.forks[i].width / 2,
                          spots.forks[i].y + spots.forks[i].height / 2),
                    cv::FONT_HERSHEY_TRIPLEX, 0.5, cv::Scalar(0, 0, 255), 0.5);
            cv::Rect rect(spots.forks[i].x, spots.forks[i].y,
                          spots.forks[i].width, spots.forks[i].height);
            cv::rectangle(img, rect, cv::Scalar(0, 0, 255), 1);
        }

        // 透视变换视角
        Mat imgIpm = Mat::zeros(Size(COLSIMAGEIPM, ROWSIMAGEIPM),
                                CV_8UC3); // 创建全黑图像
        for (int i = 0; i < spots.carsIpm.size(); i++)
            circle(imgIpm, Point(spots.carsIpm[i].x, spots.carsIpm[i].y), 3,
                   Scalar(0, 0, 255), -1);

        for (int i = 0; i < spots.forksIpm.size(); i++)
            circle(imgIpm, Point(spots.forksIpm[i].x, spots.forksIpm[i].y), 3,
                   Scalar(0, 255, 0), -1);

        // 车位已停车辆绘制
        if (!spots.spotEnable[0] && spots.forks.size() > 0) // 1号车位
        {
            cv::Rect rect(0, spots.forks[0].y - 10, 60,
                          spots.forks[0].height + 20);
            cv::rectangle(img, rect, cv::Scalar(0, 0, 255), 1);
        }
        if (!spots.spotEnable[1] && spots.forks.size() > 0) // 2号车位
        {
            cv::Rect rect(COLSIMAGE - 61, spots.forks[0].y - 10, 60,
                          spots.forks[0].height + 20);
            cv::rectangle(img, rect, cv::Scalar(0, 0, 255), 1);
        }
        if (!spots.spotEnable[2] && spots.forks.size() > 1) // 3号车位
        {
            cv::Rect rect(0, spots.forks[1].y - 10, 60,
                          spots.forks[1].height + 20);
            cv::rectangle(img, rect, cv::Scalar(0, 0, 255), 1);
        }
        if (!spots.spotEnable[3] && spots.forks.size() > 1) // 4号车位
        {
            cv::Rect rect(COLSIMAGE - 61, spots.forks[1].y - 10, 60,
                          spots.forks[1].height + 20);
            cv::rectangle(img, rect, cv::Scalar(0, 0, 255), 1);
        }
        savePicture("../res/samples/imgs/", imgIpm);
        savePicture("../res/samples/imgs/", img);
        break;
    }
    case Step::ENTER:
        str = "Enter";
        break;
    case Step::PARKING:
        str = "Parking";
        break;
    case Step::EXIT:
        str = "Exit";
        break;
    case Step::TRACKOUT:
        str = "TRACKOUT";
        break;
    case Step::FORKOUT:
        str = "Forkout";
        break;
    default:
        break;
    }
    putText(img, "PARK - " + str, Point(100, 50), cv::FONT_HERSHEY_TRIPLEX, 0.5,
            cv::Scalar(0, 255, 0), 0.5);
}

/**
 * @brief 停车场数据复位
 *
 */
void FsmPark::reset() {
    countSes = 0;      // AI场景识别计数器
    timeout = 0;       // 超时计数器
    step = Step::NONE; // 停车步骤
    waiting = false;   // 停车等待使能
    countWait = 0;     // 停车等待计数器
    countOut = 0;      // 出库检测计数
    countIn = 0;       // 入库矫正计数器
    speedUp = 0;       // 出库加速延迟计数器
    countFlow = 0;     // 直行计数器
}

/**
 * @brief 设置下阶段
 *
 * @param step
 */
void FsmPark::setStep(Step st) {
    countRes = 0; // AI场景识别计数器
    countSes = 0; // 场次计数器
    timeout = 0;  // 超时计数器
    step = st;    // 停车步骤
    countIn = 0;  // 入库矫正计数器
    countFlow = 0;
    params->ctrl.back = false; // 倒车失能
}

/**
 * @brief 车道线重绘（岔路左转）
 *
 */
void FsmPark::replanTracking() {
    params->track->pointsEdgeLeft.clear();  // 清空原来数据
    params->track->pointsEdgeRight.clear(); // 清空原来数据

    // 左车道线
    PointX startPoint = PointX(ROWSIMAGE - 10, 1); // 入库补线起点:固定左下角
    PointX endPoint = PointX(ROWSIMAGE / 3, 1);    // 入库补线终点
    PointX midPoint = PointX((startPoint.x + endPoint.x) * 0.3,
                             (startPoint.y + endPoint.y) / 2); // 入库补线中点
    vector<PointX> repairPoints = {startPoint, midPoint, endPoint};
    vector<PointX> modifyEdge =
        Bezier(0.02, repairPoints); // 三阶贝塞尔曲线拟合
    params->track->pointsEdgeLeft = modifyEdge;

    // 右车道线
    startPoint =
        PointX(ROWSIMAGE - 10, COLSIMAGE * 0.8); // 入库补线起点:固定左下角
    endPoint = PointX(ROWSIMAGE / 3, 30);        // 入库补线终点
    midPoint = PointX((startPoint.x + endPoint.x) * 0.3,
                      (startPoint.y + endPoint.y) / 2); // 入库补线中点
    repairPoints = {startPoint, midPoint, endPoint};
    modifyEdge = Bezier(0.02, repairPoints); // 三阶贝塞尔曲线拟合
    params->track->pointsEdgeRight = modifyEdge;
}
/**
 * @brief 停车入库车道线重绘
 *
 * @param left true：左转 | false：右转
 */
void FsmPark::replanTracking(bool left) {
    params->track->pointsEdgeLeft.clear();  // 清空原来数据
    params->track->pointsEdgeRight.clear(); // 清空原来数据

    if (left) {
        // 左车道线
        PointX startPoint =
            PointX(ROWSIMAGE - 10, 1);              // 入库补线起点:固定左下角
        PointX endPoint = PointX(ROWSIMAGE / 3, 1); // 入库补线终点
        PointX midPoint =
            PointX((startPoint.x + endPoint.x) * 0.3,
                   (startPoint.y + endPoint.y) / 2); // 入库补线中点
        vector<PointX> repairPoints = {startPoint, midPoint, endPoint};
        vector<PointX> modifyEdge =
            Bezier(0.02, repairPoints); // 三阶贝塞尔曲线拟合
        params->track->pointsEdgeLeft = modifyEdge;
        // 右车道线
        startPoint =
            PointX(ROWSIMAGE - 10, COLSIMAGE * 0.8); // 入库补线起点:固定左下角
        endPoint = PointX(ROWSIMAGE / 3, 1);         // 入库补线终点
        midPoint = PointX((startPoint.x + endPoint.x) * 0.3,
                          (startPoint.y + endPoint.y) / 2); // 入库补线中点
        repairPoints = {startPoint, midPoint, endPoint};
        modifyEdge = Bezier(0.02, repairPoints); // 三阶贝塞尔曲线拟合
        params->track->pointsEdgeRight = modifyEdge;
    } else {
        // 左车道线
        PointX startPoint =
            PointX(ROWSIMAGE - 10, COLSIMAGE * 0.2); // 入库补线起点:固定左下角
        PointX endPoint = PointX(ROWSIMAGE / 3, COLSIMAGE - 1); // 入库补线终点
        PointX midPoint =
            PointX((startPoint.x + endPoint.x) * 0.3,
                   (startPoint.y + endPoint.y) / 2); // 入库补线中点
        vector<PointX> repairPoints = {startPoint, midPoint, endPoint};
        vector<PointX> modifyEdge =
            Bezier(0.02, repairPoints); // 三阶贝塞尔曲线拟合
        params->track->pointsEdgeLeft = modifyEdge;
        // 右车道线
        startPoint =
            PointX(ROWSIMAGE - 10, COLSIMAGE - 1); // 入库补线起点:固定左下角
        endPoint = PointX(ROWSIMAGE / 3, COLSIMAGE - 1); // 入库补线终点
        midPoint = PointX((startPoint.x + endPoint.x) * 0.3,
                          (startPoint.y + endPoint.y) / 2); // 入库补线中点
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
bool FsmPark::findSymbols(vector<PredictResult> results, int label) {
    for (int i = 0; i < results.size(); i++) {
        if (results[i].type == label) // AI标志
            return true;
    }

    return false;
}

/**
 * @brief 搜索AI标志（最顶上的一个）
 *
 * @param label
 * @param target 目标标志信息
 * @return true
 * @return false
 */
bool FsmPark::findSymbols(vector<PredictResult> results, int label,
                          PredictResult &target) {
    bool res = false;
    target.y = ROWSIMAGE - 1;
    for (int i = 0; i < results.size(); i++) {
        if (results[i].type == label) // AI标志
        {
            if (results[i].y < target.y && results[i].width < 120 &&
                results[i].height < 120) {
                target = results[i];
                res = true; // 搜索到目标AI
            }
        }
    }

    return res;
}

/**
 * @brief 搜索停车位坐标
 *
 * @param results
 */
vector<PredictResult> FsmPark::findParkStation(vector<PredictResult> results) {
    vector<PredictResult> resFork;
    for (int i = 0; i < results.size(); i++) {
        if (results[i].type == LABEL_FORK && results[i].width < 100 &&
            results[i].height < 120 && results[i].y > 15)
            resFork.push_back(results[i]);
    }

    vector<PredictResult>
        resSpot; // size=0:无停车位，size=1:停车位1/2，size=2:停车位1/2/3/4
    vector<PredictResult> resFilter; // 滤波后的坐标

    if (resFork.size() < 1) // 未检测AI标志
        return resSpot;

    resFilter.push_back(resFork[0]);
    for (int i = 0; i < resFork.size(); i++) {
        for (int n = 0; n < resFilter.size(); n++) {
            PointX centerFilter = getResultCenter(resFilter[n]);
            PointX centerResult = getResultCenter(resFork[i]);
            if (centerFilter.x > 0 && centerFilter.y > 0 &&
                centerResult.x > 0 && centerResult.y > 0) // 数据有效
            {
                if (abs(centerFilter.y - centerResult.y) > 30) // 两个坐标不重叠
                {
                    bool newFork = true;
                    for (int y = 0; y < resFilter.size(); y++) // 重复筛选
                    {
                        PointX centerF = getResultCenter(resFilter[y]);
                        if (abs(centerF.y - centerResult.y) < 30) // 数据重复
                        {
                            newFork = false;
                            break;
                        }
                    }
                    if (newFork)
                        resFilter.push_back(resFork[i]);
                    else if (resFork[i].score > resFilter[n].score) // 更新数据
                        resFilter[n] = resFork[i];
                } else if (resFork[i].score > resFilter[n].score) // 更新数据
                    resFilter[n] = resFork[i];
            }
        }
    }

    if (resFilter.size() == 1) // 一个车位
        resSpot = resFilter;
    else if (resFilter.size() == 2) // 两个车位标识
    {
        if ((resFilter[0].y + resFilter[0].height / 2) <
            (resFilter[1].y + resFilter[1].height / 2))
            resSpot = resFilter;
        else // 车位号重新排序
        {
            resSpot.push_back(resFilter[1]);
            resSpot.push_back(resFilter[0]);
        }
    }

    return resSpot;
}

/**
 * @brief 获取AI检测目标的质心坐标
 *
 * @param res
 * @return PointX
 */
PointX FsmPark::getResultCenter(PredictResult res) {
    PointX center;
    center.x = res.x + res.width / 2;
    center.y = res.y + res.height / 2;

    if (center.x < 0)
        center.x = 0;
    else if (center.x > COLSIMAGE)
        center.x = COLSIMAGE;

    if (center.y < 0)
        center.y = 0;
    else if (center.y > ROWSIMAGE)
        center.y = ROWSIMAGE;

    return center;
}

/**
 * @brief 检测停车位上是否有车
 *
 * @param results
 * @return vector<PredictResult>
 */
void FsmPark::findParkCars(vector<PredictResult> results) {
    // 数据复位
    for (int i = 0; i < spots.carPark.size(); i++)
        spots.carPark[i].score = 0;

    if (spots.forks.size() != 2) // 基于岔路标志检测车位
        return;

    if (spots.forks[1].y < ROWSIMAGE / 4) // 图像清晰度限制
        return;

    // IPM透视变换（保留，供调试显示；车壳占位判定已移除）
    int x = spots.forks[0].x + spots.forks[0].width / 2;
    int y = spots.forks[0].y + spots.forks[0].height / 2;
    Point2d spot1Ipm = ipm.homography(Point2d(x, y)); // 透视变换
    x = spots.forks[1].x + spots.forks[1].width / 2;
    y = spots.forks[1].y + spots.forks[1].height / 2;
    Point2d spot2Ipm = ipm.homography(Point2d(x, y)); // 透视变换

    spots.forksIpm.clear();
    spots.carsIpm.clear();
    spots.forksIpm.push_back(spot1Ipm);
    spots.forksIpm.push_back(spot2Ipm);
}
