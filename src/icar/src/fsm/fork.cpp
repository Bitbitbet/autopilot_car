#include "fsm/fork.hpp"

using std::vector;

/**
 * @brief Construct a new Fsm Park
 *
 * @param par
 */
FsmFork::FsmFork(std::shared_ptr<Params> par) : FSMState(FsmMode::fork, par) {}

/**
 * @brief Destroy the Fsm Park
 *
 */
FsmFork::~FsmFork() {}

/**
 * @brief 检查状态切换
 *
 * @return FsmMode 切换后的状态
 */
FsmMode FsmFork::getMode() {
    // 输出场景状态结果
    if (!params->config.fork || !enable)
        return FsmMode::normal;

    return FsmMode::fork;
}

/**
 * @brief 运行FSM状态（循环主程序）
 *
 */
void FsmFork::run(Mat &img) {
    if (!params->config.fork) // 该模式未启用
        return;

    enable = handle(img, 0); // 处理T形岔路口
}

/**
 * @brief 图形化显示FSM数据
 *
 * @param img
 */
void FsmFork::show(Mat &img) {
    if (params->mode != FsmMode::fork)
        return;

    putText(img, "[4] Fork", Point(COLSIMAGE / 2 - 50, 20),
            cv::FONT_HERSHEY_TRIPLEX, 0.5, cv::Scalar(0, 255, 0), 0.5);
}

/**
 * @brief 重置FSM状态
 *
 */
void FsmFork::reset(void) {
    step = Step::NONE; // 岔路处理阶段
    counterFork = 0;   // 图像状态计数
    lastForkL = PointX(0, 0, 0);
    lastForkR = PointX(0, COLSIMAGE - 1, 0);
    repairing = false;
}

/**
 * @brief 处理T形岔路口
 *
 * @param img 所传递的图像
 * @param type 0: 岔路补直线通过; 1: 进行卜型路左转; 2: 丁字路左转
 * @return true
 * @return false
 */
bool FsmFork::handle(Mat &img, int type) {
    if (params->track->pointsEdgeLeft.size() !=
            params->track->widthBlock
                .size() // 只有在左右边界点数量都和白块数相同时才找T形岔路
        || params->track->pointsEdgeRight.size() !=
               params->track->widthBlock.size() ||
        params->track->widthBlock.size() <
            ROWSIMAGE / 6) // 点太少了，不找特征了
        return false;
    if (step == Step::NONE) {

        if (type < 2) { // 左侧可能出现T形岔路口（卜型）
            // 先找到左侧道路上下直道段上下直道段
            PointX leftFilletUp;    // 左上圆弧点
                                    // 对于圆弧拐点，slope代表其index而不是斜率
            PointX leftFilletDown;  // 左下圆弧点
            leftFilletUp.slope = 0; // 初始化index
            leftFilletDown.slope = 0;
            leftFilletDown = searchFilletLeftDown(
                params->track->pointsEdgeLeft, params->track->pointsEdgeRight);
            if (leftFilletDown.slope) // 找到下拐点后找上拐点
            {
                leftFilletUp = searchFilletLeftUp(params->track->pointsEdgeLeft,
                                                  leftFilletDown);
                bool legal = 0;
                for (int i = leftFilletDown.slope; i <= leftFilletUp.slope;
                     i++) {
                    if (leftFilletDown.y - params->track->pointsEdgeLeft[i].y >
                        20) {
                        legal = 1;
                        break;
                    }
                }
                if (!legal) {
                    reset();
                    return false;
                }
            }
            if (leftFilletDown.slope && leftFilletUp.slope &&
                posInRange(gradientCal(leftFilletDown, leftFilletUp), -1,
                           0)) { // 直行不转弯
                if (posInRange(gradientCal(leftFilletDown, leftFilletUp), -1,
                               0)) {
                    // 找到上下拐点且斜率符合要求
                    double k = gradientCal(leftFilletDown, leftFilletUp);
                    double b = leftFilletDown.y - k * leftFilletDown.x;
                    for (int i = leftFilletDown.slope; i <= leftFilletUp.slope;
                         i++) {
                        params->track->pointsEdgeLeft[i].y =
                            (int)(k * params->track->pointsEdgeLeft[i].x + b);
                    }
                    step = Step::ENTER; // 更新当前状态
                    counterFork = 0;
                }
            }
        } else if (type == 2) {
            if (params->track->widthBlock.size() >
                ROWSIMAGE - params->track->rowCutBottom - 5) {
                return false; // 对于该部分判断一个重要依据即为搜索行数的减少
            }
            PointX Right_Fillet_DOWN;    // 左上圆弧点
            PointX leftFilletDown;       // 左下圆弧点
            Right_Fillet_DOWN.slope = 0; // 初始化index
            leftFilletDown.slope = 0;
            leftFilletDown =
                searchFilletLeftDown(params->track->pointsEdgeLeft,
                                     params->track->pointsEdgeRight, -1, 0);
            Right_Fillet_DOWN =
                searchFilletRightDown(params->track->pointsEdgeRight,
                                      params->track->pointsEdgeLeft, -1, 0);
            if (leftFilletDown.slope &&
                Right_Fillet_DOWN.slope) { // 找到左右下拐点
                int countThroughoutBlocks = 0;
                int countSTpointLR = 0; // 统计左右两侧均有贴边的情况
                for (int i = leftFilletDown.slope > Right_Fillet_DOWN.slope
                                 ? leftFilletDown.slope
                                 : Right_Fillet_DOWN.slope;
                     i < params->track->widthBlock.size() - 1; i++) {
                    // 从俩下拐点的最低点往上看大白块数量，一旦有一个不满足马上break
                    if (params->track->widthBlock[i].y >
                        Right_Fillet_DOWN.y - leftFilletDown.y +
                            10) { // 下面行的宽度大于俩下拐点间距则++
                        countThroughoutBlocks++;
                    }
                    if (params->track->pointsEdgeLeft[i].y < 3 &&
                        params->track->pointsEdgeRight[i].y > COLSIMAGE - 3)
                        countSTpointLR++;
                }
                if (countThroughoutBlocks > ROWSIMAGE / 6 &&
                    countSTpointLR > 5) { // 许多行满足该情况
                    step = Step::ENTER;   // 判定通过，进入补线阶段
                    counterFork = 0;
                }
            }
        }
    } else if (step == Step::ENTER) {
        counterFork++; // 记录处理场数
        if (type < 2) {
            { // 左侧可能出现T形岔路口
                // 先找到左侧道路上下直道段上下直道段
                PointX leftFilletUp; // 左上圆弧点
                                     // 对于圆弧拐点，slope代表其index而不是斜率
                PointX leftFilletDown;  // 左下圆弧点
                leftFilletUp.slope = 0; // 初始化index
                leftFilletDown.slope = 0;
                bool LastFlag = 0;
                leftFilletDown = searchFilletLeftDown(
                    params->track->pointsEdgeLeft,
                    params->track->pointsEdgeRight,
                    (ROWSIMAGE - params->track->rowCutBottom) * 0.8);
                if (!leftFilletDown.slope ||
                    leftFilletDown.x <
                        40) { // 此时的Left_Fillet_DOWN不可能特别靠上
                    leftFilletDown = lastForkL;
                    LastFlag =
                        1; // 如果没有找到有效拐点就以上一次的拐点为基准补线
                }
                leftFilletUp = searchFilletLeftUp(params->track->pointsEdgeLeft,
                                                  leftFilletDown);
                if (leftFilletUp.slope < leftFilletDown.slope ||
                    !posInRange(gradientCal(leftFilletDown, leftFilletUp), -1,
                                0)) // 下拐点不可能比上点高
                {
                    leftFilletDown = lastForkL;
                    LastFlag = 1;
                }
                int countSL = 0;
                if (!type && LastFlag) {
                    countSL = countFormerStickpointL(
                        params->track->pointsEdgeLeft, 0,
                        leftFilletUp.slope); // 统计上拐点下方有多少贴边左边点
                } else if (type && LastFlag) {
                    countSL = countFormerStickpointL(
                        params->track->pointsEdgeLeft, 0,
                        params->track->pointsEdgeLeft.size() /
                            2); // 统计上拐点下方有多少贴边左边点
                    if (params->track->spurroad.size() > 0) {
                        for (int i = 0; i < params->track->spurroad.size();
                             i++) { // 太靠右侧以后出现了错误分岔点
                            if (params->track->spurroad[i].x > ROWSIMAGE / 3 &&
                                params->track->spurroad[i].y > COLSIMAGE / 3 &&
                                params->track->spurroad[i].y <
                                    COLSIMAGE * 7 / 8) {
                                lastForkR =
                                    params->track->spurroad
                                        [i]; // 出现符合要求的分岔点后改为补一条直线到分岔点+进入EXITING状态
                                lastForkR.slope = 1; //
                                step = Step::EXIT;
                                counterFork = 0;
                                break;
                            }
                        }
                    }
                }
                int FirstLS = -1;
                for (int i = 0; i < params->track->pointsEdgeLeft.size(); i++) {
                    if (params->track->pointsEdgeLeft[i].y < 3) {
                        FirstLS = i;
                        break;
                    }
                }
                int FirstRS = -1;
                for (int i = 0; i < params->track->pointsEdgeRight.size();
                     i++) {
                    if (params->track->pointsEdgeRight[i].y > COLSIMAGE - 3) {
                        FirstRS = i;
                        break;
                    }
                }
                if (!type || ((FirstRS > 10 || FirstRS == -1) &&
                              (FirstLS > 40 || FirstLS == -1))) { // 直行不转弯
                    if (leftFilletUp.slope) // 此阶段只要找到上拐点即可开始补线
                    {
                        // 找到上下拐点且斜率符合要求
                        double k = gradientCal(leftFilletDown, leftFilletUp);
                        double b = leftFilletDown.y - k * leftFilletDown.x;
                        int startIndex = 0;
                        if (!LastFlag ||
                            leftFilletDown.slope >
                                30) // 如果该图没能找到下拐点，则下方所有点一起拟合为一条直线
                            startIndex = leftFilletDown.slope;
                        for (int i = startIndex; i <= leftFilletUp.slope; i++) {
                            params->track->pointsEdgeLeft[i].y =
                                (int)(k * params->track->pointsEdgeLeft[i].x +
                                      b);
                        }
                        return true;
                    }
                } else if (
                    type &&
                    (repairing ||
                     (FirstLS < 40 &&
                      countSL >
                          30))) { // 如果之前已经开始转弯补线，则在退出状态之前都需补线
                    int x_end = 0;
                    for (int kkk = FirstLS + 1;
                         kkk < params->track->pointsEdgeLeft.size(); kkk++) {

                        if (params->track->pointsEdgeLeft[kkk].y >
                            3) // 找到第一个 左侧贴边白点
                        {
                            x_end = kkk;
                            break;
                        }
                    }
                    PointX startPoint =
                        params->track->pointsEdgeRight[0]; // 补线：起点
                    PointX midPoint1;
                    PointX midPoint2;
                    PointX endPoint =
                        params->track->pointsEdgeLeft[x_end]; // 补线：终点
                    midPoint1 = PointX(endPoint.x + 10,
                                       startPoint.y - 30); // 补线：中点
                    midPoint2 =
                        PointX(ROWSIMAGE - 30, startPoint.y - 15); // 补线：中点
                    vector<PointX> input = {startPoint, midPoint2, midPoint1,
                                            endPoint};
                    vector<PointX> b_modify = Bezier(0.01, input);
                    params->track->pointsEdgeRight.resize(0);
                    params->track->pointsEdgeLeft.resize(x_end);
                    for (int kk = 0; kk < b_modify.size(); ++kk) {
                        params->track->pointsEdgeRight.emplace_back(
                            b_modify[kk]);
                    }
                    vector<PointX> repair0;
                    repair0.push_back(endPoint);
                    repair0.push_back(params->track->pointsEdgeLeft[x_end - 1]);
                    repair0 = smoothLine(repair0);
                    for (int i = 0; i < repair0.size(); i++) {
                        params->track->pointsEdgeRight.push_back(repair0[i]);
                    }
                    repairing = true;
                    return true;
                }
            }
        } else if (type == 2) {
            int StickL = 0;
            int StickR = 0;
            for (
                int i = 0; i < params->track->pointsEdgeLeft.size();
                i++) { // 统计贴边点需要在一开始统计，否则之后的统计均会被补线所影响
                if (params->track->pointsEdgeLeft[i].y < 3) {
                    StickL++;
                } else
                    break;
            }
            for (int i = 0; i < params->track->pointsEdgeRight.size(); i++) {
                if (params->track->pointsEdgeRight[i].y > COLSIMAGE - 3) {
                    StickR++;
                } else
                    break;
            }
            PointX
                Right_Fillet_DOWN; // 左上圆弧点
                                   // 对于圆弧拐点，slope代表其index而不是斜率
            PointX leftFilletDown; // 左下圆弧点
            bool lastLFlag = false;
            bool lastRFlag = false;
            Right_Fillet_DOWN.slope = 0; // 初始化index
            leftFilletDown.slope = 0;
            leftFilletDown =
                searchFilletLeftDown(params->track->pointsEdgeLeft,
                                     params->track->pointsEdgeRight, -1, 0);
            Right_Fillet_DOWN =
                searchFilletRightDown(params->track->pointsEdgeRight,
                                      params->track->pointsEdgeLeft, -1, 0);
            if (!leftFilletDown.slope) {
                leftFilletDown = lastForkL;
                lastLFlag = true;
            }
            if (!Right_Fillet_DOWN.slope) {
                Right_Fillet_DOWN = lastForkR;
                lastRFlag = true;
            }
            if (lastLFlag && lastRFlag && StickL > 30 &&
                StickR > 30) { // 俩点都找不到了
                step = Step::EXIT;
                counterFork = 0;
                return true;
            }
            if (!lastLFlag && !lastRFlag && leftFilletDown.slope > 10 &&
                Right_Fillet_DOWN.slope > 10 &&
                !repairing) { // 距离弯道距离相对较远，先补直线便于接近
                // 将拐点上方使用直线进行补齐
                double k = gradientCal(
                    leftFilletDown,
                    params->track->pointsEdgeLeft[leftFilletDown.slope - 10]);
                double b = leftFilletDown.y - k * leftFilletDown.x;
                for (int i = leftFilletDown.slope;
                     i < params->track->pointsEdgeLeft.size(); i++) {
                    params->track->pointsEdgeLeft[i].y =
                        (int)(k * params->track->pointsEdgeLeft[i].x + b);
                }
                // 同理补充右侧直线
                k = gradientCal(
                    Right_Fillet_DOWN,
                    params->track
                        ->pointsEdgeRight[Right_Fillet_DOWN.slope - 10]);
                b = Right_Fillet_DOWN.y - k * Right_Fillet_DOWN.x;
                for (int i = Right_Fillet_DOWN.slope;
                     i < params->track->pointsEdgeRight.size(); i++) {
                    params->track->pointsEdgeRight[i].y =
                        (int)(k * params->track->pointsEdgeRight[i].x + b);
                }
                repairing = true;
            } else if (leftFilletDown.slope && Right_Fillet_DOWN.slope) {
                int x_end = 0;
                for (int kkk = leftFilletDown.x - 50; kkk > 0;
                     kkk--) // 起点需要加15屏蔽与其y值相同部分产生的白到黑跳变
                {
                    if (img.at<uchar>(kkk, leftFilletDown.y) < 255 &&
                        img.at<uchar>(kkk + 1, leftFilletDown.y) >
                            0) // 找到从左下拐点到上方第一个白到黑跳变点
                    {
                        x_end = kkk;
                        break;
                    }
                }
                PointX startPoint = Right_Fillet_DOWN; // 补线：起点
                PointX midPoint1;
                PointX midPoint2;
                PointX endPoint = PointX(x_end, leftFilletDown.y); // 补线：终点
                midPoint1 = PointX(endPoint.x, startPoint.y - 30); // 补线：中点
                midPoint2 =
                    PointX(startPoint.x - 30, startPoint.y - 15); // 补线：中点
                vector<PointX> input = {startPoint, midPoint2, midPoint1,
                                        endPoint};
                vector<PointX> b_modify = Bezier(0.01, input);
                params->track->pointsEdgeRight.resize(Right_Fillet_DOWN.slope);
                if (lastRFlag && Right_Fillet_DOWN.slope < 15) {
                    if (b_modify.size() > 3) {
                        double k = gradientCal(b_modify[0], b_modify[3]);
                        double b = b_modify[0].y - k * b_modify[0].x;
                        for (int i = 0; i < Right_Fillet_DOWN.slope; i++) {
                            params->track->pointsEdgeRight[i].y =
                                (int)(k * params->track->pointsEdgeRight[i].x +
                                      b);
                        }
                    }
                }
                for (int kk = 0; kk < b_modify.size(); ++kk) {
                    params->track->pointsEdgeRight.emplace_back(b_modify[kk]);
                }
                vector<PointX> repair0;
                repair0.push_back(endPoint);
                for (int i = params->track->pointsEdgeLeft.size() - 1;
                     i > leftFilletDown.slope; i--) { // 将左侧有效点加入
                    if (params->track->pointsEdgeLeft[i].y > 5) {
                        if (endPoint.y > params->track->pointsEdgeLeft[i].y)
                            repair0.push_back(params->track->pointsEdgeLeft[i]);
                    } else
                        break;
                }
                repair0 = smoothLine(repair0); // 光滑补线
                for (int i = 0; i < repair0.size(); i++) {
                    params->track->pointsEdgeRight.push_back(repair0[i]);
                } // 完成右侧补线，接下来进行左侧补线
                for (int i = leftFilletDown.slope;
                     i < params->track->pointsEdgeLeft.size(); i++) {
                    if (params->track->pointsEdgeLeft[i].y < 3) {
                        endPoint = params->track->pointsEdgeLeft[i];
                        params->track->pointsEdgeLeft.resize(i);
                    }
                }
                repair0.resize(0);
                repair0.push_back(
                    params->track
                        ->pointsEdgeLeft[params->track->pointsEdgeLeft.size() -
                                         1]);
                repair0.push_back(endPoint);
                repair0 = smoothLine(repair0);
                for (int i = 0; i < repair0.size(); i++) {
                    params->track->pointsEdgeLeft.push_back(repair0[i]);
                } // 完成所有补线
                repairing = true;
            }

            if (repairing)
                return true;
        }
    } else if (step == Step::EXIT) {
        // 能进入Exiting状态那肯定是需要转弯了
        counterFork++; // 记录处理场数
        if (type < 2) {
            int countSL =
                countFormerStickpointL(params->track->pointsEdgeLeft, 0,
                                       params->track->pointsEdgeLeft.size() /
                                           2); // 统计上拐点下方有多少贴边左边点
            lastForkR.slope = 0;               // 开始重新找分岔点
            if (params->track->spurroad.size() > 0) {
                for (int i = 0; i < params->track->spurroad.size(); i++) {
                    if (params->track->spurroad[i].x > ROWSIMAGE / 3 &&
                        params->track->spurroad[i].y > COLSIMAGE / 3) {
                        lastForkR =
                            params->track->spurroad
                                [i]; // 出现符合要求的分岔点后改为补一条直线到分岔点+进入EXITING状态
                        lastForkR.slope = 1;
                        break;
                    }
                }
            }
            if (countSL < 10 || lastForkR.slope == 0) {
                step = Step::END;
                return true;
            } else {
                // 找到上下拐点且斜率符合要求
                double k =
                    gradientCal(params->track->pointsEdgeRight[0], lastForkR);
                double b = params->track->pointsEdgeRight[0].y -
                           k * params->track->pointsEdgeRight[0].x;
                int startIndex = 0;
                for (int i = startIndex;
                     i <= params->track->pointsEdgeRight[0].x - lastForkR.x;
                     i++) {
                    params->track->pointsEdgeRight[i].y =
                        (int)(k * params->track->pointsEdgeRight[i].x + b);
                }
                vector<PointX> repair0;
                repair0.push_back(lastForkR);
                params->track->pointsEdgeLeft.resize(
                    params->track->pointsEdgeRight[0].x - lastForkR.x + 1);
                params->track->pointsEdgeRight.resize(
                    params->track->pointsEdgeRight[0].x - lastForkR.x +
                    1); // 类似种树问题，从0开始到index结束
                params->track->handle(true,
                                      params->track->pointsEdgeRight[0].x -
                                          lastForkR.x + 1);
                for (int i =
                         params->track->pointsEdgeRight[0].x - lastForkR.x + 1;
                     i < params->track->pointsEdgeRight.size();
                     i++) // 从重搜第一点开始存储
                {
                    repair0.push_back(params->track->pointsEdgeRight[i]);
                }
                params->track->pointsEdgeRight.resize(
                    params->track->pointsEdgeRight[0].x - lastForkR.x + 1);
                repair0 = smoothLine(repair0);
                for (int i = 0; i < repair0.size(); i++) {
                    params->track->pointsEdgeRight.push_back(repair0[i]);
                }
                return true;
            }
        } else if (type == 2) {
            if (params->track->widthBlock.size() >
                200) { // 白块数量较多，退出状态
                step = Step::END;
                return true;
            }
            int StickL = 0;
            int StickR = 0;
            for (
                int i = 0; i < params->track->pointsEdgeLeft.size();
                i++) { // 统计贴边点需要在一开始统计，否则之后的统计均会被补线所影响
                if (params->track->pointsEdgeLeft[i].y < 3) {
                    StickL++;
                } else
                    break;
            }
            for (int i = 0; i < params->track->pointsEdgeRight.size(); i++) {
                if (params->track->pointsEdgeRight[i].y > COLSIMAGE - 3) {
                    StickR++;
                } else
                    break;
            }
            if (!StickR && !StickL) { // 左右均无前沿贴边，退出状态
                step = Step::END;
                return true;
            }
            PointX leftFilletDown = lastForkL;
            PointX Right_Fillet_DOWN = lastForkR;
            if (leftFilletDown.slope && Right_Fillet_DOWN.slope) {
                int x_end = 0;
                for (int kkk = leftFilletDown.x - 50; kkk > 0;
                     kkk--) // 起点需要加15屏蔽与其y值相同部分产生的白到黑跳变
                {
                    if (img.at<uchar>(kkk, leftFilletDown.y) < 255 &&
                        img.at<uchar>(kkk + 1, leftFilletDown.y) >
                            0) // 找到从左下拐点到上方第一个白到黑跳变点
                    {
                        x_end = kkk;
                        break;
                    }
                }
                PointX startPoint = Right_Fillet_DOWN; // 补线：起点
                PointX midPoint1;
                PointX midPoint2;
                PointX endPoint = PointX(x_end, leftFilletDown.y); // 补线：终点
                midPoint1 = PointX(endPoint.x, startPoint.y - 30); // 补线：中点
                midPoint2 =
                    PointX(startPoint.x - 30, startPoint.y - 15); // 补线：中点
                vector<PointX> input = {startPoint, midPoint2, midPoint1,
                                        endPoint};
                vector<PointX> b_modify =
                    Bezier(0.01, input); // 拟定右边第一段补线
                // 对右边界进行处理
                params->track->pointsEdgeRight.resize(Right_Fillet_DOWN.slope);
                if (Right_Fillet_DOWN.slope < 15) {
                    if (b_modify.size() > 3) {
                        double k = gradientCal(b_modify[0], b_modify[3]);
                        double b = b_modify[0].y - k * b_modify[0].x;
                        for (int i = 0; i < Right_Fillet_DOWN.slope; i++) {
                            params->track->pointsEdgeRight[i].y =
                                (int)(k * params->track->pointsEdgeRight[i].x +
                                      b);
                        }
                    }
                }
                // 接下来进行尾端处理
                vector<PointX> repair0;
                repair0.push_back(endPoint);
                bool LeftPushRightFlag =
                    false; // 使用是否有左侧点加入右侧点判断是否使用另一种补线方法
                for (int i = params->track->pointsEdgeLeft.size() - 1;
                     i > leftFilletDown.slope; i--) { // 将左侧有效点加入
                    if (params->track->pointsEdgeLeft[i].y > 5) {
                        if (endPoint.y > params->track->pointsEdgeLeft[i].y) {
                            repair0.push_back(params->track->pointsEdgeLeft[i]);
                            LeftPushRightFlag = true;
                        }
                    } else {
                        break;
                    }
                }
                if (LeftPushRightFlag) // 如果曾有左侧点加入右侧，那么此时上方必为凸字型，需要使用大弯弧补过去
                {
                    for (int kk = 0; kk < b_modify.size(); ++kk) {
                        params->track->pointsEdgeRight.emplace_back(
                            b_modify[kk]);
                    }
                    for (int i = leftFilletDown.slope;
                         i < params->track->pointsEdgeLeft.size(); i++) {
                        if (params->track->pointsEdgeLeft[i].y < 3) {
                            endPoint = params->track->pointsEdgeLeft[i];
                            params->track->pointsEdgeLeft.resize(i);
                        }
                    }
                    // 找到track.pointsEdgeLeft最上面点上方第一个白到黑的跳变点加入到repair0队列中
                    x_end = 0;
                    for (
                        int kkk =
                            params->track
                                ->pointsEdgeLeft
                                    [params->track->pointsEdgeLeft.size() - 1]
                                .x -
                            30;
                        kkk > 0;
                        kkk--) // 起点需要加15屏蔽与其y值相同部分产生的白到黑跳变
                    {
                        if (img.at<uchar>(kkk, 0) < 255 &&
                            img.at<uchar>(kkk + 1, 0) >
                                0) // 找到从左下拐点到上方第一个白到黑跳变点
                        {
                            x_end = kkk;
                            break;
                        }
                    }
                    repair0.push_back(PointX(x_end, 0));
                    repair0 = smoothLine(repair0); // 光滑补线
                    for (int i = 0; i < repair0.size(); i++) {
                        params->track->pointsEdgeRight.push_back(repair0[i]);
                    } // 完成右侧补线，接下来进行左侧补线
                    repair0.resize(0);
                    repair0.push_back(
                        params->track->pointsEdgeLeft
                            [params->track->pointsEdgeLeft.size() - 1]);
                    repair0.push_back(endPoint);
                    repair0 = smoothLine(repair0);
                    for (int i = 0; i < repair0.size(); i++) {
                        params->track->pointsEdgeLeft.push_back(repair0[i]);
                    } // 完成所有补线
                    return true;
                } else { // 否则则需要使用重搜
                    // 找到track.pointsEdgeRight第一个白到黑的跳变点并以它为基准进行重搜
                    PointX TempPoint = b_modify[b_modify.size() - 1];
                    for (int kk = 0; kk < b_modify.size(); ++kk) {
                        if (img.at<uchar>(b_modify[kk].x, b_modify[kk].y) > 0)
                            params->track->pointsEdgeRight.emplace_back(
                                b_modify[kk]);
                        else {
                            TempPoint = b_modify
                                [kk]; // 记录第一个处于黑色区域的点，接下来会从该点进行重搜
                            break;
                        }
                    }
                    int sizeLNow = params->track->pointsEdgeLeft.size();
                    params->track->pointsEdgeLeft.resize(
                        params->track->pointsEdgeRight[0].x - TempPoint.x + 1);
                    for (int i = sizeLNow;
                         i < params->track->pointsEdgeLeft.size();
                         i++) { // 如果有左侧点被错误补出
                        if (params->track->pointsEdgeLeft[i].x == 0) {
                            params->track->pointsEdgeLeft[i].x =
                                params->track->pointsEdgeLeft[i - 1].x - 1;
                            params->track->pointsEdgeLeft[i].y =
                                params->track->pointsEdgeLeft[i - 1].y;
                        }
                    }
                    int sizeRNow = params->track->pointsEdgeRight.size();
                    int deltaPoint =
                        params->track->pointsEdgeRight[0].x - TempPoint.x + 1 -
                        params->track->pointsEdgeRight
                            .size(); // 记录此时打算resize的大小与如今右边点数量的差异
                    params->track->pointsEdgeRight.resize(
                        params->track->pointsEdgeRight[0].x - TempPoint.x +
                        1); // 类似种树问题，从0开始到index结束
                    for (int i = sizeRNow - 1; i < sizeRNow + deltaPoint;
                         i++) // 如果有右侧点被错误补充，先进行初始处理
                    {
                        params->track->pointsEdgeRight[i].y =
                            params->track->pointsEdgeRight[sizeRNow - 1]
                                .y; // 暂时先使用最后一个点的y值把所有（0，0）补齐
                        params->track->pointsEdgeRight[i].x =
                            params->track->pointsEdgeRight[sizeRNow - 1].x - i +
                            sizeRNow - 1;
                    }
                    params->track->handle(
                        true,
                        params->track->pointsEdgeRight[0].x - TempPoint.x +
                            1); // 可能出现多点的情况，补充了一部分（0，0）
                    // 对这部分(0，0)点进行重新定位
                    double k = gradientCal(
                        params->track->pointsEdgeRight[sizeRNow - 1],
                        params->track
                            ->pointsEdgeRight[sizeRNow - 1 + deltaPoint]);
                    double b =
                        params->track->pointsEdgeRight[sizeRNow - 1].y -
                        k * params->track->pointsEdgeRight[sizeRNow - 1].x;
                    for (int i = sizeRNow - 1; i < sizeRNow - 1 + deltaPoint;
                         i++) {
                        params->track->pointsEdgeRight[i].y =
                            (int)(k * params->track->pointsEdgeRight[i].x + b);
                    }
                    // 需要对之前补零点的地方进行修正
                    repair0.resize(0);
                    repair0.push_back(TempPoint);
                    for (int i = params->track->pointsEdgeRight[0].x -
                                 TempPoint.x + 1;
                         i < params->track->pointsEdgeRight.size(); i++) {
                        repair0.push_back(params->track->pointsEdgeRight[i]);
                    }
                    repair0.push_back(
                        params->track->pointsEdgeLeft
                            [params->track->pointsEdgeLeft.size() - 1]);
                    repair0 = smoothLine(repair0);
                    params->track->pointsEdgeRight.resize(
                        params->track->pointsEdgeRight[0].x - TempPoint.x + 1);
                    while (
                        repair0[0].x >
                            params->track
                                ->pointsEdgeRight
                                    [params->track->pointsEdgeRight.size() - 1]
                                .x &&
                        params->track->pointsEdgeRight.size() >
                            10) { // 一直跳到x连续
                        params->track->pointsEdgeRight.pop_back();
                    }
                    for (int i = 0; i < repair0.size(); i++) {
                        params->track->pointsEdgeRight.push_back(repair0[i]);
                    }
                    return true;
                }
            }
        }
    } else if (step == Step::END) {
        reset();
        return false;
    }
    if (counterFork > 100) {
        reset();
        return false;
    }
    return false; // 默认返回false
}

/**
 * @brief 搜索左上拐点，必须找到下拐点或者前一下拐点才会补线
 *
 * @param pointsEdgeLeft
 * @param leftFilletDown
 * @return PointX
 */
PointX FsmFork::searchFilletLeftUp(vector<PointX> pointsEdgeLeft,
                                   PointX leftFilletDown) {
    PointX leftFilletUp; // 左上圆弧点
    leftFilletUp.slope = 0;
    for (int i = leftFilletDown.slope + 10; i < pointsEdgeLeft.size() - 2;
         i++) {
        if (pointsEdgeLeft[i].y > leftFilletDown.y &&
            pointsEdgeLeft[i + 1].y > pointsEdgeLeft[i].y &&
            pointsEdgeLeft[i + 2].y >= pointsEdgeLeft[i + 1].y &&
            pointsEdgeLeft[i + 2].y - pointsEdgeLeft[i].y <
                3) // 连续三点大于下点，开始补线
        {
            leftFilletUp = pointsEdgeLeft[i];
            leftFilletUp.slope = i;
            return leftFilletUp;
        }
    }
    return leftFilletUp;
};

/**
 * @brief 搜索左下拐点，可加入条件是否判断右直线侧
 *
 * @param pointsEdgeLeft
 * @param pointsEdgeRight
 * @param rowsEnd
 * @param straightSideJudge
 * @return PointX
 */
PointX FsmFork::searchFilletLeftDown(vector<PointX> pointsEdgeLeft,
                                     vector<PointX> pointsEdgeRight,
                                     int rowsEnd, bool straightSideJudge) {
    // 先找到左侧道路上下直道段上下直道段//
    // 对于圆弧拐点，slope代表其index而不是斜率
    PointX leftFilletDown; // 左下圆弧点
    leftFilletDown.slope = 0;
    if (rowsEnd == -1 || rowsEnd > pointsEdgeLeft.size() - 4) {
        rowsEnd = pointsEdgeLeft.size() - 4;
    }
    for (int i = 5; i < rowsEnd; i++) { // 先找下拐点
        if (!leftFilletDown.slope && pointsEdgeLeft[i].y > 10 &&
            pointsEdgeLeft[i].y >
                pointsEdgeLeft[i + 1]
                    .y) // 第一次找下拐点，不找贴边点，找第一个反向点
            if (posInRange(pointsEdgeLeft[i].slope, -1, 0) &&
                posInRange(pointsEdgeLeft[i - 1].slope, -1, 0) &&
                posInRange(pointsEdgeLeft[i - 2].slope, -1,
                           0)) // 前三点是一个直线
            {
                bool ileagal = false;
                for (int pj = i + 1; pj < i + 3; pj++) { // 左侧多点反向
                    if (pointsEdgeLeft[pj].y < pointsEdgeLeft[pj + 1].y) {
                        ileagal = true;
                    }
                }
                if (straightSideJudge) {
                    for (int pj = i + 1; pj < i + 3; pj++) {
                        if (!posInRange(pointsEdgeRight[pj].slope, 0,
                                        1)) { // 右侧必须全部为有效直线
                            ileagal = true;
                        }
                    }
                }
                if (ileagal == true) {
                    continue;
                }
                // 如果连续三点反向且前三点均为直线
                for (int k = i; k > 0; k--) {
                    if (pointsEdgeLeft[k].y != pointsEdgeLeft[i].y) {
                        leftFilletDown = pointsEdgeLeft[k];
                        leftFilletDown.slope = k;
                        if (lastForkL.x <
                            leftFilletDown
                                .x) // 一轮循环中找到的下拐点只可能越来越靠下
                            lastForkL = leftFilletDown;
                        return leftFilletDown;
                    }
                }
            }
    }
    return leftFilletDown;
};

/**
 * @brief 搜索右下拐点，可加入条件是否判断左直线侧
 *
 * @param pointsEdgeRight
 * @param pointsEdgeLeft
 * @param rowsEnd
 * @param straightSideJudge
 * @return PointX
 */
PointX FsmFork::searchFilletRightDown(vector<PointX> pointsEdgeRight,
                                      vector<PointX> pointsEdgeLeft,
                                      int rowsEnd, bool straightSideJudge) {
    // 先找到左侧道路上下直道段上下直道段//
    // 对于圆弧拐点，slope代表其index而不是斜率
    PointX Right_Fillet_DOWN; // 左下圆弧点
    Right_Fillet_DOWN.slope = 0;
    if (rowsEnd == -1 || rowsEnd > pointsEdgeRight.size() - 4) {
        rowsEnd = pointsEdgeRight.size() - 4;
    }
    for (int i = 5; i < rowsEnd; i++) { // 先找下拐点
        if (!Right_Fillet_DOWN.slope && pointsEdgeRight[i].y < COLSIMAGE - 10 &&
            pointsEdgeRight[i].y <
                pointsEdgeRight[i + 1]
                    .y) // 第一次找下拐点，不找贴边点，找第一个反向点
            if (posInRange(pointsEdgeRight[i].slope, 0, 1) &&
                posInRange(pointsEdgeRight[i - 1].slope, 0, 1) &&
                posInRange(pointsEdgeRight[i - 2].slope, 0,
                           1)) // 前三点是一个直线
            {
                bool ileagal = false;
                for (int pj = i + 1; pj < i + 3; pj++) { // right侧多点反向
                    if (pointsEdgeRight[pj].y > pointsEdgeRight[pj + 1].y) {
                        ileagal = true;
                    }
                }

                if (straightSideJudge) {
                    for (int pj = i + 1; pj < i + 3; pj++) {
                        if (!posInRange(pointsEdgeLeft[pj].slope, -1,
                                        0)) { // left侧必须全部为有效直线
                            ileagal = true;
                        }
                    }
                }
                if (ileagal == true) {
                    continue;
                }
                // 如果连续三点反向且前三点均为直线
                for (int k = i; k > 0; k--) {
                    if (pointsEdgeRight[k].y != pointsEdgeRight[i].y) {
                        Right_Fillet_DOWN = pointsEdgeRight[k];
                        Right_Fillet_DOWN.slope = k;
                        if (lastForkR.x <
                            Right_Fillet_DOWN
                                .x) // 一轮循环中找到的下拐点只可能越来越靠下
                            lastForkR = Right_Fillet_DOWN;
                        return Right_Fillet_DOWN;
                    }
                }
            }
    }
    return Right_Fillet_DOWN;
};

/**
 * @brief 搜索左侧贴边点数量函数，可设定起始和终止点
 *
 * @param pointsEdgeLeft
 * @param StartLine
 * @param Endline
 * @return int
 */
int FsmFork::countFormerStickpointL(vector<PointX> pointsEdgeLeft,
                                    int StartLine, int Endline) {
    if (pointsEdgeLeft.size() < ROWSIMAGE * 2 / 3) {
        Endline = pointsEdgeLeft.size();
    }
    int countStickL = 0;
    for (int i = StartLine; i < Endline; i++) {
        if (pointsEdgeLeft[i].y < 3) {
            countStickL++;
        }
    }
    return countStickL;
}

/**
 * @brief 搜索右侧贴边点数量函数，可设定起始和终止点
 *
 * @param pointsEdgeRight
 * @param StartLine
 * @param Endline
 * @return int
 */
int FsmFork::countFormerStickpointR(vector<PointX> pointsEdgeRight,
                                    int StartLine, int Endline) {
    if (pointsEdgeRight.size() < ROWSIMAGE * 2 / 3) {
        Endline = pointsEdgeRight.size();
    }
    int countStickR = 0;
    for (int i = StartLine; i < Endline; i++) {
        if (pointsEdgeRight[i].y > COLSIMAGE - 3) {
            countStickR++;
        }
    }
    return countStickR;
}