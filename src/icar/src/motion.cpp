#include "motion.hpp"
#include "params.hpp"
#include "tools.hpp"
#include <cmath>

using std::shared_ptr;

void Motion::poseControl(shared_ptr<Params> &params) {
    float error = params->ctrl.center - COLSIMAGE / 2.0; // 图像控制中心转换偏差
    static int errorLast = 0;                            // 记录前一次的偏差
    if (params->mode ==
        FsmMode::yfork) // yfork单边：立即跟随center，不被上一帧偏差限速拖反
        errorLast = (int)error;
    else if (abs(error - errorLast) >
             COLSIMAGE / 20.0) // 普通跟线：限速防猛打方向
    {
        error = error > errorLast ? errorLast + COLSIMAGE / 10
                                  : errorLast - COLSIMAGE / 10;
    }

    params->config.turnP =
        abs(error) * params->config.runP2 + params->config.runP1;
    int pwmDiff = (error * params->config.turnP) +
                  (error - errorLast) * params->config.turnD;
    errorLast = error;
    params->ctrl.servo = (uint16_t)(PWMSERVOMID + pwmDiff); // PWM转换
    if (params->ctrl.servo > PWMSERVOMAX)
        params->ctrl.servo = PWMSERVOMAX;
    else if (params->ctrl.servo < PWMSERVOMIN)
        params->ctrl.servo = PWMSERVOMIN;
}

void Motion::speedControl(std::shared_ptr<Params> &params) {
    if (params->ctrl.stop) { // 停车
        params->ctrl.speed = 0.0;
        return;
    }
    if (params->ctrl.crossStop) { // 斑马线停车专用：优先于任何状态速度
        params->ctrl.speed = 0.0;
        return;
    }

    if (params->ctrl.back) { // 倒车(停车场)
        params->ctrl.speed = -params->config.velPark;
        return;
    }
    if (params->ctrl.slow) { // 减速区速度
        params->ctrl.speed = params->config.velSlow;
        return;
    }
    if (params->mode == FsmMode::stop) { // 停车区速度
        params->ctrl.speed = params->config.velStop;
        return;
    } else if (params->mode == FsmMode::park) { // 停车场速度
        params->ctrl.speed = params->config.velPark;
        return;
    } else if (params->mode == FsmMode::cross) { // 斑马线速度
        params->ctrl.speed = params->config.velCross;
        return;
    } else if (params->mode == FsmMode::busy) { // 施工区速度
        params->ctrl.speed = params->config.velBusy;
        return;
    } else if (params->mode == FsmMode::yfork) { // 岔路口速度
        params->ctrl.speed = params->config.velYFork;
        return;
    }

    else if (params->ctrl.countAcc < 50) {
        params->ctrl.countAcc++;
        params->ctrl.speed = params->config.velCross;
        return;
    }

    int line =
        params->ctrl.lineArea; // 动态速度，返回点越高，速度越大，线性变化

    // 控制率
    uint8_t controlLow = 3;   // 速度控制下限
    uint8_t controlMid = 5;   // 控制率
    uint8_t controlHigh = 10; // 速度控制上限

    float upper = ROWSIMAGE * 0.35; // 84
    line = std::max(line - 20, 0);
    if (line > upper)
        line = upper;

    params->ctrl.speed = params->config.velLow +
                         std::pow(float(upper - line) / (upper), 3) *
                             (params->config.velHigh - params->config.velLow);
    if (params->ctrl.speed > params->config.velHigh)
        params->ctrl.speed = params->config.velHigh;
}

void Motion::drawImage(std::shared_ptr<Params> &params, Mat &img) {
    std::string str = "Vel: " + double2String(params->ctrl.speed, 2);
    putText(img, str, Point(COLSIMAGE - 100, 120), FONT_HERSHEY_PLAIN, 1,
            Scalar(0, 0, 255), 1); // 速度
}

void Motion::outlineCheck(std::shared_ptr<Params> &params) {
    if (outline) {                // 已出现
        params->ctrl.stop = true; // 停车
        countRes++;
        if (countRes > 15) {
            std::cout << "-----> [Stop] Game over, system exit!!! <-----"
                      << std::endl;
            std::exit(0); // 程序退出
        }
    } else { // 出线检测
        if (params->track->pointsEdgeLeft.size() < 30 &&
            params->track->pointsEdgeRight.size() < 30) // 防止车辆冲出赛道
        {
            countRes++;
            countOut = 0;
            if (countRes > 10)
                outline = true;
        } else {
            countOut++;
            if (countOut > 30) {
                countRes = 0;
                countOut = 30;
            }
        }
    }
}