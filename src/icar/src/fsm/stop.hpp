#pragma once

#include "fsm/fsm.hpp"

/**
 * @brief 停车区规划与控制
 *
 */
class FsmStop : public FSMState {
  public:
    FsmStop(std::shared_ptr<Params> par);
    ~FsmStop();
    void run(Mat &img);
    void show(Mat &img);
    FsmMode getMode();

  private:
    /**
     * @brief 场景状态
     *
     */
    enum Step {
        NONE = 0, // AI未识别
        ENABLE,   // 场景使能
        STOP      // 停车
    };
    Step step = Step::NONE;          // 场景状态
    uint16_t countRec = 0;           // AI场景识别计数器
    uint16_t countSes = 0;           // 场次计数器
    int timeout = 0;                 // 超时计数器
    std::vector<cv::Point> polyRoad; // 赛道多边形点集
    std::vector<cv::Point> polyCar;  // 智能车多边形点集
    double overlap = 0;              // 重合度
    float scoreLap = 0.5;            // 智能车在赛道上的置信度

    double getRoadCarPloy(PredictResult result);
    double getOverlapArea(const std::vector<cv::Point> &polyA,
                          const std::vector<cv::Point> &polyB);
    void drawPolygon(Mat img, std::vector<Point> poly, bool fill,
                     bool close = true);
    void setStep(Step st);
};
