#pragma once

#include "fsm/fsm.hpp"

/**
 * @brief 停车场控制
 *
 */
class FsmPark : public FSMState {
  public:
    FsmPark(std::shared_ptr<Params> par);
    ~FsmPark();
    void run(Mat &img);
    void show(Mat &img);
    FsmMode getMode();

    bool stopping = false; // 停车等待标志
    bool fitting = false;  // 控制中心拟合标志
    uint16_t speedUp = 0;  // 出库加速延迟计数器
    float spotUp = 0.3;    // 上停车位距离像素百分比[0,1]
    float spotDown = 0.6;  // 下停车位距离像素百分比[0,1]

  private:
    /**
     * @brief 车位信息
     *
     */
    class Spots {
      private:
        PredictResult carNone;

      public:
        std::vector<PredictResult>
            forks; // 停车位箭头信息（size=0:无停车位，size=1:停车位1/2，size=2:停车位1/2/3/4）
        std::vector<Point2d> forksIpm; // 停车标志IPM坐标
        std::vector<Point2d> carsIpm;  // 车辆IPM坐标
        int counter[4] = {0};          // 车位检测计数器
        bool spotEnable[4] = {true};   // 停车位使能标志
        int countRes = 0;              // 车位检测计数器
        bool checked = false;          // 已确定停车位编号标志
        int times = 0;                 // 临时计数器

        std::vector<PredictResult> carPark = {
            carNone, carNone, carNone, carNone}; // 1/2/3/4号停车位检测车辆信息

        void reset() {
            for (int i = 0; i < 4; i++) {
                counter[i] = 0;
                spotEnable[i] = true;
            }
            forks.clear();
            countRes = 0; // 车位检测计数器
            checked = false;
            times = 0;
        }
    };

    /**
     * @brief 停车步骤
     *
     */
    enum Step {
        NONE = 0, // 未知状态
        ENABLE,   // 停车场使能
        FORKIN,   // 入库岔路转向
        TRACKIN,  // 入库巡线
        ENTER,    // 入库
        PARKING,  // 停车
        EXIT,     // 出库
        TRACKOUT, // 出库巡线
        FORKOUT,  // 出库岔路转向

    };
    Spots spots;            // 车位信息
    Step step = Step::NONE; // 停车步骤
    uint16_t countFlow = 0; // 直行计数器
    uint16_t countRes = 0;  // AI场景识别计数器
    uint16_t countSes = 0;  // 场次计数器
    uint16_t timeout = 0;   // 超时计数器
    int countOut = 0;       // 出库检测计数
    bool waiting = false;   // 停车等待使能
    int countWait = 0;      // 停车等待计数器
    int countIn = 0;        // 入库矫正计数器
    std::vector<std::vector<PointX>> pointsEdgeLeftPast,
        pointsEdgeRightPast; // 记录赛道入库车道线

    void setStep(Step st);
    void reset();
    void replanTracking();
    void replanTracking(bool left);
    bool findSymbols(std::vector<PredictResult> results, int label);
    bool findSymbols(std::vector<PredictResult> results, int label,
                     PredictResult &target);
    std::vector<PredictResult>
    findParkStation(std::vector<PredictResult> results);
    PointX getResultCenter(PredictResult res);
    void findParkCars(std::vector<PredictResult> results);
};
