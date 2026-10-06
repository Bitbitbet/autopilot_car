#pragma once

#include "fsm/fsm.hpp"

/**
 * @brief 停车场岔路图像识别与规划
 *
 */
class FsmFork : public FSMState {
  public:
    FsmFork(std::shared_ptr<Params> par);
    ~FsmFork();
    void run(Mat &img);
    void show(Mat &img);
    FsmMode getMode();

  private:
    /**
     * @brief 场景状态
     *
     */
    enum Step {
        NONE = 0, // 未知类型
        ENTER,    // 入环
        EXIT,     // 出环
        END       // 环任务结束
    };
    bool enable = false;    // 状态使能
    Step step = Step::NONE; // 环岛处理阶段
    bool repairing = false; // 判断是否正在补线
    int counterFork =
        0; // 记录一个状态的图像场数，如果卡某个状态控制可通过此标志调整出状态时间
    PointX lastForkL;       // 左下有效连接点
    PointX lastForkR;       // 右下有效连接点
    uint16_t countRec = 0;  // AI场景识别计数器
    uint16_t countSes = 0;  // 场次计数器
    uint16_t countExit = 0; // 程序退出计数器

    void reset(void);
    bool handle(Mat &img, int type = 0);
    PointX searchFilletLeftUp(std::vector<PointX> pointsEdgeLeft,
                              PointX leftFilletDown);
    PointX searchFilletLeftDown(std::vector<PointX> pointsEdgeLeft,
                                std::vector<PointX> pointsEdgeRight,
                                int rowsEnd = -1,
                                bool straightSideJudge = true);
    PointX searchFilletRightDown(std::vector<PointX> pointsEdgeRight,
                                 std::vector<PointX> pointsEdgeLeft,
                                 int rowsEnd = -1,
                                 bool straightSideJudge = true);
    int countFormerStickpointL(std::vector<PointX> pointsEdgeLeft,
                               int StartLine = 0,
                               int Endline = ROWSIMAGE * 2 / 3);
    int countFormerStickpointR(std::vector<PointX> pointsEdgeRight,
                               int StartLine = 0,
                               int Endline = ROWSIMAGE * 2 / 3);
};
