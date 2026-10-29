#pragma once
#include <string>

// 比赛链路统一按 640x480 处理；像素阈值为 640x480 口径。
constexpr int RACE_COLS = 640;
constexpr int RACE_ROWS = 480;

// 任务决策参数。和 trackview 的巡线参数一样用编译期常量，现场改完重编即可。
struct RaceConfig {
    // 检测
    float confidence = 0.65f;
    float maxDetectionAge = 0.40f; // 超过就当作没有 AI 结果
    int confirmFrames = 3;         // 同一类别连续多少帧才算确认
    float confirmGap = 0.30f;      // 两次确认之间允许的最大间隔
    // 速度上限 (m/s)
    float limitedSpeed = 0.40f; // 限速区
    float forkSpeed = 0.15f;    // 岔路
    float approachSpeed = 0.12f; // 斑马线前
    // 岔路
    float directionMemory = 4.f;
    float forkConfirmTime = 0.12f;
    float forkMinTime = 0.40f;
    float forkMaxTime = 4.f;
    float forkExitTime = 0.35f;
    float cooldown = 2.f;
    int minBranchWidth = 60;      // 单条支路的最小横向宽度（px）
    int minBranchGap = 24;        // 分隔带的最小宽度（px）
    int minBranchRows = 3;        // 分叉至少连续多少扫描行
    int nearCutRow = 20;          // 靠车头多少行不参与几何判定
    // 触发行（像素，0 在图像顶部）
    int triggerRow = 290;
    int obstacleRow = 310;
    float clearTime = 0.60f; // 障碍/道闸消失多久才算清空
    // 斑马线后停车
    bool stopAfterCross = true;
    int crossPassedRow = 410;
    float crossClearTime = 0.25f;
    float stopHoldTime = 1.f;
    // 道闸：默认要求同一个道闸由横变竖才算抬起
    float gateOpenAspect = 0.75f;
    // 复核开关
    bool verifyForkArrows = true;
    bool verifyStopCandidates = true;
    float stopCandidateConfidence = 0.35f; // 仅停车牌候选
    bool verifyCrosswalkGeometry = true;
    float crossCandidateConfidence = 0.10f; // 仅斑马线候选
    std::string modelDirectory = "../res/models/yolov3_mobilenet_v1";
};
