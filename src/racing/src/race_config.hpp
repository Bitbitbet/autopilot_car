#pragma once
#include <string>

// 比赛链路统一按 640x480 处理；像素阈值为 640x480 口径。
constexpr int RACE_COLS = 640;
constexpr int RACE_ROWS = 480;

// 任务决策参数
struct RaceConfig {
    // 检测
    float confidence = 0.65f;
    float maxDetectionAge = 0.40f; // 超过就当作没有 AI 结果
    float confirmGap = 0.30f;      // 两次确认之间允许的最大间隔
    // 速度上限 (m/s)
    float limitedSpeed = 0.40f;  // 限速区
    float approachSpeed = 0.12f; // 斑马线前
    float targetSpeed = 0.20f;   // 走目标点（停车场）时
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
    std::string modelDirectory = "../res/models/yolov3_mobilenet_v1";
};
