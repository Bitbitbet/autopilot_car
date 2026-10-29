# 智能小车比赛程序

`racing` 是比赛入口：一个循环里完成取图、巡线、AI 检测和模式切换，组织和 `icar` 一样直白。图像、标定和识别口径统一为 640×480，与 `trackview` 共用同一套 `Camera` / `Track`。检测结果只用来改运行模式（限速、停车、选路、绕障停车），不做离线记录，关键状态只走 `cout` / `cerr`。

## 可执行文件

| 程序 | 用途 |
|---|---|
| `racing` | 比赛主程序：巡线 + AI 检测 + 任务状态机 |
| `trackview` | 只做巡线，不带 AI；调参和验证巡线用 |
| `icar` | 组员的 FSM 版本，保留原样 |
| `camera_tool` / `calibration` / `calibrate_homography` / `detection_tool` / `collection` / `img2video` | 相机、标定、检测和采集的小工具 |

## racing

```bash
./racing              # 带图像窗口
./racing --headless   # 不开窗口，只跑车辆控制
```

没有串口时自动退化为不驱动车轮，只跑图像和决策。没有手柄时只能用图像窗口的按键。

按键：`a` 手动/自动切换（自动要求检测新鲜且路径稳定），`空格` 停车并锁定，`+`/`-` 巡航上限，`]`/`[` 弯道速度，`.`/`,` 降级速度，`w` 转向权重，`q` / `Esc` 退出。

- 巡线参数（速度、转向、路径判定、门限）在 `src/tool/trackview_pilot.hpp` 的 `namespace cfg` 里，编译期常量。
- 比赛参数（检测阈值、触发行、岔路/斑马线/道闸时间）在 `src/racing/src/race_config.hpp`，同为编译期常量。
- 黑色赛道把 `src/camera/include/camera.hpp` 里的 `BLACK_ROAD` 改成 `true` 重编；二值图始终是"白=可走"。

## 构建

需要 Linux ARM64 工具链和车载依赖：OpenCV 3.4、PPNC、ONNX Runtime、LibSerial、libevdev、libudev。C++20。

```bash
cmake --preset debug
cmake --build build-debug -j4
```

## 运行资源

- `../res/models/yolov3_mobilenet_v1/`：完整模型部署文件。
- `../res/models/config_ppncnna.json`、`config_ppncnms.json`：检测器的两个 PPNC 配置。
- `../res/calibration/calibration.xml`：本车相机标定（含地面单应），不能借用其他车的。
- 检测器启动时会用 `g++` 现场编译 NMS 后处理库，运行目录需要有 `g++`。

模型、标定和日志不进仓库（`res/` 已忽略）。缺资源时程序在打开串口之前报错退出。

## 代码位置

| 文件 | 内容 |
|---|---|
| `src/racing/src/main.cpp` | 主循环：取图、巡线、按键/手柄、串口、显示 |
| `src/racing/src/race_manager.hpp` | 路牌连续确认与任务状态机（限速、停车、选路、绕障停车） |
| `src/racing/src/race_perception.hpp/.cpp` | 单帧异步推理，只保留最新一帧结果 |
| `src/racing/src/race_detection_review.hpp`、`race_crosswalk.hpp`、`race_direction.hpp`、`race_appearance.hpp` | 停车牌/斑马线/箭头方向/顶部红牌的原图像素复核 |
| `src/tool/trackview_pilot.hpp` | 巡线转向与速度规划（640×480 像素口径） |
| `src/camera/` | 相机：去畸变、二值化、图像↔地面坐标 |
| `src/track/` | 从二值图取左右边缘（按地面距离限定搜索范围） |
