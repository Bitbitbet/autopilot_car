# 比赛基础集成版使用说明

程序：`trackview_race`。分支：`trackview/race-integration-20261016`。新增代码注释为中文。

保留 `trackview` 与组员 `icar` 的原始入口，原 v6 的配置和转向实现未修改。比赛入口是从本次 v6 基线独立派生的文件，以后如修改公共巡线流程，需同步核对两个入口；不要把它误当成原 v6 已自动增加全部任务。

## 文件

- `src/tool/trackview_race.cpp`：摄像头、键盘/手柄、串口与任务总入口。
- `src/tool/race_perception.hpp/.cpp`：最新帧异步队列，复用组员的检测后端，初始化时检查模型配置和 15 类标签顺序。
- `src/tool/race_backend_loader.cpp`：仅实车推理时加载独立 AI 动态库，帮助和回放不执行设备认证。
- `src/tool/race_manager.hpp`：岔路扫描、任务状态和速度/停车仲裁。
- `src/tool/race_config.hpp`、`config/race.json`：任务策略参数。
- `src/tool/race_replay.hpp`：离线读取检测记录；禁止使用离线记录控制实车。
- `src/tool/race_preview.hpp`：与推理帧对应的检测框、类别、置信度和样本保存。
- `tests/race_regression.cpp`：无实车回归测试。
- `docs/比赛准备与代码差距-20261016.md`：规则来源、准备及待完成项。

## 配置和显示

沿用原 `trackview.json`：maxSpeed=10、巡航初值 0.25、弯道 0.15、降级 0.12，原有键盘调整规则不变。

新增 `race.json` 只在比赛程序中启用：测速区 0.40，岔路 0.15，信号接近 0.12。最终发送速度受当前巡航、道路质量、弯道和任务速度共同约束。UI 的 speed 是指令值，不是编码器实测。

任务参数不是新的硬件能力承诺。`triggerRow`、`crossPassedRow`、分叉宽度等均为 320×240 图像上的像素阈值，需要按安装位置调试。

| 状态 | 含义 |
|---|---|
| AI_WAIT | 没有新鲜 AI 快照，自动模式停止 |
| FORK_ARMED | 已确认方向牌，尚未开始分叉选路 |
| FORK_LEFT / FORK_RIGHT | 跟随指定连通道路，交给 v6 转向 |
| FORK_WAIT | 发现分叉但没有确认方向 |
| LIMIT_ZONE | 保持测速区限制，等待解除牌 |
| CROSS_APPROACH | 依据本届规则等待经过斑马线的图像证据 |
| SIGNAL_WAIT | 停车并等待新鲜通行牌 |
| GATE_WAIT | 等待可核验的道闸抬起证据 |
| OBSTACLE_WAIT | 前方障碍等待；尚不是自动绕行 |
| REVIEW_STOP | 入库/施工规划待验证，或选路故障锁定；不会按 a 直接绕过 |

限速区状态不会因为手动/自动切换而清除。其他定时动作在人工介入后取消，避免继续已过时的动作。`REVIEW_STOP` 锁定需退出程序、确认现场和解决原因后再测试。

## 模型与资源

比赛程序与普通 v6 不同，还需要同目录 `librace_npu_backend.so`、完整的 `../res/models/` 以及车载 PPNC/ONNX 库。已有检测器会在启动时解压并编译 NMS，因此初始化可能较慢，需要车载 g++；这个步骤在打开电机串口前完成。没有模型不能仅复制本二进制就运行 AI。

本地已有参考模型保存在 `<实验资料目录>/github_review_20261010/赛道试跑准备包/res/models/`，标签顺序核验通过不代表其在当前场地的准确率已验证。不要将用于离线回放的合成标定文件上传到小车。

## 启动顺序

上传之前，先保留现有能跑的 v6；比赛程序和 race.json 使用独立文件名。仍需要同目录 trackview.json 及上一级真实相机标定文件。

先预览识别，不发送串口控制：

```bash
cd /root/track-trial-20261010/bin
chmod +x ./trackview_race
./trackview_race --config ./trackview.json --race-config ./race.json --no-control
```

这一步不会打开电机串口或手柄；AI 模型仍在车载板运行。确认检测框与真实目标对应，再逐项低速测试：

预览增加 `AI detections` 窗口：左侧画检测框，右侧显示类别/置信度，顶部显示帧号和结果年龄。`FRESH` 表示结果未过期，`STALE / WAIT` 表示不能作为当前有效结果。框必须落在对应实物上；高置信度本身不表示一定识别正确。

在该窗口按 `s`，只在 `--no-control` 模式保存当前 AI 原图 `camera.png`、画框图 `detections.png` 和 `detections.json`，目录为本次日志中的 `ai-sample-N`。没有目标的画面也能保存，便于分析漏检。绘图和保存都在 GUI 线程执行，不阻塞巡线线程；图像属于原推理帧，不能把旧框画到新图上误导判断。

```bash
./trackview_race --config ./trackview.json --race-config ./race.json
```

默认手动，点击图像窗口后按 `a` 自动，空格急停，Esc 退出。`+/-` 调巡航，`[/]` 调弯道，`,/.` 调降级速度，`w` 切权重。比赛基础版有尚未完成的任务，不应直接用于完整比赛。

## 离线回放

```bash
./trackview_race --replay sample.avi --detections detections.jsonl --headless \
  --config ./trackview.json --race-config ./race.json
```

每行记录一张原视频帧，编号从 1 开始递增，检测框对应已去畸变的 320×240 图像。例如：

```json
{"frame":1,"detections":[{"type":13,"score":0.95,"x":200,"y":60,"width":30,"height":40}]}
{"frame":2,"detections":[]}
```

省略某帧表示没有新推理结果；空数组表示已完成该帧推理但未检测到对象，两者不同。视频文件和检测文件都只在 `--replay` 下使用，此时不会打开串口和手柄。

日志目录增加 `race.csv`（状态、原因、快照编号、AI 年龄、速度限制）、`detections.jsonl`（实际收到的模型结果）和 `race-config.json`。检测日志不能直接与只有事件片段的视频配对：还必须匹配完整原视频的帧号与相机标定流程。

## 本地编译

本机已有 WSL/ARM64 SDK；运行 `<实验资料目录>/race_build_20261016.sh`，输出在 `<实验资料目录>/race_update_20261016/`。该脚本运行 ARM64 回归与离线回放，不连接车辆。

构建目标为 `trackview_race`，测试目标为 `race_regression`。原有源码、模型和运行包的 Git 规则仍遵守，不把私有测试视频或大型模型自动加入仓库。
