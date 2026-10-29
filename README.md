# 智能小车比赛程序

当前开发入口为 `trackview_race`，固定启动脚本为 `tools/run_car.sh`。程序复用原车的 YOLOv3 MobileNetV1 模型，在巡线基础上加入路牌任务决策、识别结果复核及运行诊断。后续继续更新同一入口，预览与实跑通过参数切换。

2026-10-29 源码同步：包含黑白路面模式、有限反光修补、米制直道锥桶绕行实验、DS v5 运动参考恢复整合，以及方向记忆续期和选路自动取证。用户已反馈黑色赛道反光弯道能通过；实际岔路与完整车尾越桶仍未验收。本轮已完成 ARM 编译及方向状态、异步样本保存检查。现场步骤见[下午实跑计划](docs/下午实跑计划.md)，外部方案仅作调研，见[绕行开源调研](docs/绕行开源调研.md)。

## 当前验证范围

- 原实验室 T710 已多次完成 0.10 m/s 的红牌停车、绿牌恢复。低分 stop 候选可通过同帧红白图形复核，仍保留真实分数、连续确认和接近条件。
- 斑马线增加同帧条纹几何识别：低分 cross 候选经条纹复核，模型漏检时可用独立几何证据。绿牌及其他类别继续使用原检测阈值。真实岔路、斑马线停车组合、道闸及另一辆车适配需要现场验证；施工绕行、自动泊车和起终点计圈尚未完成。
- 启动巡航、弯道和降级速度均为 0.10 m/s；最高可调上限为 10 m/s。这个上限不代表已验证的实车速度。
- 绕障路径规划已接入 `preview` 和 `drive` 的后台观测，普通 `drive` 仍遇障停车。显式 `bypass competition` 使用当前车四点地面标定、运动估计和车尾净空判断执行直道静态锥桶绕行实验，完整通过尚待现场验证；详见[绕行实验](docs/绕行实验.md)。
- 主程序和 AI 接入库需配套编译、部署。模型权重没有重新训练或替换。

## 源码位置

| 目录或文件 | 用途 |
|---|---|
| `src/tool/trackview_race.cpp` | 摄像头、巡线、键盘手柄、串口和任务总入口 |
| `src/tool/race_manager.hpp`、`race_config.hpp` | 任务状态、路牌连续确认和策略参数 |
| `src/tool/race_detection_review.hpp`、`stop_candidate_review.hpp` | 停车牌候选接受条件及图形复核 |
| `src/tool/race_perception.cpp`、`race_backend_loader.cpp` | 原模型接入和配套库检查 |
| `src/tool/race_preview.hpp`、`race_ui.hpp`、`race_trace.hpp` | 匹配图像、原始分数、接受原因和诊断记录 |
| `tools/run_car.sh` | 日常统一入口 |
| `tools/read_car_compatibility.py` | 换车前的只读环境盘点 |
| `tests/` | 无硬件回归、模拟串口、启动脚本和可选样本回放 |
| `config/` | 通用配置模板，部署时需要核对现场参数 |

旧日期版脚本及相关说明保留用于复现历史测试，不作为日常启动入口。独立 NPU 图片诊断工具需要对应样本和部署环境；它们不参与主程序日常启动。

## 构建

需要兼容的 Linux ARM64 工具链和车载依赖，包括 OpenCV 3.4、PPNC、ONNX Runtime、LibSerial、libevdev、libudev。使用 C++20。具体库位置按目标车或交叉编译 sysroot 配置，不复制覆盖车辆系统库。

在目标车或已配置工具链的 Linux 环境，从仓库根目录执行：

```bash
cmake -S . -B build -DTRACKVIEW_BUILD_TESTS=ON
cmake --build build --target trackview_race -j4
```

交叉编译时通过 `-DCMAKE_TOOLCHAIN_FILE=/path/to/toolchain.cmake` 指定自己的工具链。`trackview_race` 的构建依赖会同时产生 `librace_npu_backend.so`。

## 准备独立运行目录

先确认 `car-test` 是本次新建或已归属本项目的目录。以下从仓库根目录准备同一套运行文件，模型和车辆标定需另外放入，不覆盖组员工程。

```bash
mkdir -p car-test/bin car-test/res/models car-test/res/calibration
cp build/trackview_race build/librace_npu_backend.so tools/run_car.sh car-test/bin/
cp config/trackview.json config/race.json config/navigation.json car-test/bin/
```

在 `bin` 内生成主程序与接入库的校验清单：

```bash
cd car-test/bin
python3 - <<'PY'
import hashlib, json
from pathlib import Path
files = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
         for name in ('trackview_race', 'librace_npu_backend.so')}
Path('release-manifest.json').write_text(
    json.dumps({'files': files, 'backend_abi': 3}, indent=2), encoding='utf-8')
PY
chmod +x ./trackview_race
```

运行前还需要：

- `../res/models/yolov3_mobilenet_v1/` 中完整、配套的原模型部署文件。
- `../res/models/config_ppncnna.json` 和 `config_ppncnms.json`，其 `model_dir` 指向 `../res/models/yolov3_mobilenet_v1`。
- 该车适配的 `../res/calibration/calibration.xml`；不能把另一辆车的标定当作已验证替代品。
- 车载设备认证环境及 `python3`、`g++`。原检测器启动会构建 NMS 后处理库。

模型、标定、实拍样本和日志不随源码仓库发布，`res/` 已忽略。不要从源码 checkout 缺少这些资源推断程序可以直接开车。

## 固定启动方式

在运行目录的 `bin` 内执行：

```bash
bash ./run_car.sh preview signal
```

预览不驱动车轮；识别无需按 `a`，按一次 `a` 后开始计算任务状态。确认预览与车辆适配后，使用 `bash ./run_car.sh drive signal` 实跑，按一次 `a` 启动自动，空格停车，`q/Esc` 或终端 `Ctrl+C` 退出。远程按键可能延迟，现场需保留停车手段。

场景参数：`lane` 为普通巡线并保留原任务规则；`signal` 为独立停车放行；`limit` 为独立限速解除测试；`competition` 保留原任务配置。signal 和 limit 的配置副本跳过先经过斑马线的条件，原配置不改写。

第三参数为 `white` 或 `black`，例如 `bash ./run_car.sh preview competition black`。二值图统一以白色表示可行驶道路；省略时沿用车辆配置 `roadSurface`，旧配置缺省为 white。黑路面支持有限内部反光/漆面空洞修补，保留连接图像边界的背景排除区。模式另有 `calibrate`（只标定）和 `bypass`（显式绕行实验）；上传或覆盖编译后的主程序后先执行一次 `chmod +x ./trackview_race`。

`+`/`=` 与 `-` 调巡航，`]`/`[` 调弯道，`.`/`,` 调降级速度，每次 0.05 m/s。重启后恢复初始值；任务和道路质量仍约束最终速度指令。

完整操作与验证边界见 [运行与验证](docs/运行与验证.md)。

直道锥桶绕行实验通过同一个入口执行 `bash ./run_car.sh calibrate competition` 做当前车的四点地面标定，再用 `bash ./run_car.sh bypass competition` 低速实跑。`bypass` 沿用初始速度0.10 m/s、最高可调上限10 m/s，视觉里程或路线中断锁停；`drive` 仍保留原障碍停车行为。操作、标点距离及测试边界见 [绕行实验](docs/绕行实验.md)。发布直接更新固定目录中的文件，不再生成压缩包。

## 无硬件测试

在仓库根目录运行 Python 测试；Linux 上的启动器测试使用模拟程序，不控制车辆：

```bash
python3 -m unittest discover -s tests -p 'test_*.py' -v
cmake --build build --target trackview_regression race_regression serial_regression gui_stall_regression race_ui_regression race_trial_regression stop_candidate_review_regression race_crosswalk_regression race_obstacle_regression race_async_navigation_regression race_bypass_regression -j4
ctest --test-dir build --output-on-failure
```

交叉编译产物需在 ARM64 主机或正确配置的 QEMU/sysroot 下执行。上述命令不能直接在 Windows 运行 ARM64 二进制。

以下回归需要外部保存的样本，不把私有图片自动纳入仓库：

```bash
./build/tests/race_stop_regression /path/to/stop-preview-samples /path/to/output
./build/tests/race_appearance_regression /path/to/fork-trace
./build/tests/stop_candidate_review_regression /path/to/probe.jsonl /path/to/saved-run
./build/tests/race_crosswalk_regression /path/to/crosswalk-saved-run
```

`race_stop_regression` 需要 `sample-1`、`sample-2`、`sample-4`、`sample-5` 等原测试目录和 JSON；`race_appearance_regression` 需要 `trace.json` 及其引用图像；最后一项需要保存运行中的 `ai-sample-N/camera.png`。不带参数的 `stop_candidate_review_regression` 只运行自包含图形回归。
