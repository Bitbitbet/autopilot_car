# trackview v6：速度上限 1.5 m/s

本分支保存用户已完成实车测试的 v6，并仅将其可配置速度上限调整为 1.5 m/s。不是已完成 1.5 m/s 实车验证的声明。

## 分支与改动

- 分支：`trackview/v6-max-speed-1p5-20261015`。
- v6 基于原项目提交 `b99bc26`；本分支的第一个提交 `0a97e8c` 保存此次调整前完整 v6 源码及已有测试。
- 组员近期对相机等模块的重构保留在原分支；本分支独立保存 v6，未将不同版本混合，也未合并到 master。
- `src/tool/trackview_config.hpp`：默认 `maxSpeed` 从 0.5 改为 1.5，同时将配置校验允许的最高值改为 1.5。
- `config/trackview.json`：只将 `maxSpeed` 从 0.5 改为 1.5。
- 新增相应回归检查，验证 1.5 可加载、速度指令不超过上限、降低 limit 立即生效、弯道/降级目标及丢线停车保持原行为。

启动巡航上限仍为 0.25 m/s；弯道目标仍为 0.15，降级目标仍为 0.12。加减速率、转向、图像处理、识别、串口保护、按键和日志逻辑均沿用 v6。

## 使用现有配置

如果车上已有自己调好的 `trackview.json`，只修改其中一项，保留其他参数：

```json
"maxSpeed": 1.5
```

旧 JSON 的 `maxSpeed: 0.5` 会覆盖新版程序默认值；仅更换可执行文件不能解除该配置限制。仓库中的完整示例配置也已更新，但不必用它覆盖自己的全部调参。

点击 track_view 图像窗口，使用英文输入法：`+` 或 `=` 将 limit 每次增加 0.05，`-` 降低；最高 1.5 m/s。`a` 切换自动模式，空格停车，Esc 停车退出。`[`/`]` 调整弯道目标，`,`/`.` 调整降级目标，`w` 切换中心线权重；规则与旧 v6 相同。

1.5 是允许设置的上限，不是启动速度或保证达到的实际车速；弯道与识别质量限制仍参与控制。界面 speed 为速度指令，不是编码器实测车速。运行中按键调参不写回 JSON。

## 构建

源码为 C++，使用项目原有 CMake 构建。需提供目标平台匹配的 OpenCV、libserial、libevdev、libudev 及 C++20 工具链。示例：

```bash
cmake -S . -B build-v6 -DTRACKVIEW_BUILD_TESTS=ON
cmake --build build-v6 --target trackview trackview_regression serial_regression gui_stall_regression -j4
ctest --test-dir build-v6 --output-on-failure
```

交叉编译时需额外指定已有的目标工具链和车辆 sysroot；不要用 Windows 可执行文件代替车载 ARM64 Linux 程序。车上运行仍使用原有相机标定资源，工作目录与旧 v6 一致。

## 本次验证

已使用匹配车辆的 ARM64 工具链交叉编译，并在 QEMU 下通过控制算法、串口故障与超时、显示阻塞及 120 帧离线视频回放检查。新增检查确认 1.5 m/s 可作为配置上限和直道速度指令，超限请求被限制，原有弯道/降级目标及丢线停车有效。

没有连接或操作小车；真实车辆在 1.5 m/s 下的循迹表现仍需实车测试。
