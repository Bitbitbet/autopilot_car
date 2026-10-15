# trackview v6：最高可调 10 m/s，保留原版初始速度

本版按用户澄清的要求，将最高可调值保留为 10 m/s，恢复原版初始速度：启动巡航 limit=0.25、弯道目标 turn=0.15、降级目标 qualityLimit=0.12。C++ 默认值与示例 JSON 保持一致。

分支：`trackview/v6-speed-ceilings-10-20261015`。基于已经保存的 v6 / 1.5 m/s 分支独立提交，不合并或修改组员的 master。

## 改动

默认 C++ 配置和 JSON 同步设置：

```json
{
  "cruiseSpeed": 0.25,
  "maxSpeed": 10.0,
  "curveSpeed": 0.15,
  "degradedSpeed": 0.12
}
```

这只是四个相关字段的展示；已有 JSON 中的其他字段保留。C++ 配置校验允许 maxSpeed 最大为 10。

弯道目标、降级目标和巡航 limit 都可以通过原按键调到 10；实际速度目标仍不超过 limit。默认弯道目标和降级目标仍参与减速；用户若把 turn、qualityLimit 都设为 10，则在可信路径有效时跟随 limit。原有加速过程保留，速度指令不会瞬间升到 limit。

路径无效、跨度不足、丢线、手动停车、通信超时等停车条件保留；将速度参数提高不等于让车辆在没有可信路径时继续行驶。图像识别、转向参数、加减速率和日志实现均未修改。

10 m/s 是软件参数上限和目标值，不是实际测得的车速，也不保证车辆能达到或稳定循迹。没有刷写下位机、修改其限速或操作实车。

## 上传和运行

1. 上传新程序 `trackview_fixed_v6_max10_defaults` 到个人目录 `/root/track-trial-20261010/bin/`，保留旧程序。
2. 在车上现有 JSON 中设置 maxSpeed=10、curveSpeed=0.15、degradedSpeed=0.12、cruiseSpeed=0.25，其他个人调参保留。只换程序而保留旧 JSON 会继续使用旧值。
3. 退出旧程序后执行：

```bash
cd /root/track-trial-20261010/bin
chmod +x ./trackview_fixed_v6_max10_defaults
./trackview_fixed_v6_max10_defaults --config ./trackview.json
```

启动文字应包含 `default limit=0.25 m/s | max=10 | turn=0.15 quality=0.12`，具体参数以个人 JSON 为准。

点击图像窗口并使用英文输入法：

| 按键 | 作用 |
|---|---|
| a | 切换自动/手动模式，道路条件满足后才能进入自动 |
| + 或 = / - | 提高/降低 limit，每次 0.05 m/s |
| ] / [ | 提高/降低 turn，每次 0.05 m/s |
| . / , | 提高/降低 qualityLimit，每次 0.05 m/s |
| 空格 | 停车 |
| Esc | 停车退出 |
| w | 切换中心线权重 |

运行时调参不写回 JSON。要预设某个参数，可在程序退出后修改对应 JSON 字段再启动。

## 检查

新增离线回归覆盖原版初始速度、10 上限、JSON 加载、默认弯道及降级减速、显式设置 turn/qualityLimit=10 后的直道/弯道/有效单边路径的用户 limit、运行中降低参数和无效路径停车。原 v6 记录测试改用显式的历史速度配置，继续验证原来的可调减速与识别行为。

构建方法及环境依赖见历史说明 `trackview-v6-max1p5.md` 的构建部分。交叉编译使用已配置的 ARM64 车辆工具链和 sysroot；离线测试与真实车辆试跑是不同的验证。

历史三个速度值都设为 10 的版本保留在提交 `2a609cf` 及原本地文件 `max10/trackview_fixed_v6_max10` 中；当前版本恢复了原版初始目标。

2026-10-15 已交叉编译 `trackview_fixed_v6_max10_defaults`，并通过 QEMU 下的控制回归、串口故障/超时检查、显示阻塞检查和 120 帧离线回放。启动输出核对为 `max=10 | turn=0.15 quality=0.12`。本次没有改变识别、转向或停车算法，也未进行实车操作。
