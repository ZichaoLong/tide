# 多payload owner的张量窗口：实现与CPU资格

生产实现固定 `796baafa3a0d580e6722cd28814bf23dafe60e8b`，SDK peer CLI后继固定
`efc2776b7262dc82ada1508e396cb686a296fb3f`。数学仍为固定core-5 `0770593`。
[机器记录](column-owners-20261111.json)包含原构建manifest与字节核验、固定CPU结果、准入超时和有限成本；
[实际审阅清单](column-owners-20261111-reviews.json)包含11个文件，和前份清单有重叠，不是全仓已审。

## 实现

Python与独立C++列调度按节点解析Read/payload位置、按区域参数owner解析完整control/selection位置。
真实消息在Aggregate之前整列复制到目标owner，同一消息到相同owner复用副本；重复边仍各参与一次。
完整区域描述汇集后才求解，归一化、活动支持、发送强度不按设备拆分。控制返回各节点owner，
raw KV/循环状态保持私有。输出保留各端口owner，公共readout选择归约设备，任务仍逐端口计算loss后汇总。

参数仍由原canonical owner和optimizer管理。`place_payloads`既有“共享节点参数共置”约束保留；
没有为跨卡静默克隆权重或拆分optimizer。KV区域base共享使同一区域成员共置；
独立节点源参数的SSD用例则覆盖一个区域跨owner的结构。普通copy不宣称P2P、collective或resident推进。
native能力版本升至2，旧库明确要求重建，不把旧二进制当成新多owner列实现。

## 实际验证

| 项目 | 结果与边界 |
| --- | --- |
| 冻结开发源码 | Python-owned NPU与独立SDK构建成功；CPU94 passed/9原数值xfail |
| 固定796baafa CPU | 243 passed/13原数值xfail；包含上述范围，不相加 |
| 固定efc2776b SDK CPU | 完整C++源码/binary hash核验后，五记忆/位置/独立scoring、streaming/raw/VJP/两SGD窗通过 |
| 双NPU准入 | 等待120秒后exit3，未进入模型计算 |
| CPU+单NPU补验准入 | 等待60秒后exit3，未进入模型计算；观察到短暂空闲卡时锁仍由其他任务持有 |
| 新多owner NPU数值/训练/恢复 | **未验证**；不能从编译、CPU退化或以前单payload资格推断 |

`test_column_payload_owners`用独立标量调度比较同秩区域、重复边、真实消息/控制/raw状态以及输入/参数VJP，
两个AdamW窗后换owner加载checkpoint，再比较下一窗参数、梯度、optimizer槽、完整raw状态和输出。
readout另与逐端口字面公式比较。CPU运行将两个owner都放CPU，只验证逻辑与原行为，不验证设备copy。
SDK peer参数通过现有runtime resolver解析并初始化，开发/固定CPU编译运行与实际peer设备运行分别记录。

没有使用共享NPU授权、没有绕过设备锁、没有把准入超时描述为模型数值/通信失败。
所有作业已终态，原失败记录保持；无需继续排队才能阅读或复现本报告。

## 默认CPU有限回归

受owner分派影响的默认窗口各做一次FC16完整训练流程，原下游0439efd、FP32、B8/L128/d128、KV qry、
H2/FFN512、原head/loss/AdamW；一窗warmup后测一窗。CPU1线程，实际affinity80–95。
本项目重构建在计时结束后开始；同时只有NPU准入等待，无NPU模型计算。外部共享主机争用未排除。

| 实现 | 完整训练秒 | 输入位置/秒 | 候选/Full/边消息 |
| --- | ---: | ---: | --- |
| PyTorch | 1.234671 | 829.37 | 16384/8344/25060 |
| Python调用native | 1.225463 | 835.60 | 16384/8344/25060 |

分子1024。完整成本含准备、调度、packing、forward/loss、backward、检查、optimizer与同步。
与前一提交PyTorch 1.244159秒的有限观察相近，没有观测明显退化；不作统计加速结论。
这里不新增完整尺度独立数值、精确恢复、NPU成本或TTFT资格，原严格失败保持。

## 干净源码复现及待验库存

按[开发入口](../development.md)构建对应backend，设置正确native库后运行CPU门禁。
实际设备分配在独立background.slice服务内完成，再运行：

```sh
TIDE_COLUMN_PEERS=npu:0,npu:1 python -m pytest -q tests/test_column_payload_owners.py --dtype float32
python examples/consumer_core5/tensor_window.py --device npu:0 --peer-device npu:1 --dtype float32 --client python --placement mixed-a --no-deterministic-algorithms --output-dir NEW
TIDE_COLUMN_PEER_DEVICE=npu:1 tidegraph-column-check --device=npu:0 --dtype=float32
```

独立小图库存为KV/SSD×Python/native四项；SSD区域跨owner，KV保留base共享共置。
安装消费者库存为KV/SSD×Python/native×mixed-a/c八项；native增加对应`--native-library BUILD`。
SDK库存为五记忆×三放置15项。这些命令和库存已提供，**均未取得本阶段的双NPU结果**。
CPU+NPU补验可显式设置`TIDE_COLUMN_PEERS=cpu,npu:0`，但不能替代双NPU资格。
后续先完成上述值/VJP/恢复及异步完成检查，再选有限跨卡性能；不先扩大规模。
新默认单NPU回归、resident列路径、FP16及旧资源边界仍见ROADMAP。
