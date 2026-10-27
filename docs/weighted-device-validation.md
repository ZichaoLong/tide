# 带权消息的 eager 设备验证

数学与完整类型对应从 [weighted-message](weighted-message.md)、[sending](weighted-sending.md)、
[recurrent](weighted-recurrent.md) 和 [placement](weighted-placement.md) 进入。
本检查使用独立 CPU 标量调度，与纯 PyTorch/LibTorch 在线候选分别比较。
参考的消息、路由和候选集合不提供给候选执行器。实际完成状态及固定证据由
[STATUS](STATUS.md) 与 [证据索引](evidence/README.md) 给出；源码入口不是设备资格。

```sh
python scripts/check_weighted_device.py --device npu:0 --implementation python \
  --placement mixed-c --output-dir artifacts/weighted-python-001
python scripts/check_weighted_device.py --device npu:0 --implementation native \
  --native-library build/npu-python --placement mixed-c --output-dir artifacts/weighted-native-001
```

CUDA 使用对应 device 与匹配构建。显式缺失后端失败，不跳过、不改用 CPU。
每个输出目录必须新建。`--case` 可重复选择开发子集；省略时执行 19 个有限代表案例，
并非所有开关的笛卡尔积。FP32 atol=1e-6、rtol=1e-5；整数、消息存在性、平票次序、
参数别名和 None/连接零保持严格。图与配置不因设备修改。

`--dtype float16 --scoring-dtype float32` 显式选择半精度payload与FP32评分；
未声明评分精度会在初始化设备前拒绝。`--optimizer adamw|sgd` 默认AdamW，
half使用FP32 master和静态`--loss-scale`（默认16）；FP32不使用loss scaling。
入口记录请求和实际精度、优化器、学习率与scale。不会为GDN自动改用SGD。
[半精度算术](weighted-precision.md)的atol=1e-3/rtol=2e-2用于所有浮点观察量，
包括从half操作数产生的FP32描述量、未缩放VJP和master slots；离散与连接性严格。
checkpoint恢复的raw state与optimizer/master逐值精确，第四步跨设备更新单独比较。
CPU与目标都采用同一量化初态；CPU FP32是另一个精度基准，不能冒充half数学路径。
`test_weighted_device_precision.py`先检查此入口的完整记录/独立根/四步与配置拒绝；
实际NPU/CUDA及默认AdamW的已知严格失败分别保留在证据中。

| 范围 | 入口及检查 |
| --- | --- |
| CPU 创建的同一参数/输入 | [cases](../scripts/weighted_device_cases.py)，记录 fixture；FC/HB、反馈重复边 PDG、非对齐 DAG、稀疏独立端口、五记忆及异构组合 |
| 完整 forward 与一阶训练 | [checks](../scripts/weighted_device_checks.py)，独立参考与 streaming/在线 prefill、实际 tensor owner、完整返回/状态/消息/control、独立 descriptor/output/raw State 根 |
| 续接与 optimizer | 保留计算图切点、三步 AdamW、含 pending 精确 checkpoint、换 owner 后第四步更新、CPU handoff；按实际候选结果直接保存与解码 |
| 设备敏感局部反例 | [primitives](../scripts/weighted_device_primitives.py)：大于 2^54 的 ID 平票、fractional total、全非正 hard fallback 的 literal VJP，packed copy 的 int64/别名/None/连接零 |
| 独立 C++ | [training executable](../cpp/test/weighted_graph_training.cpp)，CPU Streaming reference 与目标 Greedy、24 个五记忆/异构 × 四图案例、三步及恢复第四步、完整 TIDEGCK1；`--placement` 与可重复 `--case TOPOLOGY/FAMILY` |

标量 hard/total-zero 在加速器上使用单行 device sorted batch；CPU 标量排序仍是独立锚。
排序依次使用精确 int64 节点编号与稳定 score 排序，fractional fill 和指定 VJP 在目标设备计算。
静态 IDs 可以在主机准备，实际 score 不导出到 Python/std::sort 排序。
窗口就绪、记录分配与推进仍由 host 调度；本检查不认证 resident。

`--peer-devices npu:0 npu:1` 显式请求多个逻辑设备，首项须与 `--device` 相同。
按 region 循环放置节点，使区域参数共享保持同一叶、私有 state 分别归属各节点；
共享参数与 owner 冲突沿用公共 `place_payloads` 的提前拒绝。验证真实跨设备消息，
所有参数/输入/状态的反向汇总、optimizer 边界、同 owner/交换 owner/CPU checkpoint 恢复。
物理卡由外部 launcher 动态分配；不把本机暂时空闲的物理 ID 固化进代码。
这是一进程内的多设备模型切分，不声称完成 HCCL 数据并行或 resident 多卡。

当前 Torch-NPU 2.12 / CANN 9.1 的完整两卡 Python 检查在默认异步 dispatch 下
出现 `rtStreamWaitEvent` invalid event；显式 `ASCEND_LAUNCH_BLOCKING=1` 的开发检查
通过。两种配置分别保留结果，同步通过不解除异步限制。入口记录 launch-blocking
与 task-queue 环境，不自动修改这些设置。普通两卡复制与两个根的小探针不足以
认证完整图训练；固定提交资格、native 与 standalone 结果分别记录。

`profile_weighted_device.py` 在独立进程对一个已检查组合执行一次 warmup 和一次完整训练 profile。
包括输入准备/上传、在线调度、packing、forward/loss、backward/有限检查、AdamW 与同步；
构造和 warmup 单列。记录真实 kernel 类型/名称、显式传输/主机标量、candidate/Full/消息工作量，
保留原始 trace 并拒绝观测到的意外 host fallback。该小 profile 不作吞吐排名，也不证明所有
算子在任何形状都不会回退。NPU 的设备 AiCPU 算子与主机 CPU fallback 分开记录。
