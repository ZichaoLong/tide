# Resident 块调度与 KV/区域/消息组合资格

固定数学为 core-5 `0770593838a37c790a568226930567ed60f99b83`。
本轮通过 **17 项 CPU 元数据检查和 3 个 standalone NPU FP32 组件检查**。
它们建立设备在线块调度及 Read→区域选择→Full→消息列的独立数值边界；完整图 owner 尚未接入。
[机器记录](resident-blocks-20261112.json)保留实际源码、构建、日志摘要/hash、配置与逐项结果，
[块契约](../resident-block-execution.md)和[消息契约](../resident-block-kv-messages.md)定义适用域。

| 内容 | 实现/资格身份 | 实际结论 |
| --- | --- | --- |
| 独立块调度、静态 Settle lowering、Read-selection | 实现 `a50420bd` | 元数据与数值组成已实现 |
| 原 standalone 构建 | 固定 `d69025e4`，含 a504 | `tide-block-dispatch-check` 链接失败；记录保持 failed |
| 构建修复 | `a4b71bfc` | check target 显式 `-fPIC` |
| 新 SDK/CANN 构建、CPU 与两组件真机 | 固定 `a95b8987` | 构建/loader、CPU 17 项、dispatch 和 Read-selection 分别 passed |
| 独立 proposal Full 与消息列/VJP | 实现 `bbea0c09` | 五类独立根、参数角色归约、c/G 与 cG 分离 |
| 消息组件新 CANN 构建/真机 | 固定 `7d5e05c3` | 新组件构建/loader 和消息组件 passed；core 复用 a95 |

原构建错误为 `R_AARCH64_ADR_PREL_PG_HI21 ... recompile with -fPIC`，发生在检查程序链接，
不属于数学或设备数值失败。后继在新构建目录成功；没有把原任务改写为通过。
原 `gef-rewrite-shared-qual` 外层仍 failed，原因包含其他区域测试的设备绑定和 Scalar-floor 诊断；
其两个 resident 子检查通过不能覆盖这些失败。消息专用 `gef-rewrite-messages-7d5` 已 passed，
结束时 MainPID=0、ControlGroup 为空。

## 实际执行与构建身份

三项设备检查均使用 standalone LibTorch 2.12.0、aarch64 ABI1、CANN 9.2.0、
AscendC `Ascend910_9392`，host ATen/interop 各 1 线程，设备为 `npu:0`。
前两组件映射共享物理 NPU1，消息组件映射共享物理 NPU9；有本次对话明确授权和有界准入记录。
这些是组件正确性检查，不是独占性能测量。

新消息组件 source/hash 为 `7d5e05c3` / `84a609f6798281a02bcea9f661a178ef4a1736cdfa4989bb06baad875b8b0b1b`；
实际 core 仍为 `a95b8987` / C++ hash `f4596708e67ba24ba1d844452cb551593e8a46db25e641f3ce8c02304dc0130b`。
构建入口核对当前 C++ hash 和复用 core 各 binary hash；不能称为重新构建了 7d5 core。
SDK和前两组件实际 build jobs=160；新消息组件 jobs=320、CPU0–319、256GiB、1800 秒。
构建/loader成功与真机结果在机器记录中分列。未运行 CUDA 构建或 GPU 真机。

## 独立检查与数学对应

CPU 17 项来自固定 a95 的 `batched-cpu.xml`：11 项 dispatcher、6 项 selection/lowering，
是该作业 150 passed 的子集，不另加成 167。
[dispatcher 检查](../../tests/test_resident_block_dispatch.py)把同一设备元数据语义源编译到 CPU，
与独立递归 DAG 求值比较，覆盖随机重编号、同秩、重复边、B/L 坐标、正延时反馈的有限展开、
缺席与零消息、control-only 依赖、巨大 int64 时钟、窗口复用、迟到/重复完成、容量与数值拒绝不提交。
[selection 检查](../../tests/test_resident_block_selection.py)核对完整区域、独立根连接及逐节点 Gate，
尤其拒绝用整个区域的 any(active) 广播成每个节点的 Full 支持。这些 CPU 检查不冒充设备 KV 数值。

| NPU 组件 | 实际工作 | 独立锚点与覆盖 |
| --- | --- | --- |
| `resident-block` | 一次 DeviceProgram 提交内由设备 choose→vendor BMM→数值 presence→commit | 独立 C++ 递归 DAG；输入值改变真实后继，主机不预先提供已选轨迹；int64、空窗、额度/数值拒绝 |
| `resident-block-selection` | 整块 KV qry→Describe→完整区域 SelStep，control-root VJP | 字面带权 Attention、网络公式和两候选 quad 闭式解；完整 memory/描述/control、active/history、raw/输入/参数梯度、None/连接零 |
| `resident-block-kv-messages` | 上述选择→独立 proposal Full→结构化消息及参数角色归约 | 独立 proposal Attention/位置/残差/SwiGLU；五类根各自、全根、无连接共七模式，另验完全未定义根、共享参数、十项拒绝 |

数值材料为 R4/N3、两个同秩独立区域、两个共享参数 bank；d=2、summary=2、region hidden=3、
KV qry、cat/ind、regional_total/context/cat/quad、k=1、gamma=2、temperature=0.8。
原始时间从 `2**54+17` 开始、帧间隔 3；raw 浮点和 int64 元数据分开。
Read cache=2；消息检查另外供应 cache=3 的实际 proposal，包含当前时刻新条目和不同内容。
R4 可以解释为 B2×L2 的布局，但逐帧快照独立供应，**没有执行跨时间 Upd/scan 或验证跨序列状态隔离**。
完整 fixture 和可重生参数见[Read 公式/材料](../../tools/device_online/block_selection_formula.cpp)、
[Read fixture](../../tools/device_online/block_selection_fixture.cpp)、
[Full 公式](../../tools/device_online/block_kv_message_formula.cpp)与
[消息 fixture](../../tools/device_online/block_kv_message_fixture.cpp)。

消息返回 `(c,G)`，其中 G 未乘发送强度；显式请求才额外返回 cG。
五类根为 candidate control、selected Full G、message c、message G 和 cG。
Read/proposal raw 梯度独立返回，共同 content 梯度合并；direct control 根不被 Full active 错误遮罩。
Read.query/Full.query、Full.gate/up 真实共享，Full.out 数值相等却保持独立身份；
trainable 子集检查汇总所选物理 owner 的全部使用角色。
还检查不同 Full owner、默认沿用 Read owner、默认不算投影及拒绝投影根、半定义根、坏名称/形状、
forward/VJP/参数归约预算拒绝，以及存在且 G=0 的消息。
独立 reverse 在原始 raw/content/时钟/参数缓冲被覆盖后读取真实 saved tapes，未调用参考补写。

Read-selection 浮点阈值为 rtol=3e-5、atol=3e-6；消息组合为 rtol=4e-5、atol=4e-6。
支持、连接位和 int64 时间独立精确比较。Read-selection 日志的三次小 replay 共 0.0102373 秒
只是该检查的诊断字段，不能转换为模型吞吐、TTFT 或完整训练成本。

## 范围与继续接口

本轮证明组件能在设备实际运行并组成数值块，没有宣称整段 resident 图执行已经完成。
静态 Settle lowerer 当前接受 ports/aligned 域；有限潜在 DAG 原语尚未接一般动态 PDG 槽位生成。
新的完整 owner 仍需实际 Aggregate/Upd/Next、choose/commit 循环中的数值接入、逐物理边/端口路由、
完整中间观察根、状态/tape 生命周期、optimizer、精确 checkpoint 及普通异步多卡验收。
其他记忆/异构配置和 FP16 也不能继承这一新组合的 FP32 通过。

raw 快照仍逐帧物化，尚未使用共享 KV 前缀 bank；Full 虽保留实际 active 语义，
当前 vendor 计算仍执行固定容量行，padding 的算术和存储不能因发送稀疏就算作已经省去。
元数据闭包为单 worker 扫描。没有新增完整 FC16、LH、大尺度吞吐或训练/恢复结果。
[旧 FC16 的 EE1019/task-buffer/507002](resident-shared-bank-20261112.md)保持未解除；
旧执行器的小图 optimizer/checkpoint 资格也不自动成为新块执行器资格。
当前待办与停止/重启条件统一见 [ROADMAP](../ROADMAP.md)。

## 从干净源码重现

固定 7d5 源码或明确记录的后继，按[开发入口](../development.md)准备匹配 standalone SDK core。
长构建和真机检查使用独立有界 background.slice 服务、当次设备授权与准入；以下为内部命令：

```sh
python -m pytest -q tests/test_resident_block_dispatch.py tests/test_resident_block_selection.py
python scripts/build_device_control.py --core-build SDK_CORE --build-dir NEW_DEVICE_BUILD \
  --ascendc-soc Ascend910_9392 --jobs all \
  --checks resident-block resident-block-selection resident-block-kv-messages
python scripts/verify_device_control.py --build-dir NEW_DEVICE_BUILD --output-dir NEW_RESULTS \
  --device npu:0 --checks resident-block resident-block-selection resident-block-kv-messages
```

实际历史 gate 的两个 source 身份分别保留；在新提交一起运行三项会生成新的资格，不能回写原记录。
材料由 tracked fixtures 生成，复现不依赖旧 /dev/shm 构建或已退役 raw 目录。
