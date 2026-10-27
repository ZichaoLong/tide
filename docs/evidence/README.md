# 历史验证与测量证据

[Core-5 有限完整调用性能](core5-weighted-performance-20261028.md)：ceae7bf；24配置22通过，CPU反馈两项严格浮点失败保留；CPU/mixed/resident、Python与standalone分列，独立profile及工作量，冷启动/SDK allocator指标仍有限制。

[Core-5 常驻样本副本与master复验](core5-resident-weighted-peers-20261028.md)：7db0b08；standalone两dtype各20训练更新及6次双卡共同更新，Python-owned672e948四项通过；CUDA三架构构建。原检查器失败保留；单样本节点切分未由此实现。

[Core-5 FP16公共全图/训练资格](core5-resident-half-training-20261028.md)：21ed86d；CPU32、Python-owned NPU52、standalone两dtype各21图/15配置/20更新、完整VJP与精确恢复；CUDA三架构构建，peer与性能另验收。

[Core-5 FP16四循环/KV Full资格](core5-resident-half-memory-20261028.md)：bcef2c4；CPU24、NPU两dtype四组合、CUDA三架构构建；完整half图另验收。

[Core-5 FP16原始状态/投影/卷积资格](core5-resident-half-primitives-20261028.md)：2e4d445；CPU32、NPU两dtype六组合、CUDA三架构构建；明确FP32算术/伴随，完整half图另验收。

[Core-5 weighted公共训练资格](core5-resident-weighted-training-20261028.md)：343429d；CPU28、Python-owned NPU28、standalone五配置20更新、保留窗口/累计/精确恢复与两安装消费者，CUDA三架构构建；FP16/peer另验收。

[Core-5 weighted完整窗口VJP资格](core5-resident-weighted-window-20261028.md)：2622e60，NPU27配置/324根模式与36前向回归、CPU17项及CUDA三架构构建；公共retained训练另验收。
- [Core-5 消息祖先反向资格](core5-resident-weighted-route-20261028.md)：ad8e7b8；CPU12/NPU12、精确时间和独立边界根，CUDA三架构构建；完整窗口继续。

- [Core-5 常驻完整数值阶段 VJP](core5-resident-weighted-stage-20261028.md)：ebe5471；NPU24配置/342根模式/完整前向回归，CUDA三架构与安装loader；跨阶段训练继续。

- [Core-5 完整 weighted 常驻 CUDA 构建](core5-weighted-flow-cuda-20261028.md)：8eebcec；显式sm80/90/100、架构8项、DSO及安装loader通过；无GPU真机。

- [Core-5 完整 weighted 常驻推理](core5-resident-weighted-flow-20261028.md)：5efce32；CPU50、standalone NPU21图/15配置、Python-owned23项、安装消费者与精确续接。训练另验收。

- [Core-5 常驻批量head资格](core5-resident-head-20261027.md)：161ccea；CPU8/NPU完整共享VJP与实际版本，CUDA三架构构建。

- [Core-5 常驻发送原语资格](core5-resident-sending-20261027.md)：8e7298d；CPU34/NPU八配置完整VJP，CUDA三架构构建。

- [Core-5 常驻Alloc组件资格](core5-resident-allocation-20261027.md)：2814b87；CPU29项、NPU FP32完整VJP，CUDA三架构构建，未认证完整图。

[Core-5 CUDA链接构建与CPU24项资格](core5-cuda-eager-20261027.md)：670ead3；GPU真机待验。

- [Core-5 standalone master/schema2资格](core5-native-master-20261027.md)：670ead3；CPU24项，NPU FP16显式SGD24图/AdamW2图及设备→CPU精确恢复续接。

- [Core-5 eager FP16 NPU有限资格](core5-half-device-20261027.md)：01689c0；CPU12项、Python/native mixed与显式同步双卡、master恢复及归属profile。

- [Core-5 带权 FP16 CPU 资格](core5-half-20261027.md)：8c8824f；99项通过、2项GDN AdamW strict xfail单列。

- [Core-5 eager NPU 混合与两卡同步资格](core5-device-20261027.md)：470ce06；CPU200项、Python/native有限矩阵通过，standalone严格失败单列。

- [Core-5 带权执行位置 CPU 资格](core5-placement-20261027.md)：9234c13，508 项；typed Read/control/selection、packed copy VJP 与训练恢复。

- [Core-5 standalone 完整训练/checkpoint CPU 资格](core5-native-training-20261027.md)：a7106fc，20 项；每 dtype 24 图案例、三步训练与恢复后第四次更新。

- [Core-5 KV 基座投影复用 CPU 资格](core5-kv-sharing-20261027.md)：6dc2665，309 项；私有状态、完整使用者 VJP、训练与恢复。

- [Core-5 Aggregate 精确计算复用 CPU 资格](core5-sharing-20261027.md)：52dee1e，543 项；物理来源、所有使用者反向、训练和恢复。

- [Core-5 来源绑定与稳定一阶归约 CPU 资格](core5-source-20261027.md)：503f609，665 项；四种别名/宽度、独立公式、AdamW 与恢复。

- [Core-5 区域实际批量/VJP CPU 资格](core5-region-20261027.md)：5391017，536 项；完整候选、精确解反向、独立根、在线训练/续接。

- [Core-5 Read/Full 批量 VJP CPU 资格](core5-query-vjp-20261027.md)：67854b3，647 项；独立字段/行根、共享参数、AdamW 与恢复。

- [Core-5 Read/Full 数值批量 CPU 资格](core5-query-20261027.md)：7b8b469，377 项；实际 Mem/Full、完整快照与独立根。

最新：[core-5 Aggregate 批量计算/VJP](core5-aggregate-20261027.md) — `d90cfbc`，704 项相关 CPU 资格。

[core-5 State 批量 VJP](core5-state-vjp-20261027.md) — `7a017bf`，343 项相关 CPU 资格。

[core-5 精确联合记忆](core5-packed-20261027.md) — `26dad7d`，378 项相关 CPU 资格。

[core-5 完整评分/证据/key](core5-scoring-20261027.md) — `cacb645`，593 项带权支线 CPU 资格。

[core-5 带权循环记忆](core5-recurrent-20261027.md) — `442e8b2`，544 项定向 CPU 资格。

[core-5 带权 KV 首组合](core5-weighted-kv-20261027.md) — `6fb467c`，129 项定向 CPU 资格。

[core-5 结构化表示](core5-structured-20261027.md) — `714d5f9`，212 项定向 CPU 资格。

[core-5 一般 SettleGraph](core5-settle-20261027.md) — `efb9d62`，247 项定向 CPU 资格；不认证尚未完成的带权局部函数或新设备路径。

本目录保留已审阅的实现提交、验证范围、结果、失败和性能限制。
它是历史证据，不是待执行清单；当前任务只由 [STATUS](../STATUS.md) 和
[ROADMAP](../ROADMAP.md) 决定。

2026-10-12 用户确认旧实验默认不再复核，原始构建/运行产物可退役。
报告中的绝对路径、`TASK/...` 和 `artifacts/...` 仅标识当时的来源，
**不承诺文件仍在线可用**。必要记录与退役说明见 [产物保留规则](../artifact-retention.md)。
历史报告保留原文和原失败；原始文件移除不等于原验证未发生，也不构成新源码通过证据。

## 主要结论入口

- 基座：[最终资格](foundation-final.md)、[跨图资格](cross-family-qualification.md)、[公共库](library-foundation.md)。
- eager 加速器：[范围与环境](accelerators-20260928.md)。
- 最新批量实现：[初始 CPU/CUDA 资格](batched-vjp-cuda-local-20261010.md)、[Add 后续](add-batch-followup-20261010.md)。
- NPU：[公共模块](npu-module-batching-20261011.md)、[常驻完整组件](resident-composite-20261011.md)、[stream/精度](precision-and-stream-gates-20261011.md)。
- 性能：[旧全尺寸结论](selection-terminal-20261010.md)、[六组件](npu-batched-components-20261011.md)、[最新完整选型](native-selection-20261011.md)。
- 不能忽略的限制：[原始近似平票路由失败](original-add-route-witness-20261004.md)、最新完整选型中的两项容量拒绝。
- 接续与清理：[固定教材/干净 clone 起步验证及历史产物退役](artifact-retirement-20261012.md)。

查看某个结果时先确认源码、环境、模块、图类别、精度、输入和观察量范围。
跨硬件浮点结果、共享服务器争用与历史计时不能承诺原样重放。
默认基于当前源码创建新的独立验证/测量，不重启旧 job 或补写旧结果。

- [Core-5 熵分配 CPU 定向资格](core5-entropy-20261027.md)：a05d3ff，226 项通过。

- [Core-5 十种发送函数 CPU 定向资格](core5-sending-20261027.md)：4a08588，320 项通过。

- [常驻完整带权区域组合](core5-resident-region-20261027.md)：ce63aef，CPU33/NPU14配置完整VJP与保存版本，CUDA三架构构建。

- [常驻完整带权Aggregate](core5-resident-aggregate-20261027.md)：92751c0，CPU15/NPU物理原子及完整VJP/保存版本，CUDA三架构构建。

- [常驻来源低秩参数链](core5-resident-source-20261027.md)：6f709f8，CPU18/NPU完整因子链与保存版本，CUDA三架构构建。

[带权常驻参数归约与局部训练](core5-resident-parameters-20261027.md)：2e335e7，CPU3/NPU八步更新与新owner恢复、CUDA三架构构建。

[常驻原始状态与缓存](core5-resident-raw-20261027.md)：cb68d91，CPU14/五schema三dtype精确codec/NPU完整字段VJP及CUDA三架构构建。

[常驻固定时间短卷积资格](core5-resident-convolution-20261027.md)：f89bdfe，CPU/NPU完整组件及CUDA三架构构建。

[常驻共享投影资格](core5-resident-projection-20261027.md)：2af401d，CPU/NPU完整组件及CUDA三架构构建。

[常驻 KV 接收更新资格](core5-resident-kv-update-20261027.md)：681a7e9，CPU/NPU完整组件及CUDA三架构构建。

[常驻带权摘要与 Attention 归约资格](core5-resident-pool-20261027.md)：afe7b5e，CPU/NPU完整组件及CUDA三架构构建。

[常驻 KV 原始时间窗口与 sum 记忆资格](core5-resident-kv-sum-20261027.md)：fea6fa7，CPU/NPU完整组件及CUDA三架构构建。

[常驻相对旋转与极大逻辑年龄数值转换资格](core5-resident-rotary-20261027.md)：1eb4b72，CPU/NPU完整组件及CUDA三架构构建。

[常驻 KV 内容检索完整 Mem 资格](core5-resident-kv-query-20261027.md)：8b1be80，CPU/NPU完整组件及CUDA三架构构建。

[常驻 KV 完整内容与公共残差资格](core5-resident-kv-full-20261027.md)：386970a，CPU/NPU完整组件及CUDA三架构构建。

[常驻完整 Describe 资格](core5-resident-describe-20261027.md)：cd576d2，CPU/NPU完整组件及CUDA三架构构建。

[常驻四循环 Upd/Mem/Full 固定源码资格](core5-resident-recurrent-20261028.md)：6e2ecb7，CPU/NPU完整组件及CUDA三架构构建。

[已安装带权配置检查接口](core5-weighted-binding-20261028.md)：ce21a83，CPU Python/native 45 项与 C++ 两种 dtype。

[常驻半精度KV读取](core5-resident-half-kv-read-20261028.md)：db85db6，CPU27、NPU sum/qry两dtype及CUDA三架构构建。

[常驻半精度KV更新](core5-resident-half-kv-update-20261028.md)：69bf4f0，CPU26、NPU更新/raw两dtype及CUDA三架构构建。
