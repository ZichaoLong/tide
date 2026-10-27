# Core-5 带权执行位置 CPU 资格 — 2026-10-27

固定实现 `9234c13ed7bba0d258503de6db80e4805201e409`，固定上游 `0770593`。
**508 passed / 0 failed / 0 skipped；CPU FP64/FP32、aarch64。**
干净源码、workload exit 0、服务 inactive/success 已核对；[命令/环境/哈希](core5-placement-20261027.json)。

[契约](../weighted-placement.md)将实际 Read^0 / Read^- / Read^+ 的临时参数、内容和状态视图
移至显式 Read 精度/位置，区域 control/query 与 selection 可分别配置。参数身份、别名、
私有原始状态和 checkpoint 不变；int64 时间/计数/KV 与卷积戳保持精确。
同形行按最多 8 MiB 分组 stack/copy/unbind；独立行 VJP 保留 None/连接零、原参数版本和共享贡献。
发送正支持在 cast 后若变为零明确拒绝，不隐式删除消息。

Python / independent native 与独立标量调度对照五种记忆、三种快照、十种发送，
检查完整记录、独立 descriptor/输入/参数根、保留图切点、异构 Attention 与共享参数、
三步 AdamW、含 pending checkpoint 与第四步恢复。CPU FP32 payload / FP64 scoring 验证实际 cast；
literal Read^0 公式、超过 2^54 metadata、组容量与 standalone 解析 copy VJP 是局部独立锚。
同批回归包括既有 placement、区域/查询 batched VJP 与 standalone 完整图训练/checkpoint。

开发第二轮 tree `259db933bc5d1bc55c62fb52e861d39d5220bb45` 同范围 508 项通过。
第一轮 tree `2cb09da838d5e1335e691dc2fa83a1a31cad4150` 在测试缺少 ATen/core/grad_mode.h 处编译失败；
原 job 保持 failed/exit 1。补头文件后另建输出验证，没有改写旧失败。早期短公式测试的根清单和
connected-zero 项错误已修正，并由完整套件覆盖。

本资格仅验证 CPU 位置和评分转换，不认证 NPU/CUDA 跨设备、resident、跨卡或性能。
后续设备验证脚本和标量硬发送的设备排序改动不属于本固定提交。
复现：检出固定实现，按 [development](../development.md) 构建，运行 JSON 的 pytest 列表，
`--dtype both`、新 basetemp 与 TIDE_BUILD_DIR。Python3.12.13、Torch2.12.0+cpu、GCC10.3.1，
ATen/BLAS 单线程、两个 build workers；unit tide-core5-weighted-placement-qualify-20261027，
background.slice、Nice10、1200秒上限。原始输出 artifacts/jobs/core5-weighted-placement-qualify。
