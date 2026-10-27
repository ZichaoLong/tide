# Core-5 KV 基座投影复用 CPU 资格 — 2026-10-27

固定实现 `6dc2665c4fbd9053173bbd9d26c7e1b03b62abae`，固定上游 `0770593`。
**309 passed / 0 failed / 0 skipped；CPU FP64/FP32、aarch64。**
干净源码、workload exit 0、服务 inactive/success 已核对；[命令/环境/哈希](core5-kv-sharing-20261027.json)。

[数学及公共能力对应](../weighted-kv-sharing.md)限定在当前合法 packed 块内，
相同正接收 Content、实际 base Parameter 与 epsilon 才共同计算 `N(u)` 和 `base@N(u)`。
每个节点继续使用自己的低秩适配器、summary 和 KV；准备张量作为显式 State VJP 操作数，
不进入数学 Content 或持久状态。独立同值输入/参数、零接收与无能力程序不虚报复用。

Python、独立 native 的 streaming/frontier/greedy 对照标量参考，检查完整快照、
各 keys/values/summaries/weights 独立根、None/连接零、全部使用者反向及保留图切点。
推理、混合 prepared/unprepared/零接收行、过期原始条目、超过 2^54 的 int64 时间、
原位修改版本与一阶边界均覆盖。相关 Aggregate 三步 AdamW 和含 pending checkpoint
换 owner 恢复经过共享基座路径；通用 State VJP、五种记忆和来源/查询训练回归同批通过。

独立 standalone 手算检查 RMS 与基座/私有适配器的 literal Jacobian，两个使用者系数
1 与 3 的共享贡献合计 4；私有适配器仍保留未使用梯度 None。此局部 C++ 证据不替代
下一阶段的 standalone 完整图训练/checkpoint 矩阵。准备调用减少不意味着候选、Full 或消息数减少。

开发 tree `ac222fb4b72840e69884a795be9009064e2055e3` 已构建并通过同范围 309 项。
开发前定向续接测试曾因错误计入中间状态 objective 失败 2 项，改为完整末态和拼接输出后通过，
未因此修改实现；原失败记录保留。本资格不混入后续完整原生 checkpoint 草稿。
复现：检出固定实现，按 [development](../development.md) 构建，运行 JSON 测试列表，
`--dtype both`、新 basetemp 和当前 TIDE_BUILD_DIR；不依赖旧运行目录。
Python 3.12.13、Torch 2.12.0+cpu、GCC 10.3.1；ATen/BLAS 单线程、两个 build workers。
unit tide-core5-kv-sharing-qualify-20261027：background.slice、Nice=10、20 分钟上限；
原始输出 artifacts/jobs/core5-kv-sharing-qualify，build manifest 已独立保存。
本报告不认证 mixed/resident、CUDA/NPU 真机或性能。
