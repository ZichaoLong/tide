# Core-5 standalone 训练与完整 checkpoint CPU 资格 — 2026-10-27

固定实现 `a7106fc0dc90d1eefdd77662d9df392b7f2b36cc`，固定上游 `0770593`。
**20 passed / 0 failed / 0 skipped；CPU FP64/FP32、aarch64。**
干净源码、服务 inactive/success、exit 0 已核对；[命令/环境/哈希](core5-native-training-20261027.json)。

[完整原生格式与数学对应](../native-graph-checkpoint.md)保存具名参数/固定值/别名、
SGD/AdamW 与 raw Continuation，使用独立 `TIDEGCK1` schema 1；内嵌原 `TIDENCK1` 参数格式。
整数时钟与 metadata 不经浮点转换，存储不使用有效状态投影。加载先完整校验/分配，
再复制到原参数身份；持久化为明确 autograd 边界，不恢复 tape、RNG 或下游训练控制器。

独立 C++ 可执行文件每 dtype 自建 24 个图案例：kv、affine、delta、gated_delta、ssd、
异构组合 × FC、HB depth=1/同秩分支、非对齐 DAG、带重复边的反馈 PDG。
独立 Streaming/Greedy 比较全部记录，包括结构化 Describe、原始状态/整数槽、控制、
逐边/端口输出与 pending。三步 AdamW 比较全部具名参数和输入梯度、None/连接零、
共享 owner 和 optimizer 槽；保留图切点比较完整末态与拼接输出 VJP。
含 pending checkpoint 换模型 owner 后，比较精确状态及恢复后的第四次更新。
Python pytest 只启动并检查独立进程 JSON，没有提供参数、路由、梯度或参考执行结果。

另验证超过 2^54 的逻辑时间、初态 -1 时钟、非空自定义 History 的有符号标量、
巨大 int64 node map 与 Tensor。拒绝别名变化、损坏校验和、无效 pending 和目标覆盖。
校验和正确但状态时钟非法的记录也拒绝，且全部参数保持原值。
原生参数/optimizer、accelerator CPU、greedy 和 standalone 局部手算回归同批通过。
局部数学真值由已有独立公式/Jacobian 检查锚定；调度对照本身不单独证明局部公式。

开发 tree `4c70968719d7235a2459a72baab94a044fb60ebd` 的 43 项编译/链接及同范围 20 项已通过。
本固定资格与后续 placement 草稿隔离，未发生该增量的构建或测试失败。
复现：检出实现，按 [development](../development.md) 构建，运行 JSON 测试列表
（`--dtype both`），或独立运行 `tidegraph-weighted-training-check --device=cpu
--dtype=float64 --output-dir NEW`，FP32 另运行。无需旧运行目录。
Python 3.12.13、Torch 2.12.0+cpu、GCC 10.3.1；ATen/BLAS 单线程、两个 build workers。
任务 background.slice、Nice=10、20 分钟上限；原始输出 artifacts/jobs/core5-native-training-qualify。
本报告不认证 FP16/master 格式、新支线 mixed/resident、CUDA/NPU 真机或性能。
