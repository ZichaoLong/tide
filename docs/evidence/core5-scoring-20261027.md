# Core-5 完整评分/证据/key CPU 资格 — 2026-10-27

实现/测试固定 `cacb645adc64ed06e454f7ac8d8572bf09b98ee2`，上游 `0770593`。
**593 passed / 0 failed / 0 skipped；CPU FP64/FP32、aarch64。**
干净源码、workload 退出 0、服务终态成功已核对。[记录与哈希](core5-scoring-20261027.json)。
[评分契约](../weighted-scoring.md)给出定义→配置/维度/快照→符号→独立测试；
[分配契约](../weighted-allocation.md)给出显式共同阈值和精确映射 VJP。

本次接入 match、att、grouped evidence、content_stats/content key、received/mu query 权重；
独立参数及显式同形共享、适用配置拒绝与既有图 hash 保持。独立逐标量 Encoder 公式、
归一化偏置/门控残差/四投影手算、key 的直接统计依赖与 query 权重公式各自检查。
alpha_sel=0 的评分投影等价与完整函数记录等价分别验证。

评分整图覆盖五种异构记忆、FC/HB、反馈 PDG、三种 Read 快照和完整返回；
独立描述/输出/初态根、共享参数全部使用者、三步 AdamW、每切点续接、checkpoint 与
optimizer 恢复后再更新、全参数/输入方向差分通过。standalone C++ 的局部评分公式/VJP
单独运行。联合门禁同时运行所有既有带权公式、发送、分配、卷积及图执行测试。

解析共同 score_shift VJP 在活动总量约束上返回连接零，其他 bounds/caps 路径仍保留；
松弛分支为评分梯度和的负值。中心取非零响应斜率坐标，防止不相关零 cap 污染。
独立平移、上下界/等式/slack、共同 cotangent、版本检查覆盖该修复。
未调整容差、dtype、前向或使用梯度裁剪。
一维 Encoder/Query 统一矩阵乘列向量顺序，避免近零 FP32 梯度被优化器放大。

开发失败保留：scoring-dev **345/3**（共同阈值相消和矩阵顺序）；
scoring-vjp **246/40**（C++ undefined shift 占位无 device）；
scoring-vjp-fixed **301/1**（FP32 Read− key 偏置梯度绝对差 2.0266e-6）。
修复后主工作区评分定向 **66/0**，再固定本实现运行独立 593 项资格；不累计为额外样本。
原失败源码/日志/status 保留在各 `artifacts/jobs/core5-scoring-*` 目录，原结果不改写。

本证据不认证后续真正 packed/联合前缀优化、批量 VJP、mixed/resident、跨卡、GPU、
完整 standalone 训练/checkpoint 或性能，也不是整个仓库的全量资格。
CUDA 源码、构建与真机证据仍分列。

复现：固定提交按 [development](../development.md) 构建后：

```sh
TIDE_BUILD_DIR="$PWD/build/cpu" python -m pytest -q --dtype both \
  tests/test_weighted*.py tests/test_cli.py::test_cpp_custom_state_kernel
```

该固定提交的 glob 恰含 16 个带权测试文件；确定展开列表见 JSON。
Python 3.12.13、Torch 2.12.0+cpu、GCC 10.3.1，ATen/BLAS 单线程。
服务 tide-core5-scoring-qualify-20261027，background.slice、Nice=10、15 分钟上限；
复用前核验 C++ 源码和全部二进制 hash。原始记录 artifacts/jobs/core5-scoring-qualify；复现无需该目录。
