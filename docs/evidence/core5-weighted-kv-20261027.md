# Core-5 带权 KV 首组合 CPU 资格 — 2026-10-27

固定实现/测试提交 `6fb467c25755a2b3f8938952f1b2ec1011cbcd11`；数学来源 `0770593`。
**129 passed，0 failed，0 skipped；CPU FP64/FP32，aarch64。**
干净 checkout、workload 退出 0、服务终态成功均已核对。
[机器记录与哈希](core5-weighted-kv-20261027.json)。

契约：[带权消息](../weighted-message.md)、[精确分配 VJP](../weighted-allocation.md)；
固定定义对应从 [semantic-map](../semantic-map.md) 进入。

## 实际范围

- 首组合 kv/sum/+、cat/ind、regional_total/context/cat/quad；另含 qry、Read 0/-/+、fixed 总量。
- 独立 quad 活动集枚举、精确解映射方向差分、KV 手算与直接 Attention 公式；
  与 C++ 不同的求解过程，VJP 身份为 quad-exact-map-v1。
- 原始逻辑时间窗口、int64 超过 2^54、多头部分 RoPE/距离偏置、初态独立根。
- Python/native 完整记录及独立输入/参数/状态/pending 根；FC/HB 正式共享、位置及来源绑定，
  同时保留 PDG 反馈、非对齐 DAG、一般端口；多步 AdamW、每切点续接、checkpoint 新 owner 恢复。
- standalone C++ 手算、局部 VJP、共享根及时间/状态检查，由 tidegraph-kernel-check 执行。
  这不是完整 standalone C++ 训练/checkpoint 矩阵。

此报告不认证其他正式记忆/分配/评分候选、新 packed/批量 VJP、mixed/resident 或 GPU。
prefill 使用已声明的合法局部递归后备；不能据此宣称完成新联合计算优化。
完整记录等价、指定投影与不同函数性质检查在测试中分别表达。

## 复现

按该提交的 development.md 构建对应 CPU 库，然后执行：

```sh
TIDE_BUILD_DIR="$PWD/build/cpu" python -m pytest -q --dtype both \
  tests/test_weighted_allocation.py tests/test_weighted_kv_formulas.py \
  tests/test_weighted_execution.py tests/test_weighted_native.py \
  tests/test_weighted_instances.py tests/test_structured_records.py \
  tests/test_semantics.py tests/test_cli.py::test_cpp_custom_state_kernel
```

Python 3.12.13、Torch 2.12.0+cpu、GCC 10.3.1、C++17/CXX11 ABI，ATen/BLAS 单线程。
资格通过 develop.py --reuse-build 验证 C++ 源码和所有二进制哈希；构建清单基线字段
不是本次资格的源码身份。服务 tide-core5-weighted-kv-qualify-20261027，
background.slice、Nice=10、10 分钟上限；原始记录在 artifacts/jobs/core5-weighted-kv-qualify。
复现无需依赖这个旧目录。

首轮 core5-weighted-kv-dev 的 C++ 测试缺 Options 构造参数导致构建失败，原记录保持失败。
后续修复、125 项开发检查及本次 129 项不可变资格各自独立记账。
