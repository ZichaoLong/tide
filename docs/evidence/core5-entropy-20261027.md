# Core-5 熵分配 CPU 资格 — 2026-10-27

固定实现/测试 `a05d3ffc946985cfeb124a5e3b3074ceeece306d`，上游 `0770593`。
**226 passed / 0 failed / 0 skipped，CPU FP64/FP32，aarch64。**
干净源码、workload 退出 0 与服务终态成功已核对；[记录及哈希](core5-entropy-20261027.json)。

[分配契约](../weighted-allocation.md)逐项关联定义、参数、精确映射 VJP 和独立检查。
Bernoulli、Shannon 和固定 alpha∈[1,2] 的 ent 使用相同盒/总量约束；
Python 受保护 Newton/二分与 C++ 独立二分使用各自实际响应点。
反向针对最优解映射，非有限迭代导数或 HST。

独立 80 位 Decimal 标量求解、隐式 Jacobian、gradcheck 与 ent(alpha=2) 移位等式
覆盖值、cap/总量/温度导数；原地版本修改与数学正支持下溢明确拒绝。
默认 quad 配置身份保持；Python/native 完整区域链、独立根、三步 AdamW、
pending/checkpoint 恢复通过。standalone C++ 单列等评分 Bern 手算与 Shannon/softmax。
KV、实例与循环执行定向回归通过；这些结果不代表所有配置的组合资格。

本报告不认证其他发送/评分、packed 联合算法、mixed/resident、GPU 或完整 standalone
C++ 训练/checkpoint。旧失败保持原记录；本次开发与不可变资格分别各 226 项通过。

复现：在固定提交按 development.md 构建 CPU 库后运行：

```sh
TIDE_BUILD_DIR="$PWD/build/cpu" python -m pytest -q --dtype both \
  tests/test_weighted_allocation.py tests/test_weighted_entropy_allocation.py \
  tests/test_weighted_entropy_native.py tests/test_weighted_kv_formulas.py \
  tests/test_weighted_native.py tests/test_weighted_instances.py \
  tests/test_weighted_recurrent_execution.py tests/test_cli.py::test_cpp_custom_state_kernel
```

Python 3.12.13、Torch 2.12.0+cpu、GCC 10.3.1，ATen/BLAS 单线程。
服务 tide-core5-entropy-qualify-20261027：background.slice、Nice=10、15 分钟上限；
复用构建前核验源码及全部二进制哈希。原始记录 artifacts/jobs/core5-entropy-qualify，
复现无需此目录。
