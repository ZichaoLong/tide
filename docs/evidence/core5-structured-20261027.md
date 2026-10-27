# Core-5 结构化表示 CPU 资格 — 2026-10-27

实现/测试固定提交 `714d5f963b676d95a0ae763171a7d93806f183d3`；数学来源 `0770593`。
**212 passed，0 failed，0 skipped；CPU FP64/FP32，aarch64。**
干净 checkout、workload 退出 0、服务终态成功均已核对。
[机器记录与哈希](core5-structured-20261027.json)。

契约：[structured-records](../structured-records.md)；定义对应从
[semantic-map](../semantic-map.md)进入。此项验证公共表示及已有调度回归；
不认证带权 KV/分配公式、整个旧 CPU 门禁或任何设备实现。

## 实际范围

- Python 自定义有限产品 Read：绑定初态/旧态/proposal、Full comparison，反馈 PDG 与非对齐 DAG。
- 独立字段 VJP 根与无连接参数、typed packing、值解码及元数据损坏拒绝。
- int64 时间超过 2^54、状态校验、旧边界节点兼容及 checkpoint v5 精确原始状态。
- SrcClass 同类别的三个物理原子各参与，独立手算 9*x 及导数。
- standalone LibTorch `check_structured_records` 自有字面公式与标量/frontier 调度；
  Python/native 的 SettleGraph、坐标、Read/状态时钟/批量 VJP 回归。

精确字段域、整数、路由与 None/零梯度单独检查；浮点容差沿用 semantics。
具名字段 batch 当前仍使用明确的局部语义反向重放，不宣称完成新批量 VJP 优化。

## 复现

在该提交上按 development.md 构建对应 CPU 库，然后执行：

```sh
TIDE_BUILD_DIR="$PWD/build/cpu" python -m pytest -q --dtype both \
  tests/test_structured_records.py tests/test_read_contract.py \
  tests/test_state_clock_contract.py tests/test_coordinate_checkpoints.py \
  tests/test_cli.py::test_cpp_custom_state_kernel tests/test_settle_ports.py \
  tests/test_read_batched_vjp.py
```

环境为 Python 3.12.13、Torch 2.12.0+cpu、GCC 10.3.1、C++17/CXX11 ABI，
ATen/BLAS 单线程。资格通过 develop.py --reuse-build 校验所有原生源码和二进制哈希，
复用了同一 C++ 内容构建；开发构建的基线字段不是本次资格源码身份。
服务 `tide-core5-structured-qualify-20261027` 位于 background.slice，Nice=10，时限 10 分钟。
日志/完整记录在 `artifacts/jobs/core5-structured-qualify`；复现无需旧目录。

原 `core5-structured-dev` 保持 **210 passed / 2 failed**：Python 边界节点没有 kernel，
新增槽校验最初没有兼容该类型。修复后新的开发树与本次不可变源码各自 212 项通过。
先前缺 TIDE_BUILD_DIR 的 24 项前置条件失败亦保留；本报告不将这些旧失败改写为通过。
