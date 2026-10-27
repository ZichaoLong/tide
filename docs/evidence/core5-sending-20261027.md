# Core-5 发送函数 CPU 资格 — 2026-10-27

实现/测试固定 `4a0858880831cb46ad96191ab9dd58fca5260223`，上游 `0770593`。
**320 passed / 0 failed / 0 skipped；CPU FP64/FP32、aarch64。**
干净源码、workload 退出 0、服务终态成功已核对。[记录与哈希](core5-sending-20261027.json)。
[契约与定义→符号→测试](../weighted-sending.md)列明定义域、字段、适用配置和反向。

十种正式发送、固定常量/可训练 logit/上下文节点增益接入公共 Python/C++；
regional 描述四字段，节点增益描述三字段。k 的发送强度界与硬选择的活动数 K 分开。
独立 exp 商公式/全部输入根、Attention 零基权重右导数及全零方向反例、
LP 顶点枚举、零温小数发送、稳定 ID、hard 全非正退化与 cap 分支反向均覆盖。
全上限端点直接返回 caps，支持下溢与输入版本变更明确失败。
阈值头不误用 sigmoid；gain 独立标量导数、constrained/hard 弱总量反例另有手算。

完整记录/独立描述与输出根、HB/非对齐 DAG、PDG 反馈、每切点 pending 和 VJP、
各发送函数三步 AdamW、参数/optimizer/checkpoint 恢复与 no_grad 推理通过。
KV、熵分配和循环执行定向回归保持。standalone C++ 有独立 Attention/填入/hard/比例公式与 VJP。
原先 quad 默认身份保留，不把其他新函数缓存解释为原函数的续接。

此证据的评分是 cat/ind、证据为 cat。它不认证待实现评分/key、真正 packed 联合算法、
mixed/resident、GPU、完整 standalone 训练/checkpoint 或性能。
开发快照 304 项和端点补充快照 70 项各自保留，正式 320 项包含新增检查且不重复累计。

复现：固定提交按 development.md 构建后运行：

```sh
TIDE_BUILD_DIR="$PWD/build/cpu" python -m pytest -q --dtype both \
  tests/test_weighted_sending.py tests/test_weighted_sending_native.py \
  tests/test_weighted_sending_execution.py tests/test_weighted_entropy_allocation.py \
  tests/test_weighted_entropy_native.py tests/test_weighted_kv_formulas.py \
  tests/test_weighted_native.py tests/test_weighted_recurrent_execution.py \
  tests/test_cli.py::test_cpp_custom_state_kernel
```

Python 3.12.13、Torch 2.12.0+cpu、GCC 10.3.1；ATen/BLAS 单线程。
服务 tide-core5-sending-qualify-20261027，background.slice、Nice=10、15 分钟上限；
复用前核验 C++ 源码及全部二进制哈希。原始记录 artifacts/jobs/core5-sending-qualify，复现无需该目录。
