# Core-5 带权循环记忆 CPU 资格 — 2026-10-27

固定实现/测试提交 `442e8b2701164eb7e4241da56c8a395a832ebeb0`，数学来源 `0770593`。
**544 passed，0 failed，0 skipped；CPU FP64/FP32，aarch64。**
干净 checkout、workload 退出 0、服务终态成功均已核对。
[机器记录与哈希](core5-recurrent-20261027.json)。

契约及双向定义→符号→测试对应：[循环/卷积](../weighted-recurrent.md)、
[共同带权绑定](../weighted-message.md)、[semantic-map](../semantic-map.md)。

## 实际范围

- affine/delta 的完整双时间尺度矩阵、摘要、sum/state、当前特征与 Full；
  GDN 的多头/固定时间卷积/L2/保留及 Delta 门/每头 RMS 后输出门；
  SSD 的分组 B/C、dt 外积写入、直通项、先门控再全内部宽 RMS。
- 独立教材标量：G=2/1.5、GDN H=.5/23/32、SSD H=1/1.5；基础多维 A/B 公式；
  独立 grouped conv1d 与普通序列 mixer，逐参数/输入 VJP；全参数方向差分。
  序列对照验证声明的实数公式，没有执行外部 FLA/Mamba 程序或宣称整模型等价。
- 固定逻辑时间槽、缺失/弱/零事件、快照等式、不同余数、>2^54/MAX 时间；
  零接收保留 exact raw tuple，未来有效状态的弱极限与完整记录等价分开检查。
- 由手算摘要区分三 Read，完整描述四字段，未激活候选仍写入；Full 独立读 proposal。
- 纯 PyTorch、Python/native 独立调度的完整记录与独立状态/描述/输出根，PDG 反馈、
  非对齐 TimedDAG、FC/HB ports 与异构 KV/循环；shared parameter/private state。
- 每切点的记录/pending/跨窗口梯度、checkpoint 新 owner 恢复；三步 AdamW 与恢复后再更新，
  native streaming/prefill、no_grad 推理。standalone C++ 另有独立标量、局部 VJP、时钟与卷积检查。
- KV 相关回归、结构化记录、公共语义与旧 isolated-gradient 的定向回归。

新记忆当前使用声明的合法序列递归后备。此报告不认证新 packed 联合算法、批量 VJP 优化、
mixed/resident、GPU 或完整 standalone C++ 训练/checkpoint 矩阵；其余正式发送/评分亦未在此完成。

## 复现

在该提交按 development.md 构建对应 CPU 库，然后执行：

```sh
TIDE_BUILD_DIR="$PWD/build/cpu" python -m pytest -q --dtype both \
  tests/test_weighted_allocation.py tests/test_weighted_kv_formulas.py \
  tests/test_weighted_execution.py tests/test_weighted_native.py tests/test_weighted_instances.py \
  tests/test_weighted_convolution.py tests/test_weighted_recurrent_formulas.py \
  tests/test_weighted_recurrent_execution.py tests/test_compare_objective.py \
  tests/test_structured_records.py tests/test_semantics.py tests/test_isolated_gradients.py \
  tests/test_cli.py::test_cpp_custom_state_kernel
```

Python 3.12.13、Torch 2.12.0+cpu、GCC 10.3.1、C++17/CXX11 ABI。
ATen/BLAS 单线程；开发构建两个 worker。资格复用前验证 exact C++ 源码及所有二进制哈希。
服务 tide-core5-recurrent-qualify-20261027，background.slice、Nice=10、15 分钟上限。
原始记录位于 artifacts/jobs/core5-recurrent-qualify；复现无需该目录。

## 保留的失败与修复

core5-recurrent-dev 因 launcher 预创建 job.py 要求独占的新输出目录而退出 1，构建和测试未开始。
core5-recurrent-build 构建成功，测试 **212 passed / 1 failed**。SSD FP32 Read0 的前向逐位一致，
output/state/pending 各组 VJP 逐位一致；验证目标按 Python 字典与 C++ map 的不同字段次序
累加，目标差 1.9e-6，平方目标中的相消梯度差 7.15e-6 超过既有容差。
修复 compare.objective 的 owner/字段归约顺序并加入排列不变性检查，没有改模型或容差。
后续冻结开发树 536 项通过，补非对齐 DAG 8 项短测，本次固定提交合并 544 项通过。
这些独立运行保持各自原始状态，旧失败不改写为通过。
