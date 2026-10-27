# Core-5 State 批量 VJP CPU 资格 — 2026-10-27

固定实现/测试 `7a017bf4eea93787ca2d8e72baf436b128266afd`，上游 `0770593`。
**343 passed / 0 failed / 0 skipped；CPU FP64/FP32、aarch64。**
干净源码、工作退出 0 与服务终态成功已核对；[命令/环境/哈希](core5-state-vjp-20261027.json)。
[契约与逐项对应](../weighted-state-vjp.md)说明函数、实际参数、分组、槽与独立测试。

五种带权记忆的 Upd 接入批量一阶 VJP，数值仍由精确联合候选计算。
Python 私有 module 树捕获全部注册参数/缓冲区并保持共享关系；C++ 独立捕获 NodeWeights::extra。
结构分组加入实际零/正接收分支；按未定义 cotangent 模式返回参与行梯度，
不按梯度数值猜测连接。空卷积左缓存的 H 根保留无连接，与非空但不命中的连接零区别。
int64 元数据不可求导，原始参数/状态版本变化和高阶请求明确失败。

新增独立根检查覆盖四个 sample 的零/正接收、所有浮点槽、空 KV/卷积槽、各输入、
完整参数、独立输出、混合槽与连接零目标。key/value/summary 同形共享累计全部贡献，
不改变活模型的别名。Python 禁止 scalar Upd 调用的用例通过；Python/native 的实际
batched_state_events 非零、semantic_state_replays 为零。private batch graph 的构图与反向成本仍存在。
standalone C++ KV 的 packed 输出根对共享 base 的导数与独立 sqrt(2) 手算相符，并检查实际能力统计。
其他 C++ 局部公式不自动认证所有 standalone 完整训练/恢复组合。

完整相关回归包括旧 State VJP、五种记忆、异构配置、Read0/-/+、评分/发送函数、
独立图调度、完整记录/状态/pending/控制、每切点 VJP、多步 optimizer 与 checkpoint 恢复。
这是相关模块资格，不是整个仓库全量矩阵；Read/Full/Aggregate/Region 批量反向、
设备真机/常驻/跨卡、完整 standalone 训练/checkpoint 和性能须分别记录。

开发 state-vjp-dev 保留 **337 passed / 6 failed**：FP32 affine Read0 的 score.bias 差 7.63e-6，
SSD Read−/+ 的 input.2 差 2.13e-6 / 3.87e-6，Python/native 各三项。
独立诊断显示共享 norm 先合并 cotangent 再做 RMS VJP，改变 scalar 分支分别反传的归约顺序。
保留写入、GDN 控制、p_v 摘要各自的批量 norm 后，state-vjp-norm 开发 **81/0**，
再固定本提交运行独立资格。未改变数学/数值前向、输入、dtype、容差或裁剪梯度。
原失败 status/source/log 和 diagnosis.txt 保留，不以新通过覆盖。

复现：固定提交按 [development](../development.md) 构建（StateKernel 虚接口改变须重编全部依赖），然后：

```sh
TIDE_BUILD_DIR="$PWD/build/cpu" python -m pytest -q --dtype both \
  tests/test_weighted_state_vjp.py tests/test_weighted_packed.py \
  tests/test_state_batched_vjp.py tests/test_weighted_recurrent_execution.py \
  tests/test_weighted_native.py tests/test_weighted_scoring_execution.py \
  tests/test_weighted_sending_execution.py tests/test_cli.py::test_cpp_custom_state_kernel
```

Python 3.12.13、Torch 2.12.0+cpu、GCC 10.3.1；ATen/BLAS 单线程，build workers=2。
服务 tide-core5-state-vjp-qualify-20261027，background.slice、Nice=10、15 分钟上限。
复用前校验源码和全部二进制 hash；原始 records 在 artifacts/jobs/core5-state-vjp-qualify，复现无需该目录。
