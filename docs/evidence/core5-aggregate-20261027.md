# Core-5 Aggregate 批量计算与 VJP CPU 资格 — 2026-10-27

固定实现/测试 `d90cfbc99bc9af821d24dec11cc27cc59926b02e`，上游 `0770593`。
**704 passed / 0 failed / 0 skipped；CPU FP64/FP32、aarch64。**
干净源码、工作退出 0 与服务终态成功均已核对；[命令/环境/哈希](core5-aggregate-20261027.json)。
[契约与数学对应](../weighted-aggregate-batch.md)给出原子、参数、布局、零分支和返回。

Python 与独立 C++ 都按来源类别合批非线性投影，再按实际 fiber 长度完整归约。
SrcClass 只选参数；同类不同物理原子各参与一次，没有填充原子或同纤维唯一性假设。
alpha=0 不调用源变换，零总量返回零内容，载荷强度只在聚合公式乘一次。
候选实际结果直接解码，不调用 scalar Aggregate 补写数值。

显式 `aggregate_autograd="batched"` 保留逐纤维公开根；按来源类别顺序、
零/正总量及输入求导元数据分组，用实际批量图求一阶 VJP。只对已定义 cotangent 的行
返回输入梯度，不按梯度数值判断连接。零总量不连接源变换参数，正总量中的零质量原子
保持连接零；参数别名和重复使用的同一输入累计全部贡献。
原始版本变化和高阶请求明确拒绝，默认 replay 策略仍可配置。

独立加权均值 Jacobian、含非线性来源变换的方向差分与 literal SiLU/归一化手算
核验局部公式；standalone C++ 另核对重复原子导数及输入/参数的 None/零边界。
两 sample 的反馈 PDG、非对齐 DAG 对照独立标量调度，核对完整记录、控制、
状态、pending、输出和内容独立根、切点继续，并检查实际批量计数与零 replay。
HB/PDG 的 Python/native batched 策略各比较三步 AdamW 参数、梯度与 optimizer slots，
在含 pending 的切面保存、换 owner、恢复后继续更新。

相关回归覆盖旧 Aggregate 公式/调度/近零 optimizer/配置拒绝，以及 KV、所有循环记忆、
完整评分/发送、多步训练和恢复。该资格不是整个仓库的全量矩阵；Read/Full/区域
批量 VJP、设备真机/常驻/跨卡、standalone 完整训练恢复与性能仍分别验收。

开发数值树 d9758c37765f4bda35f4a91bd32ba42402d49ddd 为 **288/0**，
VJP 树 4ebfbaeb9848217a423a922c101c2615913190f1 为 **696/0**，额外训练检查 **8/0**。
局部短测最初 4 项因固定 scale 不需要梯度却列为根而报错；改为显式可求导的独立
未使用 scale 根后 9 项通过，未修改数学实现或容差。最终本资格包含全部修正与新增测试。

复现：固定提交，按 [development](../development.md) 构建后执行：

```sh
TIDE_BUILD_DIR="$PWD/build/cpu" python -m pytest -q --dtype both \
  tests/test_weighted_aggregate_batch.py tests/test_weighted_aggregate_vjp.py \
  tests/test_weighted_aggregate_training.py tests/test_aggregate_autograd.py \
  tests/test_aggregate_formulas.py tests/test_aggregate_schedules.py \
  tests/test_aggregate_nearzero.py tests/test_aggregate_contract.py \
  tests/test_weighted_execution.py tests/test_weighted_native.py \
  tests/test_weighted_recurrent_execution.py tests/test_weighted_scoring_execution.py \
  tests/test_weighted_sending_execution.py tests/test_cli.py::test_cpp_custom_state_kernel
```

Python 3.12.13、Torch 2.12.0+cpu、GCC 10.3.1；ATen/BLAS 单线程，build workers=2。
服务 tide-core5-aggregate-qualify-20261027，background.slice、Nice=10、15 分钟上限。
原始 status/log/source/development/build manifest 保留于 artifacts/jobs/core5-aggregate-qualify；
复现无需该旧目录。复用 build 前核验源码及全部二进制 hash，不以旧 manifest baseline 代替资格提交。
