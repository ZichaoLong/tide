# 联合KV准备的固定检查与成本（2026-11-10）

实现`aab8cb7b262acfeddd947587026555eff095e6bd`，数学0770593；
[机器可读配置/工作量/hash](joint-kv-preparation-20261110.json)。
公共入口与独立根契约见[KV准备](../weighted-kv-sharing.md)。

将同块中的相同基座、不同样本/时间输入形成矩阵批；私有低秩与summary投影按有界矩阵预算合批。
相同参数和输入才共享计算，不同输入只是批量计算。保留全部使用者贡献、原始int64时间、
None/连接零、独立字段根和参数版本检查。此提交仍用原greedy调度。

## 验证范围

- 开发快照全320 CPU构建完成；固定提交重新安装wheel，核对复用构建的C++源码与二进制hash。
- 固定安装CPU FP32/FP64相关80项通过；NPU相关33项通过（含CPU辅助断言）。
- 安装消费者Python/native在真实NPU上分别完成三步独立数值/离散/SGD检查及新进程精确恢复。
- 服务`gef-joint-prefill-fixed-block-a`于2026-11-10 02:57:08 UTC结束，exit0；逐项记录均已审阅。
- Python-owned NPU构建不等于standalone SDK或resident/GPU资格；本轮没有这些新证据。

重建与检查：按development选择私有CANN9.2环境，从固定提交安装wheel并运行
`tests/test_weighted_block_preparation.py test_weighted_kv_sharing.py test_weighted_compute_sharing.py
 test_weighted_private_projection.py test_weighted_independent_preparation.py --dtype both`；
设备整链用`scripts/qualify_dense_query.sh NEW npu MATCHING_BUILD auto`。
全路径前缀为`tests/`，资格时使用`python -m pytest -o pythonpath= ...`避免误用工作区源码。

## 原FC16任务的有限成本

固定下游0439efd、原FC16/B8/L128/head/loss/AdamW、FP32、native、auto查询；
CPU1线程，NPU mixed-c/2 host线程。各一进程、一预热一测量，预热后恢复相同初始参数/optimizer，
测量前后参数hash核对在JSON。完整wall包含准备、调度/packing、反向、检查、更新、清理和同步。

| 设备 | 前一固定默认完整步/s | 本提交完整步/s | forward/s | backward/s | 实际Full/消息 |
| --- | ---: | ---: | ---: | ---: | --- |
| CPU | 38.953 | 35.486 | 14.660 | 20.291 | 8344 / 25060 |
| NPU | 101.756 | 96.776 | 54.258 | 41.472 | 8345 / 25064 |

前一来源见[prefill-default](prefill-default-20261110.md)。私有投影提交12288→96，最大私有批744；
基座投影4次、最大批1024。CPU峰值RSS8277MiB；NPU峰值allocated5481MiB/reserved5576MiB，
包含预热。设备映射保留在JSON；未排除外部共享主机争用，单次观测不证明统计稳定。

复现成本：`scripts/benchmark_dense_query.py --downstream-repo READ_ONLY_DOWNSTREAM
 --device cpu|npu:0 --query-execution auto --native-library MATCHING_BUILD --batch 8 --length 128
 --warmup-state reset --output-dir NEW`。按执行契约放独立有界服务，设备准入单独记录。

本配置未做原任务完整独立数值/离散验收，因此这些是实际成本，不是严格等价吞吐或学习结论。
投影次数大减而总耗时仅有限下降；状态快照/一阶根逐时绑定、动态索引、守卫与大量小提交仍需profile，
不能因外层区域仅4次或投影已合批就宣称完成全sequence内部求值。
