# 独立输入KV投影合批与有限对照 — 2026-10-29

实现固定`f03e9bee0f069d48f5f1544d3821b7781409a27f`，对照固定`38ebf422b198d7cc041b6238fb954cf9775c02c0`。
数学仍为`0770593838a37c790a568226930567ed60f99b83`，图v13/checkpoint v5不变。
[同名JSON](core5-independent-projection-20261029.json)保存源码、构建/运行身份、完整命令、终态与原始记录hash。
这是已有低秩/共享组内私有BMM之后的增量；不覆盖旧失败或未验证平台。

## 定义与独立证据

[KV §1.2及FC §6.2对应](../weighted-kv-sharing.md)区分两件事：相同参数和输入许可计算复用；
不同输入/基座的同形行可以批量计算，但必须保留每行参数、epsilon和输入身份。
既有共享准备先执行，剩余正接收行以四次真实BMM计算base、adapter V/U和summary。
归一化逐行执行，私有KV缓存、时间与State写入保持原路径。该实现不减少候选更新、Full或消息。

新增独立矩阵批按四倍逐行矩阵字节数限制在8MiB；不足两行或末尾单行继续原逐行计算。
这是局部打包预算，不是总计算图/HBM上限，不改变dtype或截断逻辑时间/状态。
`linear_rows`只对有根的输出返回VJP，未使用行保持None、连接零保持零，共享参数汇总全部使用者。
原参数/输入版本检查及明确的一阶限制保持。准备字段仅来自候选实际计算，不进入checkpoint。

独立手算映射/Jacobian覆盖不同epsilon、别名贡献、零接收、异形、预算、版本和None/零；
Python与C++分别验收。完整图检查覆盖独立调度、保留图、逐边消息、raw state/control/pending及训练恢复。
旧C++检查用“准备字段为空”代替“无共同计算”，因此曾失败；修正后直接要求零复用计数、
实际独立批、手算输入梯度和另一行输入/private adapter无根，未放宽容差或删除无共享要求。

## 固定源码资格

| 入口/环境 | 固定源码资格与实际范围 |
| --- | --- |
| CPU aarch64 / Torch2.12 / Python与native | **273 passed、2个原FP16 strict xfail**；独立公式/Jacobian/None根、共享、在线完整图、三步AdamW、真实checkpoint与standalone相关检查 |
| 纯PyTorch NPU / Torch2.12.0+cpu、Torch-NPU2.12.0.post2、CANN9.1 | FP32/FP16各3图（KV FC/HB/一般ports）通过；streaming/prefill、推理、独立字段根、4更新、恢复与CPU交接；物理1映射逻辑0 |
| Python/native NPU / 同上、全新core与常驻adapter | 上述六组eager通过，另有公共KV常驻前向/训练/磁盘恢复**12项通过**；物理1映射逻辑0 |
| standalone SDK / Torch2.10.0、CANN9.0、Python3.11配套运行时 | 两dtype各3图（FC/HB/非对齐KV）×4次默认AdamW更新通过；half显式FP32 scoring/master、loss_scale16；完整原生checkpoint/CPU交接 |
| standalone SDK常驻 / 同上 | flow、training、node-training × FP32/FP16 **六项通过**；每dtype前向21案例，单卡训练5配置/20更新/3窗口，双卡节点训练5配置/20更新/换放置恢复/共同拒绝；物理9/13映射逻辑0/1 |

这是受影响组件的资格，不是全部配置/平台矩阵。CPU两个strict xfail仍严格绑定原GDN/AdamW见证。
SDK核心还通过CPU FP32/FP64公式检查；不得把它与新Python/native ABI的设备结果混记。

CPU core复用开发构建前后，完整C++内容hash与23个binary hash逐一核对；
构建manifest保留原始ead6f34开发来源，正式资格来源是f03e9be，不能混为同一身份。
新增State可选virtual接口后，两个NPU运行时均重新构建核心与相关常驻消费者，未把旧ABI当新资格。
GPU按用户要求延后，本提交没有新的CUDA构建/真机证据；历史CUDA构建只证明其对应提交。

NPU raw checkpoint及optimizer解码逐值核对；跨设备后续更新按原精度容差核对。
D64新CPU FP64两步诊断也通过完整记录、离散结构、参数梯度、参数和SGD槽检查。
第一步实际保存并换owner解码；第二步两条独立调度各自的记录、梯度、参数及optimizer全部逐值恢复。
SDK日志中的unused field padding差异只涉及未使用填充，有效字段逐值相同；不宣称整块缓冲区字节相等。
两个原FP16 GDN/默认AdamW strict xfail保持；D64/D2048 FP32、原FP32反馈与默认异步双NPU边界
仍从[前一轮证据](core5-nongpu-boundaries-20261029.md)导航，未用本轮KV通过替代。

## 有限性能和实际工作量

每配置一次独立进程，一次warmup和一次连续测量窗；profile另开进程。
完整训练计时包含输入准备、调度/packing、完整返回、loss、反向、finite checks、SGD、detach和同步；
构造、warmup、最终梯度/optimizer导出、独立参考核对及profile解析单列。
候选不消费参考轨迹。项目重型构建和计时不重叠，其他用户争用未排除；单次差异不证明稳定收益。

| 相同材料/完整训练一窗 | baseline 38ebf42 | f03e9be | 单次耗时变化 | 完整核对 |
| --- | ---: | ---: | ---: | --- |
| CPU native，HB medium，FP32 | 388.129 ms | 405.010 ms | +4.35% | 返回、最终梯度、参数及SGD槽通过 |
| CPU native，ranked D64，显式FP64 | 2991.719 ms | 2996.551 ms | +0.16% | 同上；另有两步真实恢复 |
| NPU纯PyTorch mixed-c，HB medium，FP32 | 1825.444 ms | 1850.355 ms | +1.36% | 返回、最终梯度、参数及SGD槽通过 |

三组before/after各自fixture hash相同，均使用prefill、一个warmup和一个测量窗。
中型材料hash为`e6e7b7d9f3f12040bb34638de8cd4ffc0d152c5fb452d41f16dae3bd0e836f42`；
D64 FP64材料hash为`07a775da9626754a12db7d2daaad4733b81afaa9bd64c71b72be4dd5300ff01c`。
NPU前后位于同一物理1/逻辑0。CPU主机load约35.11→33.91，NPU流程约33.64→34.09；
这些值只是争用背景，不是独占或争用相同的证明。

| 实际工作量 | HB medium（CPU/NPU均相同） | ranked D64 FP64 |
| --- | ---: | ---: |
| 候选更新 / Full / 消息 / 输出 | 128 / 128 / 192 / 32 | 479 / 344 / 1604 / 19 |
| 原共享准备覆盖写入 | 96 | 34 |
| 新独立准备覆盖写入 | 32 | 445 |
| 新投影调用 / 投影行 / 最大批 | 4 / 128 / 32 | 112 / 1780 / 24 |
| 原私有投影调用 / 投影行 / 最大批 | 144 / 288 / 2 | 6 / 102 / 32 |

投影行按四种线性映射分别计数，因此1780行对应445次写入。两类准备合计覆盖全部479次D64写入；
State仍调用479次，最大State batch仍1，不把投影合批称为完整State或窗口反向已经融合。
三对候选的逻辑工作量、pending=0完全相同，没有借发送变稀疏制造计时收益。

D64独立profile完整通过，汇总了203类事件，其中有17697次mm、236次bmm、69342次copy、12325次stack、
6046次index_select、27526次clone。`BatchedQueryVjp`出现823次、`BatchedAggregateVjp`447次。
实际user_annotation inclusive跨度：forward 1.886s、backward/finite 3.069s、optimizer .163s、
detach/返回/同步 .329s；mm/bmm的self CPU合计约.125s。这些数含profile开销，不能代替上表2.997s，
也不能据此断言某一个私有图或最终梯度累加点是唯一瓶颈。下一次有依据的审查方向是
Read/Full/Aggregate私有VJP构造、字段拆装和通用调度/返回复制，而非继续只增加投影BMM数。

profile原trace约739.66MiB，进程历史RSS峰值约14.51GiB，整个独立采集/解析/核对流程230.49s；
普通D64测量阶段结束时历史RSS峰值约781.87→776.88MiB，不能用profile峰值冒充训练峰值。
NPU测量阶段peak allocated为39255040→39651328字节（+396288），reserved峰值增加2MiB，
结束live allocated同为1669120字节。本次未证明内存收益；局部8MiB packing预算不是全图HBM上限。

D64静态计划hash为`6c8cc36ae97287133d40e67ccc8e137574b80e2df420feff6adfef3395397bb3`，
480节点/2208边、B1/每窗1位置、9076975主体参数，对照core-3 Add主体9076928，差+47。
匹配使用真实私有FFN宽4/5；无虚设参数或任务头。相同参数预算不等于相同函数、激活量或训练工作量。
原D2048没有为本次接续重跑；其FP32失败和既有CPU FP64推理成本保持原范围。

本轮没有观察到明确的端到端加速：中型CPU单次略慢，D64约持平，NPU单次略慢。
这确认了新矩阵批的实际覆盖与正确性，同时留下性能负结果，不宣称“低垂果实已处理完”。
这些中型完整返回任务仍倾向CPU；D64显式FP64为已验证可用配置。此判断不推广到大型设备训练、
不同loss/返回模式或跨卡吞吐，也不把相同参数量解释成与core-3相同函数或同训练工作量。
按约定停止扩大本轮性能矩阵，不重复抽样寻找有利结果。

## 终态、复现与边界

开发原构建+测试在1170秒超时failed/124；测试专用任务235 passed/2旧strict xfail/2旧断言失败，
修正任务50 passed。SDK第一次在配置阶段缺少Ninja而failed/1，未编译/未用设备；新任务仅修正私有工具PATH，
SDK配套Python仍优先，没有修改公共环境。Python/native首次设备任务因driver覆盖模块PYTHONPATH，
CANN初始化找不到tbe，两个eager进程和12个resident测试均未执行模型；修正任务环境后单独验收，
复用前再次核对core/resident内容与全部binary hash。所有原失败与对应archive、manifest、日志保持原终态。

| 作业前缀（unit后缀均为-20261029） | 持久终态 |
| --- | --- |
| core5-independent-dev | failed/124；构建完成、测试未完成 |
| core5-independent-dev-tests | failed/1；235 passed、2 strict xfail、2旧断言失败 |
| core5-independent-fix-dev | passed/0；50项 |
| core5-independent-qualify | passed/0；273 passed、2 strict xfail |
| core5-independent-npu | failed/1；缺Ninja，配置阶段 |
| core5-independent-npu-build | passed/0；新SDK构建、CPU核检查及全部约定设备检查 |
| core5-independent-python-npu | passed/0；纯PyTorch六组 |
| core5-independent-pynative-npu | failed/1；构建完成，设备初始化缺tbe |
| core5-independent-pynative-device-fixed | passed/0；六组eager与12项公共常驻；记录目录名省略-fixed |
| core5-independent-study-cpu | passed/0；计划、四项计时、两步恢复、独立profile共7个stage |
| core5-independent-study-npu | passed/0；前后两项完整计时 |

全部约定作业已终态，未留下排队或运行任务。上述结果不解除D64/D2048及反馈FP32、
eager FP16 GDN/default AdamW、默认异步双NPU的原限制；后续触发条件在[ROADMAP](../ROADMAP.md)。

从干净固定提交按[开发入口](../development.md)构建；完整CPU测试清单与运行参数见JSON。
下面是公共入口，输出目录必须为新目录，不依赖旧worktree或私有driver：

```sh
TIDE_BUILD_DIR=CPU_CORE python -m pytest -q tests/test_weighted_independent_preparation.py \
  tests/test_weighted_kv_sharing.py tests/test_weighted_compute_sharing.py \
  tests/test_cli.py::test_cpp_custom_state_kernel --dtype both
python scripts/check_weighted_device.py --device npu:0 --implementation python \
  --placement mixed-c --dtype float32 --scoring-dtype float32 \
  --case kv-fc --case kv-hb --case ports-sparse --output-dir NEW_CHECKS
python scripts/build.py --backend npu --npu-runtime python --build-dir NPU_CORE --jobs 2
python scripts/build_device_control.py --core-build NPU_CORE --build-dir NPU_RESIDENT \
  --python-component weighted --ascendc-soc TARGET_SOC --jobs 2
TIDE_BUILD_DIR=NPU_CORE TIDE_RESIDENT_LIBRARY=NPU_RESIDENT TIDE_RESIDENT_DEVICE=npu:0 \
  python -m pytest -q tests/test_resident_weighted_flow.py::test_public_full_records_raw_checkpoint_and_private_windows \
  tests/test_resident_weighted_training.py::test_python_owned_complete_cut_updates_and_disk_resume -k kv
python scripts/prepare_weighted_workload.py --topology ranked --layers 15 --region-width 32 \
  --width 64 --samples 1 --positions 1 --match-core3 add --output-dir NEW_PLAN
python scripts/benchmark_weighted.py --device cpu --implementation native --native-library CPU_CORE \
  --dtype float64 --workload NEW_PLAN/workload.json --steps 1 --output-dir NEW_TIMING
python scripts/diagnose_weighted_workload.py --workload NEW_PLAN/workload.json --dtype float64 \
  --implementation native --native-library CPU_CORE --steps 2 --checkpoint-check --output-dir NEW_RESUME
```

native NPU检查将`--implementation`改为`native`并明确对应的`--native-library`；half检查显式
`--dtype float16 --scoring-dtype float32`。standalone SDK须使用配套环境，构建`--npu-runtime standalone`，
其`weighted-flow/weighted-training/weighted-node-training`由`build_device_control.py --checks ...`与
`verify_device_control.py`验收。上述指令是workload；长任务须遵守[有限资源/固定源码](../execution-flows.md)。

所有长任务使用独立background.slice/Nice10，ATen/BLAS1、build2；CPU资格10分钟/24GiB，
两套NPU构建60/75分钟、各32GiB，队列180秒、priority0。CPU测量25分钟、逐进程RLIMIT_AS32GiB，
NPU测量队列180秒/设备阶段660秒；实际终态见JSON。systemd属性不表示独占硬件。
CUDA、x86_64、大型设备完整训练和跨卡吞吐仍未由本轮取得资格；默认异步双NPU旧运行时失败仍独立保留。
