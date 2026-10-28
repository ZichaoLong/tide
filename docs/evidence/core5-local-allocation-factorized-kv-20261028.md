# 局部分配坐标、eager KV 低秩与有限复核 — 2026-10-28

数值实现固定为 `474f9af32d35ca7d8bc7bf6359dbddb3ec91dfb3`，包含
`56631eb9c48fc006d0df00b3a259aee10e04314b` 的分配修复与失败捕获。
数学仍为 `0770593838a37c790a568226930567ed60f99b83`，图 v13/checkpoint v5 不变。
[同名 JSON](core5-local-allocation-factorized-kv-20261028.json)保存来源、材料、逐项终态、
独立诊断与测量。原始失败保持原记录；本报告不把局部通过扩大成全部大图资格。

## 实现与独立数学证据

[分配契约](../weighted-allocation.md)对应固定区域分配 §4.1/§6/附录 S.2。
原 wide2048 和 matched2048 的实际拒绝行已提交到 `tests/data/quad-*-failure.json`，
包括原 dtype、完整评分/上限/总量/温度、实际响应、残差、区域节点及 int64 时间。
公共 `AllocationFailure` 在 Python/C++ 批量分配失败时保存首个实际失败行；benchmark 单独导出。
失败快照不是部分成功返回，前置域拒绝、scalar 与常驻失败尚不承诺同一快照接口。

独立标准库 120 位 Decimal 从捕获的二进制浮点值出发，以扩张区间/二分求解，
不调用生产求解器。wide 原控制第二项为 .7806396484375，独立解为 .7806066274642944，
误差约 3.3021e-5；matched 原第二项为 .6962890625，独立解为 .6964603960514069，
误差约 1.71334e-4。两份输入均有可行解；这两个拒绝与远处饱和评分造成的局部相消有关。

quad 现在按评分降序累计 cap 质量，选择首次达到目标总量 T 的评分 p。
根在 `[p−tau*max(caps), p]` 内，先以 p 为原点；Python 断点扫描只处理该区间的可变项，
C++ 和设备在局部区间求根。全部原始响应、正支持、残差守卫和精确解映射 VJP 保留。
区间在请求 dtype 下塌缩明确拒绝。约束分支先消去共同阈值，避免 `scores-eta` 预先抹去小分差；
松弛分支仍保留其真实依赖，原坐标 multiplier 只是浮点报告。

独立检查包括大分差下单/双自由坐标闭式解及 VJP、两个捕获行的手算解、
不同正则的大共同阈值与连接零导数、域/支持拒绝、参数版本和共享依赖。
入口为 `test_weighted_quad_scale`、`test_weighted_large_shift`、`test_allocation_failure`、
`test_allocation_decimal_diagnostic`；其余完整区域与反向门禁见同名 JSON 的精确测试清单。

[KV §1.2 的实现](../weighted-kv-sharing.md)把所有 eager 标量、独立 packed 行与共享组统一写为
`base@x + U@(V@x)`，避免未共享行反复构造稠密 `U@V`；归一化操作数供 summary 复用。
共同计算仍要求相同参数与输入，状态始终私有；已有私有 BMM 和独立行 VJP 不改变。
`test_weighted_kv_formulas` 用独立 dense 公式/多维字段根，`test_weighted_kv_sharing` 用解析 Jacobian，
配合独立调度、完整梯度、optimizer 和恢复检查。resident 组合矩阵/tape 不在这次优化范围内。

## 固定源码资格

| 范围 | 终态、身份及边界 |
| --- | --- |
| CPU 数值与训练 | 687 passed、2 strict xfail；另 59 项失败快照/Decimal/大共同阈值通过，共 746 passed。包括独立公式、字段根/None/零、别名/版本、完整调度、optimizer/恢复、一般 State 与 typed clock；两个原 FP16 GDN/AdamW 见证不解除 |
| standalone CPU | 上述门禁中 FP32/FP64 各 24 图×4 次更新，精确 checkpoint 恢复；独立 benchmark 重建。复用 core 前重新核验完整 C++ 内容与每个 binary hash；开发 manifest 的 56631eb 基提交不是资格身份 |
| standalone NPU | 新 core/resident 构建，物理 1/13；allocation FP32 与 flow/config/training/node-training 各 FP32/FP16，共 9 项真机通过；含单/双卡完整训练、恢复。额外 SDK eager fc/kv FP32 四次更新/恢复通过 |
| 纯 Python NPU | Torch2.12/CANN9.1，eager KV 的独立根、streaming/prefill、4 次更新与设备→CPU恢复通过；不以 standalone 结果替代 |
| CUDA | 新 core/resident、sm80/90/100 构建、安装 loader/help 与 core CPU 检查通过；无 NVIDIA 设备节点，GPU 真机仍未验 |

资格作业 `core5-factorized-qualify-{cpu-v2,npu-v3,cuda-v3}` 均 exit0。
初批因预建输出目录拒绝、设备 v2 因已有 resident 目录拒绝而失败，全部保留。
开发 job 中的局部回归通过与大图严格失败分别记录；后续资格成功不覆盖旧失败。
这些是与本次改动相关的定向资格，不是重跑所有历史模块的全量矩阵。

## 同材料有限测量

每配置一次新进程、一次 warmup 加一次连续测量窗，不为取得有利结果重试。
训练计时包含输入准备、调度/packing、完整返回、loss、反向、finite checks、SGD、detach 和同步；
静态构造、warmup、事后独立完整核对分列。推理无反向/更新，不导出未更新的参数副本；
验证前释放候选再构造参考。候选不消费参考轨迹，完整返回仍逐项保留并比较。

| 用例 | 结果 | 实际工作与结论 |
| --- | --- | --- |
| CPU standalone HB medium，D32/B4/4位置，102604 参数 | 56631eb→474f9af：366.831→363.974 ms，完整返回/SGD参数均通过 | 128候选/128 Full/192消息；单次约−0.8%，接近持平，不能认定稳定收益 |
| 单卡 NPU Python mixed-c，同一 HB medium | 0503593→474f9af：2189.634→2145.125 ms，完整返回/SGD参数均通过 | 物理13→npu:0；同128/128/192；约−2.0%，包含私有BMM/分配/低秩累计变化，不能单独归因 |
| CPU ranked64，480节点/2208边，9076975 参数 | 训练窗3.132 s已执行，事后独立核对失败 | 首个报告为node84 keys，最大差2.26498e-6；479候选/344 Full/1604消息，不计等价吞吐 |
| CPU matched2048，同拓扑、9262006511 主体参数 | 构造184.376 s、warmup130.412 s、推理窗130.351 s已执行；事后独立核对失败 | node0 keys最大差4.48227e-5；424候选/199 Full/946消息，不再在该窗被分配残差拒绝，但仍无大模型等价吞吐资格 |

matched2048 仍匹配旧 core-3 Add 主体 9262008512，差−2001（约−2.16e-7相对误差），
使用实际FFN宽319/320，B1/1位置推理；二进制材料37048052160字节。
这不是旧B512/12位置完整训练复刻，更不等于两个数学模型函数或训练成本相同。
该进程测量后报告生命周期RSS峰值50671738880字节（约47.19 GiB）；不是大型训练或设备峰值。
wide2048 在开发检查中也越过分配拒绝，完成1.17269 s窗口后因keys最大差4.95911e-5失败。

CPU study 总体 failed/exit1（五项通过、两项严格失败），NPU study passed/exit0（两项通过）。
NPU构造allocated两侧均538624字节，测量peak allocated由38882304变为39255040字节，
reserved峰值均41943040字节；没有获得峰值内存下降。外部共享负载未隔离，项目重型测量串行，
没有统计稳定或普遍提速结论。历史各次变慢仍保留于[性能经验](../performance-lessons.md)。

## 剩余数值分歧与 profile

同一 D64 两步实际调用中，30组 Python scalar/batch 分配的分支和 free 集合一致；
分别以其实际输入独立求解，最大局部误差8.33111e-8。
两条路径的评分在进入分配器之前已经不同，最大差5.21183e-4，不能继续全部归因于迭代残差。
完整 FP32 两步诊断保留返回、梯度及SGD参数的严格失败；首步共享base梯度最大差
约6.53267e-4（Python）/7.48873e-4（native）。离散结构一致不能代替完整训练一致。

同一FP32种子材料在两侧开始前显式提升FP64后，Python及native各自两条独立调度的两步
完整记录/梯度/参数均通过。它是有限精度敏感性诊断，不是默认FP32修复。
实际wide2048操作数的重复行BMM探针仍有2.86102e-6差异；这只是局部算术证据，
不是实际图批、吞吐测量或整图差异的完整因果解释。

独立 HB medium CPU profile 完整核对通过；1394次mm、288次bmm、14545次copy、14760次select。
self CPU事件总计约1.159 s，mm约42.15 ms、bmm约2.59 ms，forward作用域自身约95.18 ms；
profile采集/分析与非profile计时分开。trace121039409字节（约115.43 MiB）、进程RSS峰值约2.55 GiB，
比之前大型profile更适合有限诊断，但不能与不同图/不同粒度的旧profile直接推算加速。
ranked64本轮仍仅34/479次State进入共享准备的私有投影批；最大批32，State最大批1。
跨不同base/输入的同形合批、设备小算子提交、物理缺席行及窗口反向程序构造仍有空间。

## 测量阶段记录

设施提交 `b62f2db7e9bd72b652cdcb97b11fa6e1317b396a` 仅修正 standalone 进度持久化：
在fixture读取、构造、warmup、每个测量窗及验证前写phase/window，全部持久写入在计时外；
成功终态写complete。FIFO中断反例验证强制终止后仍能读取最后阶段，外层job拥有退出终态。
开发与固定提交的独立构建/6项检查均终态passed/exit0；JUnit、源码/binary hash和终态在同名JSON，
其范围为CPU FP32消费者设施，不替代474f9af数值或设备资格。
本次CPU/NPU性能均运行于474f9af，没有把新记录设施混入前后测量。

## 公共复现与资源

按[开发入口](../development.md)从上述固定提交构建core。局部独立检查无需旧运行目录：

```sh
python scripts/diagnose_allocation_failure.py --failure tests/data/quad-wide2048-failure.json --output NEW_WIDE.json
python scripts/diagnose_allocation_failure.py --failure tests/data/quad-matched2048-failure.json --output NEW_MATCHED.json
TIDE_BUILD_DIR=CPU_CORE python -m pytest -q tests/test_weighted_quad_scale.py \
  tests/test_weighted_large_shift.py tests/test_allocation_failure.py \
  tests/test_allocation_decimal_diagnostic.py tests/test_weighted_kv_formulas.py \
  tests/test_weighted_kv_sharing.py tests/test_cpp_weighted_training.py --dtype both
python scripts/build_weighted_benchmark.py --core-build CPU_CORE --output-dir NEW_BUILD
TIDE_WEIGHTED_BENCHMARK=NEW_BUILD/client/weighted-benchmark \
  python -m pytest -q tests/test_weighted_benchmark_material.py
python scripts/benchmark_weighted.py --device npu:0 --implementation python \
  --placement mixed-c --topology hb --size medium --steps 1 --output-dir NEW_MIXED
```

可扩展拓扑/参数预算/二进制材料及两步dtype诊断入口见[workloads](../weighted-workloads.md)；
具体计划参数和材料hash见同名JSON。性能比较分别固定前后源码，读取同一材料；
新clone可重建材料，不依赖旧worktree。设备资格入口/组件列表见[设备契约](../device-control.md)。

长任务均为固定源码的background.slice/Nice10服务，算子线程1、build2。
CPU资格30min，NPU/CUDA资格90min（NPU队列180s、设备1800s）；CPU测量总40min，
一般进程32GiB地址空间/360s，大模型128GiB/1200s，RLIMIT_AS实际应用于子进程。
NPU测量总20min、队列180s、设备900s、每配置240s、MemoryMax32G；阶段记录门禁各15min/32G。
系统服务属性不证明独占CPU/内存/设备。Torch CPU/Python为2.12，standalone NPU为2.10/CANN9.0，
Python mixed为Torch-NPU2.12/CANN9.1；所有有效配置/命令/终态在证据JSON保留。

原eager FP16 GDN默认AdamW、默认异步双NPU VJP、SDK FP32 affine和反馈PDG严格见证保留。
CUDA真机、x86_64、大规模设备完整训练/跨卡吞吐仍未验证；有限容量和正支持数值拒绝继续生效。
本轮完成实际失败定位、可证明的数值/低秩修复、相关资格及有限选型，不宣称全数值域覆盖、
大图严格资格或低成本优化已经穷尽。
