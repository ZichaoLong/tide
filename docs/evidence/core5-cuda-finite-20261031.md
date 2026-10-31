# RTX3090 时间线与有限完整训练对照 — 2026-10-31

[逐配置终态、原始三窗口标量、资源、命令和hash](core5-cuda-finite-20261031.json)。
实现/构建固定 `32edb0f2b035ef2b732b6f27cf577f9e98058eb6`，目录配置固定
`a864fef414266db9ae348d0b4bf2cfbf7b093e48`；数学固定
`0770593838a37c790a568226930567ed60f99b83`。环境、四阶段构建与独立设备资格分别见
[eager](core5-cuda-eager-20261031.md)和[resident](core5-cuda-resident-20261031.md)报告。
本次不继承旧CPU/NPU数值结论，也不修改原输入、容差、精度、checkpoint或图版本。

首轮37项全部终止：18项完整通过、19项失败，外层作业failed/exit1且无存活后代。
18个失败为完整记录的严格浮点差异；另一个为LH medium resident的容量拒绝。
每个通过项同时满足完整返回/状态/history/pending、最终参数梯度、SGD动量槽/masters和磁盘新owner精确下一窗。
失败候选的计时只保留为成本观察，不授予等价吞吐资格。小型单卡成功不解除其他训练边界。
随后两个小型FC scalar补充均完整通过，故有限成本记录共39项、20项合格；
另一次queue452容量诊断仍失败，不混入吞吐样本。所有作业均终态，无存活后代。

## 配置与计时

同一CUDA/Torch私有环境，物理GPU0，FP32 payload/scoring、TF32关闭，ATen/BLAS/interop/native workers均1。
SGD使用原lr=.001、momentum=.3、weight_decay=.01；一warmup后连续三个测量窗口，B1、每窗两个位置、seed7。
完整同步计时包含输入准备/上传、调度、packing、forward/loss、backward/finite检查、更新/detach和完整CPU返回。
构造、warmup、独立核对及精确恢复均单列。不是裸kernel或仅forward时间。

| 模型 | 宽度 / 区域宽度 | 实际主体节点 / 物理边 | trainable参数 |
| --- | --- | --- | --- |
| FC small / medium | 8/2；64/4 | 4/4；8/16 | 8,190；651,946 |
| HB small / medium | 8/2；64/4 | 20/48；40/96 | 41,030；3,448,274 |
| LH启发 small / medium | 8/2；64/4 | 20/64；40/222 | 41,158；3,449,234 |

全部用KV自然维度，depth2；small为baseline，HB/LH medium为textbook配置，实际参数/来源分类/发送配置在JSON的models中。
HB保留同秩节点，LH使用明确的额外长边；这不是原始LH实现性能或约1B/8B/16B的扫描。
内存/材料准入上限及每阶段allocator峰值都保留；静态下界不是训练峰值保证。
项目重型构建与GPU profile先终态，再运行串行测量。离线单CPU SQLite分析曾与后段HB/LH重叠；
外部负载未隔离。每配置仅一个进程，三个连续状态窗口不是独立重复试验，不宣称稳定倍率。

## 通过项的完整窗口成本

单位ms，为三个连续测量窗口的中位数；JSON保留每个窗口、最小/最大值、实际工作及内存。
空白表示未选择，失败保持失败。

| 客户端 / 执行 | FC small | HB small | LH small |
| --- | ---: | ---: | ---: |
| PyTorch CPU packed | 56.936 | 312.581 | 303.634 |
| Python native CPU packed | 37.267 | — | — |
| standalone CPU packed | 35.960 | — | — |
| PyTorch CUDA mixed-c packed | 146.745 | 612.096 | 669.956 |
| Python native CUDA mixed-c packed | 138.775 | 540.777 | 583.699 |
| standalone CUDA mixed-c packed | 168.463 | 598.005 | 697.881 |
| Python native CUDA resident | 235.470 | 失败 | 3,159.071 |
| standalone CUDA resident | 283.294 | 失败 | 3,148.093 |
| Python native CUDA mixed-c scalar（补充） | 133.620 | — | — |
| standalone CUDA mixed-c scalar（补充） | 133.508 | — | — |

每窗FC有8候选/8 Full/8消息，HB有40/40/96，LH有40/40/128；同输入不同客户端的这些工作量一致。
resident FC实际12条物理局部行，LH120条，均多于实际作用；窗口分别有3/6个resident阶段。
standalone resident的测量allocator峰值分别约24.1/101.5MB，初始构造约106/793ms；
驱动/厂商额外分配不在allocator指标内。初始构造被排除，不表示每个训练窗口没有反向程序构造。
这些小图未观察到GPU相对CPU的收益；不能据此判定更大批量或已发生严格失败的medium配置谁更快。
scalar补充与原packed使用完全相同的模型/输入/窗口，完整核对及同栈恢复均通过，三个窗口分别为
native135.966/124.576/133.620ms、standalone133.508/133.224/133.946ms。
本次观察没有显示小型FC的packing收益；单次非交错流程不足以改变默认策略或宣称稳定加速。

## 保留的严格边界

FC medium七种选择全部失败：CPU packed的node4 keys差1.31e-6；两Python CUDA客户端在node5 values差1.55e-6，
standalone同字段差1.67e-6；resident在node4 keys差1.25e-6。显式scalar也失败，
Python native在trace content差1.46e-6，standalone在output差2.86e-6。没有用scalar或提高容差冒充修复。

HB small resident两客户端均在node18 values同一坐标失败，差1.1175871e-6；对应eager完整通过。
HB medium五项和LH medium四项eager/CPU比较分别在完整keys/values失败，具体字段、坐标、阈值在JSON。
这些是本次材料的实际失败，不由旧aarch64/NPU失败推断；也没有定位到相同局部算术原因的证据。

LH medium standalone resident在原queue64下完成warmup和三个更新，写出checkpoint后报
`weighted resident transaction refused, device error=1`。队列proposal与broadcast输出容量使用该错误码；
原日志阶段为validation-preparation，完整下一窗恢复和独立参考尚未通过。已有测量不构成成功。
显式queue452的单次复验保持其他预算不变：452=B1×P2×(222物理边+4输入端口)，
由Settle每节点/位置最多一次且窗口完整结束推导，没有CPU轨迹预跑。
容量拒绝消失，实际磁盘新owner下一窗记录/梯度/SGD/master逐值通过；随后独立第0窗在node12 keys差2.26498e-6失败。
原queue64失败保留，增加队列只解除容量障碍；它不认证独立数值或等价吞吐，也不是总HBM不足结论。

## 实际resident时间线

Nsight Systems2025.3.1使用`--trace=cuda,nvtx --sample=none --cpuctxsw=none --cuda-graph-trace=node`。
控制例、Python调用resident和standalone resident各独立进程；两个消费者使用同一FC small材料，
各完整核对/梯度/SGD/master/精确恢复通过。profile包含构造、warmup、核对及checkpoint，不能作为吞吐样本。

| trace | 所有CUDA kernels | graph-node kernels | 不同执行node ID | 顶层graph launches |
| --- | ---: | ---: | ---: | ---: |
| control | 238 | 192 | 11 | 8 |
| standalone resident | 94,037 | 68,540 | 8,170 | 16 |
| Python resident | 92,404 | 68,540 | 8,170 | 16 |

真实graph-node活动按时间出现`queue_propose → closure → ready_pack → weighted_node_pack →
weighted_region_pack/allocate/result → broadcast_route → queue_propose → owner_adopt → closure`，
同一closure node ID在提交状态后再次执行；两客户端都观察到reverse_route和optimizer_commit。
对应[在线循环](../../tools/device_online/weighted_flow.cpp)从实际队列产生branch，
[CUDA控制](../../tools/device_online/cuda/program.cpp)把它降为设备WHILE/SWITCH；CPU没有提供事件轨迹。
该源码和trace共同支持所测完整窗口内就绪、选择、合批、推进在设备上的结论。
主机构造、边界上传/完整解码、loss根和顶层提交仍存在，不声称整个进程无CPU或无同步。

条件graph活动在本次导出中没有可连接到runtime launch的correlation；保留这一缺口，未按最近时间猜归属。
graph-node kernel覆盖并集约208.6/211.9ms是全profile范围，既非单窗口耗时，也非关键路径归因。
全profile有15/14次graph实例化、16次launch；indexing/elements、row累加和head/分配guard占较多调用，
单凭这些计数不能推出某项融合的端到端收益。

离线观察器固定`601a9e83dac5a5c52a33bc0b539b6561cadab75a`，7项独立SQLite见证通过。
原c96ff85的无索引相关查询55秒超时，reader停止且PID消失；修订为同进程ID集合后约6秒读完原三份trace。
未重跑GPU profile，未修改数据库；重叠区间用并集，未知correlation和非graph活动均保留。

## Eager profile与后续选择依据

两个独立完整训练profile均为heterogeneous-attention、mixed-c FP32、AdamW，warmup一次、采集一次，
每次32候选/31 Full/44消息。PyTorch/native分别有28,634/27,705 kernel事件，设备覆盖47.53/46.31ms，
`_local_scalar_dense`为1,697/1,544次；显式复制和主机标量成本仍存在。
私有Query backward的inclusive host时间约320/294ms，不能与嵌套父事件相加当独占时间。
没有观察到明确标记的host fallback不等于证明所有算子没有回退。

这些观察支持先考察实际私有VJP提交/装配、局部行合批与反向程序模板复用，
不支持优先优化仅12次的closure，或先扩到1B。模板复用须重新绑定实际journal、参数/独立根、版本及窗口，
不能按shape复用旧owner；数值失败仍须先有新的局部算术/伴随方案。当前没有据profile擅自新增融合核或更换通信协议。

## 无旧运行目录的复现

仅clone授权新分支。复演历史结果时在固定32edb0f建立新检出，另取配置提交的JSON；
按[resident报告](core5-cuda-resident-20261031.md#干净检出复现)重建匹配core/resident，
并在私有环境提供nlohmann_json3.11.2。所有输出路径必须是新的，长任务用[持久作业机制](../verification.md)。

```sh
git show a864fef:benchmarks/core5-cuda-finite-v1.json > NEW_STUDY.json
python scripts/build_weighted_benchmark.py --core-build NEW_CORE --resident-build NEW_RESIDENT_CPP --output-dir NEW_BENCH --jobs 56
python scripts/run_core5_study.py --study NEW_STUDY.json --case fc-small-kv --execution native-cuda --native-library NEW_CORE --output-dir NEW_NATIVE
python scripts/run_core5_study.py --study NEW_STUDY.json --case fc-small-kv --execution cpp-resident --standalone-binary NEW_BENCH/client/weighted-benchmark --export-library NEW_CORE --output-dir NEW_CPP
python scripts/run_core5_study.py --study NEW_STUDY.json --case fc-small-kv --execution cpp-cuda-scalar --standalone-binary NEW_BENCH/client/weighted-benchmark --export-library NEW_CORE --output-dir NEW_SCALAR
python -S scripts/index_core5_study.py docs/evidence/core5-cuda-finite-20261031.json --state passed --format json
```

`--case hb-small-kv/lh-small-kv/fc-medium-kv/...`与目录内执行名显式选择；不自动展开全部排列。
失败复现同样使用原目录配置和严格检查。分析器在当前分支，用
`nsys export --type sqlite --output NEW.sqlite TRACE.nsys-rep`及
`python scripts/inspect_cuda_timeline.py --database NEW.sqlite --output NEW_OBSERVATION.json`。
精确复演本报告的standalone profile，先从目录静态导出FC-small workload，再导出同源fixture；没有参考轨迹：

```sh
PYTHONPATH=python python - <<'PY'
import json
from pathlib import Path
from tidegraph.planning.study import case
study = json.loads(Path('NEW_STUDY.json').read_text())
with Path('NEW_WORKLOAD.json').open('x') as stream:
    json.dump(case(study, 'fc-small-kv', 'cpp-resident')['plan'], stream)
PY
python scripts/export_weighted_benchmark.py --workload NEW_WORKLOAD.json --native-library NEW_CORE --binary --output NEW_FIXTURE.json
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --cuda-graph-trace=node --force-overwrite=false --stats=false --output NEW_TRACE NEW_BENCH/client/weighted-benchmark --fixture NEW_FIXTURE.json --device cuda:0 --dtype float32 --scoring-dtype float32 --implementation resident --schedule prefill --mode training --packing packed --steps 1 --checkpoint-check true --output-dir NEW_PROFILE_RUN
```

Python profile把末尾consumer换成`python scripts/benchmark_weighted.py --workload NEW_WORKLOAD.json`
并传同一device/dtype/scoring/implementation/schedule/mode/packing/steps，及`--checkpoint-check
--native-library NEW_CORE --resident-library NEW_RESIDENT_PY --output-dir NEW_PY_PROFILE_RUN`。
两个eager profile用`profile_weighted_device.py --device cuda:0 --implementation python|native --placement mixed-c
--dtype float32 --scoring-dtype float32 --case heterogeneous-attention --output-dir NEW_PROFILE`，native另给匹配库。
这些命令不依赖旧profile或私有driver脚本，profile仍与正式计时分进程。
容量定位使用`--case lh-medium-kv --execution cpp-resident --resident-limits NEW_LIMITS.json`，
文件内容为`{"queue":452}`；它是明确的另一资源配置，原queue64无需该覆盖文件。
当前下一动作仅见[STATUS](../STATUS.md)，停止/重启条件仅见[ROADMAP](../ROADMAP.md)。
