# LH/HB 同材料三层成本（2026-11-12）

固定源码 `c83a2820149e69c3a0571dc61765836173bfaea1` 的 176 节点代表已经完成
CPU 12 点、NPU 10 点。主要差距在公共张量计划内部就已出现：完整训练的直接公式／公共
native／默认 native，CPU1 为 **0.880／2.964／3.063 秒**，NPU 为
**0.608／7.123／7.496 秒**。这支持继续检查节点维度合批、区域候选布局及其反向，
不能把全部差额归为默认入口包装，也不能据此断言数学要求必然造成该差距。

本报告记录重写前的一个有限比较边界；不包含后继节点组、区域组优化的收益。
定义域和职责见[独立 LH 张量基线](../lh-tensor-baseline.md)，
当前计划与限制仍由 [ROADMAP](../ROADMAP.md) 管理。

## 固定来源与实际执行

- 数学：`0770593838a37c790a568226930567ed60f99b83`，core-5 带权 KV 和区域分配。
- 22 个 `result.json` 均记录上述 `c83a2820`、`dirty=false`、`execution=completed`；
  逐项退出码均为 0，不能仅以外层作业 passed 代替这项核对。
- native 二进制来自 `129bbbdce4aea56367561f64ba5ec76f6e370a0c`。
  `git diff --exit-code 129bbbdc c83a2820 -- cpp CMakeLists.txt cmake` 无差异；
  在固定 c83 检出按 `scripts/build_identity.py` 重算源码 hash，并实际重算 `.so`、`.a` hash，
  均与 manifest 一致。13 个 native 测量点携带同一 manifest，没有把旧构建默认为新源码构建。
- 构建：aarch64、Release、GCC 10.3.1、C++17、ABI1、320 build jobs，
  backend=npu、runtime=python；Python 3.12.13、Torch 2.12.0+cpu、TorchNPU 2.12.0.post3、
  CANN 9.2.0。CPU 点也使用这套 Python-owned NPU 构建执行 CPU Tensor，
  不是另一次 standalone CPU 资格。

| 身份项 | SHA-256 |
| --- | --- |
| C++/CMake 源码 | `37c39a9c0ca8ca320519c3be69c4c1daa2179ca8ebb025febbd2669efef94089` |
| `_tide_native.so` | `ba08b9cd4a9a93a7bb99bfc54613479c4ec22a0b28230b89f262eb17e0dc7189` |
| `libtidegraph.a` | `96761849c559cc96c9285568217905b98db04b3e700ac28bb784b230a37521e3` |
| 展开 workload | `5539e0fdb716b75d1003b8d200f7c3f68cafde515503d01a62ab2b3b1c8025d7` |
| 初始模型张量，按公共角色名及值散列 | `939fbb108327068ae9c9230e5f70d926c9e4bc4af7a55f6932667d6d553aa57f` |
| 输入数值和物理端口支持 | `6fb027600264ffb28fbd08cf4daeb3d76495690f636d6ad50ea31bdc4f8022fc` |

三项材料 hash 在全部 22 点完全一致。可训练参数 **14,902,694**，冻结参数 2,132；
按实际别名去重的初态重置快照 62,730,984 bytes。

CPU 实际 affinity=0–63，主表 ATen intra-op=1，补点为32；NPU 主机 affinity=144–159，
ATen intra-op=2；所有点 inter-op=1，启动器的 OMP/MKL/OpenBLAS 环境值均为1。
这些是实际线程设置，不是仓库构建/测试上限。NPU 经空闲卡准入，物理11映射 `npu:0`，
未设置 `ASCEND_LAUNCH_BLOCKING`；FP32 IEEE 策略禁用 matmul/conv HF32，CPU matmul 为 highest。
两个 LH 服务与其他设备上的 REQ-006 诊断并行，CPU 池分开，正式计时期间本任务重型构建暂停；
共享主机、内存及外部争用未被排除，因此本次单点不是统计稳定或独占最优结论。

## 代表材料与计时职责

`--case many-lh` 是 core-5 LH 启发的 HB 扩散、收拢及 skip/cross 图，
不是原 LH Add/Attention 函数。176 个计算节点，22 个区域，每区8节点；7个 rank 的区域数为
`1,2,4,8,4,2,1`，1,058 条内部物理边，输入/输出各8端口，逻辑时间 stride=9。
编码后的虚拟边界节点数不作为本表计算节点数。

| 局部部分 | 本次配置 |
| --- | --- |
| 公共尺寸 | d64、H1、FFN256、summary32、adapter rank32、source hidden32 |
| 来源绑定 | region 分类；同类不同原子分别计入，alpha_agg=1 |
| 扩散56节点 | KV qry；Full RoPE64、Read RoPE32；window4608 |
| 中间64节点 | KV sum；无 RoPE；window0 |
| 收拢56节点 | KV sum；Full RoPE32；window4608 |
| KV 时序 | beta_short=log(2)/9、position_scale=9、origin 按 rank、distance slopes=0 |
| 全部区域 | Read+、local cat、regional_total/context/cat/quad、温度1、gamma_max2、k1 |
| 区域评分 | 扩散/收拢14区域 att/content_stats、selection width32、query_weight=received；中间8区域 ind |

B4/L16 表示4个样本、每样本最多16个全局输入位置；`staggered-v1` 的样本长度为
16、15、14、16，还按物理端口省略部分输入。实际非空 sample-position 共 **61**，
吞吐统一为 `61 / measured_seconds`，不以64或内部节点/边事件数作分子。
输入 seed=7、单位发送强度、共同正弦内容、空 KV 左态，无下游任务头。
名义 k/区域宽度=1/8 是发送强度预算，不是硬 Top-k 或实际 Full 活动比例。

| 层级 | 执行与返回职责 |
| --- | --- |
| direct | 独立 Python 张量公式 `independent-static-kv-dense-full-v1`；静态区域顺序，区域内 `[node,B,L,…]`，跨区域 gather/incidence；dense Full 后 mask，保留 padded raw 状态 |
| public | Python 调用 native `SettleColumnPlan.run_ports`；实际候选和 selected Full，公共节点/消息/区域列及终态 |
| default | Python 调用 native `EagerExecutor.run_ports`；实际 `tensor-window-v1 / settle-columns-v1`，`roots=columns`，`fallback_reason=null`，workspace 上限4GiB；默认能力分派和 continuation |

另测默认入口的纯 PyTorch training。NPU 的 public/default 是 **mixed-c**：
payload、Read、control、selection 在 NPU，事件推进在 CPU；不是 resident、standalone
或多卡结果。direct 也由主机遍历静态区域并调用 NPU 张量算子，不称为设备常驻调度。

每点独立进程，恢复相同模型、清空 SGD 槽后一次 warmup，再重置并测一次；
单步训练采用 `SGD(lr=1e-5,momentum=.3)`，loss 是逐物理输出端口
`1e-6 * sum((cG)^2)`。计划的 `memory_floor` 容量估算标签不改变这里实际使用的 SGD。
完整区间包括输入生成/上传、packing、前向、输出投影/loss、有限检查、实际反向、SGD、
结果观察/清理及最终同步；构造/编译、初态 hash、重置和 warmup 在计时外。
无梯度推理不构图；grad-forward 构图但不反向、不更新。没有测生成 token 或 TTFT。

## 完整成本与实际工作

每格为 **秒（输入位置/秒）**，CPU 主表1线程，NPU2主机线程。

| 设备／模式 | 直接公式 Python | 公共 native | 默认 native |
| --- | ---: | ---: | ---: |
| CPU 无梯度推理 | 0.211718（288.119） | 1.232480（49.494） | 1.304207（46.772） |
| CPU 含 grad 前向 | 0.249200（244.784） | 1.466995（41.582） | 1.540591（39.595） |
| CPU 完整训练 | 0.879711（69.341） | 2.963964（20.581） | 3.062758（19.917） |
| NPU 无梯度推理 | 0.176198（346.202） | 3.444114（17.711） | 3.828965（15.931） |
| NPU 含 grad 前向 | 0.204018（298.993） | 4.017925（15.182） | 4.214591（14.474） |
| NPU 完整训练 | 0.607549（100.403） | 7.123101（8.564） | 7.495768（8.138） |

| 补充训练点 | 秒 | 输入位置/秒 |
| --- | ---: | ---: |
| CPU1 默认纯 PyTorch | 4.478691 | 13.620 |
| NPU2 默认纯 PyTorch | 8.972462 | 6.799 |
| CPU32 直接公式 | 0.983061 | 62.051 |
| CPU32 默认 native | 3.882528 | 15.711 |

CPU32 两点均慢于相同候选的 CPU1；这个 d64 小矩阵代表没有因增线程获益，
不推导宽节点、1B/9B或其他 CPU 的最佳线程数。Python native 快于默认纯 PyTorch，
但把调度层切到 C++ 本身仍不能解释或消除与直接公式的差距。

全部测量点的候选数 **8,368**、正支持选中数 **2,572**、内部边消息数 **15,340** 一致。
选中/候选=30.736%；实际 physical Full 则直接为 **11,264=176×4×16**，公共/默认为2,572。
直接公式计算约4.38倍 Full行仍更快，提示较粗张量粒度值得采用；这些计数不是 FLOPs，
也不能证明 dense Full 对任何稀疏度、规模都划算。公共/默认实际各176次 state column 调用、
307次 region column 调用；调用数不是算子或设备 kernel 数。

NPU allocator 的进程高水位包含构造和 warmup，没有在正式测量前清零：
直接推理 allocated/reserved 为110.710/148 MiB，公共/默认为119.720/148 MiB；
直接训练为311.916/346 MiB，公共/默认 native 为251.092/280 MiB。
直接候选更快同时保存更多 dense 激活；该权衡与省略观察职责必须一并考虑。

## 差额能说明什么

以下是完整计时区间的实测相减，不是每项开销的因果分解，也不是严格同工作加速比。

| 设备／模式 | 公共减直接，秒 | 默认减公共，秒 |
| --- | ---: | ---: |
| CPU 无梯度推理 | 1.020761 | 0.071727 |
| CPU 含 grad 前向 | 1.217796 | 0.073596 |
| CPU 完整训练 | 2.084253 | 0.098794 |
| NPU 无梯度推理 | 3.267916 | 0.384851 |
| NPU 含 grad 前向 | 3.813907 | 0.196665 |
| NPU 完整训练 | 6.515552 | 0.372667 |

训练总差距中，CPU95.5%、NPU94.6%在进入默认包装之前已存在。
默认减公共仍包含不同 continuation、检查/发布和构造调用职责，不能全称为纯调度时间。
共同入口不是唯一瓶颈；仅替换最外层调度不足以达到直接公式的成本。

CPU1 的训练阶段主机计时如下；CPU 同步执行使它适合定位阶段差额，但它不是算子 trace。

| 层级 | forward | backward | finite gradients | optimizer | finite parameters | 其余完整区间 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 直接 | 0.245726 | 0.326203 | 0.173189 | 0.052549 | 0.066817 | 0.015227 |
| 公共 native | 1.403080 | 1.306486 | 0.070409 | 0.052320 | 0.068697 | 0.062971 |
| 默认 native | 1.489721 | 1.298973 | 0.071845 | 0.053274 | 0.070460 | 0.078485 |

公共减直接的 forward 为1.157354秒、backward为0.980284秒；其余阶段合计反而少0.053385秒。
这把主要差额限定到公共前后向内部；不能只指责 optimizer，也不能把 backward 全归为 autograd
引擎成本。本矩阵未开 `--phase-timing`，NPU 的 `host_submission_phase_seconds`
混合主机提交与等待，**不能**据它给设备 kernel 分摊耗时；本轮 LH 没有新 kernel trace 或图计数。

固定源码给出的可复查结构线索是：

- [独立公式](../../scripts/lh_tensor_baseline.py)的 `StaticKV._aggregate/__call__` 与
  [project/describe/select/full](../../scripts/lh_tensor_math.py)跨节点保留整块，实际同参数且同输入才
  共享来源变换；`project` 对不同矩阵用 bmm。独立 quad 求解枚举 O(N²) 仿射区间，
  其分支/伴随也与生产求解器不同，不能把算法差额一律当布局成本。
- c83 的 [native Settle 列执行](../../cpp/src/weighted_settle_columns.cpp)已按节点合并 B×L，
  但在每个节点分别 Aggregate、prepare、Read、Full；区域又按**完整候选成员列表**分组。
  不同成员集合产生307个区域块，而图只有22个区域。这是节点/候选模式维度的剩余分拆，
  不能把现有路径再描述为纯逐 token decode。
- 公共列还发布节点描述、状态/消息与独立列观察；直接候选声明输出与 raw 终态的数值投影，
  dense mask 的梯度连接域也不同。其额外职责、布局、求解和图管理尚未逐项隔离计时。

因此后续可在保留实际成员、稳定平票、结构梯度与私有状态的前提下，
把相容节点投影合批、把区域布局从“成员集合键”改为带真实成员索引的相容块，再用同一 runner
验证实际收益。这里尚未测这种修改，不用本报告替后继源码背书。

## 正确性证据与范围

固定 c83 CPU 作业的 JUnit **26通过、0失败、0跳过**：
`test_lh_tensor_baseline.py` 16项、`test_req006_diagnostics.py` 10项，均使用 `--dtype both`。
LH 的16项是 FP32/FP64 的独立标量调度对照和手算：输出/控制/活动、raw字段/时钟、共享参数梯度、
非空独立左态、重复边同类原子、跨窗保留图、超过2^53的 int64 时间、静默窗口，
以及三步 AdamW、磁盘读回新 owner 后的精确继续。baseline检查的梯度比较取 None/零的数值投影；
它不认证逐事件独立根或任意 optimizer 下两种连接策略的长期轨迹相同。

22个性能点全部将 `strict_numerical`、`restore` 标为 **not_checked_at_measured_scale**。
CPU 直接／公共 loss 为0.073713399470／0.073713317513，NPU 为
0.073713734746／0.073713652790；同设备直接与公共差约8.20e-8，public/default完全同 loss。
有限值检查、相近标量 loss、相同计数不能代替完整输出/支持/状态/全部梯度或精确恢复资格。
本报告没有新增176节点完整尺度等价、原LH速度比、1B/9B、standalone、resident、多卡或GPU结论。

## 从干净源码复现

采用新的固定 c83 检出，按[开发入口](../development.md)准备匹配的私有环境；
本机本次栈为 `ascend/dev-workspace-9.2.0`。长构建/矩阵放入有界 `background.slice`
服务，固定源码及资源，并按当前本机规则分配空闲 NPU。下面是 workload 命令，
`build/lh-c83` 和 `artifacts/lh-repeat` 是本次新建目录，不依赖上述旧运行目录：

```sh
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PWD/python:$PWD/tests:$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"
unset ASCEND_LAUNCH_BLOCKING
python scripts/build.py --backend npu --npu-runtime python --jobs all --build-dir build/lh-c83
python -m pytest -q tests/test_lh_tensor_baseline.py tests/test_req006_diagnostics.py --dtype both
python scripts/benchmark_lh_tensor.py --level direct --case many-lh --mode training --device cpu --threads 1 --output-dir artifacts/lh-repeat/cpu-direct
python scripts/benchmark_lh_tensor.py --level public --client native --native-library build/lh-c83 --case many-lh --mode training --device cpu --threads 1 --output-dir artifacts/lh-repeat/cpu-public
python scripts/benchmark_lh_tensor.py --level default --client native --native-library build/lh-c83 --case many-lh --mode training --device cpu --threads 1 --output-dir artifacts/lh-repeat/cpu-default
```

三种模式分别使用 `--mode inference|grad-forward|training`，每点使用新的输出目录；
分配 NPU 后改 `--device npu:0 --threads 2` 即复现同样的 NPU 三层。
默认纯 PyTorch 补点用 `--level default --client python`，CPU32补点用 `--threads 32`。
B4/L16、FP32、staggered-v1 已是该固定 runner 默认值，也可显式传入；
`--plan` 接受自己保存的展开材料，而不是要求找回旧 trace。
纯 CPU 机器可按开发文档改建 `--backend cpu`，那是新的构建环境对照，不能冒称本次栈的重复测量。

原始记录在 ignored `artifacts/rewrite-20261112/`：
`lh-{cpu,npu}-matrix.json` 及各 `lh-{设备}-{层级}-{客户端}-{模式}-t{线程}/result.json`；
CPU 作业 `artifacts/jobs/rewrite-lh-cpu/status.json` 于03:27:09 UTC终态passed，
NPU作业 `artifacts/jobs/rewrite-lh-npu/status.json` 于03:28:38 UTC终态passed，
均从03:25:12 UTC开始。队列终态completed/return_code0，JUnit为 `diagnostic-tests.xml`。
这些路径是历史溯源；本报告、固定源码、生成器与上述命令已保留复现需要的配置和结果。
