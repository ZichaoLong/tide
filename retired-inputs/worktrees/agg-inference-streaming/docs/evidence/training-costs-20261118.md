# 完整训练成本、实际设备轨迹与 Transformer 有界候选 · 2026-11-18

四份终态作业完成18个正常观察、90个正常计时样本，另有两份实际设备轨迹。
范围是 HB compact-9、188节点、9层、X内存、Settle、B8/L256、BF16；本页不是全规模、全精度或连续训练资格。
结论依据实际记录：区域合批减少调用；Reception局部重算降低保存内存；K=1 SSD摘要可脱离矩阵扫描；
优化后的剩余差距仍含实质数学工作、矩阵扫描、索引与细粒度调度。Transformer本身也有约束明确的改进空间。
全部样本、配置、哈希、实际dtype/shape/device、工作量、阶段和原始文件身份见 [JSON](training-costs-20261118.json)。

数学固定 `4f5b9f20a4030d68ee66ae2d26e34659a5e37234`，本次108文件核验通过。
证据整理基于 `22527cf472fc4d6ea9ca7ce22528b7e4ca9cc96a`；测量源码分别为
旧成本 `d5c7f30ba7e47e4bae774de75bb92dc022542980`、基线候选 `007a382883d4cc9268ae968e40e72d4abe0aa7cb`、
新成本及轨迹 `ae247dee82c60b63495e9abea7f703abd84ac3c7`。测量提交不由整理提交替代。

## 边界、恢复与资源身份

正常样本是两次预热更新后的五次完整AdamW foreach更新，每步私有状态从零开始。
计入zero_grad、准备/packing、forward、必要flush、readout/loss、backward、AdamW、同步/标量与释放；
这组为Settle，实际不执行flush。吞吐是2048个实际输入位置除以完整时间，不用节点、tick或padding替代输入数。
构造、输入传输、预热、CPU anchor快照/恢复/哈希、语义审计、profiling和checkpoint IO均不在正常计时内。
这不是带历史增量状态的多步续接；连续训练、恢复、路径数值资格由对应专项记录承担。

阶段、saved、runtime、work、trace分别执行。每次诊断先精确恢复首个正常样本前的model、gradient、AdamW与RNG；
本页所有已记录诊断的起点哈希匹配，带loss的诊断也等于该行第一个正常样本的loss。
不同Tide实现虽共享初始参数和输入，却在预热后已有不同参数/梯度/优化器哈希与活动轨迹，因此跨实现时间比不是严格同轨迹加速。
同一Transformer类别的三个候选则共用standard的精确预热后anchor，候选自身预热更新被丢弃。

正常峰值为首样本前重置、后审计前读取的累计 **allocated** 峰值，包含持久参数、梯度、优化器和输入。
reserved缓存可跨预热保留。BF16的实际矩阵路径已观察到BF16，参数、梯度、AdamW槽仍为FP32；
新Tide这些持久张量合计约0.234 GiB，十余GiB峰值不能归结为参数持有量。

| 记录 | 物理NPU | CPU affinity / 线程 | 外层时限 / RSS预算 | 终态 |
|---|---:|---|---|---|
| training-cost-settle-base | 4 | 24–27 / 4 | 2400s / 100 GiB | passed |
| training-cost-settle-batch | 14 | 28–31 / 4 | 2400s / 100 GiB | passed |
| training-transformer-strength | 2 | 48–51 / 4 | 2400s / 80 GiB | passed |
| training-cost-candidate-trace | 14 | 48–51 / 4 | 3000s / 128 GiB | passed |

各作业是冻结源码、`background.slice`服务，allocated同步检查预算48 GiB；RSS按进程树采样，不是cgroup硬限制。
设备身份为协作队列独占，外部作业和主机负载不受本项目控制。卡号、时间和主机争用差异保留；同卡也不意味着同时测量。
本页四份记录没有失败pass；这不改变已有数值超差、失败记录或后端资格。

## Tide：正常成本、已尝试的改变与剩余工作

候选定义：serial为region_batch1；batch4仅合并相容区域；组合为batch4、`ssd_summary_backend=prefix`、
`checkpoint_reception=true`，区域整体重算仍关闭。组合只对K=1短/长摘要使用affine prefix，完整SSD矩阵状态扫描未修改。

| 观察 | 正常中位s | 输入位置/s | 正常峰值GiB | saved唯一峰值GiB | 完整训练dispatch调用 |
|---|---:|---:|---:|---:|---:|
| serial | 1.971500 | 1038.80 | 13.5211 | 12.6549 | 94,554 |
| batch4 | 1.532954 | 1335.98 | 13.6518 | 12.7697 | 60,393 |
| batch4 + prefix + Reception重算 | 1.470717 | 1392.52 | 10.2131 | 9.4036 | 59,522 |

batch4/serial正常时间比0.7776、allocated峰值比1.0097。组合/batch4时间比0.9594、峰值比0.7481，
saved唯一峰值比0.7364；组合/serial时间比0.7460。上述是本形状的有限观察，不能归因为严格相同训练轨迹上的纯执行优化。
组合同时改变摘要与重算，本页不把组合差值拆成两者独立收益。

**保存内存定位。** serial的Reception保存站点同时存活storage峰值4.6918 GiB，前向同步阶段仅0.04437s；
它处理每输入位置1392个消息slot，两层NodeLinear各执行2,850,816行。norm、残差、逐边特征和反向保存使该站点值得局部重算。
但4.6918 GiB包含重算仍需的输入及跨站点别名，不是可独占释放量。新组合的外层saved hooks不再看到Reception内部保存，
checkpoint输入改在父站点保留；实际应该看全局saved union与正常allocated下降，而不能把缺失的Reception行写成“内存为零”。
原始参数storage之外的BF16权重副本、输入等仍计入“additional”，该列也不是纯激活。

**前向热点与重算代价。** 下表全部是另一次同步侵入式观察，不能当普通计时份额，也不能把父子inclusive相加。
反向列包含重算，单列的重算行只是其子范围。

| 观察 | 侵入式forward s | 侵入式backward s | SSD观察 | allocation s | messages.gather s |
|---|---:|---:|---|---:|---:|
| serial | 1.24719 | 0.76848 | 旧合并scan 30次 / 0.43936s | 0.14338 | 0.07101 |
| batch4 | 0.99266 | 0.56931 | 旧合并scan 16次 / 0.39133s | 0.09696 | 0.06987 |
| 组合 | 0.86568 | 0.64914 | matrix 8次 / 0.25349s；prefix 8次 / 0.01018s | 0.09314 | 0.06982 |

旧`ssd.scan`把真实矩阵递推和K=1摘要共用路径合并；0.43936s或0.39133s都不能全算作摘要开销。
新组合中，反向Reception重算19次耗时0.13345s，已包含于0.64914s反向；不能再加一次。
局部重算保留大部分前向流程，同时增加反向工作，这解释了内存下降远大于正常时间下降。
新prefix已避免摘要的32×32块传播矩阵；真实矩阵scan仍是可见的前向大项。

**工作量与padding。** 有效参数去重不等于计算量。serial在诊断anchor有384,720/385,024正候选，
270,583次Full；Full占candidate slots约70.28%，不是预算k/n=12.5%的硬活动上限。
K的147,703个selected rows扩为200,704个executed rows。batch4的K调用由32降至22，但executed rows升到205,568；
组合为147,793 selected / 205,888 executed。serial、batch4、组合NodeLinear前向multiply-add分别为122.158、123.383、123.464 G，
包括padding和adapter，尚未计入Attention、普通regional Linear、SSD扫描、归一化和反向。
所以合批收益伴随更多padding，不能描述为数学工作减少；不同anchor的正消息/Full差异也保留在JSON。

## Transformer：四类匹配、真实实现与有界候选

A/B匹配188个节点/局部block，C/D匹配9层深度；A/C匹配15,694,224总参数，B/D匹配实际有效触及均值。
新组合的初态活跃目标为14,467,742.1924，strength原校准为14,468,012.6802。差值改变偏差，**没有改变任何选定构造器**。

| 类别 | blocks | d / FFN / head维度 | 实际参数 | 对新组合参数目标偏差 |
|---|---:|---|---:|---:|
| A | 188 | 80 / 240 / 16 | 15,671,760 | -0.14314% |
| B | 188 | 64 / 312 / 32 | 14,366,272 | -0.70135% |
| C | 9 | 368 / 1088 / 16 | 15,692,624 | -0.01019% |
| D | 9 | 320 / 1248 / 64 | 14,475,200 | +0.05155% |

strength与新成本的baseline、workloads、matching、Attention、precision五个冻结文件SHA逐一相同。
A再次测量的初始参数、输入、model/gradient/AdamW/RNG anchor及五步loss全部相同。
B/C/D的构造器、B/L、精度、种子与实现也相同，因此可复用其已有基线观察并按新目标更新偏差；无需仅为配置文件名称重新运行。
这允许按相同工作负载解释已有结果，不会把NPU2上的测量变为NPU14上的同时对照。
跨不同宽度的输入按feature-wise种子保持公共特征前缀配对，完整输入hash各自保留；Tide与Transformer数学不同，不能比较模型质量。

基线原本已经有融合QKV、融合SwiGLU gate/up、完整B/L/head张量计算、缓存RoPE/窗口mask和native unit-prior Attention。
仅检查两个候选：query chunk128→256（窗口仍128），以及每层在切tile前一次contiguous Q/K/V。
每类别的三者初始参数、输入和训练anchor完全相同；不随计时更新重新匹配。

| 类别 | standard中位ms | chunk256中位ms / 时间比 | contiguous中位ms / 时间比 | chunk256峰值GiB |
|---|---:|---:|---:|---:|
| A | 837.904 | 677.961 / 0.8091 | 847.881 / 1.0119 | 2.0498 |
| B | 839.338 | 660.274 / 0.7867 | 830.810 / 0.9898 | 2.0191 |
| C | 43.536 | 40.168 / 0.9226 | 43.536 / 1.0000 | 0.6553 |
| D | 44.587 | 33.786 / 0.7578 | 42.400 / 0.9509 | 0.6238 |

chunk256把A/B前向Attention调用从376降至188，C/D从18降至9；有效可见对数不变，操作数矩形容量却增加33.68%。
A/B完整训练dispatch从47,977降至39,705；C/D从2,332降至1,936。更多masked工作与更少launch可以同时发生，有限结果支持批量粒度值得检查。
contiguous候选没有清晰的跨类别改善，切片仍可能需要native adapter拷贝，故不继续扩大该候选搜索。

样本range/median完整保留：C chunk256为33.87%，D standard为21.49%，D contiguous为13.45%，D chunk256为7.36%；
其余范围见JSON。C虽有较小中位数，不据此称最佳；本轮停止在这两个有限候选，不额外重测以筛出好看的样本。
这些是性能观察，CPU FP64同函数/全部梯度与续接单元测试不能替代NPU后更新数值轨迹资格。

新组合在同一成本作业内对standard A的正常时间比为1.7891；与strength的chunk256 A/B/C/D作跨卡描述性比较，
时间比分别为2.169、2.227、36.615、43.530。后两项尤其不应理解为“同深度同函数”：C/D把总参数集中在少量宽block，
A/B使用188个窄block。A/C的前向线性multiply-add都约32 G，但层调度粒度和Attention对数不同。
Tide仅NodeLinear部分就约123 G，且额外承担逐边接收、区域分配、记忆摘要/矩阵和packing；这些差异不能仅用参数量解释，
也不能把未计入的其它计算当零。A的基线强度改善还说明旧chunk128并非最优基线，但两个候选不证明全局最优。

## 实际设备轨迹：kernel、主机和重叠分别解读

新成本作业分别导出了Tide与standard A的原始Chrome trace：
[Tide](../../artifacts/training-cost-candidate-trace/costs-tide-standard-trace.json)（156,012,890 bytes）与
[A](../../artifacts/training-cost-candidate-trace/costs-transformer_A-standard-trace.json)。完整SHA和字节数见JSON。
两份轨迹确实包含`Ascend Hardware`进程；按`process_name`元数据选择硬件事件，再用`Task Type=KERNEL*`选择kernel。
不能把无`cat`的事件全视为kernel：其中还包含`Overlap Analysis`生成的Computing/Free图、EVENT WAIT/RECORD等。

| 轨迹 | 主机cpu_op事件 / inclusive总s | 硬件任务 / 区间并集s | KERNEL任务 / 区间并集s | 硬件首末跨度s |
|---|---|---|---|---:|
| Tide组合 | 229,940 / 4.22423 | 59,219 / 1.23954 | 58,672 / 1.23877 | 1.78383 |
| standard A | 230,490 / 2.98882 | 43,708 / 0.45726 | 42,204 / 0.45579 | 1.36909 |

主机inclusive事件嵌套，硬件跨stream可重叠；这些列不能相加，也不能除以普通训练中位数形成份额。
区间并集是对trace时间戳区间求union，不是设备利用率、算力利用率或正常关键路径。
例如Tide硬件任务duration求和1.35862s，而union为1.23954s；其EVENT WAIT在另一stream上与计算重叠。
trace诊断外层elapsed还含导出与分析（Tide约18.51s），不能当单步训练耗时。原始host/device事件窗口与正常样本亦不相同。

Tide主要实际硬件任务如下；以下是该任务名duration之和，仍不构成可加的正常时间分摊。

| 任务名缩写 | 次数 | duration总ms | 含义 |
|---|---:|---:|---|
| Cumprod_Cumprod | 128 | 233.860 | `KERNEL_AICPU`，属于Ascend设备任务，不是主机CPU fallback |
| BatchMatMulNd | 2007 | 112.868 | 多个矩阵/反向收缩，不能全归给SSD |
| IndexAiCore | 1786 | 73.761 | 索引工作，另有slice/transpose任务 |
| MulAiCore | 3527 | 73.549 | 多处逐元素工作 |
| ScatterElementsV2 | 8 | 63.874 | `KERNEL_MIX_AIV`，不是主机任务 |
| InplaceCopy/Transpose | 2855 | 47.141 | layout materialization之一 |
| FlashAttentionScoreGrad | 38 | 22.158 | 实际native Attention反向任务 |

matrix阶段的runtime记录仍见FP32 cumprod；prefix摘要已单列，不需要把完整矩阵scan改造成同一种递推。
Cumprod设备开销与matrix路径定位共同支持后续有界研究，但runtime dispatch和未加这些hooks的trace具有不同观察边界，
这里没有建立每个host调用到kernel的完整关键路径映射。Tide主机`aten::to`/`_to_copy`/`copy_`inclusive时间约375/364/343ms，
三者有嵌套，不能相加；逻辑materialization字节也不是硬件总线流量。

A的设备轨迹确有376次FlashAttentionScore和376次Grad，总duration约19.408/29.345ms，另有2256次MatMul约23.921ms。
其大量StridedSlice、transpose、mul和逐层调度说明native Attention并不消除所有小算子开销；这也支持已做的chunk候选检查。
这些是standard A轨迹，不能冒充chunk256轨迹或A–D全部设备trace。

## 为什么在这里停下，以及复现入口

本轮已经实施并测量合批、Reception局部重算和K=1摘要prefix；检查了Transformer自身的两种有限布局/分块候选，
补齐普通计时、按anchor恢复的诊断、实际dtype/device、逻辑工作/padding以及真实设备任务证据。
当前证据足以说明主要成本方向，不能用一次trace解释全部正常关键路径或推出理论速度上限。

暂缓全矩阵SSD scan重写、allocation/index/packing专用融合及自定义反向：这些涉及更宽的边界/导数/状态续接验证，
本轮已有独立数值路径工作，不能用大kernel duration直接替代语义证明。任何后续方案应先保持矩阵状态、零衰减/零质量、
完整参数/输入梯度和恢复后更新，再做有限同条件测试。更大批量/不同形态的同预算吞吐与这里的B8成本归因是不同问题。

FC BF16 B128/L256、Settle FP32组合B24/L256、Timed FP32组合B12/L128的A–D chunk256正常训练比较已单独归档于
[模式主表](training-modes-20261118.md)。连续训练与数值见证另有资格，不把本页B8成本直接套到这些不同负载。

原始终态入口：

- [serial成本](../../artifacts/training-cost-settle-base/costs.json)
- [batch4成本](../../artifacts/training-cost-settle-batch/costs.json)
- [组合成本及设备轨迹](../../artifacts/training-cost-candidate-trace/costs.json)
- [A–D有界候选](../../artifacts/training-transformer-strength/costs.json)

只读核验所有被本页引用的原始文件，不运行模型：

```sh
python - <<'CHECK'
from pathlib import Path
import hashlib, json
record = json.loads(Path('docs/evidence/training-costs-20261118.json').read_text())
for item in record['verified_files']:
    data = Path(item['path']).read_bytes()
    assert len(data) == item['bytes']
    assert hashlib.sha256(data).hexdigest() == item['sha256']
print('verified', len(record['verified_files']), 'files')
CHECK
```

JSON的`runs[].launch_arguments`保存每项冻结提交、完整命令、资源和新输出参数入口。
重跑必须使用新观察身份并重新按本机队列准入；本证据生成未运行NPU、未重写原始结果。
