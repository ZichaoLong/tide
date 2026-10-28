# 参数匹配、性能经验迁移与严格验证 — 2026-10-28

固定数学`0770593838a37c790a568226930567ed60f99b83`。主要实现`2da3f2c76c13cff0c41af45fddb5f24f2e99421f`，
诊断/设备scratch修复`0503593d71a64d26f24e0a4168e16af8c1633baa`。
[同名JSON](core5-performance-transfer-20261028.json)保留源码、材料/二进制/日志身份、完整计时边界、
工作量、终态和严格失败。本报告完成一轮有限设施与选型调查，不声明所有数值限制已消除或所有低成本优化已完成。

## 实现与独立资格

- 五族静态参数形状/清单逐角色对照实际模块，按真实别名计数，冻结参数、角色展开、实际存储分别表示。
  FC/HB、一般ranked、非对齐DAG、反馈PDG可扩展生成；不执行参考轨迹来决定候选日程。
- 用真实私有SwiGLU宽度匹配旧core-3主体预算，不增加无用参数；二进制v2材料保存实际FP32值/别名，
  双层SHA256校验、冻结零编码与明确准入。v1诊断解码不执行数值模型，旧消费者自行校验适用域。
- eager共享KV路径计算`U(Vx)`，独立标量/未共享路径保留组合矩阵；WeightedModel不再随机初始化随后废弃的旧权重。
  resident每阶段复用一次ready值副本，冻结参数池只复用同TensorImpl/切片/设备/版本；发布去重而梯度仍汇总全部使用贡献。
- CPU固定2da3f2c相关门禁376 passed、2 strict xfailed；另有24图用例各4次更新及精确checkpoint。
  0503593设施14项通过，小型native FP64两步开发诊断通过。独立手算、方向差分、完整独立调度的作用沿[语义索引](../semantic-map.md)区分。
- 固定0503593的standalone NPU四组件×FP32/FP16八项全部通过，覆盖完整flow、配置、训练与双卡节点训练/恢复；消费者build/loader同样通过。
- CUDA sm80/90/100四个相关常驻组件编译完成。第一次作业被NPU在共同冻结源码目录产生的`kernel_meta/`
  改变Git身份而拒绝；原作业仍failed。0503593同内容hash增量build、loader/help、CPU FP32/FP64核验通过。
  新设备验证子进程cwd固定到独立输出目录，回归同时覆盖成功/失败。没有GPU真机证据。

复用开发二进制时，固定checkout重新核对全部core/component内容hash和binary digest；消费者从固定checkout重新构建。
不是把dirty开发提交当固定源码资格，也没有修改活动任务的源码。源码/构建/真机分别记录。

## 参数匹配与实际用例

旧D2048、480节点/2208边：Add总参数9468053696，Attention总17521117376；embedding/head为206045184。
本基座无任务头，所以主体目标为9262008512和17315072192。五族均生成两种完整预算的静态计划，
实际误差在879至2001个参数内；静态计划不是十种大型运行通过。循环族在这些计划中明确使用key_width64/value_width128。

| 实际材料 | 节点/边；D；B×每窗位置 | 可训练参数 | 对照口径 |
| --- | --- | ---: | --- |
| 原HB medium | 8/12；32；4×4 | 102604 | 同一材料的旧/新消费者 |
| ranked64 | 480/2208；64；1×1 | 9076975 | 对应core-3 Add主体9076928，相差47 |
| wide2048 | 4/4；2048；1×1 | 201463846 | 同宽宽算子探针，无参数匹配宣称 |
| matched2048 | 480/2208；2048；1×1 | 9262006511 | 旧Add主体相差−2001，真实FFN宽319/320 |
| HB64 KV，上限k=区域规模 / k=1稀疏 | 8/8；64；2×2 | 402060 | 同参数、不同发送函数与实际工作 |
| HB64 SSD | 8/8；64；2×2 | 401980 | 匹配上述KV参数，相差−80；不同数学函数 |

新HB64的8条边与原medium的12条边不同；只在各自相同材料内比较，不把两者当作单纯放大宽度。

matched2048已实际导出约35GiB二进制并加载，构造约184.49秒，warmup约134.43秒；
进程生命周期RSS峰值50392788992字节（约46.93GiB）。随后测量窗分配残差拒绝，metrics为空，
因此没有大型吞吐或完整等价结论。静态导出约365.63秒单列，不算训练/推理窗口。
这项仅B1/每窗1位置推理，不是历史B512/12位置完整训练复刻，也不能用旧计时除以新计时。

## 有限成本与选型

每配置独立进程、1 warmup+1测量窗口；状态连续，训练每窗SGD并显式detach。
时间含输入准备、调度、packing、前向、完整返回、loss、反向/finite checks、更新及同步。
构造/编译/捕获、warmup、事后独立验证分列。所有比较在候选完成之后进行，不向候选提供参考轨迹。

CPU同材料原HB medium旧/新为391.432/365.834毫秒，均128候选/128 Full，完整严格对照通过。
这是一次约6.5%的下降观察，不是稳定收益；旧baseline为f72a6cc消费者及其记录的旧core，
跨越本轮性能改动及此前正确性修复，不能逐项归因。

CPU新HB64 SSD为155.713毫秒、32候选/32 Full，完整对照通过。KV的HB64/稀疏、ranked64均严格失败，
原耗时与工作量保留JSON，不用它们宣称等价加速。ranked64 streaming/prefill约3.205/3.400秒，
纯Python约5.915秒，三者都是失败候选的成本观察。稀疏HB64实测20候选/16 Full，普通HB64为32/32；
k为区域规模也不保证全活动，ranked64为479候选/344 Full。

NPU原HB medium旧/新为2576.159/2795.223毫秒，均128候选/128 Full、192消息且完整对照通过。
这次新版本约慢8.5%，没有获得设备端到端提速证据。构造后allocated由162296320降到162108928字节，
测量峰值由514766848降到514392064字节；reserved峰值同为532676608字节。
真实存储节省与计时负结果同时保留，不能将两者混写为“内存大幅下降、性能提升”。
新统计中角色参数492848字节、唯一bank410416字节；当前窗口ready值复制76032字节。
原逐节点实现按同静态布局需要8次相同复制，复制减少不意味着所有packing或设备工作减少。
物理local/Full行仍为192/192，多于128次实际作用；反向程序构造仍存在。

HB64 SSD resident为2859.207毫秒，32候选/32 Full，完整对照通过；本次同材料CPU为155.713毫秒。
在这两个通过的有限窗口上CPU更合适；不能外推D2048或把不同Torch/CANN栈差异归因于语言。
HB64 KV resident普通/稀疏的完整严格对照失败，时间和allocator峰值保留但不作等价加速依据。
Python mixed-c普通/稀疏分别约607.708/413.936毫秒，均完整严格失败。单独mixed profile观察到18625个
设备kernel事件，含674个AICORE、100个AICPU、1289个host scalar读取及1182个异步copy；
设备kernel覆盖并集约63.48ms，forward/backward作用域约609.63/543.04ms（含嵌套事件，不相加作独占时间）。
索引/Nonzero及大量提交仍值得处理；AICPU是设备CPU，不等于回退主机，本trace未观察到host fallback事件。
profile不代替对照通过，也不能从kernel计数直接推断优化倍率。

独立CPU profile在ranked64上观察到22905次`aten::mm`、100590次`aten::copy_`；约8.94秒CPU self事件中，
矩阵乘self约0.265秒、forward作用域self约1.350秒，Query/自定义反向/装配亦明显。
实际479次State、最大State batch=1、15个区域阶段；“使用packed接口”没有使不同私有模块自动组成大GEMM。
profile最终同样严格失败，它可定位这条实际路径的成本，不能认证语义或作为吞吐样本。
完整trace约818MiB，profile进程RSS峰值约16.25GiB；后续应优先小型代表用例，不能把观测开销漏出资源预算。

## 严格数值失败

同一ranked64和HB64 v2材料无损解码为v1后，旧standalone也失败（maxabs约2.086e-6与2.315e-6）。
新ranked64 v1与v2首个完整比较见证一致；失败不是v2解码改变了参数，也不是新优化后才有此类现象。
旧失败不能证明新算术分组完全无影响：首个字段/误差会随实现变化，保留两边真实结果。

两步CPU诊断使用同一FP32初值，另显式提升双方为FP64：

| 路径 | 精确离散结构 | 完整浮点记录 | 输出loss参数梯度与SGD参数 |
| --- | --- | --- | --- |
| Python FP32 | 两步相同 | 严格失败 | 两步均有严格差异 |
| native FP32 | 两步相同 | 严格失败 | 两步均有严格差异 |
| Python/native FP64 | 两步相同 | 两步通过 | 两步通过 |

FP32首步`nodes.0.kernel.base`梯度最大差约.001019（Python）/.001419（native），
不能只用最终输出的小误差作训练正确性判断。FP64结果与有限精度敏感性相符，不解除FP32失败，
也不是所有梯度根/所有规模的资格。原medium反馈PDG的两种CPU调度在新源码仍严格失败。
SDK FP32 `fc/affine`也在新源码第三轮（step=2）重现同一evidence最大差3.09944e-6；
原精度失败尚未修复。它与resident不同算术边界的八项通过分别记录。

另有独立手算反例：quad的scores=`[10000,9999]`，caps=`[1,1]`，lower=upper=`0.7`，tau=`1`，
唯一解是`[0.7,0]`。eager Python/native scalar/batch在FP32和FP64都拒绝残差；减共同偏移后的`[0,-1]`通过。
求解已居中，但残差/温度VJP仍由原坐标已舍入乘子重建响应；相消会触发拒绝。
它与正支持下溢不同，单纯升FP64并不充分；尚未证明所有D2048拒绝都是这个机制。
未改变容差、输入、活动集合或默认算法。后续修复必须同时保持实际响应、VJP、乘子解释与支持守卫。

## 复现与边界

从上述固定提交按[开发规范](../development.md)构建匹配core/可选resident，再用公共入口：

```sh
python -S scripts/prepare_weighted_workload.py --topology ranked --layers 15 --region-width 32 \
  --width 64 --samples 1 --positions 1 --match-core3 add --output-dir NEW_PLAN
python scripts/export_weighted_benchmark.py --native-library CPU_CORE --workload NEW_PLAN/workload.json \
  --binary --output NEW_MATERIAL.json
CPU_BUILD/client/weighted-benchmark --fixture NEW_MATERIAL.json --device cpu --steps 1 --output-dir NEW_RUN
python scripts/diagnose_weighted_workload.py --workload NEW_PLAN/workload.json --dtype float64 \
  --implementation native --native-library CPU_CORE --output-dir NEW_DIAG
python scripts/decode_weighted_fixture.py --input NEW_MATERIAL.json --output NEW_V1.json
```

完整Add预算用width2048；导出显式`--max-materialize-bytes 68719476736`。HB64用`--topology hb --region-width 2`
并设width64/samples2/positions2；稀疏另加`--selection regional_total_zero --k 1`；SSD使用
`--family ssd --target-parameters 402060`。四节点宽探针用`--topology fc --depth 2 --region-width 2 --width 2048`。
大容量只按实际可用资源运行，不能将本机192GiB进程预算当普遍要求。

旧消费者按f72a6cc固定源码构建，读取同一原medium fixture或经无损解码的v1；不是恢复旧实验队列。
quad反例可直接在相应dtype调用`allocate_quad(scores, caps, .7, .7, 1.)`；native用同参数的`weighted_allocate`
及`weighted_allocation_batch`入口。任何拒绝仍保留原终态。

资源：各作业background.slice/Nice10、CPUQuota200%、ATen/BLAS1、build2。CPU有限测量总3600秒、
单进程900秒、MemoryMax192GiB；CPU诊断总1200秒/单进程240秒/32GiB；NPU资格2400秒、队列180秒、
设备1800秒/64GiB；NPU测量3000秒、队列180秒、各配置300秒/64GiB；CUDA复核300秒/16GiB。
CPU测量与本项目NPU/CUDA有限构建/资格有CPU争用；项目重型计时串行，其他用户争用未排除。
开发NPU门禁物理1/13，固定资格物理1/3映射逻辑0/1；设备测量物理1映射逻辑0。
standalone为LibTorch-NPU2.10/CANN9.0，Python mixed为Torch2.12/Torch-NPU2.12.0.post2/CANN9.1；
设备Ascend910_9392。两个baseline/new resident使用相同SDK及物理设备。
systemd资源属性不代表本机legacy cgroup获得独占隔离。每个case的实际退出码和配置在JSON中。

CUDA真机、x86_64、未测的大规模NPU/跨卡吞吐仍待独立验证；eager half GDN/AdamW、默认异步双NPU
纯Torch反向失败见[数值隔离](core5-numerical-isolation-20261028.md)。共享参数、私有状态、int64时间/计数、
None/零梯度、精确checkpoint及原有容量/正支持拒绝均保持。连续存储不等于充分打包：不同节点的物理行、
逐窗口反向程序构造、未共享KV的组合矩阵仍有优化空间。下一步优先稳定数值边界，再以本轮profile为依据
研究按函数/形状分组的跨节点批量运算，单独验收所有使用贡献与完整训练；没有“所有低垂果实已处理”的结论。
