# 同模型安全容量代表点（2026-10-31）

本报告包括CUDA4.561B同模型单/双/三/四卡与NPU12.802B单/双/三卡，以及CUDA9.123B单卡拒绝/双卡实际点；其他NPU卡数待各自实际结果，不由已测点推断。
4.561B组固定Python实现bd2ff2e，core/resident构建de999c2；配置31afefb，实际冻结输入由ad93848导出。
RTX3090/sm86、Torch2.12.1+cu130、FP32 payload/scoring、TF32关闭，native mixed-c、ordinary复制、packed/scalar完整返回；ATen/interop/worker1。
FC16/8区域、KV、d4352、FFN4d，4,561,319,928个去重可训练参数，8个不可拆共享组；B1/P2/span20，staggered-v1，warmup1+连续2窗+精确恢复窗，1800秒上限。

## 单卡实际准入与结果

实际参数+冻结bank19,458,826,704 bytes；逐卡另预留1GiB workspace、256MiB activation、64MiB通信及3GiB不用。
静态峰值20,941,417,120 bytes；现场扣除3GiB后的usable21,798,912,000 bytes，预算剩余857,494,880 bytes。
主存预算91,660,099,776 bytes；两代checkpoint磁盘预算42,482,056,608 bytes，其中已含1GiB空盘保留。
先审阅并退役28.505GB终态可再生checkpoint，保留结果/配置/log，再以live free43,976,364,032 bytes准入。不是反复OOM搜索，也没有缩小batch/cache或改变精度。

| 结果 | 实际值 |
| --- | --- |
| 执行 / 独立数值 / 离散 / 精确恢复 | 完成 / 超差探索 / 通过 / 通过 |
| 两个测量窗 / 中位 | 0.595189 / 0.755775秒；中位0.675482秒 |
| 每窗candidate / Full / 消息 | 31 / 18和16 / 32和28 |
| 构造 / 独立参考 / 恢复检查 | 36.822 / 41.689 / 314.967秒 |
| 全程allocator peak allocated / reserved | 20,058,310,144 / 20,245,905,408 bytes |
| RSS高水位 | 74,476,953,600 bytes |
| 最后checkpoint单代 | 20,699,246,016 bytes；next_window3 |

allocated峰值约占扣除余量后usable的92.0%；reserved约92.9%，属于该口径下的单卡近界代表。allocator不含驱动/厂商隐藏内存；不称硬件绝对最大模型。
逐节点参数、实际state/message/pending及计算统计与请求单卡owner映射相符。完整返回、状态/控制/逐边端口/history/pending比较与磁盘下一窗各自记录。
原数值首差window0/sample0/node0 KV keys，133/4352坐标超原atol1e-6/rtol1e-5，最大2.81334e-5；六份比较累计1512数值差异、0离散差异。不能称严格合格吞吐。
完整checkpoint恢复远大于单窗计算成本，实验选型须同时考虑持久化和恢复；0.675秒不能代表整个资格流程耗时。

任务`jobs/platform-cuda-capacity4561m-c1-bd2ff2e`于06:53:40Z exit0，独立SSH检查supervisor/workload/remaining全部为空。
保存配置、结果、来源与hash后才退役可重建checkpoint，为同模型下一点释放盘；活动输入不退役。

## 同模型双卡实际点

双卡沿用完全相同model_id/workload_id、B1/P2/span20与完整返回，8共享组交替放在GPU0/1，每卡8节点。
执行完成、离散语义和磁盘精确续接通过；1517数值差异、0离散差异，首差仍为原keys坐标，保持FP32探索。

| 指标 | 双卡实际值 |
| --- | --- |
| 两测量窗 / 中位 | 0.683779 / 0.825447秒；0.754613秒 |
| 每卡峰值 allocated / reserved | 10.299098 / 10.441720GB；两卡近似相同 |
| 主存RSS峰值 / 恢复检查 | 74.808754GB / 320.717秒 |
| 原生跨卡复制 | 各窗7组；557184 / 487536 bytes；ordinary生效 |
| 实际位置 | 参数、私有状态、消息目标和计算与共享组映射相符；无参数独立复制替代共享 |

候选/Full/消息与单卡相同，31/(18,16)/(32,28)。相同模型双卡降低逐卡存储但本次单窗成本高于单卡；
不将两卡空闲显存之和称容量，也不由ordinary复制推断直接P2P或resident协议。
任务`jobs/platform-cuda-capacity4561m-c2-bd2ff2e`于07:11:16Z exit0，独立SSH核对supervisor/workload/remaining全空。
结果原字节SHA256 `3bb5e0ac7fe851677fd47329472fd31aeb656de1cb96c829a84384b6089054d6`。

## 同模型三卡实际点

同一模型/工作负载/任务与完整返回，三卡6/6/4节点；执行、离散、精确恢复通过，1517数值差异/0离散差异保持探索。
两测量窗0.719262/0.839067秒、中位0.779165秒；candidate/Full/消息与前两点相同。
allocated峰值7.859294/7.859295/5.419492GB，reserved7.990149/7.992246/5.540676GB；RSS74.883752GB。
构造40.413秒、独立参考42.921秒、精确恢复316.800秒；原生跨卡两窗仍各7组、557184/487536 bytes。
逐节点实际state、message目标、pending与有效设备映射核对相符；共享组未拆成独立参数。
任务`jobs/platform-cuda-capacity4561m-c3-bd2ff2e`于07:23:53Z exit0，SSH检查supervisor/workload/remaining全空。
结果原字节SHA256 `19a099693a6d09634fb8e65c53f6dee17843759b8da010e4e49bf878f384334d`。

## 干净安装复现

```sh
# 依human-workflow安装，site指向匹配的实际库。
python -c 'import json,sys; m=json.load(open(sys.argv[1])); print(json.dumps(next(c["config"] for c in m["cases"] if c["name"]==sys.argv[2]),indent=2))' configs/matrices/cuda-capacity-calibrated.json cuda-capacity-4561m-cards-1 > selected.json
tidegraph plan selected.json --site site.json
tidegraph job start selected.json --site site.json --output-dir jobs/new-capacity
tidegraph results docs/evidence/core5-platform-capacity-20261031.json.gz --format table
```

多卡同一模型继续保持共享组同卡、全区域函数与实际位置；普通复制不认证直接P2P/原子或resident协议。下一尺度仍需逐卡、主存、两代checkpoint共同准入，不把空闲GPU显存直接相加。

## 同模型四卡实际点

同一4.561B模型/任务/输入与完整返回，四卡各4节点，实际参数/私有state/消息目标/pending与放置映射核对相符。
两测量窗0.721320/0.847173秒，中位0.784246秒；每窗31 candidate，Full18/16、消息32/28。
执行、完整离散检查和精确下一窗恢复通过；1517数值差异、0离散差异，首差仍为原keys，保持FP32探索。
四卡allocated峰值5.419491/5.419492/5.419492/5.419491GB，reserved各5.540676GB，RSS74.944528GB。
构造39.900秒、独立参考42.857秒、精确恢复281.042秒；复制每窗7组、557184/487536 bytes，ordinary生效。
同模型1→4卡降低逐卡存储，本次成本0.675→0.784秒，没有速度优势；不是四卡总容量极限。
任务07:34:04Z终态exit0，独立SSH检查无supervisor/workload/remaining后代。
结果原字节SHA256 `1794416ad9ce21ed4b60a8a22fe4bd90f1b88da6509684ad8ca5dcbfc2093ba3`。
该四卡证据提交时checkpoint保留；随后用户授权继续，9.123B现场准入见后节。

## CUDA9.123B双卡现场准入

固定bd2ff2e，`cuda-capacity-sharded.json`对应9.123B双卡配置现场运行公共准入，未构造模型。
逐卡估计20.942GB、host174,739,737,664 bytes，现场host可用228,634,873,856 bytes；磁盘需83,895,945,824 bytes而现场只有23,246,536,704 bytes。
错误明确为`disk: planned reservation exceeds live free memory/storage`，子任务exit1、无剩余后代；不是GPU OOM或数值失败。
**磁盘估计已包含1GiB reserve**；4.561B原表中42,482,056,608 bytes也已含该reserve，不能再加一遍。
83.896GB是两代checkpoint和保留空间的总预算，超过整个50GiB盘；当前C4 checkpoint占盘不是唯一限制。
即便扩30GiB仍须扣掉环境、保留证据和余量；不因此重启机器，也不直接将所有shm当额外物理内存。
下一步只对既有4.561B做有界保存/恢复分阶段磁盘/tmpfs成本对照；tmpfs未来文件必须同时计入容器内存。


## NPU12.802B单卡校准

固定Python实现bd2ff2e、NPU core/resident eca070f；同一FC16/8区域、KV、d7296、
12,801,882,104去重参数、B1/P2，native mixed-c、FP32 payload/scoring、ordinary、packed/scalar完整返回，线程1。
参数、bank、状态、workspace、通信逐卡预算58.565GB，另保留6GiB；host241.331GB、两代checkpoint117.232GB。
这是独立于CUDA模型的NPU近界代表，不能把不同参数量作同模型跨平台加速比。

| 结果 | 实际值 |
| --- | --- |
| 执行 / 独立数值 / 离散 / 精确下一窗 | 完成 / 超差探索 / 通过 / 通过 |
| 两测量窗 / 中位 | 1.577240 / 1.318192 s；1.447716 s |
| 每测量窗 candidate / Full / 消息 | 31 / 16 / 28 |
| allocated / reserved峰值 | 54,911,347,200 / 55,194,943,488 bytes |
| RSS高水位 | 231,085,862,912 bytes |
| 构造 / 独立参考 / 恢复检查 | 158.557 / 156.683 / 502.872 s |
| 三次保存 | 269.215 / 357.505 / 281.708 s |
| 整流程 / 单代checkpoint | 1737.708 s / 58,070,735,884 bytes |

原1800秒内完成，没有延长超时；单窗成本不包括单列的资格/保存过程，不能称严格合格吞吐。
子监督于08:13:51Z终态exit0，重新核对supervisor与后代均为空；父任务继续其他卡数，二者不混算。
原结果SHA256 `7ae9a7e9ba87cf1c3cd5228680b9488b45661b5959798e13a54fbaaab13669f6`。
其条件化后继可以使用本单卡有限执行和精确恢复事实，但仍须各自逐卡/live准入与独立检查。

## NPU12.802B同模型双卡

同一model/workload/task，8个共享组在两卡交替放置，每卡8节点；沿用bd2ff2e安装与原每窗保存。
执行完成、完整离散检查和精确下一窗通过，FP32数值仍超差探索。
两窗1.850058/2.052191 s，中位1.951125 s；各窗candidate31/Full16/消息28，与单卡一致。
每卡峰值allocated27.579406 GB、reserved27.814527/27.816624 GB，RSS232.777716 GB。
原生ordinary各窗7组/817264 bytes，实际参数、私有状态、消息目标及计算位置符合映射；
没有把共享参数复制成独立参数，也不由普通复制推断P2P或resident资格。

构造148.978 s、参考155.350 s、精确恢复422.883 s、整流程1573.816 s，单代checkpoint仍58,070,735,884 bytes。
该单次同模型双卡降低每卡存储，但窗口成本高于单卡1.447716 s；恢复差值不能单独归因于卡数。
子监督终态exit0、无后代；父任务继续三卡。结果SHA256
`46782a73551d5de21cc3fe65bed4e572836d9a5404d1beb021a78f50c39fb4d8`。

## CUDA9.123B：同一模型跨越单卡容量

固定Python包36baeb7、配置`configs/checkpoint-cost/cuda-capacity-periodic.json`；CUDA核心仍是已核对的de999c2。
FC32/16区域、d4352、KV、9,122,639,888去重参数、16个不可拆共享组，B1/P2/span36；
FP32 payload/scoring、TF32关闭、native mixed-c、ordinary、packed/scalar完整返回、线程1。
interval3只发布本次最终窗，warmup1+连续2测量窗+精确下一窗及独立CPU对照；与原两代磁盘请求是不同持久化配置。

单卡实际准入在模型分配前拒绝：bank38,917,653,552 bytes、估计峰值40,474,596,032，
均超过保留3GiB后的21,798,912,000 bytes。双卡同一model/workload/task通过独立live准入；
每卡16节点/8共享组，静态峰值20.941/20.942GB，host174.740GB；磁盘单代预算42,484,843,824 bytes，
含1GiB保留，现场free43,878,969,344。没有扩盘、重启、缩小batch/cache或改变共享绑定。

| 双卡实际结果 | 数值 |
| --- | --- |
| 执行 / 独立数值 / 离散 / 精确下一窗 | 完成 / 超差探索 / 通过 / 通过 |
| 两计时窗 / 中位 | 1.115031 / 1.125218 s；1.120125 s |
| 每窗candidate / Full / 消息 | 63 / 32 / 60 |
| 每卡allocated / reserved峰值 | 20.058312 / 20.245905 GB |
| 原生ordinary通信 | 每窗15组、1,044,720 bytes；请求与生效一致 |
| RSS高水位 | 116,935,979,008 bytes |
| 构造 / 唯一保存 / 恢复 / 独立参考 | 74.833 / 112.787 / 582.984 / 85.927 s |
| 整流程 / 最后单代checkpoint | 863.134 s / 41,400,830,286 bytes |

这是真实同模型双卡容量点：单卡不能容纳bank，双卡分别接近有余量的容量，完整状态/消息/计算位置符合映射；
不是多个独立模型，也没有将共享权重拆成独立参数。FP32数值仍超原阈值，不能称严格合格吞吐或认证该规模训练。
旧每窗两代83.896GB磁盘拒绝保持；显式保存间隔是公开耐久性取舍。
恢复582.984秒远高于1.120秒窗口成本，不把4.561B的恢复比例线性外推到更大规模。

外层矩阵08:45:38Z终态failed/exit1，原因是预期的单卡拒绝；双卡子任务exit0，所有监督者/后代均终态。
单/双卡原结果SHA256分别为`9c64ae8aa2e0abe754481f375aa058d53f45b4a4c84022082619dfccf0e416ea`、
`8ff07ffa1141beaf39f5ce1c45cd89bbca9278865b999a6971f4be7a76875f30`。

## NPU12.802B同模型三卡

固定bd2ff2e、同一model/workload/task和原每窗保存；6/6/4节点及全部8个不可拆共享组实际分布于三卡。
完整执行、离散检查和磁盘精确下一窗通过，FP32独立数值继续超差探索；原阈值、记录及共享绑定保持。
两测量窗2.030038/2.051598 s，中位2.040818 s；各窗candidate31/Full16/消息28，与单/双卡一致。
allocated峰值20.746420/20.746421/13.913436 GB，reserved20.969423/20.969423/14.124319 GB，RSS231.292989 GB。
ordinary每窗7组/817264 bytes；实际参数、私有状态、消息目标和计算符合请求的三卡映射。

构造145.170 s、两计时窗后的保存293.781/279.141 s、完整精确恢复397.596 s、独立参考182.053 s，整流程1578.266 s。
同一模型更多卡降低各卡存储，但本次窗口成本高于单卡1.448 s和双卡1.951 s；不称多卡加速或严格合格吞吐。
子监督09:06:51Z终态exit0，重新检查supervisor/后代均为空；父任务已转入八卡资源队列，排队不算执行。
结果SHA256 `7c18c8fba7c53752f0b00248ea2382a854543268582972101cb82e91e9b47bb8`。

## NPU八卡资源等待与38.406B现场准入

12.802B八卡请求以priority0等待607.801秒后退出3，原因`insufficient_free_devices`；
期间候选4–11只见6至7张空闲卡，不抢占其他项目。该case未构造/未运行，不能标数值失败或容量不支持。
final父服务09:17Z已终态exit1、控制组为空，单/双/三卡的各自通过与数值探索记录保持。
后继1B七卡是独立的明确配置，不改写原八卡请求。

38.406B使用3e4f6b2跟踪配置与bd2ff2e固定包：FC48/d7296、38,405,646,376去重参数、24个不可拆共享组，
B1/P2/span52、FP32、native mixed-c/ordinary、packed/scalar完整返回，原每窗保存和1800秒上限。
其单/双卡现场准入已实际完成，不分配模型，不以OOM搜索容量：

| 卡数 | 每卡bank bytes | 每卡估计峰值bytes | 每卡usable bytes | 结果 |
| ---: | --- | --- | --- | --- |
| 1 | 163,850,061,456 | 168,046,924,512 | 59,201,015,808 | 分配前拒绝 |
| 2 | 81,925,030,664 / 81,925,030,792 | 85,935,187,640 / 85,936,942,120 | 59,192,193,024 / 59,202,494,464 | 分配前拒绝 |

两个子监督终态exit1、无后代。推理/训练/独立数值/离散/恢复均未执行，不能从静态表得到这些资格。
三卡候选09:18Z已通过自己的live准入并构造；结果待其终态，不能由12.802B校准点或两个拒绝项推断通过。

单/双卡原结果SHA256：`0d62371a922594add1d0f314a90d761a4b52f78b2445a820dea06cde74241765`、`8fe4d985de154292733f2e1549dad7407b0f95ebc97f03883c30643110607751`。

归档SHA256：`e9a97842e5a24c9a3eb0a99768a127bc18aaf6007b7c55148c46abfd0a0f221a`。
