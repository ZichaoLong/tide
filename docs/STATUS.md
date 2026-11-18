# 当前交接：Reception、FP32 训练与大参数常驻推理已交付

用户已确认本轮七阶段工作，实际训练优先 FP32；持续 FC/HB 实例路线见
[project scope](project-scope.md)。分支 `graph-execution-foundation`，起点 `df2a0f18`。
数学固定 `4f5b9f20a4030d68ee66ae2d26e34659a5e37234`，108 文件核验通过。
生产 `src/tide` tree 为 `539c5eb15176827ca14a19c7de82f08df5b6c1d1`。
实现、测量与后续文档分别保留源码身份，不用整理 HEAD 追认旧测量。

## 当前执行：固定预算下的 decode batch 搜索

用户已同意继续搜索各自吞吐最优 batch。沿用9B单卡、17B双卡/四卡的Tide，
及同规模设备上的近总参数Transformer C/32/48层，共12个系列；不重跑原A/B/D，
不把这轮辅助比较改称四类联合匹配。FP32、prompt128/chunk32/decode16、每卡
48/56GiB allocated/reserved预算固定，四卡也继续扩大batch。

复用有效旧点；新增点先warmup1/samples2筛选，必要最佳/邻点warmup2/samples5确认。
按实测内存预测扩大并细化边界，吞吐连续趋平或预算相邻边界成立时停止；不靠反复OOM。
每系列最多8个新增筛选点、2个确认点，单作业2700秒、总墙钟4小时，最多并用7张卡。
原始方案与配额/资源/用户文件身份在`artifacts/decode-capacity-20261118/preflight.json`。
当前为准备阶段，结果和停止理由完成后归档；此前交付的终态事实保持原身份。

## 追加测量已完成：Transformer深度与Tide batch/多卡

用户追加的约9B/17B固定32/48层Transformer及Tide batch/多卡问题已完成有限补测。
源码冻结 `237cc8d4`，生产树不变；入口及原推理回归20项通过，12个作业全部passed，
60个正常样本。全部FP32、prompt128/chunk32/decode16、warmup2/samples5；
原B1/C保留旧源码、物理卡和样本身份。[完整证据及命令](evidence/transformer-depth-batch-20261118.md)。

Transformer固定32/48层、近总参数的decode分别为：9B单卡20.29/15.41位置/s；
17B双卡8.79/9.52。形状及头宽随近参数匹配变化，不把胜负归因为纯层数影响。
Tide 9B单卡B2/B4/B8为5.19/7.98/12.34位置/s；17B双卡为3.76/6.31/10.07。
17B四卡B4/B8为4.46/8.52，低于同batch双卡，但每卡峰allocated降至19.52/22.93GiB。
9B单卡B8、17B双卡B8峰allocated分别46.83/47.32GiB，接近本轮48GiB采样预算。
总吞吐提高伴随单步时延增加；短请求与有限batch阶梯不认证最大吞吐或长上下文容量。

17B B4两卡/四卡Full相差1且输出支持摘要不同；B8的工作、实际触及、支持、时间摘要
一致，但packing仍不同。远程静态计划条目16→54，不能当作实测通信耗时比例。
没有新增四卡训练/恢复或副本并行资格。A–D定义与训练预设保持原定位。

`artifacts/depth-batch-preflight-20261118`保留preflight、扩展容量依据、提交与汇总；
终态/冻结源、68份原始文件指纹和用户文件保护核对已并入证据JSON。
本次没有待运行作业；所有MainPID与已记录PID/start后代均已终态。

## 已集成与核验

三种 incoming 接收 self-attention 已完成正式路径；保留原 source_residual 数学选择。
执行候选包括融合 QKV、结构 padding mask、接收/整区域重算、相容区域合批、
SSD block/incremental/doubling、prefix summaries、可选静态参数行视图，以及同参数
分区合批后跨设备区域并发。普通实现与未选候选保留，不自动改变数学绑定。
连续推理保留 KV/SSD/pending，仅有限流末尾 drain；独立缓存观察器验证正写入、
原始 int64 时间、实际保留集合、淘汰与 SSD/Conv 时钟。

冻结 `1fbe15c9` 的完整 CPU 检查 **783 passed / 1 warning**；隔离新 wheel 的
公开入口、三更新及新进程恢复已完成。`ffeca387` 的训练预设相关 **34 项通过**。
[集成与后端](evidence/reception-validation-20261118.md)保留实际 wheel、测试范围及失败。

[三种接收 FP32](evidence/reception-numerics-20261118.md)9候选中1 within/8 finite，
99比较41原阈值超差；无观察到的 support/int64/dtype/None 差异，9次各自新进程
恢复更新均阈内。NPU 内部捕获到界限，未建立全部超差的最早因果起点。
BF16 接收微测48行27 within/21 finite；组合路径第2更新出现2/16输出支持差异。
双NPU FP32同进程恢复仍有13参数字段超差，安装 placed BF16 保留恢复超差。
static_view 单独对比的 FP32 推理 exact、VJP 阈内，连续更新仍有有限超差。
CPU FP64/Decimal只支持各自有界见证；没有修改原阈值或把所有差异归为已证明的舍入。

## FP32 主力训练

五组100次真实连续更新均有限、窗口间状态detach；各自配对恢复实际更新阈内，
五组新解释器都已从100推进到101。[训练证据](evidence/reception-training-20261118.md)
区分这两种恢复事实与计时。稳定吞吐是第3–100次输入位置总数/完整更新时间。

| 配置 | B/L | 输入位置/s | allocated/reserved GiB |
|---|---|---:|---:|
| FC-K 9.33M | 128/256 | 30333 | 37.36/39.58 |
| HB source 15.69M Settle | 24/256 | 2644 | 39.28/43.51 |
| HB source 15.69M Timed | 12/128 | 370 | 42.19/47.25 |
| HB incoming light64 17.88M | 24/256 | 1618 | 38.54/44.90 |
| HB source 494.11M | 4/128 | 267 | 33.84/35.62 |

`configs/recommended` 已选入五个 FP32 预设，0.5B由B1改为B4，默认索引仍gather。
[执行建议](execution-recommendations.md)分别选择训练、含图前向、首输出、decode和批量流。
三种接收的B8/L128三模式A–D完成；0.5B B4/L128完整训练的 Tide/A/B/C/D 为
293.76/89.42/90.94/4711.94/10275.27输入位置/s。这些是零态短测，不能替代连续吞吐。

[性能归因](evidence/reception-performance-20261118.md)包含公式、伪代码、双方形状与
设备轨迹。SSD完整小模型训练吞吐约+19.25%；接收组合约+13.17%。融合不总是更快，
重算的内存/时间交换仍保留。0.5B分配约占该次侵入式前向27.69%，K packing执行/选择
约5.25；Transformer A 的1712窄层有显著调度间隙。诊断与正常时间不拼成占比。

## 连续推理与缓存

0.5B、9B、17B短请求A–D及FC约1B持续A–D均已完成；9B/17B的Tide主干
另已完成256次实际增量，不能把短请求A–D时间套到长负载。
9B B1 gather/static_view稳定decode为3.130/3.428位置/s，allocated峰基本不变；
物理卡和源码不同，工作/支持/时间摘要相同，保留测量边界。
17B双卡per_device/batch4持续decode约2.086位置/s，每卡峰35.89/38.00GiB。
17B串行batch1→batch4与batch4串行→并发分别约+33%/+64%；前者有一项Full活动差异，
后者记录的工作/支持/时间摘要完全一致。不能把两种改动的活动差异混为并发影响。
9B B2有限点完成，decode5.208、流293.46位置/s，峰39.77/42.84GiB；只有16次decode。

FC/Settle/Timed缓存观察器三项全部通过，均有真实逐owner窗口淘汰证据，原时刻和
SSD/Conv/pending检查0失败；小配置诊断不认证每个大模型bank都已填满。
HB约1B批量流与Timed长流的A–D也已全部完成。Timed Light64 B4的Tide/A/B/C/D
稳定decode约6.86/9.58/9.63/197.00/197.19位置/s；该点Tide仍慢于A/B。
[推理证据](evidence/reception-inference-20261118.md)给出首输出、prompt就绪、
首增量、p95、持续吞吐、实际活跃量、A–D构造器及显存。它们不含任务头，不能称语言模型TTFT。

完整模型仍为PyTorch，LibTorch仅区域评分切片；CUDA、HCCL/peer DMA、全模型C++、
compiled/resident训练没有新增资格。任务头、数据、loss和任务效果仍属下游。

## 资源、用户文件与接续

本轮全部作业使用协作队列独占准入，但主机共享、外部任务不受控制；没有把线程内
共享NPU授权变成仓库默认。冻结源、checkpoint、trace、环境、工作树均保留。
账户最近约152GiB/300GiB软限、700GiB硬限。原始命令和身份在各`artifacts/agg-*`下；
`artifacts/agg-inference-preflight-20261118/jobs.json`已刷新终态。52个已提交作业均无
活动MainPID或仍存活的已记录后代；外层50 passed、2 failed。两项原失败为首轮CPU集成
与BF16接收微测，修复/超差各按证据保留，没有抹去失败。全部冻结归档常规文件与各自
提交一致；最终核对在同目录`release-audit.json`，简表并入[验证证据](evidence/reception-validation-20261118.md)。

四份用户未跟踪文件保持原字节、不纳入提交：
`docs/tide-learning-lab-TL-REQ-004-handoff.md`、`docs/tide-learning-lab-TL-REQ-005-handoff.md`、
`docs/tide-learning-lab-TL-REQ-006-handoff.md`、`docs/upstream-lh-semantics-discussion-20261113.md`。
SHA在本轮preflight。参考仓库、公共环境和下游未修改；没有下游发布或验收。
本次有限计划已完成，阶段实现与证据均分开提交并正常推送。接续以实际HEAD/工作区为准，
不重交已完成任务；[ROADMAP](ROADMAP.md)只保留交付索引和具体后续机会。
