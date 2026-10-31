# 当前路线图与验收范围

本页源码定位使用清理后的提交号；历史实测身份仍在原报告中，见[迁移与映射](history-migration.md)。

数学固定`0770593838a37c790a568226930567ed60f99b83`的core-5与带权局部定义。
[STATUS](STATUS.md)是唯一当前交接；[execution-flows](execution-flows.md)是持续契约。
2026-10-31新阶段正式覆盖CPU/NPU/GPU有限矩阵、同模型多卡放置与人类CLI。允许有限浮点超差候选在显式探索模式继续成本观察；原严格失败/阈值保持。下表旧“停止扩大”指严格资格结论，不阻止本轮已授权的探索观察；非有限、非法配置、通信、求解器与容量错误仍失败。
原始大文件的在线保留按[产物规则](artifact-retention.md)；退役后复查旧失败须重新生成材料。
问题到代码、输入和证据的入口集中于[复现导航](reproduction.md)；下表仍是唯一状态与停止/重启条件。
任务终态、相关资格通过和限制解除是不同结论。完成记录由Git与[证据索引](evidence/README.md)保存。

## 本轮已交付

1. 已交付构建资源入口（2ccae1a）：默认全部可用CPU，可选half/正整数；测试资源与性能线程独立声明。
2. 公共planning与配置默认展开、模型/工作负载/执行/测量身份、公共eager分派已实现并定向检查；固定CPU资格1709b13/336通过。保持旧材料可复现。
3. core-5 FC/HB/LH启发拓扑、自然维度与约1B/8B/16B生成器、有限目录/运行/查询已交付；[固定设施资格](evidence/core5-study-clients-20261029.md)与[自然规模测量](evidence/core5-natural-study-20261029.md)分列。
4. 三客户端精度/线程/packing/peer、standalone mixed和resident实际梯度/SGD/master/磁盘恢复已验；372 CPU365，878a5a4相关CPU43、CPU6/NPU8消费者、Python NPU3配置。旧24个recipe完整训练包含于前述检查，不重复累计；其中FC单层/混合退化范围已在原报告更正。
5. 25个有限配置中7项完整通过：两种中型FC×三个CPU FP64客户端及LH约1.051B CPU FP64。区域批B4实际生效，另有独立NPU profile；FP32/资源失败不作吞吐，8B/16B仅静态计划。
6. [导航与修改维护映射](core5-study.md)、可直接查询仓库审阅证据的入口已交付；LH/core-3与本轮经验集中于[performance-lessons](performance-lessons.md)。
7. [显式native checkpoint预算](evidence/core5-checkpoint-budget-20261029.md)已交付并相关资格通过：默认4GiB保留，5.25GB实际文件保存/加载成功；该1B后续窗口非有限梯度另列，未通过完整续接。

8. [实际覆盖与导航修正](evidence/core5-coverage-correction-20261030.md)：f456160相关73项通过，含26个真实配置×PyTorch/native的52项CPU FP64训练/恢复；自动配置事实与逐配置正确性/性能索引已交付，旧性能和设备边界不变。

9. [多实验仓库协作与安装消费者](evidence/downstream-contract-20261030.md)：e17f9ea交付依赖锁、请求/交付/验收及AGENTS模板；4项单元、8组CPU安装/独立训练/新进程精确恢复、4项CLI拒绝通过。下游自行验收再升级；workers2原失败另列。

阶段提交后继续。旧边界的停止条件仍有效；有新方案和证据才重启。CUDA新任务不继承旧CPU/NPU失败或通过。
不以不支持声明代替约定功能，不把小规模通过称为全部平台/尺度通过。

## 已有公共能力与入口

| 范围 | 契约和证据入口 |
| --- | --- |
| PDG反馈、非对齐TimedDAG、一般SettleGraph，同秩区域/独立端口/原样嵌入 | [总对应](semantic-map.md)、[Settle](settle-embedding.md)、[结构化表示](structured-records.md) |
| 五记忆、原始时间短卷积、十种发送、适用的评分/证据/key、来源分类与共享 | [KV](weighted-kv-sharing.md)、[循环](weighted-recurrent.md)、[分配](weighted-allocation.md)、[发送](weighted-sending.md)、[评分](weighted-scoring.md)、[来源](weighted-source-binding.md) |
| 在线streaming/prefill、实际矩阵批与独立VJP根 | [packing](weighted-packed.md)、[区域批](weighted-region-batch.md)、[计算共享](weighted-compute-sharing.md)、[8d97020资格](evidence/core5-independent-projection-20261029.md) |
| CPU、混合、常驻FP32/FP16、单/双NPU训练和精确续接 | [能力矩阵](execution-capabilities.md)、[常驻训练](resident-weighted-training.md)、[节点训练](evidence/core5-weighted-node-training-20261028.md) |
| 参数清单、core-3参数量匹配拓扑/同材料、有限完整训练计时和诊断 | [工作负载](weighted-workloads.md)、[精度](weighted-precision.md)、[历史经验](performance-lessons.md) |

上述入口说明已实现能力和限定资格，不构成所有配置/拓扑/尺度/平台的笛卡尔资格。
下游拥有数据、任务头、loss设计、训练配方和效果搜索；基座验证机制与数学对应。

本轮GPU有限推理的结果与六项处置见[规模推理报告](evidence/core5-cuda-inference-20261031.md)；三客户端、显式精度、单卡mixed/resident各自有实际范围。旧训练停止条件不自动否定独立推理资格，也不因推理通过而解除。

[三平台首组固定矩阵](evidence/core5-platform-baseline-20261031.md)已完成98项正式流程和4项独立观察，含三平台B128/B512及A/B/C；43项严格通过、其余保留原数值失败。大参数、多卡、resident容量与人类完整演练继续如下，不能由首组外推。

## 当前授权待办

- 三平台小中矩阵、B1/8/32/128/512及A/B/C、FC/HB/LH约1B均已入库；NPU1B/B512旧边界重放已逐位不同、采样布局相同；同进程即时解码诊断待终态。不重跑原矩阵，最终汇总189目录的已测/拒绝/替代范围及目录外后继。
- [同模型容量](evidence/core5-platform-capacity-20261031.md)：CUDA4.561B单/2/3/4卡、9.123B单卡拒绝/双卡实际点，NPU12.802B单/双/三卡已审阅。NPU八卡队列超时、38.406B单/双卡拒绝均保留；三卡已构造并保存首个有限窗口，第二次保存时1800秒超时，原I/O进程已排空；1B七卡已完成/离散/恢复通过，数值探索；按用户计算优先指令不强求38.406B完整存储流程重跑。
- [训练与通信](evidence/core5-platform-training-capacity-20261031.md)：原1B配方非有限保留，新loss系数1e-9的CUDA双卡完整训练/离散/精确恢复完成、独立数值超差；NPU同一新任务排队。6095c6a已补齐训练CPU快照主存预算、固定45项检查通过。NPU原blocking单次诊断完整通过但未复现旧崩溃，不称修复；三通信full独立根已分别EE1017失败，完整独立推理/恢复先通过；4485697已定位ordinary候选all根autograd阶段SIGSEGV；单卡all根为输入梯度数值超差，显式同步诊断与独立loss训练另列。
- [checkpoint成本](evidence/core5-checkpoint-cost-20261031.md)：磁盘/tmpfs及36baeb7优化后CUDA对照已完成，少保存不称算子加速。CPU固定110项、CUDA返回6项及两卡full通过；NPU新实现小型回归继续；用户最新优先计算，未开始的NPU大tmpfs对照及38.406B周期保存后继取消，不作为完成门槛。暂不扩盘/重启，任何tmpfs仍与主存联合准入。
- 公共CLI、下游模板与CPU/NPU/GPU[人类完整流程](evidence/core5-human-workflow-20261031.md)均已实际演练；原base独立根失败不由下游通过解除。六项packing已按新范围重审，待上述结果后完成最终选型；当前停止条件不盲目扩成融合工程。
- 7×T4站点/迁移步骤已交付；实际driver/runtime/Toolkit/LibTorch、sm75算子、通信和训练/恢复待真机，不由模板认证，也不移除当前resident能力门禁。

## 逐项推进与停止条件

| 项目 | 当前终态与证据 | 停止或重启条件 |
| --- | --- | --- |
| NPU FP32约1B/B512推理续接 | [本轮规模证据](evidence/core5-platform-scale-20261031.md)：执行完成、原始解码比较通过，下一完整窗sample0/node1 KV keys有3669/32768坐标逐位不同，最大1.7166e-5；独立CPU对照未执行。B128与其他规模项结论分别保留 | 保留原checkpoint/输入，不重复相同全流程；重启须比较恢复前后布局、实际算子及首字段的有界见证。不同于跨设备容差，不因explore接纳exact失败 |
| D64 FP32完整训练 | 2ada00b单行私有VJP消冗后原两步native/Python仍有记录、梯度、参数和SGD槽差异；两调度自身恢复精确。见[eager证据](evidence/core5-eager-boundary-20261029.md) | 本候选数值修复停止；不重复九项分段精度/外层汇合探针。重启须有覆盖局部算术、中间伴随与共享贡献的具体新方案及独立根证据 |
| D2048 FP32 | [实际首写入](evidence/core5-boundary-isolation-20261029.md)：来源+私有批链194坐标失败；标量FP32对局部FP64也有289坐标失败，连续局部FP64后cast为0。原quad残差拒绝已修复，原全图keys失败未解除 | 没有公共整段算术/VJP方案前停止9B全图重跑；局部更准确不保证与旧标量FP32相同。显式全CPU FP64已有独立原窗通过证据 |
| 自然FC/HB较大训练 | [878a5a4有限测量](evidence/core5-natural-study-20261029.md)：中型FP32完整keys/values失败；FC/HB约1B FP64分别有2个SGD槽坐标/59个梯度坐标超原容差。LH1.051B FP64原两窗完整通过，不能外推 | 本轮一次流程终态，停止相同失败候选的更大吞吐；重启需具体算术/伴随方案及独立根与完整更新证据，不提高容差 |
| FC1B NPU FP32下一窗 | [9bfabbb复验](evidence/core5-checkpoint-budget-20261029.md)已越过旧4GiB文件限制，5.25GB文件保存/新owner加载返回；原owner下一窗出现非有限梯度，恢复分支与CPU独立核对未执行 | 原输入配置/平方loss/SGD及结果保留；大文件已按用户授权退役，重启须重新生成。没有首个非有限伴随定位或明确新数值方案前停止重跑；修改下游loss/训练配方是另一个配置，不称原失败修复 |
| 反馈PDG FP32 | 原W32/B4四窗记录/梯度/SGD槽仍失败，参数通过；reference/streaming/prefill三者自身恢复均精确。显式FP64四窗独立通过 | 单行VJP方案到此停止；新方案须覆盖反馈递归与跨窗更新，不能仅核对最终参数 |
| FP16 GDN/default AdamW | [归一化VJP内部FP64](evidence/core5-boundary-isolation-20261029.md)仍第二步差.00198364；两个原strict xfail保留。FP32对应见证已由9e6dc52解除 | 不改epsilon/loss scale/输入或容差；仅提升归一化反向候选停止。重启需新的完整局部伴随/舍入方案 |
| NPU运行时初始化顺序 | 固定资格暴露CPU autograd后才注册NPU触发device_ready_queues_断言；独立标量进程已确认late失败/early通过，独立NPU资格14项+eager6组通过 | 应用先resolve_device再首次autograd；已启动CPU引擎的晚注册路径保留限制，等待运行时修复。当前门禁分进程，不改数学实现 |
| 下游native workers2跨进程恢复 | 本轮FC/KV FP32小型安装消费者独立两步通过，但连续三步与两步保存/新进程恢复一步的梯度、SGD槽逐值不等；关闭autograd多线程仍失败。示例显式workers1固定资格通过，见[证据](evidence/downstream-contract-20261030.md) | 原workers2失败保留；停止重复相同候选。须先定位并行构图/反向累加顺序的具体差异，再用原材料/独立根/跨进程恢复验证；本轮只交付协作规范和单worker示例 |
| 本轮NPU blocking小训练恢复 | [多卡证据](evidence/core5-platform-multicard-20261031.md)：原FC8/d8三更新完成，换owner精确恢复阶段SIGSEGV/-11；ordinary/cpu_bridge同材料严格通过，不能互相替代。报告b3dc53c补监督状态，未修数值执行 | 保留小checkpoint与配置；一次有界faulthandler诊断未复现且完整检查通过，旧SIGSEGV仍未定位；不再重复相同试验，重启修复须有流/owner生命周期新依据及原完整训练/恢复检查 |
| 默认异步双NPU eager VJP | CPU桥完整delta-pdg仍EE1017，未完成4更新/恢复；小桥接通过不足以认证。显式blocking是另一个配置 | 同环境同事件方案停止；重启须运行时修复或新的事件所有权方案，先反例再完整图/独立根/恢复 |
| 单行私有VJP | Python/C++直接建图、轻量一阶/版本守卫已交付；固定2ada00b CPU517 passed/2原strict xfail。NPU相关开发已通过；[57308e7 NPU固定资格](evidence/core5-resident-boundary-20261029.md)通过，[有限完整对照](evidence/core5-boundary-performance-20261029.md)已终态，CPU未见收益 | 以实际端到端成本决定下一步；不从减少私有图数直接推断收益 |
| 异形字段packing与小提交 | 按device/dtype合并真实字段、8MiB分组、按shape/offset恢复；int64独立，None/零/版本守卫保留。mixed-a/c完整核对通过、单次1.505→1.477/2.183→1.848s，allocator峰值不变；小算子/scalar仍显著 | 若小算子/提交仍主导，须据新profile提出具体边界；无真实可合并域不盲目扩大batch |
| resident组合矩阵/窗口反向 | 因子KV与实际分段已交付并资格通过；[57308e7](evidence/core5-boundary-performance-20261029.md)及[b7e0a29](evidence/core5-compact-fibers-20261029.md)两次有限观察，局部槽压缩后medium构造1057ms、执行/等待700ms；构造成本仍显著 | 当前测量终态；下一候选为显式模板重绑。完整adjoint绑定journal/root/参数地址，须先独立验证绑定/版本/跨窗，不能按shape缓存旧对象 |
| 较大NPU训练/跨卡 | 已测停止。D64匹配9076975参数，native单/双NPU显式blocking自身磁盘恢复精确，独立CPU均在node89 keys严格失败；[有限证据](evidence/core5-boundary-performance-20261029.md)保留15.685/16.840s失败成本 | 原eager数值失败按首行重启条件处理；不重跑同候选。resident的新布局与新资源边界分列，不以参数量匹配或自身恢复替代独立正确性 |
| resident局部纤维填充 | 已交付并[资格通过](evidence/core5-compact-fibers-20261029.md)：b7e0a29按物理入边/端口缩小槽；CPU两环境各10、公共NPU56、SDK7组件/dtype通过。medium完整窗2.794→2.470s，peak allocated514.8→376.4MB，完整核对/恢复通过 | 本候选实施、相关资格及一次对照闭环；单次观察不宣称普遍加速。跨节点数值批和减少物理缺席行仍需新的具体实现及独立根证据 |
| NPU resident流资源 | 已定位并停止。相同D64/capacity的b7e0a29构造报207005；CANN日志明确Too many streams/EE1023，实际SQ/CQ=1984。未进入warmup，尚无完整训练、恢复或吞吐结果 | 重启须有界服务分派/复用或节点合批；先验证嵌套通知、在线顺序、共享workspace、独立VJP根、peer和恢复。该观测不是通用流上限；不靠加HBM预算或主机预选序列绕过 |
| 本轮NPU resident B512构造 | [新增证据](evidence/core5-platform-resident-20261031.md)：有界布局已通过HBM准入，aclnnInplaceCopy/361001，厂商日志EE1019/507002 stream task buffer full；未进入warmup。B128完成/恢复通过、数值探索 | 相同配置停止；先定位静态bool搬运调用者/段数，再独立验有界整型暂存还原或设备服务拆分。须保持其他字段、int64、重复owner、独立根、窗口/恢复和暂存预算；不靠主机逐事件flush或增加HBM预算 |
| NPU resident在途任务资源 | [自然HB medium](evidence/core5-natural-study-20261029.md)：Python/SDK均在warmup反向服务提交报507002；短日志已复制并固定hash，明确stream task buffer full/EE1019。不同于流数量207005，也无HBM耗尽证据 | 相同原配置停止；下一候选为有界提交/服务复用，先验证通知的嵌套与所有权、窗口顺序、独立根及恢复。不缩减原输入或改成主机逐事件推进冒充修复 |
| FC1B resident局部预算 | tensor12GiB、total48GiB时构造触发KV Full guard；B1/d1024/cache32前置估算需要当前剩余约3.499GB，失败节点剩余额度未导出。不是总内存OOM结论 | 本轮容量调查终态；需局部预算/实际保留量诊断、workspace生命周期或有界节点批方案后重启，不盲目提高所有分区。8B/16B无实测吞吐 |
| native大文件checkpoint | 9bfabbb默认4GiB及显式正int64预算、原字节、拒绝原子性与实际>4GiB保存/加载已验证；CPU32+6消费者/SDK8消费者通过 | 文件硬上限已解除；整体CPU codec副本、每张量2^28元素与集合上限仍明确存在。分片/流式codec需新方案；不能将文件预算当峰值内存保证 |
| 诊断与训练观测设施 | scalar/native实际分配观察已通过；常驻20–22失败部分行、实际梯度/SGD槽/canonical共享owner解码及磁盘换owner检查[固定资格](evidence/core5-resident-boundary-20261029.md)通过 | 不补造未计算残差，不以参考重算结果。fake布局测试必须有真实公共owner消费者作独立检查 |
| 容量/正支持 | 显式正int64容量选项、静态Settle上界与分区预算；原正支持下溢/区间塌缩仍拒绝 | 保留表示下界、同时在途容量与实际预算拒绝；无静默裁剪/截断/改活动集合。不以保守fixture预设作为引擎上限 |
| CUDA/x86_64 | [RTX3090首轮证据](evidence/core5-cuda-eager-20261031.md)：三客户端构建/加载，PyTorch及native单卡两dtype各19/19完整通过，standalone FP32 23/24、FP16 22/24；普通复制72单元与显式Python CPU桥完整delta-pdg通过。[单卡resident固定资格](evidence/core5-cuda-resident-20261031.md)56组件/dtype+53公共项通过；[FC-small设备时间线与有限对照](evidence/core5-cuda-finite-20261031.md)将实际常驻、完整正确性和成本分列 | 本轮小图GPU未见对CPU收益，不外推规模。12方向无直接P2P/原子，多卡resident硬件限制不解除；完整三卡门禁不放宽，不改通信协议 |
| CUDA standalone数值/恢复 | 原FP32 fc/ssd第3步AdamW后bias差3.49e-6超原容差；scalar同样失败，CPU FP32/FP64通过。mixed-a另在恢复梯度2坐标逐值不等。FP16 feedback/affine read_out与feedback/ssd evidence_weight第三步失败在CPU复现同值；SSD显式scalar在CPU/CUDA通过，affine两种packing仍失败 | 不重跑相同候选充当修复；先定位局部算术/伴随和更新放大，保留原输入/epsilon/容差。新增独立根已随原矩阵验收，scalar通过不冒充packed修复；不继承旧CPU/NPU GDN失败 |
| CUDA native双卡精确恢复 | gated-hb第四步nodes.0.kernel.read_out梯度3/4坐标逐值不等，最大4.44e-16；前3步完整通过。稳定字段packing修复自身反例但未解除该失败 | 停止重复地址排序候选；须有具体累加/构图顺序新方案，再验独立根/共享owner/optimizer/原精确恢复。普通复制成功不认证直接P2P、NCCL或capture |
| CUDA有限图完整状态/容量 | [32edb0f有限对照](evidence/core5-cuda-finite-20261031.md)：首轮37项18通过，18项keys/values/content/output严格差异，1项LH medium resident queue64容量拒绝；FC medium显式scalar亦失败。HB small resident两客户端同一values坐标差1.11759e-6。LH queue452越过容量且磁盘精确续接通过，但独立第0窗keys差2.26498e-6 | 相同数值候选停止扩大吞吐；下一步须沿实际首个差异的投影/归一化/共享伴随定位，提出有独立根锚点的新算术方案。容量按静态物理边/端口预算，原queue64失败保留；不以自身恢复或更大容量替代独立参考 |
| CUDA resident后续性能 | [同一时间线/有限成本](evidence/core5-cuda-finite-20261031.md)：大量indexing/逐行累加/guard，完整训练有多次graph实例化；小型LH物理局部行120对实际40。边界、反向构造和解码仍有成本 | 下一候选先验证实际局部行合批或显式反向模板重绑；需保留独立根/journal/版本/跨窗/optimizer及精确恢复。不按shape缓存旧owner，不由减少kernel数推断端到端加速 |
| CUDA d256 FP32推理 | [本轮推理证据](evidence/core5-cuda-inference-20261031.md)：B1/8/32原keys3/768坐标严格失败，自身磁盘续接精确；两客户端首差诊断保留，显式FP64另配通过 | 相同FP32候选停止扩大；重启需覆盖上游fiber/Full/KV传播的具体算术方案与完整独立记录，不能仅提升KV写或放宽容差 |
| GPU规模推理与客户端选择 | [本轮有限目录/结果](evidence/core5-cuda-inference-20261031.md)独立验推理、完整返回及续接：1.010B FP64三客户端B1、native B8和2.085B native B1通过；CPU小图、mixed/resident各有范围，原训练失败不解除 | 仅按已通过的配置选型；新的长窗口、FP16规模、其他拓扑/共享或大尺度训练重新预算并独立验证，不扩成8B/16B扫描 |
| 元数据/完整返回packing | detach bool合并减少取回但无稳定整窗收益；显式return-packed在部分CUDA配置降低成本，NPU同材料反增；默认scalar与原路径保持，见[六项经验](performance-lessons.md#gpu-推理批次与完整返回) | 本轮两个有界候选闭环，不再扩统一搬运层；实际重复packing主导时才比较一个summary复用方案，异步/跨窗缓存/最小返回需新契约 |
| 大宽度批量与resident布局 | 1B实际独立投影8MiB预算回退，私有BMM仍2行；小图长度桶随B增大得到填充，resident44–47%缺席行不等于耗时浪费；Full保守守卫及FP64域拒绝保留 | 更广节点/多head ATen批或设备活动行须有占时/资源依据；不盲目增预算、不用主机轨迹冒充常驻；反向模板须独立journal/root/版本重绑方案 |

原失败记录和严格断言不变；新增错误与旧失败分别保留。前轮构建/资格与串行有限性能均已终态，
不重复旧矩阵，不把参数量匹配等同函数、工作量或模型效果匹配，也不宣称低成本优化已经全部完成。
未解除边界都有停止及重启条件；活跃作业与具体接续命令只见STATUS。新结果只更新本表、相关契约、证据和performance-lessons。
