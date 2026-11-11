# 当前路线图与验收范围

本页源码定位使用清理后的提交号；历史实测身份仍在原报告中，见[迁移与映射](history-migration.md)。

数学固定`0770593838a37c790a568226930567ed60f99b83`的core-5与带权局部定义。
[STATUS](STATUS.md)是唯一当前交接；[execution-flows](execution-flows.md)是持续契约。
2026-10-31新阶段正式覆盖CPU/NPU/GPU有限矩阵、同模型多卡放置与人类CLI。允许有限浮点超差候选在显式探索模式继续成本观察；原严格失败/阈值保持。下表旧“停止扩大”指严格资格结论，不阻止本轮已授权的探索观察；非有限、非法配置、通信、求解器与容量错误仍失败。
原始大文件的在线保留按[产物规则](artifact-retention.md)；退役后复查旧失败须重新生成材料。
问题到代码、输入和证据的入口集中于[复现导航](reproduction.md)；下表仍是唯一状态与停止/重启条件。
任务终态、相关资格通过和限制解除是不同结论。完成记录由Git与[证据索引](evidence/README.md)保存。

## 当前计划：批量张量贯穿与架构验证（已授权实施）

用户已确认[批量接口与模块替换](execution-flows.md#批量接口与模块替换)的总体约束，允许必要时
整体重写。本版P1—P6已经用户授权，正在从批量契约及区域/KV整链开始实施。
生产代码基线为`63daf0f`，计时基线和原始资格仍按下面固定报告的实际实现提交引用。
KV保持性能主线；区域分配是同等重要的完整边界，覆盖描述/评分/证据、控制、各发送与求解/VJP，
一直贯穿活动支持、Full、消息与下一阶段聚合。早期GDN/SSD小例用于检验接口，不转移FC主线。

进行中：区域列实现及固定证据保持；[张量窗口接口](weighted-tensor-windows.md) 已形成
Python/C++ KV 整链、Python 四循环/卷积与异构组合、实际消息列及 raw 恢复。默认 eager
新增直接 B×L/一般 Settle 端口输入，按能力分派并声明后备；现有事件消费者须显式迁移。
[01cde3b固定整链资格与FC成本](evidence/tensor-windows-20261111.md)已完成；
[9fa852c KV位置列](evidence/column-positions-20261111.md)的CPU/NPU安装训练与恢复亦通过。
native 循环、split placement/resident、多设备和宽/多节点代表仍未完成，
不把已有一般执行器功能当作这些新路径已经迁移或优化。

原计划的三类未完成工作继续：实现/数值/异步/容量限制，LH及core-3经验的实际迁移，
可复用正确性/性能设施与架构导航。既有skills、独立基线、诊断收集器和KV块优化复用，不重做。
已知的批量输出拆行→区域重组→反向行回填是源码审阅发现；既有profile支持其结构成本，
不把观察算作新实现或各项剩余耗时的完整因果归因。
具名入口为[查询块](../cpp/src/weighted_kv_block_vjp.cpp)、[区域VJP](../cpp/src/weighted_region_vjp.cpp)、
[行列装配](../cpp/src/weighted_private_rows.h)和[消息交付](../cpp/src/delivery.cpp)；其余调用关系见[架构导航](architecture.md)。

| 阶段 | 工作与继承的未完成项 | 交付/判断依据 |
| --- | --- | --- |
| P1 具体契约与审阅 | 沿Aggregate→记忆/Describe→区域→Full→路由/输出及反向审阅生产者/消费者；定义张量列、ragged布局、参数/状态所有权、观察根、容量和异步寿命。KV与至少一个GDN/SSD小例及混合区域检验私有状态不同。区域字段/算法按实际语义分组，不强行统一形状/求解器。补现有仓库skills的具体模式和检查项 | 可实施的输入输出/VJP/解码契约、迁移清单、独立反例和需改符号；不再要求新计算迁就旧逐事件接口 |
| P2 加法对照与紧凑实现 | 复用直接FC与Transformer及KV/SSD基线，补默认任务所需的非空左态、共享、末状态、真实训练/恢复边界；建立区域张量公式/VJP锚点。有限调查直接FC的B×L Full和带权Attention算子能力，以及各层增量。普通任务投影与完整记录/独立根分别标明 | 同材料、同观察/计时范围的直接公式→公共块接口→小组合成本；数学额外工作、布局、构图/反向、同步与内存分别有证据，Transformer不作等价oracle |
| P3 默认FC整链贯穿 | 以KV为首要候选，区域描述/控制/求解/VJP同步改造；消息及支持以批量数据进入下游，消除边界的强制拆行重拼。配置/能力按块分派，检查与optimizer纳入完整路径；保留一般图合法推进与声明观察能力，必要时迁移API/codec | FC4/FC16的CPU与NPU mixed-c三模式、阶段差额、图/算子/内存、训练与恢复；小型PDG反馈、非对齐TimedDAG、一般Settle端口、重复边/共享/None零等定向检查；Python/native与standalone实际接入。FC为第一个整链汇报点，不以局部提速代替架构验收 |
| P4 有限候选与组合扩展 | 将当前五记忆/固定时间短卷积，以及评分/证据/key、十种发送与适用正则接入新块契约。区域的总量/阈值/受约束求解、Attention份额、比例、硬选择和小数零温发送分别保留实际支持与声明反向；优先打通功能，重性能按实测排序 | 每个候选有可用实现，合法组合可配置；逐项及有判别力的小组合覆盖，无需穷举。不同私有形状不逼迫逐行返回，不同分配语义不偷换同一Top-k；KV位置/过期/sum等域与循环序列VJP分别记录 |
| P5 代表规模与执行路径 | FC差距有可复现解释后，沿用宽FC、176节点HB/LH、sum/qry、GDN/SSD、约1B/9B代表；明确LH数学差异。按受影响能力推进CPU线程、mixed-a/b/c、resident、多卡；处理已有异步反向挂起、workspace/服务/容量边界，不能只换blocking或增预算宣称解除 | 选有限CPU/NPU点回答选型，记录实际候选/Full/消息与计时；resident就绪/选择/合批/推进及前后向/optimizer/跨卡分别验收。旧限制逐项修复或有界停止，GPU真机继续延后 |
| P6 治理、交付与持续复用 | 贯穿各阶段维护语义对应/架构/复现/性能经验；按hash清单继续逐文件/函数审阅，已审与未审分列。完善安装客户端、替换局部模块、实际配置/路径可见性、诊断和恢复演练，提供消费者迁移说明 | 固定实现先提交，相关资格和性能证据随后独立记录；新线程和下游无需会话/旧raw目录即可导航复现。旧实验锁和证据不改写；全仓未审不冒充完成 |

### 测量、阶段边界与有限停止

- 小检查优先独立公式/调度、字段与共享梯度、非空左态及精确下一窗。布局变更检查外层/私有图、连接边、反向执行和保存存储，不能仅计一个Function或一次外层调用。
- FC主尺度沿用B8/L128/d128/H2/FFN512及原配置；小型profile/反例单列，不按比例外推。无梯度推理、含grad前向、完整训练分开；TTFT实际测量，吞吐按真实输入位置换算。完整成本含准备、路由、packing、反向、optimizer及同步。
- 配对尽量固定同一设备、参数、输入、执行/精度与CPU池；若不同必须显式标记。小耗时测量可有限重复，大规模仅在阶段边界或新证据下执行；不恢复旧实验矩阵。
- 各阶段记录与紧凑基线的增量和明确保留/替换理由；未解释的主要差距仍属未完成。有限调查没有闭合时交付具体证据、候选和下一边界，不无限重复同一局部优化，也不凭局部收益宣布总体完成。
- 原FP32严格失败及旧异步/容量失败保持身份；舍入按有界首差/高精度诊断继续独立工作，非有限、控制/依赖错误、求解/通信和精确恢复失败分别处理。
- 可用CPU/内存和空闲NPU按实际收益并行分配；共享NPU须当前授权。正式计时避免项目自身争用；大写入前查容量，可用有界shm暂存，存储确实阻塞则及时汇报。长任务固定源码、独立background.slice和有限预算。
- 本表是已授权计划；设计原则和进行中的实现不等于全部能力已完成。实际任务状态、下一个动作和活跃作业只见STATUS。

## 已完成：KV块接口与重复诊断

[实现及固定证据](evidence/kv-query-blocks-20261111.md)：5ff3ce2交付隐藏KV前缀bank、
Read/Full查询块与块级伴随，Python/native默认能力分派；公开末状态、独立根和一般后备保持。
598b4fb计时/范围标记；206bc9b后备借用预检、重复诊断配对与profile/JUnit/终态附件。
CPU/NPU小图独立根、三客户端多步训练/新进程恢复与实际设备构建分列通过，原启动失败保留。
同轮FC16默认/packed-v2训练CPU5.835/11.601秒、NPU10.928/15.124秒；直接FC仍1.390/0.675秒。
99.2%的Read/Full实际命中块，B2/L32含私有图节点18909→10163；性能差距尚未闭合。
原FC16 FP32任务投影严格失败保留；同FP32材料提升FP64后值/末状态及任务梯度通过。
这不代替完整尺度全部根、多步optimizer、支持轨迹、下游学习验收或其他设备/规模资格。

该有限实施已结束；后继由页首P1—P6计划整合，区域列VJP只是整条批量接口的一部分。
NPU区域求解范围约0.178秒，region nested-grad约2.108秒，不把剩余差距笼统归给优化求解器。
这些是原实现的主机范围计时，非新实现结果或精确设备归因。当前任务状态仅在STATUS。

### 前置三层调查已完成

2026-11-11授权调查的[固定结果](evidence/fc-three-layer-comparison-20261111.md)已补齐FC4/FC16三模式、
对应阶段差额、私有根省略、CPU/NPU实际算子及同材料数值诊断。默认/直接FC16训练成本比CPU8.78、
NPU22.18；去掉包装仅部分改善，主要待设计边界为状态bank—Read/Full块—块伴随汇总。
直接FC自身仍有按B循环、小投影及算子能力成本；4D SDPA未获得NPU融合Attention。
完整FP32任务投影严格失败，同材料FP64值/控制/支持/终态通过、两梯度字段各1坐标仍严格失败；
该有界数值诊断不扩大为任意模型安全结论，也不阻塞后续独立设计。未新增完整训练轨迹资格。
该前置调查未改生产计算或采用省略根变体；其后的查询块实施已见上节，新推进顺序以页首计划为准。
CPU/NPU并行、复用环境和有界shm解决了本轮存储安排，没有发生磁盘阻塞。终态及接续仅见STATUS。

## 上轮有限实施：已完成


加法基线→公共批量接口→默认调度的本轮实现/资格/有限成本已完成，见
[固定资格](evidence/additive-interface-qualification-20261110.md)与
[三层成本](evidence/additive-interface-performance-20261110.md)。数学和通用后备不变。
采用FC/KV/SSD非零raw左态/共享/训练/恢复投影基线、默认KV Full共享bank矩阵，
修复resident首次借用局部计费。原严格数值失败和原启动/见证失败保留，成功补验独立记录。
CPU/NPU/三客户端小型训练与恢复、12个代表训练/精确下一窗及NPU约1B推理/精确下一窗
分别有证据；不把投影基线、完整记录、独立根或规模成本混作一种资格。

FC16完整训练CPU1/NPU为12.283/15.511秒，首token2.737/4.055秒；NPU完整训练未改善。
局部前向收益、独立快照反向负结果和默认入口与直接公式的明显差距均保留。
本轮有限问题已有结果，不继续重复同候选或自动扩大硬件/组合矩阵。按用户要求先汇报对齐。

| 后继项 | 当前边界与重新实施依据 |
| --- | --- |
| Read/Full块内伴随及接口 | 已按顶部5ff3ce2/598b4fb证据采用隐藏bank块伴随；公开末态和trace后备保留。query仍有列/根装配成本，原适用域不自动扩至位置/过期/FP16。下一具名边界是区域/消息/发布列贯穿，按新的独立局部见证与全链成本决定，不原样重复同候选 |
| 循环序列VJP/卷积与状态装配 | SSD层链仍有明显默认/公式差距。采用显式原始时间的有状态锚点；后继按实际profile选择边界，保留Full/Read快照、None/零与精确恢复 |
| resident单bank投影和原规模资源问题 | 首次借用缺陷已修、真实小服务和两dtype组件通过；保守query/workspace和服务数量限制仍独立。须有实际新增存储分解与小数值窗口后再复验具名规模，不原样加预算 |
| 新基线设备资格与完整规模正确性 | 本轮FC16完整任务投影已检查，FP32严格失败与FP64两字段残差保持；CPU完整控制支持/int64一致，NPU同计数不认证逐事件支持。原非零左态/投影恢复与设备小图资格保持原范围。完整尺度的all-root、多步optimizer/精确恢复/学习及其他代表尺度未新增，后继按接口变化选择定向检查 |
| 部署/吞吐后继 | 本轮没有新9B、多卡、mixed-a/b或resident吞吐；GPU继续延后。沿用历史证据原身份，有影响这些路径的新方案或实际下游请求才选有限重验 |
| 广泛源码审阅 | 既有具名审阅与本轮修改覆盖不等于全仓逐函数已审完；使用固定清单和hash绑定记录真实覆盖，不因列出机会强制无限实施 |

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

10. [CANN9.2日常入口迁移](evidence/cann92-project-migration-20261106.md)：版本隔离venv、基线身份拒绝、独立SDK前缀构建及安装入口完成；9df0ecf真实构建，60e1975补同配置新进程恢复。三客户端CPU/NPU小型完整推理/声明训练/恢复通过，独立综合根失败保留如下；不重跑历史性能或公开权重。

11. [TL-REQ-003正支持数值模式](evidence/positive-support-tl-req-003-20261109.md)：2c11695实现，2502fa4/b4879ff补测与资格脚本；显式Shannon最小次正规数存储、稳定Aggregate/KV及未补值响应VJP，CPU/NPU有限整链通过。默认身份/严格拒绝保持，下游学习验收与后端扩展另列。

12. [TL-REQ-004显式密集查询](evidence/dense-query-tl-req-004-20261110.md)：75a5027实现/新构建/安装，26ca1b6诊断；CPU25通过/2保留数值xfail，NPU13实际设备及2辅助通过，四个安装消费者独立三步/新进程精确恢复通过。原FC16/B8/L128完整步CPU62.84→45.16秒、NPU252.92→183.55秒；实际Full/消息及梯度范数不同，只作有限成本观察，不认证原语言训练等价或下游验收。

13. [默认prefill固定资格与受控成本](evidence/prefill-default-20261110.md)：默认能力分派、原始KV槽gather、跨时间区域批及分层回归入口已交付；CPU/NPU安装与standalone训练/恢复分列。原FC16耗时下降37.35%/57.09%，剩余碎片化、显存取舍和原任务未独立验收均保持事实边界。

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

## 当前待办与重启条件

- **接口、性能与架构统一实施（历史阶段完成；新阶段见页首计划）**：仓库开发/性能/维护skills、分主题历史经验与固定源码审阅工具已交付。独立公式、无梯度/构图/反向/完整训练、Transformer与LH/core-3对照各自保持验证范围。接口服务性能、通用性、清晰和替换能力，历史兼容可显式迁移。[1528df8固定资格](evidence/interface-snapshot-bank-20261110.md)交付实际共享KV bank与独立snapshot伴随、公共边界和census；CPU301通过/3保留xfail，NPU配置流程72通过/1保留xfail，三客户端小训练/精确恢复分列。FC16训练CPU1/NPU14.628/20.634秒；完整尺度数值未验，不宣称NPU普遍加速。

  [调度](evidence/scheduling-interface-audit-20261110.md)、[区域](evidence/region-batching-audit-20261110.md)、[循环/Describe](evidence/recurrent-interface-audit-20261110.md)完成具名范围审阅，范围有重叠且未覆盖全仓库。[47704c9固定资格](evidence/interface-batch-publication-20261110.md)落实实时在途容量、只读发布/字段unbind、区域消冗、五记忆Describe能力和公共返回传输去重；CPU/NPU、三客户端训练/精确恢复分列。4项FP32 affine首evidence根原严格超差保留，同材料FP64通过，其余训练/恢复通过；不放宽全局容差。FC16 CPU1/NPU训练12.447/17.212秒，NPU TTFT反升5.098秒。

  [554f2f5固定资格](evidence/interface-heads-qualification-20261110.md)完成通用Describe列、循环Read/Full按用途裁剪及常驻预算/服务诊断。CPU490通过/14保留xfail；NPU配置80通过/2xfail/1原观察器失败，ebfe075修正后设备单例通过。Python/native安装消费者与两种SDK组合的三步训练/新进程恢复通过；常驻FP32/FP16各21个小图组合通过，不增加resident训练资格。实际新增8868字节小见证仍被32768字节局部余额拒绝，因为首次借用已有bank被计入局部增长；确认边界但未修改守卫，不外推全部旧规模拒绝。

  [有限成本与图统计](evidence/interface-heads-performance-20261110.md)完成宽FC、176节点HB/LH、KV sum、SSD与三客户端共14个CPU/NPU训练/精确下一窗；独立数值仍not_checked。FC16训练CPU1/NPU12.614/15.497秒、TTFT3.083/3.960秒；新层链与外层/私有图统计仍显示内部开销。原matrix启动配置失败与仅补未执行项分列，全部作业已终态。没有重测1B/9B、多卡、mixed-a/b或resident吞吐，旧证据保留原提交。

  后继[e6f7bd5资格](evidence/additive-interface-qualification-20261110.md)已修首次借用计费，采用默认Full共享bank矩阵并扩充有状态加法基线；57439d7仅修正half拒绝见证。[三层有限成本](evidence/additive-interface-performance-20261110.md)保留NPU FC16未改善的负结果、较大私有图与循环层链差距。single-bank常驻投影和其余后继按本页顶部的有限重启条件推进；全仓未审范围仍保留。全部可用CPU/内存与空闲NPU可用，共享NPU须当前对话明确授权。

- **TL-REQ-004 联合prefill、部署代表与下游接入**：默认auto按图/模块能力使用静态分量窗口，循环分量继续在线greedy闭包；无参考轨迹预跑，也不按FC/HB/LH名称决定路径。[分量窗口](evidence/component-window-20261110.md)、[联合KV准备](evidence/joint-kv-preparation-20261110.md)、[整序列VJP](evidence/kv-sequence-vjp-20261110.md)、[query根与选定返回边界](evidence/query-root-boundary-20261110.md)已分别交付和验收。完整trace保持中间State因果根；outputs/metrics公共任务使用实际最小边界，恢复不能偷偷换成traced算法。f15c4a5按owner一次装配及3d88b0b局部合批已完成[固定CPU/NPU/SDK资格](evidence/prefill-local-batches-20261110.md)。

  [有限代表矩阵](evidence/prefill-representatives-20261110.md)已完成CPU16/NPU18项运行与精确续接：宽FC、176节点HB/LH、sum/qry、SSD/GDN层链、三客户端及mixed-a/b/c，B4/P16/FP32。大尺度独立数值均not_checked，小门禁另列；auto/greedy差异有限，小规模NPU仍慢于CPU，不能声称统一默认已经消除全部碎片化。常驻CANN9.2新构建后6项小图独立推理/训练/恢复通过，整序列常驻与规模性能尚未认证。[部署后继](evidence/prefill-deployment-20261110.md)已补FP16、异构LH、CPU线程、1.051B/9.314B与双NPU有限成本/恢复。默认异步FC16反向挂起后取消；显式blocking通过不解除。resident修正预算后仍在KV qry/local-row工作区构造拒绝，无性能或恢复结论。

  原FC16/B8/L128/d128固定下游0439efd：a553/fb的完整返回CPU29.788/NPU55.482秒，最小返回CPU22.068/NPU31.822秒，完整步各自单次计量；不同返回契约不混算加速比，设备间实际Full/消息不同。原任务完整数值/离散/学习与下游升级仍未认证。原dense-v1与GDN/HB数值xfail保留；有限同材料FP64见证仅解释其指定差异。新舍入超差有界诊断后继续探索，非有限、通信、容量、求解及精确恢复错误另列处理。

  **算子级目标与剩余边界**：在依赖已知、容量允许的B×L块内，主要算子提交由节点/区域/计算阶段决定；真实反馈/历史/选择依赖允许拆块，不能承诺任意L固定kernel数。KV私有写投影12288→96、基座批已生效；反向根装配/检查继续按profile推进。重复前缀快照的读投影、循环模块逐状态VJP、主机索引/返回装配和resident每sample/region单时间轮次仍有余地。保留None/零、int64、独立根、共享全部贡献和精确恢复；不删检查或用零mask改变活动集合。先具体边界的独立反例和资格，再有限成本，不无限重复同候选。

  cat evidence共享投影、Read有限值守卫和分配分支元数据合批已完成：CPU281通过/4数值xfail，NPU目标流程68通过（含CPU辅助），安装消费者/SDK训练及新进程恢复分列通过。原FC16同minimal成本CPU18.640/NPU22.903秒。packed-v2的指定CPU FP32 evidence根约1.044e-6严格超差保留；不放宽原容差，不把有限误差自动归因为舍入。[层链门禁](evidence/layer-chain-baseline-20261110.md)已修跨设备比较错误；同材料FP64定位到SSD张量基线的vendor默认conv HF32策略，显式IEEE后原阈值7/7通过，原失败保留。resident三轮不同预算拒绝分别记录；下一步须分解数值工作区和服务生命周期，不继续统一加预算。

  [同公式层链基线](evidence/layer-chain-baseline-20261110.md)已完成CPU13项、IEEE NPU7项独立小型检查及12个完整训练成本点。B4/L32/d128的KV native CPU/NPU为0.512/0.934秒，限定张量公式为0.029/0.022秒；SSD native为0.894/2.853秒，张量公式为0.044/0.032秒。省去辅助字段的投影不认证完整图连接或标准Transformer/Mamba；默认入口离直接公式仍有明显差距，后续优先块内状态表示/反向装配与实际小算子提交，不能仅更换最外层调度。

  485a3e2已修复选定输出推理的过时全图trace预算，46项固定安装检查通过；原1B/B32 NPU从115.34GB静态拒绝越过准入，完整窗33.965秒，CPU32线程12.262秒，活动约22.2%。大尺度独立数值not_checked，原CPU计时中断、后继恢复通过但汇总脚本报错分列；NPU B32精确下一窗通过，含恢复peak allocated/reserved约6.07/6.40GB。容量预留是估算，其他图/训练/插件/常驻保守后备不解除。现有external-lm不能代替此验收。后继用全部可用CPU及有余量NPU尽快得结果；构建与重型计时隔开，资源、作业和精确下一命令只见STATUS。GPU按授权延后，下游尚未验收或升级依赖。

- **TL-REQ-003 下游接入与表示域后继**：[显式最小正值模式已交付](evidence/positive-support-tl-req-003-20261109.md)，保留cat/ind及默认strict；CPU/NPU局部、完整两区域根/AdamW/恢复通过。下游原HB16学习验收未运行，须在新私有安装显式升级；跨模式旧checkpoint不能直接冒称精确恢复。多个尾部比例、不可表示公开伴随和弱总量仍有边界；只有实际小型见证证明本模式不够时，再设计log/缩放载荷或更大范围。native/standalone/resident、FP16、多设备、其他记忆尚未实现本策略；CUDA待真机，不以本次CPU/NPU通过外推。原严格失败及CPU资格误选native的8项加载失败保持原记录。

- **用户自行启动的双NPU B512容量探索入口**：显式observe仅放行设备静态估算，保留预算/余量/实际free和全部运行失败；默认strict及旧配置身份不变。单配置队列复用空闲卡锁、时限、日志、停止和结果入口。公开B512配置保持原模型/数学/输入/精度，整流程上限另设7200秒。实现59e9aca、固定安装设施135项通过（[证据](evidence/llm-capacity-entry-20261105.md)）；实际B512容量仍未测，当前未替用户启动。后续只审阅用户提交的这一次结果，不盲目OOM搜索、不把原严格数值失败改写。

- **Dense/MoE选择性后继已终态，数值/容量边界保留**：[15项审阅](evidence/llm-isolation-terminal-20261105.md)记录13项推理成本完成、1项四NPU独立检查数值/token失败、1项双NPU B512静态容量拒绝；无中断或未提交项，当前不需重启。实现694eb60/固定设施125项资格与模型资格分列。Qwen MoE下载完整，真实NPU输出已隔离，延迟提交成功；旧下载/cwd/超时失败不改写。小检查的100组logits/KV超原容差、生成2/8 token不同，重启修复须先定位首差；13个成本点独立数值/离散/恢复均not_checked。B512需56.719GiB而当前保留余量后54.737GiB；先校准KV/激活/workspace及reserved，再考虑显式小prefill或更多设备，不减少逻辑B或安全余量、不原样盲重试。四NPU B128 reserved仍约60.9GiB，不能按allocated外推B512。vLLM/TP/EP须另建匹配A3 NNAL的软件环境，不修改公共基线。完整原配置由`lm retry-matrix --round isolation`重建；新实验先明确配置和预算，不重复已完成曲线与原严格失败。

2026-10-31的有限授权范围已收尾：三平台B128/B512、A/B/C、FC/HB/LH约1B、多卡容量/训练、
resident边界、人类及下游流程均有实际结果、拒绝或有依据的停止记录。九份可查询归档204条run，
包含失败/诊断，不称全通过；原189项配置目录与目录外后继的审阅记录分别保留，原始运行材料按授权退役。
[八卡共享后继](evidence/core5-platform-eight-followup-20261031.md)已终态：同模型1B推理和新loss训练
执行/离散/精确续接通过、独立FP32超差；小训练严格通过。原三项八卡排队超时不改写。

- 下一次数值/运行时修复须按下表具体方案重启：NPU B512 exact-next、任意独立根事件/崩溃、原loss非有限及各平台严格超差仍未解除。停止重复相同候选，探索模式不能吞掉这些非数值错误。
- 六项packing均已重审并有[具体决定](performance-lessons.md#本轮三平台六项打包处置)。后续跨输入同base矩阵批、resident服务资源或反向模板须先给新profile/可验证方案，不自动展开全部融合工程。
- 用户计算优先，未开始的NPU大tmpfs对照及38.406B周期保存后继取消，不作完成门槛；38.406B已运行的有限窗与未验完整恢复分别保留。不默认扩盘、重启GPU或恢复大文件任务。800GiB硬配额与tmpfs联合主存准入持续有效。
- 7×T4站点和迁移步骤已交付；driver/runtime/Toolkit/LibTorch、sm75算子、互联、训练/恢复仍待真机。当前没有新的GPU作业或gpu3090依赖。

- 远端dirty-dev源码包及两份早期开发patch已移到仓库外保全，按用户要求稍后分析；位置与hash见[清理记录](evidence/artifact-cleanup-tracked-only-20261031.md)。不自动合并或运行。

## 逐项推进与停止条件

| 项目 | 当前终态与证据 | 停止或重启条件 |
| --- | --- | --- |
| NPU FP32约1B/B512推理续接 | [本轮规模证据](evidence/core5-platform-scale-20261031.md)：执行完成、原始解码比较通过，下一完整窗sample0/node1 KV keys有3669/32768坐标逐位不同，最大1.7166e-5；独立CPU对照未执行。B128与其他规模项结论分别保留 | 原checkpoint已按授权退役，保留配置/输入生成器/诊断和原结果身份；不重复相同全流程。重启须重新生成有界见证，比较恢复前后布局、实际算子及首字段；新见证不冒称原字节复验。不同于跨设备容差，不因explore接纳exact失败 |
| D64 FP32完整训练 | 2ada00b单行私有VJP消冗后原两步native/Python仍有记录、梯度、参数和SGD槽差异；两调度自身恢复精确。见[eager证据](evidence/core5-eager-boundary-20261029.md) | 本候选数值修复停止；不重复九项分段精度/外层汇合探针。重启须有覆盖局部算术、中间伴随与共享贡献的具体新方案及独立根证据 |
| D2048 FP32 | [实际首写入](evidence/core5-boundary-isolation-20261029.md)：来源+私有批链194坐标失败；标量FP32对局部FP64也有289坐标失败，连续局部FP64后cast为0。原quad残差拒绝已修复，原全图keys失败未解除 | 没有公共整段算术/VJP方案前停止9B全图重跑；局部更准确不保证与旧标量FP32相同。显式全CPU FP64已有独立原窗通过证据 |
| 自然FC/HB较大训练 | [878a5a4有限测量](evidence/core5-natural-study-20261029.md)：中型FP32完整keys/values失败；FC/HB约1B FP64分别有2个SGD槽坐标/59个梯度坐标超原容差。LH1.051B FP64原两窗完整通过，不能外推 | 本轮一次流程终态，停止相同失败候选的更大吞吐；重启需具体算术/伴随方案及独立根与完整更新证据，不提高容差 |
| FC1B NPU FP32下一窗 | [9bfabbb复验](evidence/core5-checkpoint-budget-20261029.md)已越过旧4GiB文件限制，5.25GB文件保存/新owner加载返回；原owner下一窗出现非有限梯度，恢复分支与CPU独立核对未执行 | 原输入配置/平方loss/SGD及结果保留；大文件已按用户授权退役，重启须重新生成。没有首个非有限伴随定位或明确新数值方案前停止重跑；修改下游loss/训练配方是另一个配置，不称原失败修复 |
| 反馈PDG FP32 | 原W32/B4四窗记录/梯度/SGD槽仍失败，参数通过；reference/streaming/prefill三者自身恢复均精确。显式FP64四窗独立通过 | 单行VJP方案到此停止；新方案须覆盖反馈递归与跨窗更新，不能仅核对最终参数 |
| FP16 GDN/default AdamW | [归一化VJP内部FP64](evidence/core5-boundary-isolation-20261029.md)仍第二步差.00198364；两个原strict xfail保留。FP32对应见证已由9e6dc52解除 | 不改epsilon/loss scale/输入或容差；仅提升归一化反向候选停止。重启需新的完整局部伴随/舍入方案 |
| CANN9.2迁移独立综合根 | [小型迁移证据](evidence/cann92-project-migration-20261106.md)：Settle/FC/KV B2/W8 FP32，CPU native、NPU PyTorch/native在all根的input.0.1.0.1第4坐标分别差1.78814/2.86102/2.83122e-6，超原atol1e-6/rtol1e-5。NPU PyTorch与20261031旧记录同值；六项完整推理、声明loss训练/optimizer与新进程恢复均通过 | 迁移任务已终态，原综合根失败不解除；同配置不盲重跑，不从C++另一根目标或自身恢复外推。需要该根完整资格时，先沿compare.objective、归一化VJP和累加顺序定位首差，保持原材料/阈值；没有新依据不展开数值工程。新CANN版本的resident、多卡/尺度另验 |
| NPU运行时初始化顺序 | 固定资格暴露CPU autograd后才注册NPU触发device_ready_queues_断言；独立标量进程已确认late失败/early通过，独立NPU资格14项+eager6组通过 | 应用先resolve_device再首次autograd；已启动CPU引擎的晚注册路径保留限制，等待运行时修复。当前门禁分进程，不改数学实现 |
| 下游native workers2跨进程恢复 | 本轮FC/KV FP32小型安装消费者独立两步通过，但连续三步与两步保存/新进程恢复一步的梯度、SGD槽逐值不等；关闭autograd多线程仍失败。示例显式workers1固定资格通过，见[证据](evidence/downstream-contract-20261030.md) | 原workers2失败保留；停止重复相同候选。须先定位并行构图/反向累加顺序的具体差异，再用原材料/独立根/跨进程恢复验证；本轮只交付协作规范和单worker示例 |
| 本轮NPU blocking小训练恢复 | [多卡证据](evidence/core5-platform-multicard-20261031.md)：原FC8/d8三更新完成，换owner精确恢复阶段SIGSEGV/-11；ordinary/cpu_bridge同材料严格通过，不能互相替代。报告b3dc53c补监督状态，未修数值执行 | 保留配置与原结果，小checkpoint已按授权退役；一次有界faulthandler诊断未复现且完整检查通过，旧SIGSEGV仍未定位；不再重复相同试验，重启修复须有流/owner生命周期新依据及原完整训练/恢复检查 |
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
