# 当前路线图与验收范围

数学固定 `0770593838a37c790a568226930567ed60f99b83` 的 core-5 和带权局部定义。
[STATUS](STATUS.md)是唯一当前交接与作业表；本页只保留总体待办、当前判断和未解除限制。
已完成阶段的配置、源码与测量在[证据索引](evidence/README.md)，不再复制历次会话汇报。
历史实测保留原提交身份，定位见[迁移映射](history-migration.md)；复现从[reproduction](reproduction.md)进入。

## 当前计划：批量张量贯穿与架构验证（已授权实施）

固定数学是锚点，代表性能测试牵引取舍，公共实验基座是定位。用户允许整体重写、
替换旧接口并迁移消费者；旧源码、mixed-a/b/c名称和分层均不构成架构约束。
[执行契约](execution-flows.md#批量接口与模块替换)保留一般图、可替换局部模块、
完整观察/独立根、训练、精确续接与可解码状态；不要求每个数学函数对应一次调用或独立Tensor。
CPU、主机混合与设备常驻分开验收；GPU按当前授权延后。

| 阶段 | 已有基础 | 仍须完成与验收依据 |
| --- | --- | --- |
| P1 契约与审阅 | [张量窗口](weighted-tensor-windows.md)、typed KV/循环状态、区域列、真实消息/独立根及迁移契约已在公共入口使用；[固定边界审阅](evidence/p1p6-boundary-review-20261112.md)、[静态循环/形状审阅](evidence/lh-block-static-20261112.md)与后继hash记录分列 | 沿Aggregate→状态/Read/Describe→区域→Full→消息及伴随确认真正贯穿；全仓逐文件/逐函数审阅未完成。inventory不是审阅，重叠文件不累计覆盖 |
| P2 加法对照 | FC/数学不等价Transformer、KV/SSD与LH已有独立公式→公共张量接口→默认执行器三层；三模式、完整返回投影和独立梯度根分别记录 | 后继以同材料复测被修改的边界，解释主要差距。LH FP64数值投影通过不等于完整根资格；原FP32有限超差及36个None/零差异保留。直接公式不是绝对性能上限 |
| P3 默认FC整链 | Python/native及standalone公共B×L窗口、区域分配、消息、输出任务/训练/raw恢复已接通；[最近固定资格与成本](evidence/region-tensor-20261112.md) | 来源bank/incidence与Read→区域字段owner已分别交付；[c6afc20c资格/20点成本](evidence/read-region-owner-20261112.md)保留有限数值失败。继续处理prepare/raw-bank→Read/Full、typed continuation与真实跨设备块传输。保留小型反馈PDG、非对齐TimedDAG、一般Settle端口、重复边/共享/None零；FC单点不代替全域验收 |
| P4 有限模块候选 | 五记忆/原始时间短卷积、十种发送与适用评分/evidence/key/正则已接入主机张量列；[循环资格](evidence/recurrent-columns-20261111.md)、[half资格](evidence/column-half-20261111.md)、[放置资格](evidence/column-placement-20261111.md)分列 | 保持逐候选与有判别力的小组合，不穷举。正支持补值策略仍经记录后备；新resident块的half/循环记忆/多owner尚未接齐。旧一般执行器能力不能冒称新块实现 |
| P5 规模与执行路径 | 主机LH176/FC/1B/9B有有限成本；新resident独立Model、KV sum/qry、小图三步/跨窗/新进程恢复已有[a3c资格](evidence/resident-block-public-20261112.md) | LH公共层差距仍明显，优先source-owner/region-window上下文。[有界program组合](evidence/resident-services-20261112.md)已通过完整flow、小图新进程恢复及原预算FC16/LH176执行/换owner恢复；规模独立数学未检查、成本仍高，共享历史/raw布局仍须闭环。多卡/普通异步和GPU按下表独立处理 |
| P6 治理与交付 | 配置/代码/数学/测试/证据/复现已有导航；skills保留加法、减法、静态循环/形状/复制/伴随分析 | 随每次边界重写更新符号/契约、固定资格、安装消费与精确恢复。清理陈旧交接叙述，保留可直接重建的输入与命令；不以文档链接检查宣布全仓治理完成 |

本轮已完成主机字段owner与resident KV存储的当前边界及必要验证，现停下汇报、等待重新对齐；
下文是总体剩余方向，不授权重入时自动展开下一轮。已确认原则不作为每次压缩后的新任务。

### 下一实现边界

- **主机来源与区域窗口**：[1f0e6e43审阅](evidence/region-tensor-20261112.md)确认整组KV写入和中心化Aggregate VJP已经落实，不能继续把旧逐源图当根因。72e1d22c及9ae62d0d已保留唯一实际来源块与物理incidence/增量预算；c6afc20c已连接Read→区域的字段owner。剩余prepare/raw-bank→Read/Full及typed continuation；参数共享、计算共享和私有状态分别处理。mixed预设只在配置入口展开，实际设备边界才传输；区域参数视图和操作数别名应在调用级复用。
- **resident程序资源**：[042ea369](evidence/resident-services-20261112.md)将区域前后向交给设备通知调用的有界program，已越过41c完整flow反向507002与原预算两规模失败。08a349ce补已知存储/workspace预准入与真实程序/流/通知计数，[两套构建/设备资格](evidence/resident-resources-20261112.md)已通过；部分提交故障、整体等待期限和更大资源规模仍独立处理。独立FP64投影oracle通过原阈值；旧FP32参考自身累加超差和CPU harness顺序污染保留。
- **resident历史布局**：[唯一initial/write bank、快照索引及块伴随](resident-kv-bank.md)已接默认局部路径，保留三种Read、独立raw根、过期/原始int64时间与实际checkpoint。[0c974ccd资格](evidence/resident-kv-owner-20261112.md)：56 CPU、两构建、SDK11/native4设备进程通过；小图全根、三步、精确新进程恢复已验。有限窗口expiry目前为CPU局部证据，整链设备扩展和新布局尺度/性能尚待；此后工作由本轮汇报后的对齐决定。
- **循环候选与广泛审阅**：SSD层链仍有默认/公式差距；按实际调用与有状态锚点继续审阅，不把KV优化自动当循环模块已优化。全仓未审范围由固定清单/hash记录，审阅发现机会不要求无限实施。

### 测量、阶段边界与有限停止

- 小检查以独立公式/调度、共享梯度、非空左态和精确下一窗为主。布局变更观察外层与私有图、连接、反向及存储，不能只数一个Function。
- FC主成本尺度为B8/L128/d128/H2/FFN512；resident原拒绝见证另为B4/L16，二者不混称。LH176/宽FC/HB、sum/qry、循环、约1B/9B按被修改路径选择；不扫描笛卡尔积。
- 无梯度推理、含grad前向、完整训练分开。TTFT实测，吞吐按真实输入位置；训练含准备、路由、packing、反向、clip/optimizer、通信和同步。记录候选/Full/消息的实际工作。
- 同材料尽量固定设备、参数、输入、线程/精度/观察范围；不同处明确列出。profile与无插桩成本独立；小测有限重复，大测按阶段边界执行。
- 主要差距未解释就保持未完成；已知边界用新表示/算法/独立见证推进，不无限重复同一局部优化，也不以无新依据的完整重跑代替诊断。
- 有限舍入超差只阻止相应严格等价声明，有界首差/高精度诊断后继续独立工作和授权探索。非有限、依赖/控制、求解、通信、容量与精确恢复失败不能被容差或explore接纳。
- 已有有效证据按原身份复用。长任务固定源码、background.slice、有限资源/时限；全部CPU/内存及合法分配NPU按收益使用，正式计时避开本项目重型构建争用。共享许可只在当前对话有效。

## 已有公共能力与入口

| 范围 | 定义、实现、测试与证据入口 |
| --- | --- |
| PDG、非对齐TimedDAG、一般SettleGraph及端口/嵌入 | [总对应](semantic-map.md)、[Settle](settle-embedding.md)、[结构表示](structured-records.md) |
| 五记忆、短卷积、来源分类/共享、评分和发送 | [KV](weighted-kv-sharing.md)、[循环](weighted-recurrent.md)、[分配](weighted-allocation.md)、[发送](weighted-sending.md)、[评分](weighted-scoring.md)、[来源](weighted-source-binding.md) |
| streaming/prefill、模块批、VJP与执行资格 | [张量窗口](weighted-tensor-windows.md)、[区域批](weighted-region-batch.md)、[执行能力](execution-capabilities.md)、[常驻训练](resident-weighted-training.md) |
| FC/HB/LH自然维度、1B/8B/16B生成器和有限目录 | [研究入口](core5-study.md)、[参数/拓扑清单](weighted-workloads.md)、[代表回归](prefill-regression.md) |
| 三层成本、性能经验和独立比较 | [证据索引](evidence/README.md)、[经验](performance-lessons.md)、[复现](reproduction.md) |
| 干净安装、CANN9.2/Python/SDK、下游协作 | [开发](development.md)、[协作契约](downstream-contract.md)、[维护映射](core5-study.md#修改时同步维护什么) |

上述入口不构成所有组合/规模/平台的笛卡尔资格。下游拥有数据、任务头、loss、配方和模型效果搜索。

## 当前待办与重启条件

- **TL-REQ-004/006下游**：基座默认窗口和有限代表交付不等于原语言任务/学习验收或下游升级。
  [REQ006固定请求](https://github.com/ZichaoLong/tide/blob/dc943cc7ce773e3dac4d62736ab524953573decb/docs/requests/TL-REQ-006.md)
  的HB88/HB184/入口树/Transformer诊断、[常量重载与初态](evidence/req006-floor-20261112.md)已分别记录；尚未做四代表生产替换/下游验收。不得以6e7bd49e旧直接实现的计数作为当前基座回归。
- **TL-REQ-003**：[正支持模式](evidence/positive-support-tl-req-003-20261109.md)有限CPU/NPU检查已过，原HB16学习验收未运行。native/standalone/resident、half、多设备、其他记忆尚未实现该策略；弱总量/尾部比例/不可表示伴随保留边界。新表示须以实际独立见证为依据，不能静默补值或跨模式恢复。
- **其他已终态工作**：外部Dense/MoE有限矩阵[保持原数值/容量结果](evidence/llm-isolation-terminal-20261105.md)，用户B512容量入口[仅设施已验](evidence/llm-capacity-entry-20261105.md)，未替用户启动。vLLM/TP/EP需单独匹配环境；不自动恢复已取消的38.406B/大tmpfs/旧队列。
- **GPU与外部材料**：GPU待验不阻塞本机实现；7×T4迁移指南不等于sm75真机资格。远端dirty-dev源码包/早期patch按[清理记录](evidence/artifact-cleanup-tracked-only-20261031.md)在仓库外保全，不自动合并或运行。
- **当前数值、异步、容量边界**：如下逐项保留原失败和停止/重启条件。原始材料退役后按跟踪生成器重新产生新见证，不冒称复验原字节；历史限制不变成后继架构约束。

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
| 默认异步双NPU eager VJP | CPU桥完整delta-pdg仍EE1017；5959bbc3新列路径亦在反向invalid event，8独立安装消费者及SDK首例失败。新列显式launch blocking的4全根/更新/raw恢复、2安装及SDK FP32通过；另配结果不解除普通异步 | 新列减少碎片未修复该事件生命周期边界；不继续相同copy方案、不静默开blocking。重启须运行时修复或新的事件所有权方案，先纯Torch/跨设备反例再完整图、共享根及恢复；[新证据](evidence/column-half-20261111.md) |
| 单行私有VJP | Python/C++直接建图、轻量一阶/版本守卫已交付；固定2ada00b CPU517 passed/2原strict xfail。NPU相关开发已通过；[57308e7 NPU固定资格](evidence/core5-resident-boundary-20261029.md)通过，[有限完整对照](evidence/core5-boundary-performance-20261029.md)已终态，CPU未见收益 | 以实际端到端成本决定下一步；不从减少私有图数直接推断收益 |
| 异形字段packing与小提交 | 按device/dtype合并真实字段、8MiB分组、按shape/offset恢复；int64独立，None/零/版本守卫保留。mixed-a/c完整核对通过、单次1.505→1.477/2.183→1.848s，allocator峰值不变；小算子/scalar仍显著 | 若小算子/提交仍主导，须据新profile提出具体边界；无真实可合并域不盲目扩大batch |
| resident组合矩阵/窗口反向 | 因子KV与实际分段已交付并资格通过；[57308e7](evidence/core5-boundary-performance-20261029.md)及[b7e0a29](evidence/core5-compact-fibers-20261029.md)两次有限观察，局部槽压缩后medium构造1057ms、执行/等待700ms；构造成本仍显著 | 当前测量终态；下一候选为显式模板重绑。完整adjoint绑定journal/root/参数地址，须先独立验证绑定/版本/跨窗，不能按shape缓存旧对象 |
| 较大NPU训练/跨卡 | 已测停止。D64匹配9076975参数，native单/双NPU显式blocking自身磁盘恢复精确，独立CPU均在node89 keys严格失败；[有限证据](evidence/core5-boundary-performance-20261029.md)保留15.685/16.840s失败成本 | 原eager数值失败按首行重启条件处理；不重跑同候选。resident的新布局与新资源边界分列，不以参数量匹配或自身恢复替代独立正确性 |
| resident局部纤维填充 | 已交付并[资格通过](evidence/core5-compact-fibers-20261029.md)：b7e0a29按物理入边/端口缩小槽；CPU两环境各10、公共NPU56、SDK7组件/dtype通过。medium完整窗2.794→2.470s，peak allocated514.8→376.4MB，完整核对/恢复通过 | 本候选实施、相关资格及一次对照闭环；单次观察不宣称普遍加速。跨节点数值批和减少物理缺席行仍需新的具体实现及独立根证据 |
| NPU resident流资源 | 已定位并停止。相同D64/capacity的b7e0a29构造报207005；CANN日志明确Too many streams/EE1023，实际SQ/CQ=1984。未进入warmup，尚无完整训练、恢复或吞吐结果 | 重启须有界服务分派/复用或节点合批；先验证嵌套通知、在线顺序、共享workspace、独立VJP根、peer和恢复。该观测不是通用流上限；不靠加HBM预算或主机预选序列绕过 |
| FC16 resident原预算与服务停止 | [原失败](evidence/resident-shared-bank-20261112.md)保留；[042ea369后继](evidence/resident-services-20261112.md)在原18.522GB预算、B4/L16/d128完成训练与精确下一窗恢复，LH176原预算同样完成；SDK21/native4进程提供独立小图/新进程证据 | 指定录入/原预算执行见证已闭环，不继续原候选盲重跑。规模独立数学not_checked，成本仍高；共享历史/raw布局及程序构造须有新实现与对应独立根证据，不从小图恢复外推全部规模 |
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
