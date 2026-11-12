# 带权张量窗口接口

数学固定版本与图/checkpoint 版本由 [semantics](semantics.md) 管理。本接口的软件身份是
`tensor-window-v1`；不是新的数学函数。字段列可以覆盖独立样本 B 与原始逻辑时间 L，
无需为每个作用创建 Tensor 再重拼。输入、消息、Read 描述及私有状态分别有类型和尺寸。

## 公共调用与迁移

```python
from tidegraph import ExecutionPlacement
from tidegraph.weighted import eager_runner

execute = eager_runner(binding, "python", ExecutionPlacement("cpu"), trace=False)
result = execute.run_positions(spec, values)  # values: B,L,d; unit broadcast
hidden = result.readout(values)              # B,L,d
next_window = execute.run_positions(spec, later_values, start_position=L,
                                    state=result.continuation)
```

`run_ports(spec, layout, ports, stop_position=..., state=...)` 接受不齐的真实物理端口。
`SequenceLayout(offsets,times)` 的 offsets 是 B+1 host int64，times 是逐样本严格递增的
非负原始位置；允许空样本。`MessageColumns(rows,strength,content)` 的 rows 是排序且不重复
的 host int64 行号，strength 是 R 浮点，content 是 R×d 浮点。缺席由 rows 表示，
不能从 strength 是否为零推断。边/端口身份保留在字典键中，重复边各参与一次。

同一个执行器的 `execute(continuation, external_records, stop, sealed_until=...)`
继续接受一般 PDG、非对齐 TimedDAG 和任意合法 cut。记录接口保持完整事件观察能力。
张量位置窗口要求 Settle `ports` 绑定及完整位置 cut、无 pending；一般 cut 使用记录接口。
不把任意 PDG 冒充位置窗，也不从数值参考预计算真实轨迹。

新消费者优先传入批量张量，避免先调用 `layered_inputs` 将 B×L 拆成事件。
现有消费者若依赖完整 trace/中间独立根，可继续记录 API；不静默改变其损失函数。
旧实验依赖锁保持原版本，升级由消费者明确选择。

安装实验CLI的内置 `weighted-output-square` 任务也采用该边界：`return_mode=outputs|metrics`
且默认 `input_interface=auto` 时，符合能力的Settle执行直接创建共享来源列，按物理端口
gather进入 `run_ports`，loss按各输出端口分别求 `sum((cG)^2)*scale`。
不会先汇总多个端口再平方，也不在反向之前把输出拆成事件。
FP16内置loss与旧记录任务一致，在相乘之前将c和G提升至FP32，再平方归约，
满足FP32-master优化器契约，避免先在half溢出后才cast。所需CPU输出记录在反向结束后
从实际结果整列复制、无梯度解码；无逐行可求导copy。raw状态仍使用checkpoint v5精确恢复。
`effective_input_interface`、`input_fallback_reason`及逐窗口 `effective_window_execution`记录实际选择。

当前默认使用execution-v6身份及`position_policy=bounded-common-origin-v1`；
显式 `input_interface=records`可比较旧输入边界。
半精度及多payload owner的新请求也保留auto窗口策略，由实际模块决定能力；
不能把旧record查询算法的legacy后备提前当作窗口能力。其生效配置/fingerprint与旧记录不同，
旧record锁仍按原接口复现；安装源码/二进制身份不同不能冒称同栈精确恢复。
旧execution-v5列锁在规划解析或客户端分配资源前明确拒绝静默升级：用原固定源码复现，
或明确调用`ExecutionSpec.from_record(old["requested"]).record()`生成v6，再建立新依赖锁与验收。
这只迁移执行算法身份，不更改数学、图v13或raw checkpoint v5；新旧算法不承诺逐位续接。
历史锁中省略此策略时保留record路径。完整事件观察、插件任务、一般PDG/非对齐TimedDAG
或当前不支持的放置/模块仍使用原接口；插件即便只读outputs，也未默认声明张量输入/loss。
显式完整诊断仍返回原事件根。成本对照必须检查实际接口，不能只根据 `prefill` 名称推断。

独立 LibTorch 客户端也采用同一默认选择，`--input-interface auto|records`写入执行和
checkpoint身份。其 `effective_execution`记录实际interface/schedule/root与后备原因；
静态fixture的真实rank/stride决定Settle能力，不按拓扑名称分派。构造时一次装配输入列，
窗口内按列clone/transfer，loss贯穿整列，返回CPU输出在反向后无梯度解码；metrics不生成事件输出。
完整trace、显式记录/streaming/greedy/scalar及不符合Settle时间编码的图保留独立通用执行。
旧锁未声明input策略时，Python启动器显式传`records`，不把旧实验升级到新算法。

公共C++入口见 `tide/weighted_column_checkpoint.h`：

```cpp
namespace col = tide::weighted::columns;
col::WindowExecutor execute(spec, model, workspace_bytes, placement);
auto result = execute.run_ports(continuation, layout, ports, stop_position);
// result.columns: actual typed outputs/messages/state; result.continuation: exact raw cursor.
```

该owner不依赖Python，也不调用通用执行器作递归补写；只接收完整位置cut、无pending的ports绑定。
`GraphCheckpoint::save/load`直接保存其continuation、参数alias与optimizer，现有checkpoint版本不变。
任意cut或完整事件根仍选原公共记录执行器；窗口不能借由有效状态投影替代raw解码。
`tidegraph-column-checkpoint-check`对照独立Streaming的稀疏端口、零质量、同秩HB、
超大int64时间及六种记忆组合的三步SGD，并exec新进程验证raw状态、参数、optimizer和下一步精确恢复。

## 能力分派与实际身份

`weighted.execution.eager_runner` 返回 `EagerExecutor`。窗口方法按图/局部/放置能力
选择 `settle-columns-v1`，条件不满足则选择 `graph-record-adapter` 并报告
`result.execution.fallback_reason`。不按 FC/HB fixture 名称分派；容量、求解、非有限等
执行失败直接失败，不能捕获后偷换算法。显式 legacy、greedy、streaming、scalar 或
多 worker 请求保留其执行选择。记录请求/生效身份，不能由默认名推测命中。

Python 的区域 rank 顺序和 C++ 的独立 region-order 调度都由实际候选支持形成分组；
允许同秩独立区域、独立端口时刻、缺失/零消息、非平齐样本。所有候选在区域内一起计算
评分/控制/分配；活动支持决定 Full 行和随后真实消息，B/L 浮点行不作为模块交接对象。
节点维可拆，因为参数和私有状态可能不同；这不是必须逐节点提交算子的约束。
相容节点/区域可以按形状与就绪关系合批，参数身份及私有状态仍独立。
区域按每行实际候选数N分组，组内保留int64 `[R,N]` 节点身份；成员名单不同不要求拆成不同算子。
描述字段按整节点列拼接和整数索引进入区域，控制通过成组线性转置返回节点列，不填充假候选。
不同成员组的反向携带结构行支持，并按分配的有限分支调用声明VJP；未共行的独立节点/字段保持None，
被使用的零伴随保持连接。`regional_packing`拥有布局，`regional_groups`拥有组合/反向；
相同成员的组及无梯度执行保留普通张量路径，避免给密集FC引入不必要的私有图。
固定单位transport仍逐项检查`requires_grad`、逐元素检查等于1；数值比较按device/dtype及
临时存储预算合并，不逐个scale回主机，也不缓存此前窗口的数值结论。
已对齐的消息行直接作为聚合操作数，完整区域行直接交接描述/控制，不再先补零、
恒等索引和回填；非对齐部分仍按实际行域装配。重复物理来源各参与一次，全零纤维
仍与内容/变换参数断开，质量根保留。`test_column_assembly`以独立逐纤维公式检查这些边界。

Python共享Aggregate只在参数/实际输入句柄相同时复用。`column_sharing.shared_uses`
为节点使用者提供只读整列视图，在一阶反向中按稳定节点顺序归并全部贡献；
跨设备分支的到达顺序不再决定共享输入的累加顺序。不关闭全局autograd并行或确定性策略。
这是整列调用根，不能把视图之间的关系当作旧逐事件因果根；完整独立事件观察仍用记录接口。

当前张量能力：

| 层 | 实现 |
| --- | --- |
| KV | Python/C++：sum/qry、Read0/-/+、有限/无限逻辑时间窗口、相对旋转/距离项、raw 左态、选择后的 Full |
| affine/delta/GDN/SSD | Python/C++：typed H/Z/W、正写入 scan、原始时间短卷积、异构组合 |
| 区域 | Python/C++：现有十种发送及适用 quad/bern/ent，完整评分/evidence/key 列及声明一阶 VJP |
| 状态/传输 | observe-all、adopt、严格正支持策略、unit transport；SrcClass 是参数分类，不去重原子 |
| 设备 | FP16/FP32/FP64 同dtype的节点payload owner、CPU/mixed-a/b/c 主机调度与显式评分 dtype；实际设备资格单列 |

Read/control/selection按声明的位置接收整列临时视图，参数与raw状态所有者不变。
KV Read sum只读取summary/weight bank，qry读取key/value/weight bank，Full不额外读取summary；
跨设备只复制需要的Read参数和字段，不复制Full/更新矩阵，不构造B×L逐事件请求。
选择结果整列回到payload；评分转payload若丢失正支持则明确拒绝。int64布局与时钟始终保持原类型。
这不等于设备常驻推进。位置与评分精度写入实际窗口执行记录；旧记录API维持自己的实现与资格。

消息按目标节点owner整列传输，同一实际消息到相同owner的副本复用；逐边身份和重复参与不消失。
Read按节点位置计算，完整描述先汇集区域参数owner对应的control设备，再执行完整区域分配，
最终控制返回各payload owner。raw状态与参数仍属于原节点；既有共享参数共置约束保持。
多owner实际记录包含逐节点/区域位置。输出可留在各自owner，`readout(like)`指定聚合投影设备，
内置逐端口loss先在原owner计算再汇总，归约与optimizer边界不因物理放置改变。
这是普通可求导copy，不宣称P2P、collective或设备常驻调度；异步与真机资格单列。

FP16沿用[精度契约](weighted-precision.md)：显式FP32/FP64评分，half参数/raw状态/消息，int64时钟；
KV块Attention的logits、softmax、加权归约和centered一阶转置在FP32计算，输出/伴随返回原dtype。
位置旋转仍按原始整数时间差，在FP32求相位后返回half；位置内积在FP32计算。
原始版本守卫、None/零及正支持cast拒绝保持。workspace按payload与评分中较宽dtype估算，
包含Attention的FP32暂存；native接口能力版本3要求重建。具体硬件资格以固定报告为准。

正支持补值和resident当前经能力后备处理。它们是新列迁移的未完成范围，不是从基座删除已有能力，也不代表
P1—P6 总体完成。后续状态与待办只见 [ROADMAP](ROADMAP.md)。

## 状态、梯度根和寿命

KV raw 状态是 keys/values R×d、summaries R×d_summary、weights R，加 int64 entry times、
B 个 last_time/observations 和惯性 value。查询 bank 可包含在本窗末态已过期的条目；
terminal raw retention 与当前 Read 有效视图分开。零接收保留旧 raw 状态和时钟。
可见性和距离项先以int64求原始写入/查询时间之差；position_origin在匹配中抵消。
Read与Full使用各自频率，位置不能由接收序号代替。位置logits使用单个共享value bank和
显式一阶transpose；旋转的临时张量另计workspace。
Read0 使用绑定初态；Read- 只看本次之前，Read+ 与 Full 看包含本次 proposal 的快照。

KV `prepare_kv_columns_many` / C++ `columns::prepare_many` 可按相容节点的实际正写入行数、
维度、dtype/device分组。每组以4个块矩阵算子计算base、低秩两因子和summary，原始输入/参数
仍是独立梯度根；全部使用者累加到共享参数，未使用组输出不制造连接零。每节点的
int64时钟、零写入保留、bank及terminal装配分别处理，Read/Full的快照不改变。
`column_modules.prepare_many`负责有限模块分派；其他记忆继续使用各自的准备算法。

KV `read_kv_columns_many/full_kv_columns_many` / C++ `columns::read_many/full_many`
继续按完整局部配置、实际device/dtype、B及填充后的query/bank长度分组；Read另按区域描述配置
和Read0/-/+分组。默认列调度在准备后调用Read组、在完整区域分配后调用Full组，
先解析实际放置再分组；循环模块仍使用各自的整列实现。Full仅收集正控制行，
消息保留独立的`(c,G)`，不会在局部内容中提前乘c。私有bank、支持和int64时间不合并。
组内投影使用已有`linear_blocks`：整组参数同一时使用共享GEMM，否则使用私有矩阵BMM。
Attention/加权sum的node×B轴在无梯度packing中连接，多输出一阶边界只汇总实际被使用的
节点伴随；未使用输出保持None，显式零伴随保持连接，真实参数alias汇总全部使用者贡献。
全部原始输入的版本仍受检查，冻结节点输出不因其他节点求导而获得梯度身份。
位置策略在下述有限域内消除B×Q×K×D旋转暂存；节点合批及暂存减少本身不代表
融合、总体容量准入改善或端到端提速。

`bounded-common-origin-v1`对每个样本取真实query/bank最早整数时间为共同原点，
先int64相减，再分别旋转Q与K并做矩阵乘，利用KV §2的`Rot(p)ᵀRot(q)=Rot(q−p)`。
重基准只是求值表示，未改固定Pos函数或未旋转raw key。FP32/FP64的整数跨度须分别不超过
2^24/2^53；设相应epsilon为e，最大绝对相位P还须满足`8eP ≤ sqrt(e)`。
后者是两个分离相位在转换/除法/乘法中的舍入误差代理，不能保证输出误差或路由相同。
阈值只由dtype、真实时间、频率和scale决定，与拓扑名称、用例和测试结果无关。
越界保持原相对key展开；FP16保持原相对旋转/half舍入。仅有距离项时直接矩阵乘加原age偏置。
原始age仍独立决定可见性、过期与距离；极大绝对时钟而小跨度可安全进入重基准域。
旋转使用带原值/坐标版本守卫的一阶边界，伴随为反向角旋转；多节点收缩仍保留独立None/零。
observer的`kv_rope_rebased`、`kv_rope_content_distance`及
`kv_rope_relative_{half,integer_span,phase_span}`记录实际路径，不捕获运行异常来选择后备。
保守workspace仍覆盖后备暂存；跨算法或不同物理切分不承诺浮点逐位相同，实际恢复须使用同一执行身份。

循环状态是 B×matrices×value_width×key_width 的 H、B×2×summary_width 的 Z、B×2 的 W，
以及适用的卷积缓存和 int64 时间。C++ `LocalState/LocalWindow` 采用 KV/recurrent 类型分支，
`column_modules` 负责有限分派，调度不猜状态形状；C++消费者可用 `std::visit` 访问其实际类型。
Python返回的KV/循环状态类保持分别命名。正写入序列用 scalar-affine 或 matrix-affine scan；
卷积按原始时间槽查找，不用压紧后的事件编号代替时间。KV/GDN/SSD 可出现在同一图中。
卷积tap的反向使用`tap_gather` / `columns::gather_taps`：同一lag内真实缓存行至多使用一次，
不同lag的贡献先按缓存行收集，再沿lag归约，避免重复index的浮点原子累加。
这是声明的一阶线性转置，保留原值版本守卫、None/零与缺失槽；不更改全局确定性开关。

`WindowResult` 返回 output columns、实际 raw continuation、统计及执行身份；低层
`SettleColumnPlan` 还暴露实际 node columns、逐边消息和 typed 状态。普通列根、参数、
输入及非空左态的梯度通过块算子汇总所有共享使用者；整列 None 与 connected-zero 分开。
切片观察不承诺旧独立事件根的因果关系：需要该能力时明确使用记录接口。
KV Attention 的自定义一阶 VJP 汇总共享 K/V bank；不能声称支持高阶导数。

Tensor 只读借用遵守 [发布契约](tensor-publication.md)。浮点 Tensor 由 autograd 持有，
不复用仍有 reader/反向使用的 scratch。普通窗口保留跨窗梯度；optimizer 后通常对
continuation 调用 detach。更换参数对象、dtype/device 后重建 executor；常规原地 optimizer
更新同一参数所有者可继续。当前 host 路径不是设备常驻 ready queue 或异步通信新资格。

`column_checkpoint` 只解码候选实际 raw 字段，导出完整位置 cut 的状态、history、真实端口
ledger；不调用参考或重算。已有 checkpoint-v5 save/load 保留参数共享、optimizer 和精确
下一窗。低层只传 typed `left` 时，调用者还负责保存 ledger/history；优先公共窗口接口。

## 容量与检查

每节点临时 workspace 默认 4GiB，按 live B/L/history/维度保守估算。超预算明确拒绝；
循环scan按 B×最长样本长度计入填充，卷积bank/tap另计；仅按非空行数估算会低估ragged窗口。
卷积估计包括旧缓存与新行的bank×lag反向暂存、host/device int64正向和逆向索引。
该估计不代替全模型、保留 autograd、总 HBM 或多卡通信准入，也不限制累计逻辑时钟。
跨节点KV准备另用同一上限约束额外块packing/伴随：四个线性阶段各计两份矩阵、三份输入、
两份输出，并计归一化/相加和整数索引；half按至少FP32字节保守估算。按实际维度累加到预算
前分组；只有一个节点时使用原无矩阵stack的路径。此预算不改变既有每节点容量拒绝，
也不将参数共享或连续存储直接视为省去了所有临时复制。
跨节点Read/Full的额外预算包括各参数角色的packing、保存操作数、伴随、padded query/bank、
Attention/相对位置暂存及int64索引；同一Tensor出现在不同角色时分别计费。
沿稳定请求顺序在预算前切组，单节点继续使用原单列路径；这不是对总模型存储的准入。
observer的`packing/bucket/kv_{read,full}_nodes`记录实际节点组，
`column_{matmul,attention,logits}_nodes`记录真实合批收缩；投影的shared GEMM/private BMM另列。
`state_column_calls`仍是节点调用数量，不能直接用它推断矩阵算子提交数。
输出、状态、bank 和控制做有界打包 finite 检查。显式非法 owner、时间、形状或端口拒绝。

`test_kv_columns` 独立逐行公式检查快照、raw 过期、>2**54 时间及梯度；
`test_kv_column_groups` 检查相容/异形/零写入分组、预算切分、共享参数、独立None/零和版本；
`test_column_query_groups`用字面Attention/sum公式和独立标量记忆日程检查组收缩、三Read快照、
非空初态、原始位置、Full支持及参数alias；`test_column_query_dispatch`覆盖有限Describe字段、
真实默认Python/native分派、空bank、冻结输出和高阶拒绝。逐字段根参照直接消费独立mass/content的
公式，避免把旧Content拼接适配器引入的零梯度连接误认为列接口必须具有的依赖。
`test_regional_packing`用字面关联映射检查实际成员及线性转置；`test_regional_groups`用独立逐帧
区域公式检查十种发送/评分组合、不同成员、caps与一般分支、共享参数和None/零，禁止以数值零猜依赖。
独立SDK `weighted_sequence_columns`另检查实际C++分组、原始状态与未使用块的梯度连接。
其中`check_query_groups`比较组查询与原单列的逐字段根、预算切分和真实alias；完整图对照仍由
该程序的独立Streaming日程检查，组/单列对照本身不作为独立数学公式证明。
`test_column_positions` 检查相对旋转/距离、超大时钟、literal softmax 与方向导数；
`test_column_position_ops`以独立逐坐标相对公式检查重基准/有限域后备、FP32/FP64字段VJP、
真实alias、None/零及原值版本；绝对时间先转float的反例解释共同整数原点。
保存张量形状检查只证明无query倍旋转展开，默认Python/native独立标量日程另验完整结果/梯度。
`test_recurrent_columns` 检查四循环的快照、卷积和扫描；`test_native_recurrent_columns`
检查native四族两种读取/三快照、非空独立状态列根、保留窗和ragged容量；`test_settle_columns` 用独立
标量调度检查一般端口/同秩/重复边/共享/字段/消息/末态；`test_window_execution` 检查默认
分派、十种发送覆盖与代表评分/正则、容量和空样本；`test_column_training` 检查 AdamW、
保留窗及 checkpoint。`tidegraph-column-check` 是真正 standalone C++ 消费者与独立调度对照。
安装消费者的 `--positions` 覆盖带位置项的三步训练和新进程恢复；硬件范围见固定证据。
`test_tap_gather`用独立关联矩阵和方向导数检查卷积转置、缺失槽、零梯度和版本/阶数边界；
native由完整循环公式与非空缓存容量测试覆盖。实际设备重复反向/精确恢复须单独验收。
`test_half_columns`以独立标量调度核对五记忆/位置的完整字段、输入/参数VJP、三步FP32-master AdamW和
raw换owner精确恢复，包含大于2**54的稀疏逻辑时间；Attention另有溢出half点积的FP32字面公式及版本守卫。
半精度派生浮点字段使用既有half误差策略，离散/None和自身恢复保持精确；新小例不解除旧GDN AdamW见证。
安装示例支持`--dtype float16 --scoring-dtype float32 --loss-scale 16`并记录master和比较策略。
FP32 严格梯度超差保留为具名有界 witness，同材料提升 FP64 单列；不改通过阈值。

`test_column_sharing`检查共享使用的独立公式、固定累加次序、None/零、原始版本与一阶边界。
`test_column_payload_owners`使用独立标量调度检查同秩/重复边、区域共享与跨owner区域、状态/输入/参数
VJP、AdamW及新owner精确恢复；默认CPU退化不作设备资格。
实际双卡显式设置`TIDE_COLUMN_PEERS=npu:0,npu:1`；安装示例增加`--peer-device npu:1`检查新进程恢复。
独立SDK在原设备CLI上设置`TIDE_COLUMN_PEER_DEVICE=npu:1`，按区域交替owner检查五记忆与三种混合放置。
`scripts/check_column_adjoint_order.py --device npu:0 --client python --placement mixed-a --output-dir NEW`
对同一前向图重复反向，区分累加差异与前向/恢复差异；`--include-columns`记录内部边界，
`--autograd-threads serial`仅作有界诊断。它不替代安装消费者的新进程精确恢复。

成本复用 `benchmark_dense_query.py --baseline window|columns|graph|fc-tensor|transformer-16`：
window 是默认公共接口（含 raw continuation 导出），columns 是低层组合，graph 是历史
记录接口。配对固定材料、计时范围、实际候选/Full/消息和执行身份。Transformer 函数不同，
只作成本参照。固定源码资格、实测范围与源 hash 在 `docs/evidence/`，开发日志不是资格。

## 定义—实现—验证

| 固定定义 | 符号/返回结构 | 独立锚点 |
| --- | --- | --- |
| Architecture §1–2、§4.4 | `message_columns.aggregate_columns/readout_columns`，`MessageColumns` 的存在性/载荷与 cG 投影分离 | 一般端口标量调度、重复边/零强度反例 |
| KV §1.2、§3–4 | `kv_columns_state.prepare_kv_columns`、`kv_columns.read/full`；C++ `columns::prepare/read/full` | 逐行 KV 公式、非空左态、所有 Read 快照、实际 raw 解码 |
| Recurrent/GDN/SSD 与 convolution | `recurrent_column_state/math`、`recurrent_columns`及C++ `weighted_column_recurrent_*`；矩阵 scan 与逻辑时间 gather | 四族 literal/scalar 公式、>2**54 时钟与稀疏槽 |
| Allocation §2–6/S.2 | `regional_columns.region_columns`、`allocation_columns` 及 C++ 同名层 | [区域契约](weighted-region-batch.md)、手算解与方向导数；硬规则单独声明 VJP |
| Settle §9–11、PDG/Batch 因果块 | `SettleColumnPlan.run_ports` / `columns::settle`；同秩按稳定 region ID | 独立 Python scalar/native streaming、一般图回归 |
| 完整位置 cut/续接 | `column_checkpoint.import_state/export_state`、`EagerExecutor.run_ports` | 同 owner/新 owner 的 raw checkpoint、下一窗与 optimizer |

修改这些边界时同步本页、[semantic-map](semantic-map.md)、[architecture](architecture.md)
以及对应测试和实测身份；局部经验进入 [performance-lessons](performance-lessons.md)。
