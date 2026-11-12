# LH 直接公式、source-v2 columns 与 block resident 的静态成本

本次审阅确认：公共 columns 已按节点块执行主要矩阵计算；block resident v2 的外层
B×L 调度仍接着逐输入行参数 bank 和逐帧 raw 快照。这是两个不同层级的问题，不能因
一次外层调用或一个自定义 Function 就认定内部已完成张量贯穿。

本报告是源码审阅，**没有新性能测量或硬件资格**。[机器清单](lh-block-static-20261112.json)
记录实际读取的43个文件、范围和SHA256。审阅时HEAD为`838fc30c`，包括随后进入
`4248bde1`的工作区修改；因此以各文件hash为准，不能把整个审阅称为干净838fc资格。
43个已审文件hash均与后继`a3c4f62d`匹配，可从该提交复查静态结论；这不构成该提交的运行资格。
下文缓存键修复的开发检查另列，未进入已冻结的a3c设备构建。
原始机器记录保留于`artifacts/resident-block-dev/static-review.json`；本报告不依赖该目录复现。

数学固定`0770593838a37c790a568226930567ed60f99b83`，对应[共同架构§2.2](../upstream/tide-core-5/local-semantics/weighted-message/architecture-learning-note.md)、
[KV](../upstream/tide-core-5/local-semantics/weighted-message/kv-memory-learning-note.md)及
[张量窗口契约](../weighted-tensor-windows.md)。直接公式只承诺声明的数值投影；
公共观察根、resident设备推进与精确raw续接的责任仍分别存在。

## 版本与比较边界

| 身份 | 可以使用的结论 | 不可外推的部分 |
| --- | --- | --- |
| [fb83旧静态报告](lh-static-structure-20261112.md) | LH176的22区域、176节点、1058内部边；旧直接公式546次forward矩阵调用；旧columns物理来源逐项计算 | 旧`2A≤2132`来源矩阵调用不能描述source-v2 |
| `00bb5ac2` source-v2 | 来源按实际输入与完整有序参数槽精确复用，再按形状/预算合批；仍逐物理原子聚合 | 不能由来源合批推定Read/Full尺寸分组、head或resident参数展开已经消除 |
| 838fc工作区/4248 block v2后继 | B×L区域程序、qry/sum typed tape、公共Model与独立根；本报告核对其内层实际表示 | 公共入口接入、少量提交或局部通过不等于全尺度性能、全部记忆或多设备迁移完成 |

fb83的Read90组、Full159组、区域94组是一次实际材料的观察，不能当静态常量。
394是旧直接公式编译的来源槽数；首区域8端口同输入/同参数时还可条件共享至387，
该额外共享没有实施，也没有全局最小工作量证明。

`EagerExecutor.run_ports`首先要求匹配的Settle证书；一般PDG通过`__call__`的record入口
执行。模块能力不满足时，合法Settle输入才由run_ports走明确record adapter。详见
[window_execution.py](../../python/tidegraph/weighted/window_execution.py)。

## 出入口与循环位置

| 层 | 输入/输出 | 每次调用中的循环与张量职责 |
| --- | --- | --- |
| 直接公式`StaticKV.__call__` | 输入`x[B,L,d]`、逐端口support和可选left bank；返回逐端口c/G、状态bank及可选区域观察，raw由实际bank解码 | 构造时编译拓扑与来源槽；数值主链按22个区域推进，区域内保留N×B×L。按来源owner gather、参数角色stack、少量head旋转与输出端口循环仍在，但不逐B/L调用完整模块 |
| 公共`SettleColumnPlan.run_ports` | `SequenceLayout`的int64 offsets/times与物理端口`MessageColumns`；返回实际节点/区域/消息列及raw状态 | 按区域处理来源、节点私有历史与候选；数值矩阵跨相容节点分组。组内B/Q和B/K批量，组外仍有节点元数据、RMS/字段装配及反向索引循环 |
| 默认`EagerExecutor.run_ports` | 同一端口列及Continuation；返回WindowResult、输出列、Continuation与实际执行身份 | 验证输入/绑定、缓存静态plan，调用上述公共列，再导出状态；本LH材料无record fallback，不按每token重跑通用PDG |
| resident `BlockSettleSession.advance/backward` | `positions[B,L]`、`payload[B,L,P,d+1]`、presence；返回实际字段/tape，接受独立块根 | 主机建立潜在区域程序，设备choose/commit；数值内核另有按行参数gather、raw序列B/N/L及反向扫描。外部块接口与内部串行/复制须分开审阅 |

沿用“源码forward矩阵收缩”口径，当前无低秩LH176直接公式仍可手工展开为：
opening七个qry/att区域每个29次、closing七个sum/att每个25次、middle八个sum/ind每个21次，
共`7×29 + 7×25 + 8×21 = 546`。每区的来源MLP两次、incidence归约两次、
Upd四次、Full七次相同；其余是Read/Describe/区域head的不同配置。
55f8的参数槽识别修复不改变这个无低秩材料的循环。此数不包含逐元素操作、packing或反向，
也不是所有ATen调用或设备kernel总数，更不是工作量最少的证明。

source-v2公共来源MLP按实际shape、参数与输入共享条件形成S个数值组，当前非零alpha域
是每组两次投影，而非每物理边两次；低秩有效矩阵重建另计。
Read/Full的组数依赖实际查询/历史长度、活动支持和workspace切分，下面给出每组表达式。
静态可说明循环与上界，不能把某次运行的组数写成全部输入的固定LibTorch调用总数。

## 第一层：调用数量与矩阵形状必须同时说明

令区域节点数为`N`，窗口为`B×L`，resident分配行数为`R=BNL`，缓存容量为`C`；
`d/ds/df`分别为内容、摘要、FFN宽度。公共组含`G`个节点，`Q/K`是每sample实际查询和
bank长度的最大值，至少padding至1。以下只计forward源码矩阵调用，不计所有ATen或设备kernel。

| 数值边界 | 直接公式，每区域 | 公共columns，每实际相容组 | block resident，每区域程序 |
| --- | --- | --- | --- |
| KV Upd | 4次投影 | 4次投影；实际正写入数参与分组 | 4次投影，但先按R行展开参数 |
| sum Mem＋current summary | 3次 | 3次 | 2次；短/长合成2R个pool行，但复制summaries |
| qry Mem＋current summary | 7次；共享短/长QK | 无位置且有正记忆时8次，短/长重复QK；共同原点位置路径通常7次 | 6次；短/长合为一次pool，但K/V按R×C重复投影 |
| Full | 7次；H并入batch | 有正记忆时通常7次；无记忆时4次；H并入batch | `5+2H`次，C++逐head组织QK/PV |

相对位置fallback用逐head的pairwise旋转/点积，不能套用上述普通矩阵调用表达式。
无活动/无正记忆、预算切分和观察根都会改变实际执行。Describe、来源MLP和分配另计。

这说明**少一次调用不等于少做计算**：resident的qry虽把两个pool合成一次BMM，
却在每个query重新投影历史；直接公式在共享bank上只投影每条历史一次。
无位置columns重复QK是另一个明确小重复，见
[read_groups](../../cpp/src/weighted_column_read_groups.cpp)及
[contractions](../../cpp/src/weighted_column_query_ops.cpp)。

### 直接公式和公共columns的投影

[直接project](../../scripts/lh_tensor_math.py)采用
`X[N,BL,I] × W[N,I,O] → Y[N,BL,O]`；所有矩阵确为同一对象时改为一个共享GEMM。
来源轴替换N，同一矩阵的部分别名仍可能重复stack。直接公式主链按区域推进，
没有逐B/L完整模块调用；显式decode在边界按区域、节点、sample恢复实际raw。

[公共linear_blocks](../../python/tidegraph/weighted/row_linear.py)和
[C++对应](../../cpp/src/weighted_row_linear.cpp)使用
`X[G,BQ,I] × W[G,I,O]`。反向参数收缩为
`dW[g] = dY[g]ᵀ X[g]`，返回`[G,O,I]`，不物化每条输入的外积。
无梯度packing和显式VJP保留每个节点输出是否使用、原始参数版本和共享贡献。

### resident当前投影和head

[weighted_projection](../../tools/device_online/weighted_projection.cpp)的`banks>1`分支将
参数gather为`[R,O,I]`，物化转置`[R,I,O]`，执行
`[R,1,I] × [R,I,O]`。反向先生成`[R,O,I]`逐行外积，再按owner归约。
矩阵与转置两份FP32存储为`8ROI`字节；节点块方式相同两份为`8NOI`，相差`BL`倍。
`banks==1`已有真实整块GEMM/VJP，但不能代替多bank改造。

两个调用边界还会扩大复制：

- [KV update](../../tools/device_online/weighted_kv_update.cpp)先将base、left、right、summary
  gather为`[R,…]`，再给projection逐行owner。即使原模型只有一个bank，R>1时也进入多bank路径。
  反向先归约到这些行bank，再归约至原owner；base/right/summary还各请求一次输入RMS。
- [weighted_head](../../tools/device_online/weighted_head.cpp)没有单bank块分支，始终按行复制
  两层矩阵、bias及转置，反向逐行外积。Describe score/match和区域context/query都使用它。

[Model lowerer](../../tools/device_online/block_settle_model.cpp)按每个节点stack参数角色。
canonical参数别名最终会合并梯度，但没有自动压成更少的执行bank。参数共享、计算共享和
私有状态是三件事；修复需在角色和owner层分组，不能仅更改最终参数归约。

## 第二层：分组、历史bank与raw快照

### 公共columns：实际ragged分组及剩余装配

[kv_column_groups](../../python/tidegraph/weighted/kv_column_groups.py)的Read键是KV配置、
区域配置、Read模式、dtype/device、`(B,Qmax)`及`(B,Kmax)`；Full先取正control行，
再按KV配置、dtype/device、选中Qmax和Kmax分组。键不要求完整ragged分布相同。
每个等价类再按workspace预算切开；单节点组使用原单节点路径。

因此组数是实际等价类与容量的函数，同一区域最坏每节点一组。完全相同配置不保证合组：
实际活动数或历史长度不同即可分裂。当前选择了减少Full算术的紧凑行；更宽padding/分桶
可用多做部分算术换少量提交，但必须保留选中语义、独立根和原始时间可见性。

投影和contraction已经批量，以下仍逐节点执行：正写入/活动检查与host索引、
RMS、小统计/特征拼接、bank装配、返回field反向索引。源码分别为
[KV准备](../../python/tidegraph/weighted/kv_columns_state.py)、
[查询输入](../../cpp/src/weighted_column_query.cpp)和Read/Full groups。
这不是每个token重跑完整模块，也不等于全部小操作已跨节点合批。

公共raw保留一份实际写入bank；查询只gather需要的字段为`[B,K,width]`，再构造
`weights/age[B,Q,K]`，不会直接生成`[B,Q,K,width]`的完整raw。Read和Full仍各自gather
相同bank，跨节点contraction再次packing；共同原点位置域之外的relative旋转另有pairwise临时量。

### resident：完整raw逐帧物化和重复历史投影

[raw布局](../../tools/device_online/weighted_raw_state.cpp)为：

`F_C = d+1+C(2d+ds+1)` 个浮点数；`I_C=4+C` 个int64。

[block_kv_sequence](../../tools/device_online/block_kv_sequence.cpp)保存old与proposal各
`[R,F_C]/[R,I_C]`，terminal为`[BN,F_C]/[BN,I_C]`，mapping为`[2,R,C]`。
仅前两快照及mapping主要占`8R F_C + 16R I_C + 16RC`字节，尚未包括terminal、写入和查询窗口。
Read0另把绑定初态`[BN,…]`展开成`[BNL,…]`，绑定正确不代表复制必需。

[设备序列代码](../../tools/device_online/ascendc/block_kv_sequence_math.h)在一个工作block
内按`B→N→L`循环，拷贝完整old/proposal，更新mapping并重新填入历史字段。
[wrapper](../../tools/device_online/ascendc/tide_block_kv_sequence.cpp)仅让block0执行。
反向同样单block：每个快照的每个连接字段再扫描所有此前写入，最坏结构标记工作为
`O(BNL²×字段数)`。一次kernel没有消除这些内部循环；B/N所有者独立也没有被此实现并行利用。

[window](../../tools/device_online/weighted_kv_window.cpp)随后又生成keys/values/summaries
`[R,C,width]`及双通道weights/decay、ages/mapping。sum只使用summaries；qry/Full不使用
summary值，但同一helper仍搬运所有字段。Read qry把K/V输入摊成`[RC,d]`，
多bank参数进一步展开至`[RC,ds,d]`。其重复历史投影及存储不由Attention数学要求。

数学要求保留Read0/old/proposal/Full各自观察、原始int64时间、实际末raw、独立根、
版本和精确解码；不要求为每个查询复制全部历史或每行复制矩阵。可替换边界是共享
initial/write bank、每查询可见性/前缀映射、owner分组GEMM及块伴随、按需观察物化。
若调用者真的请求每帧完整raw独立根，其返回/伴随大小仍需按实际请求计费；不能用
有效状态投影或参考重放替代这些能力。这里提出替代边界，**尚未实现或测量收益**。

## 直接基线的参数槽修复与开发检查

审阅发现`StaticKV._sources`用去重的`Module.parameters()`作为完整参数身份。
例如base为`A,B,E`，因子槽分别为`(A,A,B,B)`与`(B,A,A,B)`，去重列表同为`(A,B,E)`，
实际有效矩阵却分别是`(A+AA,B+BB)`与`(A+BA,B+AB)`。
即使A和B数值相同，前向值相同也不能合并两个函数：独立输出对A/B的梯度不同。

[修复`55f8c631`](../../scripts/lh_tensor_baseline.py)按`a,b,embedding,a_left,a_right,b_left,b_right`
的完整有序角色记录身份；无低秩时为前三槽。不引入生产来源helper，保持直接公式独立。
[专属反例](../../tests/test_lh_source_slots.py)覆盖相同值不同槽、不同值不同槽，以及
全部槽真正相同仍允许共享；分别比较字面公式与每个输出根的输入/参数VJP。
无低秩LH176的旧固定结果不受此缺陷影响，不改写旧证据。

整理期间短CPU开发检查：

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=python \
  artifacts/additive-work/env/bin/python -m pytest -q \
  tests/test_lh_source_slots.py tests/test_lh_tensor_baseline.py --dtype both
```

结果为**22 passed（8.68秒）**，FP64/FP32；其中6项新参数槽反例，其余是既有直接基线
输出/raw/梯度、重复边、跨窗/大int64、三步AdamW及恢复检查。JUnit在
`artifacts/resident-block-dev/lh-source-slots-dev.xml`，摘要及修复文件hash进入机器记录。
干净环境使用[开发入口](../development.md)准备Torch后运行同一pytest路径即可；
不依赖旧二进制、日志或raw运行目录。本段检查发生于提交前工作区，修复随后提交为55f8c631；
这是开发验证，不冒称另跑了固定提交资格或重新测量性能。

后继优先边界是resident角色块投影与共享历史bank；columns的尺寸分桶、小算术和重复QK
分别处理。沿用[优化skill](../../.agents/skills/optimize-tide-execution/SKILL.md)的两层审阅：
既核对提交/分组次数，也核对每次提交内部的张量形状、实际复制、并行域和伴随归约。
具体待办和资源安排仍由ROADMAP/STATUS维护，本报告不增加第二份任务账本。
