# 三平台有限矩阵：配置、准入与停止条件

本页定义本轮可执行选点；**不是运行证据**。
固定数学为 `0770593838a37c790a568226930567ed60f99b83`。
配置在 [configs/matrices](../configs/matrices/)，[index.json](../configs/matrices/index.json)
列出逐项模型/工作负载/执行身份、去重参数量、物理节点/边、batch、时间窗、线程、布局和静态预算。
实测结果从 [STATUS](STATUS.md) 的当轮证据进入；不能把本页 planned 或预算视为 passed。

复用上一轮47项严格通过和3项FP32失败，不重跑旧矩阵。
本轮增加CPU/NPU正式测量、FP32/FP16、B128/B512、约1B的大batch及同模型多卡。
有限数值探索得到授权；原阈值、失败、strict xfail和停止条件继续保留。

## 有限列表与执行顺序

总计189个明确case：CPU48、NPU72、CUDA69，按44个独立阶段文件组织。
每个case各一次独立测量流程，warmup1窗、正式2窗、完整返回、额外一窗精确恢复；
小型/批次配置超时900秒，scale/capacity配置1800秒。
这些是单case上限，不是承诺运行全部case的总耗时；应分组执行并检查终态、争用和准入。
没有一个命令隐式展开全部开关的笛卡尔积。

| 文件组（前缀 cpu/npu/cuda） | CPU | NPU | CUDA | 本组回答的问题 |
| --- | ---: | ---: | ---: | --- |
| coverage | 12 | 12 | 12 | FC/HB/LH的Settle、真实非对齐TimedDAG、真实反馈PDG；同一FC实例跨层级嵌入；一般Settle端口/同秩区域 |
| clients | 6 | 5 | 5 | 小型纯PyTorch、Python/native、standalone共同推理/训练/恢复；CPU另有FP64小参考 |
| modules | 2 | 2 | 2 | HB的qry/位置/区域Attention；LH的KV+GDN+SSD异构和短卷积 |
| batch | 7 | 7 | 7 | 固定4节点d128模型B1/8/32/128/512；B8 streaming与更长prefill |
| packing | 2 | 2 | 2 | A/C与batch组已有B构成同材料A/B/C；B不重复测量 |
| finite | 6 | 6 | 6 | 根据GPU诊断热点新增B128 scalar/packed有限值检查、小型训练两策略、B512 packed及独立profile；默认检查与数学不变 |
| scale | 7 | 7 | 7 | B1增参数；FC少/多节点、HB/LH各约1B；固定1B的B128/B512 |
| resident-match | 2 | 2 | 2 | 与resident有界B128/B512严格同材料的d16 eager；不把d128 batch组混入布局比较 |
| scale-observation | 1 | 1 | 1 | 对已实际完成正式流程的1B/B128另采实际MM/BMM形状与桶；不重复独立参考、不作吞吐或新增数值资格 |
| threads | 3 | — | — | ATen/BLAS1/4/16线程；独立调度workers1/2 |
| placement | — | 2 | 2 | 同一B128材料mixed-a/b，复用batch组mixed-c |
| precision | — | 2 | 2 | FP16 B128推理及小型FP16一阶训练，明确FP32 scoring/master |
| resident | — | 4 | 4 | 单卡B1/128/512静态布局入口和小型一阶训练；默认布局拒绝单列 |
| resident-bounded | — | 2 | 2 | 同一FC4/P2的B128/B512，静态物理原子上界12B、stages32；保留默认拒绝 |
| multicard | — | 7 | 6 | 同一1B模型2/3/最多卡放置；同一小训练模型的ordinary/blocking/cpu_bridge |
| capacity | — | 4 | 4 | 同一1B训练模型单/双/最多卡；1B校准后一个约2B单卡推理候选 |
| capacity-sharded | — | 3 | 1 | 单卡容不下的同模型：NPU38.406B单/双卡拒绝及三卡近界候选；CUDA9.123B双卡磁盘准入拒绝 |
| capacity-calibrated | — | 4 | 4 | 依据各平台1B实际峰值新增CUDA4.561B单/双/三/四卡与NPU12.802B单/双/三/八卡；各自逐卡、主存和两代checkpoint先准入 |

先运行coverage/clients及必要基座检查，再进入batch、packing、modules和放置；
中等规模通过或得到明确可解释的探索结果后，再逐项进入scale、multicard、capacity。
resident可以独立推进，硬件peer门禁或单项严格失败只阻塞依赖该条件的结论。
正式计时固定设备、线程和可见环境；本项目重型构建不与计时重叠，profile另开任务。

## B128/B512及A/B/C的明确对应

三个平台的 `*-batch.json` 都含 `*-fc-b128`、`*-fc-b512`；其余batch点为1/8/32。
固定模型是FC两区域、每区域两节点、d128、KV、自然FFN=4d、P2。
使用 `staggered-v1`：样例真实长度和缺失输入端口有差异，原始逻辑时间不变。
不是把一份样例完全复制为512份来宣称不定长打包通过。
B8的streaming与prefill用相同模型/任务；P8长窗另有工作负载身份，检查缓存与实际时间批变化。

| 实际策略 | case | 算子packing | 完整返回packing | 同材料要求 |
| --- | --- | --- | --- | --- |
| A | `cpu/npu/cuda-packing-a-b128` | scalar | scalar | 与B/C相同模型、输入、B128/P2、精度、客户端、线程、执行位置 |
| B | `cpu/npu/cuda-fc-b128`，位于batch组 | packed | scalar | 复用已经测过的该case，不另测同样的B |
| C | `cpu/npu/cuda-packing-c-b128` | packed | packed | 仍返回完整记录，不能混入outputs/metrics的较小返回契约 |

这里 `cpu/npu/cuda-` 表示分别替换三个平台前缀，不是一个case名称。
三个策略均显式配置，`auto` 只作为请求解析概念，不能被统计为第三种算法。
每个结果记录实际candidate/Full/消息量、实际时间批/桶/缺席行、复制同步和完整返回成本。
如果路由或工作量变了，列为非等价成本观察，不计算严格等价加速比。

各平台 `*-scale.json` 又明确列出 `*-scale-fc-few-1b-b128` 和 `*-scale-fc-few-1b-b512`。
它们的存在不免除资源准入；不能因历史FP32超差自动删掉，也不能为了运行而静默改batch/缓存。
在合理预算或有依据的保守预算校准后仍不能准入，就保存所选完整配置与资源拒绝，继续可运行的B512中等模型。

## 参数规模、物理图与共享

以下是公共独立inventory算出的实际可训练参数；不是把旧core-3参数公式套到新模块。

| 约1B代表 | 节点 / 区域 / 物理体边 | d | 实际去重可训练参数 | 不可拆共享组 |
| --- | --- | ---: | ---: | ---: |
| FC少节点 | 4 / 2 / 4 | 4096 | 1,010,339,302 | 2 |
| FC多节点 | 16 / 8 / 28 | 2048 | 1,014,118,392 | 8 |
| HB扩散/收拢 | 20 / 10 / 48 | 1792 | 971,589,310 | 10 |
| LH启发连接 | 20 / 10 / 64 | 1792 | 971,589,822 | 10 |

规模来自自然内容宽度128的步长与FFN=4d，无无效参数或任务头凑数。
FC少节点的固定B1曲线为d128 → 99,487,206参数/d1280 → 1,010,339,302参数/d4096。
FC多节点、HB和LH的1B代表与FC少节点具有不同工作量；比较用于解释规模/形状取舍，不能称同一模型的等价加速。
非对齐与反馈变换改变物理图及model_id；仅改变合法执行视图的两项FC嵌入保持原model_id/workload_id。
PDG的Python运行仍是独立正确性参考；正式PDG成本候选选择native/LibTorch，不把纯Python参考当同等性能实现。

多卡使用16节点FC模型，以保留8个不可拆共享组：GPU单/双/三/四卡；NPU单/双/三/八卡。
单卡推理基线是scale组的 `*-scale-fc-many-1b`，不重复创建一份同模型单卡测量。
静态GPU2/3/4卡分别拥有8+8、6+6+4、4+4+4+4节点；NPU8卡每卡2节点。
4节点大模块模型只有两个共享组，不能用它宣称四/八卡都承担了同模型容量。
实际验收还要核对参数/私有状态/消息/计算的真实设备、全部共享梯度贡献、统一optimizer边界和恢复。
静态映射、普通复制成功、前向成功与严格多卡resident资格分别记录。

## 预算、拒绝及重启条件

所有case均预算warmup+2测量窗+精确恢复窗，`live_results=1` 表示设备端同时保留的一窗结果。
严格/探索比较保留的CPU快照由 `host_results` 独立计费，默认包含warmup、测量窗及两份恢复比较结果。
静态逐卡参数、私有状态/KV、梯度、master/slots、通信、journal、workspace、完整返回与checkpoint恢复
同时核对；未知动态项仍是unknown。主存和磁盘另计，不能把空闲HBM简单相加。
站点模板只给预留，不包含过期free-memory值；执行前用实际站点快照和当前争用重新准入。

| 静态代表（FP32、当前完整返回/恢复预算） | 已计张量预估 | 如何使用 |
| --- | --- | --- |
| FC1B B1 / B128 / B512 单卡推理 | 4.03 / 5.98 / 11.89 GiB | B512仍需独立检查workspace、主存约57.23GiB、磁盘约11.10GiB及当前HBM余量；预算不是准入 |
| FC16节点1B训练，1 / 2 / 4卡 | 30.76 / 15.38 / 7.69 GiB每卡 | 单卡3090保守预算拒绝；双卡点待真实梯度/optimizer/恢复峰值；NPU按实际单卡容量另判 |
| 单卡约2B推理候选 | 7.95 GiB，另有未知动态项 | 1B构造/窗口/恢复峰值校准之后唯一初始近界候选；预留不足就拒绝 |
| d16 resident B128 / B512 默认布局 | 16.87 / 269.61 GiB | 默认tensor_bytes仅1GiB，两项已静态拒绝；这是容量布局/默认stages条件，不是NPU/GPU不支持该逻辑batch |

Python eager推理的checkpoint权重先解码到CPU，再复制进入既有设备bank，故不再虚计第二份完整设备权重；主存解码/副本和设备continuation预留保持。C++事务恢复、resident及训练仍保留各自预算。这只是对实际分配位置的预算修正，未改变运行或恢复实现；新容量点仍须用实际峰值校准。旧测量保留其原预算与源码身份。
Python训练host预算补齐exact-next期间同时存在的四份参数、两份梯度、四份optimizer值快照，
使用FP32构造floor覆盖half快照；完整记录、workspace及主存余量另计，不按卡数平摊。
1.014B训练静态host由35.261GB修正为52.585GB；原36baeb7实测恢复RSS40.265GB仍按旧来源记录。
这是由实际欠估触发的准入修正，不重新执行原训练，不改变逐卡张量/磁盘预算或数学身份。

经实际1B单卡构造/恢复峰值约4.788GB校准，约2B FP32只作为原初始候选保留，不能称24GiB近界。
新增CUDA `capacity-calibrated` 同一FC16/d4352/B1/P2，4,561,319,928可训练参数、8个不可拆共享组。
显式每卡1GiB workspace、256MiB activation、64MiB通信预留，并额外保留3GiB空闲显存；
单卡已计峰值约20.94GB，host约91.66GB，两代checkpoint需42.49GB可用磁盘，checkpoint上限显式32GiB。
这不是将多卡显存相加；1/2/3/4卡都核对自己的bank、实际位置和峰值。1B多卡先验后才运行这些点。
4.788GB来自4节点d4096的总峰值，不能按节点数直接线性乘其临时空间；预算保留上述额外工作区而非只计bank。
下一自然宽度d4608的bank21.81GB、两代磁盘47.48GB，在50GiB盘保留环境/证据和余量后可能先被磁盘拒绝；
不分配此静态候选，不为找边界反复OOM。运行前须审阅并退役终态大checkpoint，若live准入仍拒绝则记录具体差额。
NPU近界点需自身1B实际峰值校准后选择，不能照搬3090字节或把64GiB卡上的2B称近界。

这些值来自index的当前静态计算，**不是实测峰值或安全可用上限**。
CPU返回快照与设备端存储分开预算，避免重复把所有CPU验收快照算入HBM。
逐窗checkpoint的旧/新generation同时占盘，磁盘按两份保存预算；这些修正不改变batch、返回或数学。
校准可依据真实生命周期修正未知项，不能仅为了通过削小预算。
resident默认 `stages=span×regions×B`，journal还乘B，形成显式B²预算。
重启需要实际stage计数/设备timeline及完整语义证明，采用有依据的显式limit或有限实现修改；
保留原拒绝配置，任何修改都有新配置身份，不减少逻辑batch、不CPU预跑真实轨迹。
对齐FC4/P2另有 `*-resident-bounded.json`：外部原子≤2×P×B、物理消息≤4×P×B，故queue=12B覆盖整窗全部原子。正延迟、无跨样例路由及封窗保证每活跃样例的最早时间逐阶段递增，阶段上界无需乘B；显式stages32保守覆盖span8。B128/B512的tensor_bytes分别2/4GiB，保留旧queue×B构造守卫。此配置不改默认、不用参考轨迹，不能套到反馈/非对齐图。局部Full临时矩阵、服务队列/流/任务和物理role副本也须单独检查，不能只抬高tensor_bytes。

初始checkpoint上限：小中配置2GiB；规模/容量配置16GiB，构造参数上限32GiB。
大点每次处理后先审阅/保留必要证据，按保留规则退役可重新生成的大材料；不在50GiB GPU数据盘累积整组checkpoint。
训练与推理分别接受，训练容量拒绝不否定推理；精确续接成功也不能替代独立正确性。

## 从干净安装预览、执行或重建

先按照 [human-workflow](human-workflow.md) 安装、doctor、lock并填写自己的实际site.json。
用同一已安装命令预览与执行每个平台组；target site中的库/ABI/设备身份必须实测核验。

```sh
tidegraph plan configs/matrices/cpu-batch.json --matrix --site /path/to/cpu-site.json --format json --output cpu-batch-plan.json
tidegraph matrix configs/matrices/cpu-batch.json --site /path/to/cpu-site.json --output-dir jobs/cpu-batch-new --detach
tidegraph job status jobs/cpu-batch-new
tidegraph job logs jobs/cpu-batch-new --lines 80
tidegraph results jobs/cpu-batch-new --format csv --output cpu-batch.csv
tidegraph results jobs/cpu-batch-new --format html --output cpu-batch.html
```

NPU和GPU对应替换为 `npu-*.json`、`cuda-*.json`；GPU文件及wheel可从干净检出复制到目标根，
不读取远端脏dev目录。不满足独立peer协议的多卡resident不在这些mixed复制case中获得认证。
要只运行一个待准入点，可用标准JSON操作将该case的 `config` 保存为文件，再使用 `tidegraph plan/run/job start`：

```sh
python -c 'import json,sys; m=json.load(open(sys.argv[1])); r=next(c["config"] for c in m["cases"] if c["name"]==sys.argv[2]); print(json.dumps(r,indent=2))' configs/matrices/cuda-scale.json cuda-scale-fc-few-1b-b512 > selected.json
tidegraph plan selected.json --site /path/to/cuda-site.json
tidegraph job start selected.json --site /path/to/cuda-site.json --output-dir jobs/cuda-1b-b512-new
```

静态拒绝通过该入口保留配置和失败报告；不得把启动外层作业成功改写为该配置通过。
`strict`失败；`explore`只接纳有限数值超差、保持离散语义严格；`none`明确未检查。
本矩阵large-batch/scale使用explore，但每项仍运行完整独立比较、记录原严格结果与精确续接。
非有限值、通信错误、求解器拒绝、非法配置、容量不足立即使该case失败；不吞任意异常、不放宽阈值。

少量独立观察从已选B128/B512配置复制到新文件，将 `measurement.observation` 设为 `packing` 或 `profile`，
同时 `measurement.steps=1`；另开进程、输出目录与正式计时分开。该配置明确标为观察，不是正式吞吐样本。
优先查看FC B128的A/B/C，再按实际热点选择一个1B或B512点；不展开全部平台/模型的profile矩阵。
packing观察记录可用行/桶、调用形状与真实边界活动；完整设备resident推进仍需设备时间线，不能用主机计数代替。
`scale-observation`固定1B/B128的原模型/输入和前两窗，warmup1后观察1窗；只在对应正式流程完成后运行。
显式validation=none/checkpoint_check=false避免重复昂贵参考/恢复，仍完整返回和发布checkpoint；
该观察自身未独立验收，结果只能结合原正式证据解释形状、调用、分桶与资源，不成为新的合格吞吐。

所有材料已经写入JSON，不依赖生成器或旧运行目录。
需要审查再生成时，固定安装和源码后运行一次有限静态生成器；只能写新目录：

```sh
python configs/matrices/generate.py --output-dir artifacts/reviewed-matrix-preview --sites-dir configs/sites
```

它只调用公共planning/config接口，不分配张量、不运行候选或参考。
覆盖与预算测试是开发检查，不是本矩阵的设备资格。
后续新的资源校准、profile和六项打包处置应链接当轮证据与
[performance-lessons](performance-lessons.md)，待办只写入 [ROADMAP](ROADMAP.md)。

CUDA有界resident的B128/B512已完成实际流程，但它们的d16模型与原batch组d128不同。`*-resident-match.json`补各平台d16/FC4/B128或B512/P2的packed/scalar完整返回；模型、输入、task身份与resident一致，单独独立验收后再比较成本。原B128/B512默认resident布局拒绝继续保留，不能拿容量修正称数值修复。

NPU校准来自本轮1b3e39f的实际约1B推理：FC4全程allocator reserved4.505GB，FC16已观察4.620GB，真实bank4.326GB。Torch报告总量65.796GB、当时free65.643GB，不能直接使用营销64GiB数值。新增单一自然宽度d7296、FC16/B1/P2、12,801,882,104可训练参数、8个不可拆共享组；这是满足64GiB级卡近界目标的单个模型，不是8B/16B全扫描。bank54.617GB，沿同拓扑1B reserved/bank比例估计约58.335GB；显式3GiB workspace、512MiB activation、64MiB通信给出58.565GB静态峰值，另留6GiB不用。实际仍须逐卡live准入及有限实测；不会由静态预算称通过。host241.331GB，两代checkpoint117.232GB，私有构造/checkpoint上限显式96GiB。最大FFN矩阵212,926,464元素低于原生单tensor2^28边界。下一宽度7424静态峰值60.498GB已超过该现场扣除6GiB后的约59.200GB，不分配、不用OOM搜索。1/2/3/8卡保留完全同一模型和输入，共享组不拆成独立参数。

同模型多卡容量补充：原4.561B/12.802B分散到多卡可以证明放置，但不是多卡容量边界。`capacity-sharded`另固定一个NPU FC48/d7296模型：38,405,646,376可训练参数、24个不可拆共享组，B1/P2/span52；单/双卡估算168.047/85.937GB拒绝，三卡每卡16节点、约58.566GB，另留6GiB。host预算766.974GB（含64GiB主存余量），两代checkpoint349.567GB，构造/文件上限显式256GiB；单Tensor元素上限不变。仅在12.802B校准点有限执行、精确恢复及实际峰值建立后运行三卡点，1800秒、一次流程，无OOM搜索。更大模型不展开。CUDA对应FC32/d4352的9,122,639,888参数双卡模型每卡估算20.942GB，host174.740GB，但两代checkpoint83.896GB已超过整个50GiB盘；只做现场准入并保留拒绝，不分配、不删除活动文件或偷换checkpoint/返回契约。

现场八卡队列因第八张卡被其他项目使用而在600秒后退出，未运行不算模型失败。新增明确的`npu-model-1b-cards-7`，同一16节点/8共享组模型，七卡节点分配4/2/2/2/2/2/2；保留八卡原请求与等待记录。该项等待既有固定任务全部终态后独立执行，不重新映射活动作业，不打断其他项目。只补这一1B推理点，不扩一组七卡训练/容量扫描。

有界checkpoint优化的后继配置单列于`configs/checkpoint-cost`，不重写189项原目录。
CUDA9.123B使用显式interval3，将最终发布代数降为1后重新现场准入；原每窗两代磁盘拒绝保留。
NPU38.406B三卡的`npu-capacity-periodic.json`仅在原1800秒流程已完成有限值窗口、却在保存/恢复阶段超时，
且36baeb7相关NPU检查通过时，运行一次同模型/输入/返回/精度的interval3候选，仍1800秒。
若原配置完整结束则不重复；若非有限、通信、求解器或容量失败则不启用此后继。
减少保存次数是显式耐久性取舍，精确最终恢复和独立参考仍完整执行，不算原失败修复或窗口算子加速。

训练另选一个明确新工作负载：`configs/capacity-training/{cuda,npu}-scaled-loss.json`，
原1.014B FC16/8共享组、B1/P2、FP32、packed/scalar完整返回、原SGD/momentum/decay，
仅将内置平方loss系数从0.001改为1e-9；task/config身份改变，模型/输入身份不变。
原配方warmup约6.33e6、第二窗约7.50e14、第三窗非有限的失败继续保留；
这个单一系数用于训练存储/恢复机制观察，不是调优模型效果，也不是原失败修复。
每平台先在d8小模型上`check --scope full`通过再运行：CUDA双卡、NPU单/双卡，
每项warmup1+2更新窗、完整梯度/optimizer/独立CPU检查与exact-next，仍1800秒；
不继续搜索其他系数/学习率，任一非有限、通信、容量或精确恢复错误保持失败。
默认每窗checkpoint和原严格阈值不变；全部候选使用36baeb7固定包，实际资格各自记录。


### NPU计算收尾的依赖范围

双卡公共`check full`的all独立根发生EE1017/SIGSEGV，单卡同类根有原FP32数值超差；
它们保留为严格失败，不能扩大为通过。独立的实际训练入口仍按原loss/梯度/optimizer/恢复
分别验收。新loss候选单/双卡各先运行FC8/d8/B2/P3的小型三更新完整训练，保持原阈值；
对应小训练的执行、有限值、离散、原loss梯度/槽与精确续接成立后，才运行
`capacity-training/npu-scaled-loss.json`中对应1B项（1800秒/项、一次流程）。
小型多卡独立根未通过会限制通用VJP资格，不能被大模型探索结果覆盖；单卡计算不依赖双卡根成功。
仅为定位双卡all根，允许原ordinary配置显式设置`site.environment.ASCEND_LAUNCH_BLOCKING=1`
一次有界诊断；同步提交不称默认运行时修复或性能结果，不改精度/输入/容差。
原B512旧checkpoint重放在raw边界已有分歧；另一次同进程新建checkpoint与即时解码/下一窗
可区分重放与恢复。该必要诊断使用联合主存准入后的tmpfs，不做大文件成本对照。

## 八卡共享资源的有界后继

`configs/capacity-training/npu-eight-followup.json`复用原FC16约1.014B模型的八卡推理，
并列出FC16/d8小训练及loss1e-9的八卡约1B训练。大训练只有小训练完整独立数值/离散/精确续接
通过才进入；这是新增任务，不重试原0.001非有限配方。
所有项B1/P2（小门禁B2/P3）、KV、FP32、native mixed-c、ordinary、packed/scalar完整返回；
共享参数组不可拆，实际各卡owner由公共plan/run记录。每项只运行一次，推理1800秒以内、
小门禁300秒、大训练1800秒以内；不恢复12.8B/38.4B额外大文件流程。

可在原队列之外显式授权共用低占用卡，但仍须遵守原队列。可选站点适配器
`scripts/run_shared_npu.py --help`读取调用方指定并校验SHA256的站点helper，仅在本进程替换
设备可用性谓词：PID/启动ticks/UID绑定、到期时间、健康、利用率、HBM使用和剩余预算全部校验。
其余priority/FIFO、两次采样、整组锁、子进程生命周期由原helper执行；不修改共享helper、
不绕过锁、不终止其他任务。适配器不是安装后tidegraph的依赖，也不增加新的mixed类别。
共用状态必须保存在policy/audit，成本不宣称独占或统计稳定。

固定提交后，在站点队列分配内运行安装环境的：

```sh
python scripts/run_npu_eight_followup.py configs/capacity-training/npu-eight-followup.json \
  --site site.json --source FIXED_COMMIT \
  --scratch /dev/shm/NEW_GEF_EIGHT_RUN --output artifacts/NEW_GEF_EIGHT_RECORDS
```

该计算驱动仅调用已安装公共`tidegraph run`和现有监督/报告；读取同一展开配置，不导入benchmark。
它要求tmpfs余量48GiB、可用主存128GiB（含约53GB训练host和保留checkpoint代数），
持久盘低于软配额且距硬限至少150GiB。每项还执行公共逐卡/主存/输出预算；
不因临时盘可用而忽略主存。逐项终态复制小记录到持久盘，checkpoint留tmpfs供审阅；
中断后根据准确unit/子监督核对，不能只看外层状态。
