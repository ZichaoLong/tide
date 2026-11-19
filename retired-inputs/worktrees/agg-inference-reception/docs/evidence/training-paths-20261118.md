# 固定训练起点：单卡、两卡路径与有限数值见证

八个训练路径作业完成31个候选：9个 `within_threshold`、20个
`finite_deviation`、2个 `structural_failed`。31例的共同起点、重复VJP后起点不变、
保存数据和恢复下一前向均精确；恢复后的真实更新有28例在阈值内、3例保留超差。
没有非有限值或dtype检查失败。这里的有限偏差标签包含部分实际support变化，不能
读成路径等价。原始阈值、两项整数时间失败和三项恢复超差均保留。

本页覆盖固定起点的推理、含图前向/VJP、多步更新和新进程恢复，以及一个局部
Attention见证；不报告训练质量、正常吞吐、二十更新或FC恢复调查。逐例统计、
首差坐标、原始记录/输入/命令/产物SHA256在[JSON](training-paths-20261118.json)。

## 源码、起点及阈值

主矩阵冻结 `ae247dee82c60b63495e9abea7f703abd84ac3c7`；不合批prefix隔离使用
`22527cf472fc4d6ea9ca7ce22528b7e4ca9cc96a`，把显式捕获界限提高到1200万；Attention诊断
使用 `2c011b8a4f703b5aa50ab4107bb50085e3786c56`。三者的生产包Python文件哈希
均为 `efd8a85972ea6f0d4a8b9a0f41908485da9fb8ea52ee95aa972fe8c957c72922`。
数学仍固定 `4f5b9f20a4030d68ee66ae2d26e34659a5e37234`。

所有路径作业为compact-9 Multiscale-X、n8/hubs4、d128、region_shared，
15,694,224参数，B1/L8、两个长度4的chunk；Timed另外执行flush。
baseline先做一次真实AdamW更新，产生非空私有状态和351个参数的非空slots，step=1。
每个作业内的全部候选随后加载完全相同的参数、state、moments、step和输入。
独立作业各自产生anchor；相同seed和输入hash并不保证warmup后的逐位起点相同。

每个候选依次执行推理、同owner两次VJP、连续真实更新、保存和新进程真实更新。
chunk之间保留完整梯度，一次更新/flush结束后detach。单卡和CPU诊断在anchor后
继续两次，恢复时step从3到4；两卡作业继续四次，恢复时从5到6。351个原始参数
均实际参与更新；放置后的optimizer通过原参数行归属映射，并检查活跃step恰增1。
初始VJP的原参数及实际shard均没有None梯度；本组结果不扩大到其它配置的连接性。

| 检查 | atol | rtol |
|---|---:|---:|
| FP32路径比较 | 1e-6 | 2e-3 |
| BF16路径比较 | 2e-3 | 5e-2 |
| FP32/BF16恢复后真实更新 | 1e-6 | 2e-3 |
| CPU FP64诊断 | 1e-10 | 1e-8 |

scaled error为逐元素 `abs(actual-reference)/(atol+rtol*abs(reference))` 的最大值。
JSON中的largest-errors按scaled error排序，并非整个树的绝对误差排名。
整数时间、dtype、None及保存/下一前向精确性另行检查；所有实际deterministic设置
为false，用户未显式请求开启。BF16沿用FP32参数/梯度/AdamW、BF16投影/KV payload、
FP32稳定子计算和int64时间政策；记录保留实际tensor dtype分布，不由dtype标签推断
未观测的硬件累加实现。

## 单卡路径结果

以下更新编号按JSON从0开始，即anchor后的第一次、第二次更新。表中数字为
相对同作业baseline的数值超差张量数；support和整数时间另列。

| 作业及候选 | update0 → update1数值超差 | update1 support差异张量 | 整数时间差异 |
|---|---:|---:|---:|
| Settle FP32 batch4 / batch_reception | 8→205 / 8→204 | 0 / 0 | 0 |
| Settle FP32 batch_prefix / batch_prefix_reception | 7→141 / 7→143 | 0 / 0 | 0 |
| Settle BF16 batch4 / combined / batch_reception | 各11→435 | 各1 | 0 |
| Settle BF16 batch_prefix / batch_prefix_reception | 各11→447 | 各2 | 0 |
| Timed FP32 batch4 / batch_prefix_reception | 50→169 / 53→241 | 0 / 0 | 0 |
| Timed BF16 batch4 / batch_reception | 各12→613 | 各5 | 0 |
| Timed BF16 batch_prefix / batch_prefix_reception | 各21→667 | 各13 | 各8 |

Settle BF16的单独 `recompute` 在全部检查内通过；其推理与baseline精确。
`combined` 只合批加区域重算，未像以前的混合候选额外改变local Attention backend。
`batch_reception`、`batch_prefix`、`batch_prefix_reception` 的精确定义在JSON的
resolved_candidates；相同超差计数本身不能证明两候选张量逐位相同。

Settle FP32 baseline的 `finite_deviation` 只来自推理/含图前向5个状态张量超差，
最高scaled error为4.2391：segment1 memory20 K.keys的maxabs为5.0911e-6。
另外四项是memory22 K.features、28 K.values、29 S.conv.1、30 K.values。
该比较无support、整数时间、dtype或非有限值差异。其同owner重复VJP只有阈内
embedding梯度差异，最大绝对差3.7253e-9；恢复后的实际更新也在原严格阈值内。
前向模式差异与反向重复差异是两个观测，不能相互替代解释。

Timed FP32 baseline、两个BF16 baseline均在阈值内；这两个BF16模式前向精确。
本节全部20个单卡候选自身的保存数据/下一前向精确，恢复后实际更新在严格阈值内。
各优化候选相对baseline的连续轨迹超差仍然存在，不因自己的恢复成功而撤销。

## Timed BF16整数时间失败与不合批prefix隔离

原 `training-paths-timed-bf16` 的 `batch_prefix` 和
`batch_prefix_reception` 在初始推理/VJP及update0只有数值超差；update1首次出现
8个整数metadata项，全部为memory8/10 K.times，覆盖segments0/1/2及最终state。
其首个返回树时间见证在segment0 memory10：flat533为89对88；flat537为3对94，
同位置weight为0.9092793对0。它涉及活跃支持，不能归为仅无效槽的整数差别。
存储的key-time/support见证本身不证明非法可见性；后者还需对应query时间和窗口。
原作业保留 `structural_failed/exit2`。

同作业batch4此前也已在update1出现5个support及613个数值超差，但没有整数时间
差异。加入prefix的两候选有13个support、667个数值超差，最大scaled error为
2284.15，对应pending值maxabs4.5683。实际工作也改变：首段候选2037对2035、
Full1440对1436、positive_messages8709对8661。work单列保存；comparison中的
空 `work_differences` 不是工作量相同的证据。

随后 `training-paths-timed-prefix-isolated` 仅比较baseline与
`ssd_summary_backend=prefix`，保持region_batch_size=1。初始推理/VJP有15个数值
超差，首个返回树浮点字段为memory1 S.short；update0有18个，update1有612个及
7个support差异，没有整数时间、None、dtype或非有限值失败。首个返回树support
见证是segment0 pending5 strengths，flat2为0对0.0216075；flush values的maxabs
达6.6834、scaled error3341.70。自己的恢复检查仍在严格阈值内。

这说明不合批prefix在本次有限训练轨迹中也能产生后续support分歧，不能把原组合
的全部差异都归于合批。两作业的输入hash与生产包hash相同，但独立warmup后的
fixed_start hash不同；新结果不是原失败轨迹的逐位三方消融，也不清除原整数时间失败。

主矩阵每路径200万元素的内部Trace均触顶；隔离作业显式提高到1200万仍触顶。
这些返回树首差按字段遍历排序，不是首个因果算子。现有证据没有建立prefix内部
最先数值差异到后续support/时间变化的完整因果链。CPU FP64 Settle的baseline、
不合批prefix、reception三例全部在阈值内，恢复更新精确；这是有限精度诊断，
不能替代Timed BF16失败的解释或资格。

## 两卡：首次梯度差异、重复和恢复

两个作业均为同一Settle负载、region_batch_size=1，对比单卡baseline、两卡
`placement_serial`、两卡 `placement_per_device`。实际transport由auto选择
`host_stage`，参数在两个设备上的数量为7,647,728和8,046,496；没有模型副本。
serial由caller逐设备执行；per_device配置最多两worker，反向仍由普通autograd
engine执行。本记录不把串行挂钩重放当作实测并发调度。

FP32和BF16的放置推理均与同作业baseline精确。首次返回树差异出现在梯度
`describe.encode.bias`，并非初始forward/state；以下VJP及重复的最大scaled error
均在各精度一般容差内。重复发生在同owner、无参数/状态/slot更新的同一点。

| 精度/调度 | VJP1对baseline最大scaled | 同owner重复最大scaled | 四次更新数值超差张量数 | 恢复更新超差数 |
|---|---:|---:|---|---:|
| FP32 serial | 0.03489 | 0.02760 | 4、168、192、169 | 1 |
| FP32 per_device | 0.03491 | 0.03007 | 4、176、207、181 | 0 |
| BF16 serial | 0.71710 | 0.67235 | 0、455、457、434 | 236 |
| BF16 per_device | 0.90861 | 0.66731 | 0、442、446、475 | 237 |

FP32放置路径无support或整数时间差异；两卡作业中的FP32 baseline以及放置路径
保留3个模式前向状态超差，属于该作业自己的起点，不能与另一个单卡作业的5项
混为一次逐位重复。BF16模式前向均精确；serial在update1有2个support差异张量，
per_device在update1有2个、update3有1个，但没有整数时间差异。

恢复均为真实新进程，保存数据和下一前向精确，351个活跃原参数step从5增至6。
三项保留的严格恢复超差为：

- FP32 serial：仅 `regional.0.second.bias` 一个参数张量超差，maxabs1.0602e-5、
  scaled error3.7427。per_device恢复在阈内，最大scaled error0.9038。
- BF16 serial：129个参数梯度、1个输入梯度、84个optimizer项、22个参数张量，
  共236项超差；最高scaled error714.80，来自inputs.D.weight梯度，maxabs1.2512e-3。
- BF16 per_device：128个参数梯度、1个输入梯度、86个optimizer项、22个参数张量，
  共237项超差；最高scaled error488.28，来自inputs.H.weight梯度，maxabs9.7656e-4。

同owner BF16重复已经有约1e-3量级的梯度差异；其一般BF16检查阈内与恢复的严格
FP32级阈值超差并不矛盾。数据/下一前向精确及已有重复差异支持区分checkpoint
机制与反向数值可变性，但本记录没有逐算子定位全部梯度、更新或support分歧。
serial两卡也有重复差异，不能将所有现象简单等同于per_device线程并发。

## 同输入Attention表示的定向见证

`training-attention-mode-witness`直接读取原Settle FP32的anchor.pt、inputs[1]和
原记录，校验参数/state/输入/config/生产包身份，保持原chunk4及deterministic=false。
没有重新生成warmup或执行backward/AdamW。94次推理Attention调用只留下有界内存
指纹；含图前向在第4次调用（index3、chunk0、region站点）首次出现数值差异。
此前3次的输入和输出指纹精确；这不是“region3”，也不是整个模型的首个因果算子证明。

该调用的q/k/v均为FP32 `[1,4,2,8,32]`，prior为FP32 `[1,4,8]`，mask省略。
两模式的输入数值/hash完全相同，prior.requires_grad从false变为true：

| 实际调用 | 推理 | 含图前向 |
|---|---|---|
| native API | npu_fusion_attention_v2 | npu_fusion_attention_v2 |
| prior表示 | 显式log-prior PSE | rank-one Q/K扩展 |
| native QKV形状 | `[4,2,8,32]` | `[4,2,8,48]` |
| PSE | FP32 `[4,2,1,8]`，type0 | 无PSE，type1 |
| scale | 0.1767766953 | 0.1767766953 |

原推理prefix回放精确；两个同输入局部primitive回放也分别精确复现各自捕获结果。
全调用输出差maxabs1.1921e-7、scaled error0.009095，在原FP32阈值内。
独立标量CPU FP64对8个完整查询行（35、0至6）的全部8个key及32个输出通道求
加权指数商，没有调用生产Attention、matmul、softmax或autograd：推理相对oracle
maxabs1.2039e-7/最大scaled0.004062，含图前向1.7036e-7/0.012691，均无超差坐标。

此见证支持所选调用在两种先验表示下存在有限算术差异，且其局部误差阈内。
它不说明所有后续5个状态超差均由这一次调用产生，也没有观测硬件内部累加顺序
或验证该调用VJP。诊断最多3次模型前向尝试、2次primitive回放，未改变生产策略。

## 执行身份及复现

所有任务使用冻结归档、`background.slice`、单ATen线程、固定CPU affinity、
有限运行时间和采样进程树RSS界限。NPU经本机协作队列，原身份为
`exclusive_cooperative_queue; external jobs not controlled`；共享主机及队外活动
没有被排除。这里的总时间和RSS是侵入式诊断/进程树观察，不是模型延迟或设备峰值。

| artifacts目录后缀 | 物理NPU | affinity | 服务总秒数 | 进程树RSS峰值GiB | exit |
|---|---|---|---:|---:|---:|
| training-paths-settle-fp32 | 1 | 56–59 | 336.330 | 8.617 | 0 |
| training-paths-settle-bf16 | 2 | 52–55 | 337.994 | 9.128 | 0 |
| training-paths-timed-fp32 | 4 | 64–67 | 306.725 | 10.077 | 0 |
| training-paths-timed-bf16 | 5 | 60–63 | 407.231 | 9.798 | 2 |
| training-paths-placed-fp32 | 0、14 | 68–71 | 419.018 | 10.310 | 0 |
| training-paths-placed-bf16 | 4、7 | 72–75 | 492.717 | 11.053 | 0 |
| training-paths-prefix-fp64 | CPU | 88–91 | 135.266 | 9.845 | 0 |
| training-paths-timed-prefix-isolated | 1 | 76–79 | 148.088 | 9.766 | 0 |
| training-attention-mode-witness | 5 | 116–119 | 20.044 | 2.432 | 0 |

路径作业限制为1800–3600秒、48–80GiB采样RSS；Attention为900秒/40GiB。
具体界限、协作队列终态、完整命令、环境记录及其hash在JSON。harness将有限数值
偏差以exit0结束，因此服务 `passed` 不替代内部数值结论。

原始 `source_dirty=true` 均保留：归档无独立.git，内部git status向上发现主工作树
状态，并未限定为归档自身。另行只读核验每份归档的全部Git tracked blob：八路径归档各347个、
Attention归档349个，全部与各自声明commit逐字节一致；没有缺失或差异文件。
恢复身份检查覆盖实际Tide Python包与入口harness；其字段明确排除辅助harness模块
及工具链身份，不能将它扩大成全部运行环境的逐位认证。

复现采用[训练路径入口](../training-path-checks.md)和[执行契约](../execution-flows.md)，
先冻结对应commit，再使用JSON保存的完整命令、配置及新输出目录。两个关键命令为：

```sh
python -m scripts.check_training_paths \
  --config configs/hb-multiscale-x-timed.json --device npu:0 --precision bf16 \
  --candidates baseline prefix \
  --candidate-overrides '{"prefix":{"ssd_summary_backend":"prefix"}}' \
  --batch 1 --length 8 --chunk 4 --steps 2 --threads 1 \
  --witness-elements 12000000 --output-dir NEW_PREFIX_RESULTS

python -m scripts.diagnose_attention_modes \
  --anchor ORIGINAL_TRAINING_PATHS_SETTLE_FP32/results/anchor.pt \
  --device npu:0 --output-dir NEW_ATTENTION_RESULTS
```

前者各候选共享新作业的真实anchor；后者要求原anchor及相邻result.json的hash身份，
重用原起点。新NPU执行仍须重新准入。原始checkpoint/anchor和诊断payload是复现
依赖，退休前应先审核依赖。CUDA、direct peer-DMA/HCCL及编译训练不由这些作业获得资格。
