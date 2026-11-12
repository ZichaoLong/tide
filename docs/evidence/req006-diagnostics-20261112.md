# TL-REQ-006：四代表完整更新与设备轨迹（2026-11-12）

本轮完成四个固定下游直接实现的无插桩更新，并保留四份设备 trace。
已明确定位到一个此前设备总利用率无法说明的热点：**HB88 的704个 AICPU kernel 全部来自
浮点 clamp_min 反向的 scalar 比较/转换**，不是整数逻辑时间计算或 host CPU fallback。
HB184 与入口树还存在明显的分配求解工作；本报告不把全部差距归于同一个原因。

详细材料、实际工作、配置分组、trace 摘要及三条完整 flow 见
[小型复核附件](req006-diagnostics-20261112.json)。
通用复现入口见[REQ-006 诊断契约](../req006-diagnostics.md)；请求、上游交付和下游验收分别记账。
本轮没有修改下游直接实现，也没有把它作为当前基座默认执行器的性能。

## 来源和材料核验

| 内容 | 固定身份 |
| --- | --- |
| 请求 | `dc943cc7ce773e3dac4d62736ab524953573decb` 的 TL-REQ-006 |
| 原配置/报告 | `8995958728167eee7a4bcc2139f5e1794cbf82ab` |
| 实际下游代码 | `6e7bd49eb5e35c15445826cac3a9b6ddda75aff4` |
| 本轮诊断 harness | `c83a2820149e69c3a0571dc61765836173bfaea1`，干净源码 |
| 离线 trace 分析 | `2927ba401c71dc8abd44da2a6518a7f48984d41c`，干净源码 |
| 数学 | `0770593838a37c790a568226930567ed60f99b83` |

四份配置从仓库[checksummed gzip](../../examples/diagnostics/req006/manifest.json)恢复，
与每个实际 result 的原配置相同，未覆盖 batch、length 或其他配置。
8个执行目录各27份下游 Python 文件均按记录重算 hash，一致；从固定 Git 对象提取，
未读取活动训练源码/权重。配置和每文件 hash 保留于附件，不依赖原学习运行目录。

全部输入为合成 token，shape `[64,256]`、seed17；各模型构造seed11，进度固定0，
有效目标数均 **16,384**，无 ignore 目标。输入 hash 均为
`9f839d90b9c74d984de34fa25021e612db8f93f4faf698b4cd51453eb6274167`。
每进程先做一次真实 synthetic prime update，再冻结权重、AdamW槽和 CPU/NPU RNG；
一次 warmup 后重置，完整初态 hash 检查均通过，随后一次正式更新。
这不是恢复历史 checkpoint，也没有重放原学习轨迹。

**四组 measure/profile 的 prime 后权重和完整初态 hash 均不同**，虽然配置、输入、seed相同。
每进程内部的重置成立，跨进程严格同权重配对并未成立；没有保留这些权重数值来判断差异大小，
原因仍未判定，不能只按 hash 将其称为舍入。故 profile 是同配置机制证据，
不按其阶段比例分摊 measure 秒数，不给两者严格同材料加速结论。各自 hash 原样列于附件。

实际运行是 Python/PyTorch 的下游 `DirectSemanticLM` 或 `SmallTransformer`，
不是 native/standalone/mixed-a/b/c/resident 的公共图执行器。张量在单张 NPU，
Python 控制静态区域/层顺序。设备 `Ascend910_9392`；Python3.12.13、Torch2.12.0+cpu、
TorchNPU2.12.0.post3、CANN9.2.0，FP32、matmul HF32=false，ATen intra-op2/inter-op1。
HB88/HB184 使用物理7、CPU160–175；入口树/TF使用物理5、CPU176–191；均映射 `npu:0`。
未设置 launch blocking。两组与其他设备作业并行，CPU池分开；未排除共享主机及外部争用。

## 配置和完整更新成本

| 代表 | 节点/区域、维度及参数 | 实际局部配置 | micro / global；累积次数 |
| --- | --- | --- | --- |
| HB88 pooled | 88节点/22区域、每区4；224边；d128/FFN512；24,876,408参数 | 全KV qry、Read+、local cat/region ind；pooled_attention、base=mu、fixed_gain2、温度1 | 16/64；4 |
| HB184 quad | 184节点/46区域、每区4；480边；d128/FFN512；53,021,958参数 | 全KV qry、Read+、cat/ind；regional_total/context/cat/quad、k2、温度1 | 8/64；8 |
| R64入口树 | 主体640节点/10区域、每区64；21边界节点/6区域；1,620边；d64/FFN256；51,621,168参数 | KV qry、cat/ind、regional_total/fixed/quad；进度0时主体实际k32/温度1；4个重边界区域k1/温度.5，另2区域共5节点走light路径 | 4/64；16 |
| Transformer | d512×16层、FFN2048、4 heads；52,540,416参数 | 显式 QKᵀ→softmax→AV，GELU FFN；不是之前FC对照的融合SDPA候选 | 64/64；1 |

Tide三个代表均每个KV头H1，summary/adapter/source hidden均32，无RoPE/距离斜率；
source_binding=region、聚合使用matmul。HB88/HB184/树的时间stride分别9/11/10，
窗口分别有4608/5632/5120和window0两组，按原逻辑时间定义。
入口树名字的h2不是Attention头数；配置最终k2/温度.5在进度0尚未生效，
不能把该新诊断写成目标低活动阶段。其body退火从2,097,152到8,388,608 token。

全部L256，loss为有效目标归一化cross entropy。AdamW foreach=false、betas(.9,.95)、
eps1e-8、matrix-only weight decay.1、clip1；本次有效lr Tide为1.5625e-5，TF为4.6875e-6。
全局批64一次optimizer，不能把多个microbatch当成多个独立更新。

| 代表 | 完整步秒 | 完整步有效目标/s | 原 update 子区间秒 | update 有效目标/s | allocator峰值GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| HB88 pooled | 2.612413 | 6,271.596 | 2.606859 | 6,284.957 | 14.917 |
| HB184 quad | 10.397292 | 1,575.795 | 10.384445 | 1,577.744 | 16.776 |
| R64入口树 | 16.036165 | 1,021.691 | 16.030417 | 1,022.057 | 22.140 |
| Transformer | 0.145799 | 112,373.692 | 0.144427 | 113,441.661 | 10.681 |

吞吐分子均为16,384。完整步包括分批/上传、目标计数、模型内置观测、前后向、clip、AdamW、
结果释放及终端同步；update是原 `campaign_runtime.logical_update` 的内部计时，
不含上传、计时前目标计数/进度设置。输入构造、prime、重置、warmup、hash、profile导出、
评价、checkpoint和日志I/O均不在两区间内。allocator峰值在恢复后清零，包含活跃模型；
CPU RSS仍是含构造/warmup的进程高水位。没有TTFT、生成吞吐或完整学习作业吞吐声明。

TF是不同函数、不同参数形状、不同物理批；其优势是有用的设备效率对照，不能当等价加速比。
本轮也不是原报告67M token轨迹中pooled HB184的重测。

| 代表 | 实际候选 | 正支持Full | 正支持/候选 | dense节点行数 | light行数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| HB88 | 1,441,792 | 1,441,792 | 100% | 1,441,792 | 0 |
| HB184 | 2,966,892 | 2,403,491 | 81.010% | 3,014,656 | 0 |
| R64树 | 7,133,959 | 5,649,472 | 79.191% | 10,829,824 | 81,920 |

下游的 `work/physical_full_rows` 计数器实际将全部node slots计入，树包含5个light节点；
按源码扣除light，真正经过重Memory/FFN的dense行数为 **10,747,904**，其余81,920行只做light聚合。
全部主体行仍dense计算后mask，逻辑稀疏不等于省去对应Full计算。树主体正支持/全部主体槽为51.724%。
三个代表的positive-floor事件分别33,783,993／79,582,024／107,208,749，涉及Attention坐标，
不能与节点数混为一类计数。该下游采用声明的normal-tiny floor及原边界VJP，
不冒充基座其他正支持策略或严格实数求值。正支持错误计数为0；分配残差最大分别
0／3.5763e-7／5.7221e-6，低于这份冻结代码的2e-5守卫，不能代替独立正确性。

## 设备轨迹及可定位边界

四个profile原作业均保持 **failed**：CANN的字符串时间戳触发
`TypeError("bad operand type for unary -: 'str'")`，发生在执行和导出trace之后。
固定2927的离线恢复全部成功；重算四个原trace的SHA-256与恢复记录一致，未重跑设备profile。
原 `iterations=[]`、缺失最终模型/optimizer hash仍保留；不能从离线解析反填原步骤资格。

| profile代表 | kernel总数 | AIVEC | MIX_AIV | AICORE | AICPU | MIX_AIC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HB88 | 74,999 | 62,354 | 5,957 | 5,728 | 704 | 256 |
| HB184 | 544,339 | 471,818 | 42,345 | 27,728 | 2,208 | 240 |
| R64树 | 360,926 | 321,505 | 22,973 | 14,624 | 1,344 | 480 |
| TF | 2,786 | 2,198 | 297 | 291 | 0 | 0 |

这是实际设备Task Type计数，不是各硬件利用率或耗时比例；名称中的AiCore也不能覆盖Task Type。
例如 `aclnnGeScalar_GreaterEqual_GreaterEqual` 同时有AIVEC和AICPU实例。
未观察到具名host fallback事件，不等于排除所有未标记的主机工作。

| profile代表 | CastAiCpu次数/累计秒 | GeScalar GreaterEqual累计秒 | matmul类累计秒 | 设备kernel区间并集秒 |
| --- | ---: | ---: | ---: | ---: |
| HB88 | 352 / 0.732225 | 0.679284 | 0.141788 | 2.361995 |
| HB184 | 1,104 / 1.894082 | 1.497851 | 0.432382 | 7.494613 |
| R64树 | 672 / 4.511097 | 4.458587 | 0.505050 | 15.344598 |
| TF | 0 / 0 | 0 | 0.070799 | 0.128050 |

matmul类按实际MatMul/BatchMatMul计算kernel名称求和，不计同名API内部的所有cast/slice为矩阵计算。
kernel累计和区间并集都不是独占关键路径归因，不与不同进程的无插桩wall-time相除。

HB88的 `async_npu / torch_to_npu` flow 将全部352个Cast与来源的 `aten::ge` 对齐，
均位于 `ClampMinBackward0`；再以Sequence number回到前向 `aten::clamp_min`：

| 原前向范围／输入shape | Cast次数 | Cast累计秒 | 对应AICPU GreaterEqual次数／累计秒 |
| --- | ---: | ---: | ---: |
| KV `[4,16,1,256,256]` | 88 | 0.236820 | 88 / 0.215789 |
| KV `[4,16,256,256]` | 176 | 0.457523 | 176 / 0.435932 |
| region `[16,256,4]` | 88 | 0.037882 | 88 / 0.013256 |

全部输入为float Tensor加Scalar；合计704个AICPU核，覆盖该trace的全部AICPU实例。
而前向int64 `[256,256]` 的440次 `>=` 是AIVEC，累计仅0.005637秒。
附件中的connection_id209087、209526、692699及forward序号49596、49557、61575保存完整追溯。
这三条前向相邻算子为logsumexp→sub→floor观测→clamp_min→exp，
与固定下游 `direct_semantic_ops.positive_exp(logp)` 对应；KV的两次qry及一次Full、
pooled区域各调用该权重归一化路径。它解释了**已关联的热点属于哪项操作**，
尚未解释厂商为何选AICPU、也未验证哪种替代实现保持值/边界VJP并更快。
HB184/树的相同核名还没有逐实例完成该flow关联，不将HB88结论自动扩为全部根因。

矩阵已经在节点×B×L上合批，并非单token小投影：

| 代表 | 实际前向矩阵形状例 |
| --- | --- |
| HB88 NodeLinear | `[4,4096,128] @ [4,128,32]` 440次；到FFN512的bmm176次 |
| HB184 NodeLinear | `[4,2048,128] @ [4,128,32]` 1,840次；到FFN512的bmm736次 |
| R64主体 NodeLinear | `[64,1024,64] @ [64,64,32]` 800次；到FFN256的bmm320次；另有宽4边界块 |
| TF线性层 | `[16384,512] @ [512,1536/512/2048]`，各16次；FFN下投影 `[16384,2048] @ [2048,512]` 16次 |

节点/区域数、三次KV注意力、较窄投影、microbatch累积及源码中的select/stack仍增加工作。
这与本轮发现的标量反向热点同时存在，不能只看总参数量或笼统称“全是碎片化”。

CPU接口事件中的 index/assembly 次数为4,913／20,881／95,265／81，
scalar/sync为4,935／41,767／23,471／890，autograd engine执行次数为
22,652／105,640／72,240／735（顺序HB88/HB184/树/TF）。这些不是唯一计算图节点数。
**四代表未运行census**；私有图节点/保存storage总量仍未测。
`--action census` 已有接口，小型开发记录不替代本规模结果。
分配自定义Function的forward在no-grad中求解，不因外图只有一个节点就把28轮求解当成一次kernel。

带标签的主机inclusive范围：HB184 allocation5.071秒、region5.870秒、KV2.694秒、
backward5.989秒；树allocation3.137秒、region3.466秒、KV2.712秒、backward4.633秒。
allocation嵌在region，KV嵌在memory/model，worker执行还可能落在unattributed，
不能相加成完整步或设备耗时。HB88 pooled没有allocation调用而仍有同类AICPU热点，
因此“分配求根是所有低AICore现象的原因”已不成立。

## 后继判别与复现

本轮优先得到可局部验证的候选：保持positive-floor值与相等边界VJP，
比较float scalar和同dtype零维Tensor的clamp_min，并分别观察真实反向kernel与完整算子成本。
它只验证一个算子边界，不预先承诺消除所有AICPU或改善整模型。
随后再考虑分配求解的批量精确方案、节点/区域交接与观测同步；保留负结果，
不将必要数学、厂商实现选择和基座API成本混为同一项。

后继独立[scalar-floor诊断](../../scripts/diagnose_scalar_floor.py)与
[公式/VJP检查](../../tests/test_scalar_floor_diagnostic.py)支持zero、tiny和log-tiny，
FP32/FP64、相邻可表示数、相等边界、signed zero记录及非有限分类。
0D常量在算子区间外按dtype/device构造并复用；不修改下游或生产函数，不使用global monkeypatch。
后继[常量表示与初态配对证据](req006-floor-20261112.md)已分别记录算子结果和小型CPU材料复用；
不改写本报告的原失败或跨进程不配对结论。该工具的设备效果不属于上表结果。例：

```sh
python scripts/diagnose_scalar_floor.py --device npu:0 --dtype float32 --bound log-tiny --variant scalar --shape 4,16,256,256 --mode backward --action measure --threads 2 --output-dir artifacts/floor-scalar-new
python scripts/diagnose_scalar_floor.py --device npu:0 --dtype float32 --bound log-tiny --variant tensor --shape 4,16,256,256 --mode backward --action profile --threads 2 --output-dir artifacts/floor-tensor-profile-new
```

对两variant分别使用measure/profile及新目录；小边界公式检查在计时外，profile不报告吞吐。
其结果是前向或前向加输入VJP的算子成本，不含optimizer，不宣称模型收益或公共checkpoint逐位等价。

从固定2927或含该解析修复的后继检出开始，按[开发入口](../development.md)准备匹配环境。
调用者提供包含6e7bd49e的下游Git仓库；工具只读取该对象，不读取其活动工作树。
在固定源码、有界background.slice及已分配空闲NPU的作业内执行：

```sh
python scripts/diagnose_req006.py --downstream-repo /path/to/tide-learning-lab --case hb88-pool-s11 --progress-tokens 0 --prime-updates 1 --device npu:0 --threads 2 --mode training --action measure --output-dir artifacts/req006-new-measure
python scripts/diagnose_req006.py --downstream-repo /path/to/tide-learning-lab --case hb88-pool-s11 --progress-tokens 0 --prime-updates 1 --device npu:0 --threads 2 --mode training --action profile --output-dir artifacts/req006-new-profile
```

其余case为 `hb184-quad-s11`、`tree-r64-h2-q3-s11`、`tf-hb184-s11`，各用新目录/独立进程。
需要图统计时另用 `--action census`，不把插桩吞吐写进主表。
离线处理自己的新trace：
`python scripts/req006_profile.py NEW_TRACE.json --output NEW_SUMMARY.json`。
默认每个测量一次warmup、一次实测、无microbatch覆盖；本次均显式prime1。
进度0、合成输入和跨进程prime权重不完全一致的边界必须保留。

四measure为passed；四profile原failed不改写。离线恢复服务
`gef-rewrite-trace-recovery` 从03:33:55至03:36:13 UTC完成，记录passed。
原始记录位于ignored `artifacts/rewrite-20261112/req006-{case}-{measure,profile}/`；
报告和附件保留结果、配置身份及trace hash，复现不要求找回这些目录。
没有新增独立数值/离散等价、应用恢复、学习效果、下游验收或公共执行器完整资格。
