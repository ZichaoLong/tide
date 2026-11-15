# 常规Transformer-MoE补充对照：已授权执行表

2026-11-15用户已批准执行本表。实现、检查、有限校准与有条件正式训练由独立持久控制器推进。
当前v10/修复保持原源码、表和截止。新增补充批独立固定来源、表和48h墙钟，不改旧20项表。

## 问题与边界

主问题：共同自然文本曝光及近似每token活跃参数下，当前直接HB相对标准MoE是否有质量/容量收益？
同时看MoE保持top2和专家尺寸、E20→E64增加总容量是否有收益，以及小档→大档的变化。
“纯MoE”在此指常规decoder-only Transformer的FFN均替换为独立top2专家，不含TIDE图、记忆或区域模块。
attention仍为普通因果稠密attention，专家没有独立KV历史，不先加入共享专家或MoE attention。
它是有力的条件容量基线；层数、全局attention和通信结构与HB不同，不是固定深度/拓扑下的单因素证明。

## 有限结构表

保持已校准Transformer的pre-LN、4头、位置编码/缩放、绑定embedding/head、NPU融合attention。
专家用GELU两矩阵FFN，router为无bias线性层；每token/layer选择2个专家，选中权重重归一化，FP32路由。
下表的解析参数清点已由实际构造/训练核对；[公式与逐项计数](../evidence/moe-controls-proposal-inventory.json)，
已完成批次的[结果与边界](moe-controls-results.md)。
参数M=10^6；曝光Mi=2^20有效训练目标。

|档位|宽度×层数|每专家FFN宽|每层专家E|top-k|k/E|总参数|每token活跃参数|活跃参数/总参数|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|small|192×6|384|20|2|10.000%|19.472M|3.484M|17.89%|
|small|192×6|384|64|2|3.125%|58.603M|3.535M|6.03%|
|large|256×12|480|20|2|10.000%|63.444M|10.201M|16.08%|
|large|256×12|480|64|2|3.125%|193.729M|10.336M|5.34%|

小档对应现有HB约3.7M活跃参数和Transformer3.460M；大档对应HB约9.4–10.2M及Transformer10.530M。
近似匹配，不宣称严格等FLOPs/访存/耗时；报告偏差，匹配不能只按FFN专家计算。
大档若沿用h512，E64会约206.34M，超200M，因此统一h480；E20/E64间保持专家尺寸不变。
小档配现有d192/L6/FFN768稠密骨干；大档新增d256/L12/FFN960稠密骨干（约10.136M），
以匹配top2专家FFN总宽度，现有FFN1024/10.530M也作为历史较强参照保留。

MoE的k/E=10%或3.125%仅指专家选择率。共享attention、embedding/head与全部router参数仍参与，
故表内全部活跃参数比例约16–18%/5–6%；不能与HB的Full/全图直接画等号。
所有活跃参数按同一token的计算路径计数；不把全batch专家并集、非零梯度元素数或FLOPs当作同一量。

## 配方校准与选择

所有候选从同一语料/词表/划分开始；L256/global64，物理microbatch由短实测决定，不机械沿用小smoke。
AdamW beta=(0.9,0.95)、eps1e-8、matrix-only decay0.1、clip1；初始沿用较好的GPT2+按深度残差输出缩放。
各专家独立初始化，不复制完全相同的专家。主任务NLL与辅助loss分别记录。

- 小档LR={3e-4,1e-3,3e-3}；大档LR={1e-4,3e-4,1e-3}。
- 每个MoE形状配balance系数{1e-3,1e-2}，router z-loss系数先固定1e-3：4形状×6=24个具名配方。
- 两个匹配稠密骨干各检查相同档位3个LR，最多6个控制先导；不要求旧TF自动沿用旧最佳LR。
- 共最多30个先导，调参seed33；每项先8Mi，再每形状/稠密骨干前2名继续到32Mi（最多12段），
  每种规模/专家数都保留晋级名额，不因E64早期loss高而取消该结构。
- 选择只用既定validation与实际路由/稳定性记录；不使用confirmation。负载不均是要报告的学习现象，
  不能通过丢token或隐藏冷专家把配方标成健康。若全部配方有问题，先报告和定位，不无界扩大网格。
- 最佳配方固定后，4个MoE形状各seed11/22从零128Mi，最多8条MoE主轨迹；
  两个匹配稠密骨干各双seed，最多4条主控制。与已有配置完全一致的稠密结果可按身份复用，最多12条正式轨迹。
  筛选权重不继承到正式seed，8/32/128Mi保留同口径验证；日程从起点声明256Mi cosine衰减，与v10前缀一致。
  主终点128Mi，无自动256Mi。全部合格终点评价现有262K confirmation，test密封。

建议辅助项：p为所有专家的softmax概率，f_i为分配到专家i的次数/(k×有效token数)，
L_balance=E×sum_i(f_i×mean(p_i))，分配计数stop-gradient；L_z=mean(logsumexp(router_logits)^2)。
辅助统计排除padding，定义跨梯度累积的归约口径，避免改变microbatch就改变平衡目标。
报告的质量只用next-token NLL，不把NLL+辅助项与旧NLL比较。

## 实现、准入与资源建议

直接实验模块实现，与现有Transformer共用attention及训练设施。优先按专家打包和batch/grouped GEMM，
避免每token或逐专家小算子的Python串行路径。dropless：所有选中分配都处理，不用capacity factor截掉超载token。
允许实测更快的稠密遮罩/分块参考，逻辑活动、执行的专家行数和吞吐分列；分块反向保存/重算也计峰值。
先测完整update的micro16/32/64（资源允许时再更大），保持global64；更大global batch属于另外的配方变更，不混用。

检查路由权重/支持、空专家、重复token合并、因果性、主loss及aux梯度、共享参数、optimizer、同栈新进程恢复；
用小型独立参考保留原严格阈值，有限差异与学习分列。非有限、丢token、非法路由和恢复错误阻塞对应路径。
训练记录各层专家负载、冷专家、选择熵、balance/z-loss、主NLL、激活参数、物理专家行数、吞吐和显存。

建议独立补充批48h墙钟，无累计NPU/CPU算时上限；最多8个新增单卡worker，与现有作业共用全局12slot并保持原作业固定。
新增盘目标64GiB/停止线96GiB，账号保持>=200GiB硬余量；先做完整步显存/吞吐校准再冻结预算和提交表。
200M参数上限不代表显存已验收；特别核对专家激活、dispatch/padding和反向保存峰值。没有实测前不承诺具体完成小时数。
Lens项目tide-learning-lab，建议组TL-MOE-CAL／TL-MOE-CONTROL／TL-MOE-CONFIRM；统一配置/seed/运行ID与研究图关联。
只把最终审阅过的同曝光结果用于科学支持，先导和执行完成不自动升级成MoE或HB优势。
