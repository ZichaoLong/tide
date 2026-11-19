# 接收 Attention 周期：FP32 起点与 SSD 候选

本页记录已完成的四个冻结作业；[JSON](reception-start-20261118.json.gz)给出源码、命令、
设备身份、原始结果 SHA、数值阈值与样本。后续整模型选择、持续训练和推理资格另记。
固定数学为 `4f5b9f20`。全部使用独占协作队列准入；外部不协作任务不受锁控制。

## FP32 完整训练起点

冻结 `95465188`，HB compact-9 / n8+h4 / d128 / region_shared / Settle / X，
B8/L256、region_batch4、prefix 摘要、接收局部重算、FP32参数及计算，完整AdamW。
两次预热、五次正常样本，输入搬运、审计和trace不计入正常时间。旧BF16成本不能替代此点。

| 模型/候选 | 中位秒/步 | 输入位置/秒 | allocated峰值GiB | reserved峰值GiB |
|---|---:|---:|---:|---:|
| source-residual | 1.3683 | 1496.8 | 12.376 | 14.344 |
| transformer_A/standard | 0.7596 | 2696.2 | 3.192 | 3.373 |
| transformer_A/chunk256 | 0.5843 | 3504.7 | 3.148 | 3.357 |
| transformer_C/standard | 0.0402 | 50924.3 | 0.868 | 0.982 |
| transformer_C/chunk256 | 0.0370 | 55344.0 | 0.882 | 0.982 |
| incoming_light_64 | 1.8444 | 1110.4 | 12.225 | 14.184 |

source-residual 的 A/C 同时满足约188/9 blocks与约15.694M总参数；B/D构造也在记录中，
本起点诊断只计时A/C，不能冒充完整A–D结果。Transformer chunk256相对standard的改进
说明双方都需要有限算子/分块调节；C standard的range/median约0.215，chunk256约0.005。
不同数学模型不是质量或语义等价对照。light64的参数量与实际活动由其自己的审计记录。

source-residual侵入式分阶段观测：总1.4503秒、前向0.8474、反向0.5790、AdamW0.0133；
前向矩阵SSD扫描0.2522、allocation0.1041、消息gather0.0701、查询packing0.0575秒。
light64的接收前向约0.1459秒，反向中接收重算约0.3478秒（包含Attention0.0747）。
这些数只解释**同步插桩下**的工作，不能按正常1.37/1.84秒拼接成真实占比。
source的正常训练有独立设备trace，128次Cumprod设备事件累计约233ms；host/operator
事件有嵌套和并发，不能与设备时长相加。实际NodeLinear算量、选中/padding量和完整
operator计数保留在JSON及原始文件，不能将参数去重误当成矩阵计算去重。

## SSD 同函数微测

冻结 `57eb12c3`。8节点、B8、H4、V/K32；FP32，HF32关闭。B/C由一个group广播到heads。
三次预热、九次采样；backward列包含五个输入根的VJP，没有AdamW或模型其余计算。

| L | 候选 | inference ms | 全VJP ms |
|---|---|---:|---:|
| 256 | block | 32.316 | 85.490 |
| 256 | incremental | 32.410 | 85.274 |
| 256 | doubling | 5.026 | 17.385 |
| 1 | block | 0.600 | 1.554 |
| 1 | incremental | 0.188 | 0.775 |
| 1 | doubling | 0.178 | 0.761 |

两种长度的各候选，均通过同fixture与block以及有界独立CPU FP64逐步递推的原阈值。
包含零衰减/不写入、初始矩阵与全部五根梯度。L256的doubling减少Cumprod，但保留更多
中间乘积：本微测含图前向allocated峰值由约118.5MiB到143.5MiB；这不是所有形状下的
收益保证。L1直接递推省掉pairwise传播/索引，incremental在更长L上保持block原路径。
候选继续显式可选；尚不能从局部5倍收益推断整模型加速比例或改变训练默认。

复现使用JSON内各run.command与原始launch.json，在新的输出目录按执行契约重新准入；
原始结果、trace和失败记录都保留，阅读本页不需要重新运行实验。
