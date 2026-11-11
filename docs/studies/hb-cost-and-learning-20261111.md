# HB训练问题与直接实现成本对照

2026-11-11，用户授权在实验仓库实测非TIDE直接实现，并讨论HB40/HB88的学习问题；
详细性能定位/公共执行优化仍由基座负责。本轮不追加正式语言训练，不打开密封test，不接入基座工作树新代码。

## 两个问题

1. **成本**：在同CPU/NPU、B/L/宽度、静态邻接和完整请求/更新口径下，普通Torch直接图的实测成本有多大？
   与实验实际使用的74e7bbd公共执行相距多少？更大B和CPU线程是否改变直接路径的资源选择？
2. **学习**：在相同有效训练token及相同validation块上，HB40/88的学习差异是什么？
   现有曲线、梯度范数和局部候选筛选支持优先定位什么？所有建议保持假说身份，不凭loss下降认证scaling。

固定数学0770593、基座74e7bbd、`.venv-positive-floor-v1`及`.local/native-build-v6`；
参考基座`optimize-tide-execution`的加法/减法比较方法，只读其文档，不导入benchmark或复制执行核心。

## 成本研究表（TL-HB-COST-20261111）

| 项目 | 固定条件 | 改变项/记录 |
| --- | --- | --- |
| 直接MLP图 → 直接Attention图 | FC16/HB40/HB88静态邻接、region4、d128、FFN512、tied4096词表head | 加入H2因果Attention；普通Torch矩阵批，真实参数数目不补齐 |
| 原TIDE | 同形状；原KV/qry/regional_total/quad、温度.25、strength k2 | 已安装native原公共路径、完整返回；记录candidate/Full/消息及实际批 |
| 普通Transformer参照 | 原10M配置，同B/L和head/loss计时 | 不同函数/结构的常规算子参照，不据此推断HB质量 |
| 减法观测 | 上述各候选分别测no_grad、构图前向、完整反向+clip+AdamW | 不把构图前向冒充训练；不把差值全归为调度开销 |
| 线程与batch小扩展 | 主表B4/L32；CPU1与NPU/host2 | CPU8只复核HB40直接Attention与TIDE；直接Attention另测B16/L128，NPU含HB40/HB88，CPU8含HB40 |

每格一次预热后三次重复，每次恢复相同初始权重与空AdamW状态，使用同一合成输入seed991。
计时含准备、公共返回、全位置head/loss、有限性检查、标量返回、张量清理和末端设备同步；
完整步另含backward/clip/AdamW。输入搬运、模型构造、复位、预热、日志/Lens另计。
分母为实际B×L输入位置；不是生成token/s、TTFT或语言学习速度。

**直接图是不同函数的可达成本参照。** 它保留邻接/路径深度及每节点Attention/SwiGLU，使用均匀入边均值、
独立节点权重；没有TIDE质量/发送控制、Read/Describe、带权KV、通用事件调度、独立中间梯度根和公共continuation。
原source共享亦不同。直接图计算全部节点，TIDE只对实际正支持做Full。时间倍数不是严格等价加速，
也不能单独分解必要数学成本与可删除执行开销；进一步同函数HB投影/模块边界对照交基座。
该参照由实验侧独立普通Torch公式实现，不能用作正式TIDE训练核心。

配置：[hb-cost-20261111.json](../../configs/checks/hb-cost-20261111.json)。三个后台suite，
上限90分钟wall、0.75 NPU卡时、6 aggregate CPU核时、96GiB aggregate RSS、2GiB新增产物。
CPU1主表≤2400秒，CPU8子表≤900秒，NPU主表≤2400秒，服务另留有限启动/记录余量。
单卡准入选择低负载且至少32GiB余量的卡，保留两次观测和lifetime锁；当前会话既有共享许可仍有效。
不打断其他任务；超时/容量/非有限错误按原身份失败，不扩大表来追求成功标签。

独立小检查已验证直接节点合批对逐节点/逐时间公式的FP64输出和输入/参数VJP、因果性、tied head及全部参数梯度。
不认证TIDE与直接图同函数等价；原TIDE严格失败和三步恢复边界保留。
执行、严格数值、离散语义、恢复、学习和资源分别记录；本轮无训练效果资格或新恢复声明。

Lens项目`tide-learning-lab`、组`TL-HB-COST-20261111`；run前缀`tl-hb-cost-<suite>-<case>`。
原始JSON/JSONL权威，私有入口沿用`.local/LENS.md`。服务/状态/日志及终态结论随后写入同组证据。

## 学习审阅的预先边界

主比较使用相同32768有效validation token及262144训练token；按block_id及token数核对评价一致。
初始化只有2048验证token，不能直接与32768口径作精确前后差；另取完全相同块的交集报告。
原外层已终态，部分内层running保留原事实；不得在审阅时改写`runs/`。
coverage的d32/L32/65536曝光与scale的d128/L128不能当作纯容量阶梯；单seed/多候选选择偏差保留。
先形成真实表、梯度/曲线证据和可区分解释，再提出有界后继配方；不自动追加训练。

## 有界梯度补证

已有同验证块比较确认HB88从133K到262K继续恶化，且FC32也出现较大恶化/后恢复。
为把后继建议落到可检验机制，在CPU8 suite终态后增加一个CPU1/16GiB/900秒的只读权重诊断，
仍在上述6 CPU核时与96GiB aggregate RSS总预算内：FC16/FC32/HB40/HB88各取原始初始化及262K checkpoint，
同B2/L16已见validation前缀，只做一次前向/VJP，无optimizer更新，无checkpoint改写。
记录输入/hidden/logit RMS、loss、preclip参数梯度、input-feature与hidden方向梯度范数及主要梯度注册名。
共享参数注册名不是某个节点的独占因果贡献；方向增益不是Jacobian谱范数，小前缀不是原L128质量复评。
该补证不作修复或新学习效果声明；Lens组`TL-HB-LEARNING-20261111`，run前缀`tl-hb-grad-`。
