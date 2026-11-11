# FC/HB直接图与固定TIDE安装：完整请求成本

2026-11-11，TL-HB-COST-20261111。25个成本case、八个只读权重VJP均已完成；四个服务无存活后代，33个Lens投影已核对项目、身份、事件数与读回。
**普通Torch的d128直接图在NPU上明显快于CPU；实验现用74e7bbd的TIDE路径却更慢。**
宽度128不是这类图计算不能受益于NPU的充分解释。
两者函数与职责不同，以下倍数不能解读成纯基座开销、已实现的等价加速或TIDE质量结论。

[研究表](hb-cost-and-learning-20261111.md) · [固定配置](../../configs/checks/hb-cost-20261111.json) ·
[完整测量/运行/Lens映射](../evidence/hb-cost-review-20261111.json) ·
[另行审阅的HB学习结果](hb-learning-results-20261111.md)

## 共同条件与直接实现

主表B4/L32，4096词表、全位置head/交叉熵、FP32 dtype、合成输入seed991；
TIDE与直接图width128、FFN512、H2、每区域4节点，同导出的静态邻接/输入输出端口及路径深度。
CPU主表intra/inter-op=1/1，NPU physical1→logical0、host2/interop1；不是独占设备计时。
NPU两次准入观测health OK、AICore0%、HBM2877MiB，仍保留用户已授权的共享策略和lifetime锁。
本轮没有profile插桩或重型构建；CPU suites独立并行，外部CPU/主存争用未排除。

[`direct_graph_cost.py`](../../tide_lab/direct_graph_cost.py)是独立普通Torch公式：
入边均匀平均、每节点pre-RMS因果Attention和SwiGLU、节点独立参数、tied embedding/head。
区域投影实际`bmm`形状为[4, B×L, d]乘[4, d, out]，主表每节点128行，扩展表2048行；
节点参数批4不等于独立样本B。因果attention保留整个窗口，未测逐token KV cache。
MLP锚点保留相同静态图/FFN并去掉attention，用于加法比较；三个执行模式用于减法观察。

直接图**计算全部节点**，但省略TIDE的带权payload/transport、source共享、Read/Describe、
区域求解/稀疏发送、通用调度、独立中间梯度根、完整公共记录和continuation。
它使用fan-in初始化，TIDE保留原默认初始化；没有复制或导入基座benchmark/执行核心。
这不是完整数学投影同函数实现，不能据差额推算每项TIDE职责的必要成本；
也不能用它替换正式TIDE训练核心。参考的是基座
[`optimize-tide-execution`的加法/减法比较方法](https://github.com/ZichaoLong/tide/blob/e3452962b928d8cb82d73b1b08fd1ef69e41a119/.agents/skills/optimize-tide-execution/references/additive-comparisons.md)，
没有把这些文档对应的上游新实现冒充本次已安装版本。

| 结构 | 直接Attention可训练参数 | TIDE可训练参数 | 最长串行阶段 |
| --- | ---: | ---: | ---: |
| FC16 | 4,726,784 | 5,092,788 | 4 |
| HB40 | 11,024,384 | 11,939,922 | 5 |
| HB88 | 23,619,584 | 25,633,934 | 7 |

普通Transformer另取既有9,740,544参数配置，仅作常规算子成本参照。
其width256/11 blocks与这些图不同，不据该计时判断语言质量或架构效率。

## 计时口径

每case每模式一次预热、三次计时，表中为秒数中位数，原每次值/范围均在JSON。
每次恢复相同初始权重与**空AdamW状态**；完整步包含首次optimizer state分配，
不是已经进入稳态的连续训练步。
计时包含准备、公共返回、全位置head/loss、finite检查、host标量、张量清理和末端同步；
完整步再含backward、clip和AdamW。模型构造、输入生成/上传、复位、预热和Lens在计时外。
no_grad也计算全位置loss；不是仅最后位置推理、TTFT或生成token/s。
吞吐分母为B×L输入位置；这些重复复位窗口不是语言学习轨迹。
FP32 dtype/确定性策略有固定源码，未单独序列化各后端HF32开关，不作跨后端严格数值资格声明。

<!-- COST_TABLES_BEGIN -->
| B4/L32主表 | CPU1 no_grad | CPU1构图前向 | CPU1完整步 | NPU no_grad | NPU构图前向 | NPU完整步 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FC16 直接MLP | 0.02071 | 0.02131 | 0.07646 | 0.00275 | 0.00308 | 0.01135 |
| FC16 直接Attention | 0.02856 | 0.03013 | 0.10655 | 0.00518 | 0.00680 | 0.02320 |
| FC16 TIDE | 2.10402 | 3.61585 | 7.88714 | 7.46730 | 12.90848 | 25.08114 |
| HB40 直接MLP | 0.04563 | 0.04768 | 0.16586 | 0.00485 | 0.00614 | 0.02327 |
| HB40 直接Attention | 0.06736 | 0.07064 | 0.24444 | 0.01061 | 0.01304 | 0.04684 |
| HB40 TIDE | 5.22943 | 9.23012 | 20.19748 | 23.96564 | 35.87444 | 64.03441 |
| HB88 直接MLP | 0.09403 | 0.09872 | 0.34441 | 0.01022 | 0.01294 | 0.06012 |
| HB88 直接Attention | 0.14325 | 0.15010 | 0.50896 | 0.02158 | 0.02715 | 0.10090 |
| HB88 TIDE | 11.39818 | 19.65165 | 43.75475 | 51.44958 | 77.86642 | 143.09216 |
| Transformer10M | 0.04960 | 0.05188 | 0.22457 | 0.01158 | 0.01403 | 0.08308 |

| 线程/形状扩展 | B / L | no_grad秒 | 构图前向秒 | 完整步秒 | 完整步输入位置/秒 |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU8 HB40直接Attention | 4 / 32 | 0.02413 | 0.02713 | 0.10178 | 1257.7 |
| CPU8 HB40 TIDE | 4 / 32 | 5.71430 | 9.93895 | 23.32167 | 5.5 |
| CPU8 HB40直接Attention | 16 / 128 | 0.16133 | 0.18338 | 0.69444 | 2949.1 |
| NPU HB40直接Attention | 16 / 128 | 0.02101 | 0.02412 | 0.07502 | 27299.7 |
| NPU HB88直接Attention | 16 / 128 | 0.04396 | 0.05215 | 0.16265 | 12591.8 |
<!-- COST_TABLES_END -->

## 如何理解这些差距

1. HB40直接Attention的CPU/NPU比约5.22；现TIDE的NPU反而约为CPU的3.17倍。
   这不是“d128太小就必然CPU更快”，也不是由学习质量差决定的。
2. CPU1的HB40 no_grad/构图前向/完整步分别5.229/9.230/20.197秒；
   NPU分别23.966/35.874/64.034秒。前向已经很慢，不能只归咎于反向或AdamW。
   三模式差额包含不同执行/内存生命周期，不能直接当纯autograd引擎时间。
3. CPU8的HB40直接Attention为0.10178秒，比CPU1快约2.40倍；
   TIDE为23.32167秒，比CPU1慢约15.5%。增加线程不是现安装的通用修复。
4. TIDE有真实批量路径：主表max Full batch为FC16 124、HB40 111、HB88 115，
   max state batch4/sequence32、native max region batch4。不能称全部标量计算。
   这些统计也不证明Describe/共享KV/中间反向已经组成足够大的有效设备算子。

主表最后重复的CPU/NPU TIDE计数如下；计数一致不等于逐事件支持、输出或梯度严格等价。

| 结构 | candidate / Read / state update | Full | Full比例 | 消息 |
| --- | ---: | ---: | ---: | ---: |
| FC16 | 2048 | 1039 | 0.5073 | 3120 |
| HB40 | 4587 | 2616 | 0.5703 | 6282 |
| HB88 | 10009 | 5782 | 0.5777 | 14722 |

直接图dense node-position数分别2048/5120/11264，边-position数6144/12288/28672；
TIDE有些分支候选根本不出现，且即便不Full仍可能有Upd/Read。
所以真实稀疏已存在，但不能把Full比例当作总计算或理论端到端加速比。

## 给基座的下一步用途

本页给出实验侧成本参照，不在本分支详细剖析/优化执行核心。
建议上游优先在HB40/88补**同函数输出训练投影→公共模块边界→完整公共执行**的少量对照，
对齐参数、支持、返回、前后向及optimizer职责；由上游确定需要重构的KV/Read/Describe/块反向边界。
参数别名、共享梯度、None/zero、optimizer、状态和恢复不能因省略公共记录而隐式改变。
只测core微算子不足以判断我们能否快速完成1M/4M有效目标的语言比较。

上游已有[FC三层对照](https://github.com/ZichaoLong/tide/blob/e3452962b928d8cb82d73b1b08fd1ef69e41a119/docs/evidence/fc-three-layer-comparison-20261111.md)：
其生产源码e6f7bd5、B8/L128、不同seed/返回/计时，默认FC16训练CPU12.414秒、NPU16.525秒，
同函数投影的直接FC为1.414/0.745秒。它与本页不同身份/工作量，不能直接相除算版本加速。
**本页测的是实验仍使用的74e7bbd，不是基座当前优化中的实现。**
本轮没有接入、修改或否定正在上游进行的优化；交付后再以新锁/新作业做HB下游有界验收。

优先验收问题是“能否在实际B/L下显著缩短共同曝光的周转时间”，
不是先要求稀疏率兑现同倍吞吐。随后扩大有效物理batch、并行独立轨迹；
CPU/NPU按实测含启动/评价的完成时间选，不能仅用本页小形状预测大形状TIDE成本。
直接大形状结果只认证该直接图；B和L一起改变，不能把吞吐变化全归为batch。

## 身份、记录与恢复边界

成本源码`b2e85cb972e43bffc23c6da6286c147f1b06e96c`，
固定快照`.local/source-hb-cost-20261111`；数学0770593、安装74e7bbd、native构建`.local/native-build-v6`。
Python3.12.13/Torch2.12.0+cpu/TorchNPU2.12.0.post3/CANN9.2.0，私有`.venv-positive-floor-v1`。
native库SHA256 `27ecaa1315d250d9ab2de2a42f2a2487dc4d8225d739c61ffb333b7d66b08289`，
计划SHA256 `60d3f18325070e9d50b77152cd52dd4764acf5884c62cd636f40c6ddb033f121`。
记录含冻结源码tree hash、实际安装文件身份、每格参数/活动/内存/重复和完整argv。

三suite为`hb-cost-20261111-cpu-main`、`hb-cost-20261111-cpu-eight`、`hb-cost-20261111-npu-main`；
额外只读VJP为`hb-gradient-20261111`。均进入`background.slice`，冻结源码和输入未改。
状态/日志/停止沿用`python scripts/job.py status|logs|stop artifacts/jobs/<ID>`（实际调用时三者选一）。
成本窗口不产生可续训checkpoint；若复测须新job/output身份，从固定源码及相同环境重建，不能覆盖旧记录。
可只读重建审阅：`python scripts/review_hb_cost.py`，不启动模型计算。

Lens项目`tide-learning-lab`：成本组`TL-HB-COST-20261111`，run前缀`tl-hb-cost-{cpu-main,cpu-eight,npu-main}-`；
梯度组`TL-HB-LEARNING-20261111`，run前缀`tl-hb-grad-`。
实际投影名称/run UUID/事件数和读回在审阅JSON，私有入口仅见本机`.local/LENS.md`。
项目原始记录权威，Lens不决定执行、严格数值或学习是否通过。

直接节点的FP64独立逐节点/逐时间公式核对输出atol1e-10/rtol1e-9、VJP atol1e-9/rtol1e-8，
因果性、共享head及全部参数梯度检查通过；不认证TIDE/直接图同函数或跨后端严格等价。
没有新精确恢复或语言质量资格；原TIDE严格有限失败、离散/恢复边界保留。

<!-- RESOURCE_SUMMARY -->
实际服务wall跨度30.53分钟；保守扣账CPU 1.567核时
（包含NPU host线程、启动/记录与梯度补证）、NPU 0.509卡时，
低于90分钟/6核时/0.75卡时总表。run/job及两份固定源码快照占用
45.64MiB，不含复用环境、旧数据/模型和旧checkpoint。
三个suite进程RSS高水位：cpu-eight 2.79GiB、cpu-main 5.09GiB、npu-main 4.39GiB；这是进程累计峰值，不能归给后面的单个模型。
NPU allocator最大峰值2.71GiB，不包含其他进程与全部驱动内存；梯度probe未另采集RSS高水位。
所有窗口未见非有限/容量/solver执行失败；这不改变原TIDE严格数值失败。
