# KV128 FFN执行、padding与CPU/NPU对照

这轮保持声明的数学公式和参数归属，分别实现FFN权重直读W、有效行packing P及组合WP。
NPU中尺度的三次配对训练没有加速：W/P/WP吞吐比中位为0.9910/0.8544/0.8131，
但采样allocated峰值由22.739GiB降到21.517/20.598/19.273GiB。
保留原版gather/rectangular默认。数值超差、资格制品预算拒绝与正常计时分开；
新候选未扩展到9B。大图仅继续比较已有KV128和用户指定的共同32层Transformer。

本HB图有固定层次结构；执行的不规则性主要来自动态Full选择、独立节点权重和私有历史布局。
“非结构化组织”和“复杂运算”不能从总差距唯一拆成两个相加百分比。
本次给出可复核的阶段时间、实际行/调用/矩阵、局部执行干预和设备对照；
同步诊断不是正常wall时间占比，省下逻辑行或索引字节也不是可回收的加速比例。

## 身份与范围

实现提交05e8338bf62569100d194848689d545f1140afbc；工具先后为2b8851a7、a5853660、46317a8d。
生产src/tide树始终89f2707e71d989340c12de310797a4c7f4c99f64。
数学固定4f5b9f20a4030d68ee66ae2d26e34659a5e37234，108文件原字节核验通过。
每项结果保留其实际工具提交；证据提交不追认旧测量身份。

训练为129,025,464参数Tide，192 D/40 Hub/44区域，d192、KV32、摘要48、FFN768；
固定32层近总参数Transformer为d496/FFN2048/31个16维头、129,039,856参数。
B16/L128/chunk32，两个真实warmup更新后五个继续更新，三次独立配对repeat，奇数轮反序。
目标为主干readout平方均值，chunk内保留图，更新之间detach状态，AdamW foreach=True。
资格使用另一明确输入seed和有符号余弦目标、foreach=False，不能用有限性检查补等价资格。

推理为768 D/80 Hub/44区域，d768、KV128、摘要192、FFN3072，7,268,408,784参数；
共同Transformer32为d4352/FFN15208/68个64维头、窗口128，8,778,371,328参数。
两者参数数并不相等，共同参考不替代严格A–D。已有[五类绑定和A–D证据](memory-candidates-20261119.md)及[上轮细拆](kv128-transformer-breakdown-20261119.md)保留原身份。
同B8、prompt128/chunk32/decode16；另有NPU decode256，均不搜索batch。
首输出指首个可用主干输出，不含任务头、采样或完整语言模型TTFT。

实际FP32；NPU HF32关闭，Ascend910 64GiB物理9经空闲独占协作队列准入。
Torch2.12.0+cpu/TorchNPU2.12.0.post3/CANN9.2，私有环境与公共工具链边界保持。
CPU为本机aarch64/4插槽/8NUMA/320核；每任务32线程，medium正常240–271/NUMA6，
large正常0–31/NUMA0，独立诊断120–151/NUMA3。affinity+first-touch，不称硬membind。
同一subject跨设备初参数和输入SHA匹配；训练后活动、状态和后端执行可以不同。
主机共享；不同NUMA的单次诊断和16/32线程校准不混进正式三轮结果，不称全局最优。

## 局部函数与两项干预

函数保持 `z=RMSNorm(x); y=x+Wd[i](SiLU(Wg[i]z)*Wu[i]z)`。
本次独立FFN的稠密前向乘加对数为 `3 R d F`，R为实际执行行，不能用选中行替代。
权重bank形状为gate/up `[owners,F,d]`、down `[owners,d,F]`；
所有共享基座的梯度汇总、节点私有adapter和私有历史归属不变。

W按owner组织激活并直接读取原权重bank视图。NPU forward/dX使用group_type0，
dW使用group_type2；native autograd不可用，因此显式实现一阶伴随，不承诺高阶导数。
普通FFN一次owner plan复用于三次投影；放置wrapper仍逐投影建plan。
CPU是逐group MM参考实现，不冒充NPU native grouped matmul的CPU版本。

P只在FFN边界压紧同owner有效(batch,time)行，使用1/2/4…容量桶，末桶封顶原B×Q。
`S≤E_new≤E_old`；当`S>0`时`E_new<2S`，`S=0`时`E_new=0`。每owner只进一桶，三投影复用mapping。
Attention矩形、内容/摘要/FFN宽、Full选择规则、原始int64时间、KV窗口及SSD状态均不改变。
这是执行候选，不是新的记忆数学或GQA。定义与公开配置见docs/ffn-execution.md。

## 正常训练与显存

所有吞吐先按每repeat总位置/总时间计算，再取三repeat中位数；范围是三轮最小/最大。
配对比先在同repeat相除再取中位，不能用表中两个中位吞吐的商代替。

| 设备 | 候选 | 训练位置/s，中位 [范围] | 对原版配对比中位 | NPU allocated/reserved GiB | CPU正常结束RSS高水位 GiB |
|---|---|---:|---:|---:|---:|
| NPU | baseline | 883.923 [883.702, 961.595] | 1.0000 | 22.739/23.777 | — |
| NPU | W | 882.153 [875.783, 942.152] | 0.9910 | 21.517/22.520 | — |
| NPU | P | 761.036 [747.421, 821.560] | 0.8544 | 20.598/21.494 | — |
| NPU | WP | 718.749 [707.665, 788.020] | 0.8131 | 19.273/20.291 | — |
| NPU | Transformer32 | 4501.196 [4448.307, 4593.017] | 5.0337 | 6.644/7.670 | — |
| CPU | baseline | 93.050 [90.258, 94.076] | 1.0000 | — | 30.921 |
| CPU | W | 78.856 [75.071, 78.988] | 0.8382 | — | 29.085 |
| CPU | P | 78.962 [78.805, 80.174] | 0.8616 | — | 26.671 |
| CPU | Transformer32 | 239.461 [233.510, 258.396] | 2.5735 | — | 9.578 |

CPU没有正式WP吞吐点。NPU三候选均无加速，CPU W/P的配对吞吐约下降16.2%/13.8%。

下表为全部15次normal更新的算术均值（秒），阶段加余项等于完整更新均值。

| 设备/对象 | 前向窗口 | 反向边界 | AdamW | 其余边界 | 完整更新 |
|---|---:|---:|---:|---:|---:|
| NPU baseline | 1.338519 | 0.896237 | 0.019302 | 0.000695 | 2.254753 |
| NPU Transformer32 | 0.202642 | 0.236615 | 0.014206 | 0.000298 | 0.453761 |
| CPU baseline | 4.958318 | 17.010265 | 0.185793 | 0.002226 | 22.156601 |
| CPU Transformer32 | 3.010731 | 5.136484 | 0.267641 | 0.001422 | 8.416278 |

NPU三轮配对吞吐变化：W -0.9%（范围-2.0%至-0.2%），
P -14.6%（-15.4%至-13.9%），WP -18.7%（-19.9%至-18.1%）。
原版为883.702–961.595位置/s，Transformer为4448.307–4593.017。
这是相同初参数/输入、同配置的有限执行比较；warmup后的参数、AdamW、梯度及私有状态SHA已不同，
Full量为200435/200441/200429/200441，不能称严格同轨迹或同实际负载加速。

正常训练阶段均值可相加：前向窗口包含zero_grad、准备和分块同步；反向边界可含此前排队的拼接/loss工作。
其余项含尚未落入上述区间的拼接/loss提交和必需detach等；纯loss/反向拆解来自另外的同步趟。

旧normal记录的boundary.memory文字过宽。complete_seconds在有限性检查前结束，
allocator峰值在run_stream返回后采样，包含其未计时finite检查，但早于后续continuation审计。
46317a8d修正文案，旧原始记录不改。CPU RSS是进程生命周期高水位，不与NPU HBM直接等同。

独立r0 ledger中，Tide权重0.481GiB、AdamW0.961GiB、私有状态0.195GiB；
前向exclusive saved storage为原版/W/P/WP的19.984/18.753/17.792/16.433GiB，
Transformer为4.428GiB。W的saved view可与参数alias，unique和exclusive必须区分。
allocator余项包含未观察张量及后端分配，不能全部叫workspace；各阶段峰值不可相加。
ledger目标使用按chunk求和，归约次序与normal joined readout不同，独立记录而非计时轨迹证明。

## 正常大图推理与CPU差距

短请求固定B8/prompt128/chunk32/decode16。CPU每对象1warm+2normal、NPU为2warm+5normal，均三配对repeat。
吞吐与独立序列/s按每轮汇总后取中位；首主干是每轮样本中位的中位，请求列为每轮均时的中位。

| 设备/对象 | 首主干输出 s | prefill位置/s | stable decode位置/s | 完整请求 s | 独立序列/s |
|---|---:|---:|---:|---:|---:|
| NPU baseline | 0.742524 | 337.104 | 18.449 | 9.974589 | 0.802038 |
| NPU Transformer32 | 0.129124 | 1962.448 | 137.114 | 1.458333 | 5.485716 |
| CPU baseline | 6.426261 | 38.520 | 3.843 | 59.938169 | 0.133471 |
| CPU Transformer32 | 3.681484 | 68.439 | 6.506 | 34.657180 | 0.230832 |

同repeat的Transformer/Tide吞吐比：

- NPU：prefill 5.821倍 [5.779, 5.830]；stable decode 7.435倍 [7.417, 7.440]；完整请求 6.858倍 [6.825, 6.858]。
- CPU：prefill 1.789倍 [1.758, 1.831]；stable decode 1.686倍 [1.678, 1.782]；完整请求 1.731倍 [1.712, 1.803]。

以下为所有normal短请求的可加算术均值，单位秒；首主干输出是内嵌边界，不另加。

| 设备/对象 | prefill128 | decode16合计 | 其余完整边界 | 完整请求 |
|---|---:|---:|---:|---:|
| NPU baseline | 3.037636 | 6.940499 | 0.000065 | 9.978200 |
| NPU Transformer32 | 0.522823 | 0.934462 | 0.000046 | 1.457331 |
| CPU baseline | 26.785860 | 33.835448 | 0.000081 | 60.621389 |
| CPU Transformer32 | 14.942778 | 19.718398 | 0.000068 | 34.661244 |

NPU短请求normal最大allocated/reserved：
baseline 33.491/35.854GiB；
Transformer32 34.866/37.639GiB；
CPU正常结束RSS高水位为Tide40.665、Transformer37.191GiB，含构造及先前进程分配。

NPU持续decode256采用同B8/prompt128/chunk32、每对象1warm+2normal、三次配对repeat。
下面统计口径与短请求主表相同；stable decode每个请求计后255步。

| 对象 | 首主干输出 s | prefill位置/s | stable decode位置/s [范围] | 完整请求 s | 独立序列/s | normal allocated/reserved GiB |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.746970 | 335.178 | 18.436 [18.211, 18.470] | 114.118540 | 0.070103 | 33.507/36.193 |
| Transformer32 | 0.129106 | 1965.587 | 137.226 [137.226, 137.720] | 15.445462 | 0.517951 | 34.898/37.670 |

长请求的stable decode配对差距为7.443倍 [7.430, 7.563]，完整请求为7.388倍。
没有观察到延长decode使差距缩小；此结论限于窗口128和本次256步有限负载。
独立序列吞吐为B/完整请求时间；批请求/s再除以B=8，不能混用这两个分母。

长请求的可加normal算术均值（每对象6样本、秒）：

| 对象 | prefill128 | decode256合计 | 其余完整边界 | 完整请求 |
|---|---:|---:|---:|---:|
| baseline | 3.062299 | 111.471768 | 0.000114 | 114.534181 |
| Transformer32 | 0.521492 | 14.906530 | 0.000091 | 15.428113 |

KV128仍有6,002,049,024个独立FFN参数，约占总参数的82.58%；收窄KV没有缩小这部分。
三轮CPU结果反映这组32线程、NUMA0、共享主机的实际行为；CPU诊断来自NUMA3。
同初参数/输入SHA不保证训练后活动与算术路径一致，也不足以把跨设备比值拆成纯硬件因果比例。
CPU校准只有各一轮：16线程Tide85.039/TF208.501，32线程94.556/187.657位置/s；
正式比较固定双方32线程，不以这两点宣称线程全局最优。NUMA3的TF单次训练3.640s与
NUMA6校准约10.94s差异尚未解释；后来的频率/本地页观察不能解释过去计时。

短请求推理独立r0 ledger中，Tide/Transformer权重为27.077/32.702GiB，末态私有记忆1.419/1.054GiB。
此处没有优化器或saved-for-backward；前向仍有临时张量与后端分配。live allocator与可见storage并集的余项单列，
不能把不同时间的峰值减去末态就命名为workspace，CPU RSS也不能替代这个分区。

## 为什么减少padding仍更慢

NPU r0 work中，KV FFN实际行由335104降至164244（约-51%），
但调用由28升至131；SSD仍为81920行/20次。Attention矩形仍是335104行。
三投影合计的全FFN乘加对由184.478B降至108.895B，不能据此按比例推断整步。

独立同步前向的FFN/行准备阶段：原版0.069587s、W0.102398s、P0.273599s。
W的48次owner plan复用于144次投影，其中plan0.021323s、native matmul0.018838s，
另有激活组织、反向映射和调用余项。P的bucket0.088129s、apply余项0.050298s、FFN本体0.135173s。
P同样的1,642,070,016逻辑权重gather输出字节由144次拆成453次；矩阵调用也由144增至453。
W移除了这部分ATen参数gather，但有288次激活index_select和4.804GB重复逻辑输出。
这些字节不是总线流量或带宽测量，vendor的32-padding分配仍可存在。

WP的KV-down非空group合计768，而bank视图跨度合计3920；W为832。
多桶还会扩大空group覆盖与梯度组织，不能把直读视图等同于后端零搬运。
原生trace确认GroupedMatmul在Ascend Hardware执行，也观察到int64 sort/cumsum的AICPU kernel。

## 分配、历史与反向

训练内部另做独立同步重放，下面均为秒。CPU取NUMA3的诊断锚点，不能代入NUMA6正常三轮。
原版/TF的前向为1.519/0.338s（NPU）、3.410/1.554s（CPU）；反向为0.982/0.220s、13.066/2.091s。
前向的互斥细分如下；“其余Full”已排除单列FFN。

| 同步前向阶段 | NPU Tide | NPU TF32 | CPU Tide | CPU TF32 |
|---|---:|---:|---:|---:|
| FFN含行准备 | 0.069587 | 0.060531 | 0.488093 | 0.870458 |
| 消息收集/区域分组 | 0.140843 | 0.000000 | 0.324353 | 0.000000 |
| 接收 | 0.068651 | 0.000000 | 0.433907 | 0.000000 |
| KV准备/其余Full | 0.299569 | 0.000000 | 0.662789 | 0.000000 |
| SSD准备/其余Full | 0.186868 | 0.000000 | 0.365628 | 0.000000 |
| 描述/区域评分 | 0.185609 | 0.000000 | 0.579052 | 0.000000 |
| 分配含主解/回退/face | 0.410213 | 0.000000 | 0.180589 | 0.000000 |
| 状态拆分/合并 | 0.040940 | 0.000000 | 0.051556 | 0.000000 |
| Transformer Attention及其余块 | 0.000000 | 0.272348 | 0.000000 | 0.662088 |
| 入口/readout/观察器/余项 | 0.116834 | 0.004721 | 0.323787 | 0.021476 |
| 前向合计 | 1.519113 | 0.337601 | 3.409753 | 1.554021 |

8次NPU及4次CPU训练诊断均从各自warmup后的锚点重放，对各自normal更新在原阈值内；
输出/私有状态精确，CPU全检查精确，NPU梯度/参数/AdamW存在阈内非exact。
这是进程内重放检查，新进程checkpoint恢复资格另列。

NPU baseline同步前向中，sorted0.106340s、bisection0.182795s、保存face0.011198s；
28次回退仅请求324行，却执行49152行，每次40轮二分加3轮修正，宽度8。
CPU诊断为4次回退、请求7/执行7168行；warmup后的轨迹和算术不同，不能称同工作硬件加速。
metadata观察器的标量读取单列，不能冒充生产同步成本。

CPU反向逐算子另趟观察到Tide BMM总9.351598s/2425次，index_put1.934496s/1660次；
Transformer MM2.290412s/1024次，并有128次CPU flash-attention反向。
Tide CPU Attention暴露BMM/softmax/exp，未观察到fused dispatcher；NPU Tide/TF分别28/128次native fused attention。
通用反向算子缺少前向模块归属，不能全部记作FFN反向。算子秒数不能除以正常更新时间。

大图的实际依赖顺序仍是：

```text
for compatible_regions in layer_order:
    receive(gather_messages())
    prepare_KV_and_SSD_for_receiving_nodes()
    describe_and_score_regions()
    solve_allocation()
    pack_selected_queries(); read_memory(); apply_FFN(); publish()
    split_or_join_private_state()
```

SSD准备发生在Full选择之前，包含需要更新的私有记忆；不能仅凭最终Full稀疏就跳过。
本次算子趟的16步decode：KV准备98,304行、选中9,749行、FFN执行48,512行；
SSD准备/FFN执行均10,240行。CPU/NPU这些聚合行数相同，allocation回退不同：
NPU prefill请求112/执行22,528、decode16/424；CPU为1/1,024、1/32。

下面为独立同步阶段趟的互斥分区。prefill单位为整段128位置的秒数；decode单位为16次调用的平均毫秒，
包含第一步。正常stable decode仅取后15步，不能与本表相除得占比。
每个子scope只计一次；表中只合计捕获的调用，释放/检查另列。
余项包含hook/dispatch及入口等待，不能全部视为Python时间。
CPU同步为no-op，NPU则强制设备完成；同步改变调度的影响保留。

### prefill同步互斥阶段

| 阶段 | NPU Tide | NPU TF32 | CPU Tide | CPU TF32 |
|---|---:|---:|---:|---:|
| FFN（含归一化/行准备） | 0.893961 | 0.359847 | 6.933104 | 7.074455 |
| KV历史整理 | 0.160471 | 0.000000 | 0.110633 | 0.000000 |
| KV其余准备 | 0.102356 | 0.000000 | 0.938022 | 0.000000 |
| KV读取及其余Full | 0.156428 | 0.000000 | 0.905669 | 0.000000 |
| SSD准备 | 0.196176 | 0.000000 | 0.756485 | 0.000000 |
| SSD其余Full | 0.042131 | 0.000000 | 0.250643 | 0.000000 |
| 消息收集、接收与区域分组 | 0.732902 | 0.000000 | 5.753703 | 0.000000 |
| 描述与区域评分 | 0.316979 | 0.000000 | 3.326928 | 0.000000 |
| 分配求解 | 0.389669 | 0.000000 | 0.245081 | 0.000000 |
| Full打包/发布 | 0.137295 | 0.000000 | 0.206395 | 0.000000 |
| 状态拆分/合并 | 0.033428 | 0.000000 | 0.077264 | 0.000000 |
| Transformer Attention/投影/其余块 | 0.000000 | 0.344840 | 0.000000 | 3.696293 |
| 入口/观察器/未细分余项 | 0.272436 | 0.004846 | 1.124741 | 0.023132 |
| 合计 | 3.434233 | 0.709533 | 20.628669 | 10.793880 |

### decode同步互斥阶段

| 阶段 | NPU Tide | NPU TF32 | CPU Tide | CPU TF32 |
|---|---:|---:|---:|---:|
| FFN（含归一化/行准备） | 169.699983 | 42.961359 | 602.883543 | 657.831517 |
| KV历史整理 | 49.668631 | 0.000000 | 37.164095 | 0.000000 |
| KV其余准备 | 21.737493 | 0.000000 | 159.679108 | 0.000000 |
| KV读取及其余Full | 33.314770 | 0.000000 | 121.396818 | 0.000000 |
| SSD准备 | 34.322355 | 0.000000 | 69.974367 | 0.000000 |
| SSD其余Full | 8.672237 | 0.000000 | 19.010133 | 0.000000 |
| 消息收集、接收与区域分组 | 39.134224 | 0.000000 | 96.409242 | 0.000000 |
| 描述与区域评分 | 56.502667 | 0.000000 | 269.092465 | 0.000000 |
| 分配求解 | 54.952609 | 0.000000 | 24.249208 | 0.000000 |
| Full打包/发布 | 18.966892 | 0.000000 | 14.080465 | 0.000000 |
| 状态拆分/合并 | 10.847048 | 0.000000 | 36.143676 | 0.000000 |
| Transformer Attention/投影/其余块 | 0.000000 | 58.798632 | 0.000000 | 350.662421 |
| 入口/观察器/未细分余项 | 24.468232 | 1.225149 | 80.228667 | 2.135111 |
| 合计 | 522.287140 | 102.985140 | 1530.311787 | 1010.629049 |


在同一个NPU decode阶段趟中，明确命名的消息收集、区域分组、KV历史整理、Full打包/发布、
状态拆分/合并共101.785ms，占该诊断19.49%；FFN阶段169.700ms，占32.49%。
这些是观察窗口的时间份额，不能解释为“非结构化导致19.49%的性能劣化”。
权重索引藏在FFN内，接收/区域评分又同时包含组织和数值计算，二者并无唯一可加的因果划分。

另一独立逐算子趟的NPU decode中，每次B8 decode内FFN权重gather累计时间平均150.055ms、BMM累计时间平均16.052ms；
16步累计624次gather产生207,920,037,888逻辑输出字节。它们不是总线流量，也不能与上表相加。
直接可见的FFN矩阵乘加对为Tide25.990G、Transformer50.830G，单位是每次B8 decode（8个新输入位置）；
该同步算子趟中，每次decode的矩阵累计计时平均分别16.052/35.459ms。
这是dispatcher观察窗口，不能当作kernel-only时间或相同矩阵的效率比。
具体操作数包含Tide的`[19,8,768] @ [19,768,3072]` BMM，
以及Transformer的`[8,1,4352] @ [4352,30416]` gate/up投影（后一形状每步32次）。
本诊断直接观察到weight-bank gather大项；仅以小矩阵效率解释全部差距缺少支持。
正常执行中可回收多少时间仍需已获数值资格的整体干预测量。
这也解释了为何只看KV宽或最终缓存体积会漏掉大成本，但尚未证明在9B用W可回收这些时间。

四点同设备的两诊断输出SHA一致；normal没有保留输出SHA。Tide有44个命名int64时钟值指纹，
另20个SSD卷积时钟仅metadata，浮点历史没有完整逐值hash；不能据此声称全部私有状态相等。
CPU/NPU的20次readout SHA均不同，没有做跨设备阈值资格；相同聚合工作量不等于相同浮点轨迹。


## 数值资格与恢复

完整CPU suite为979 passed、1 strict xfailed、1 warning；不能称980项数值通过。
保留共享FFN对FP32 literal公式的近零梯度超差；与FP64公式的有界诊断另列。
小图CPU（独立FFN、d64/KV32/F256、B2/L8/chunk4）：W/P在阈内；
WP同一参数字段在后两次更新比较中超差，共3次坐标出现、2个不同坐标。四者各自新进程恢复精确。
小图NPU：W阈内，P/WP各11次更新/状态字段比较超差，包含重复状态表示；
四者各自新进程恢复更新阈内，只有W实际更新精确。该小图不覆盖共享FFN的NPU集成资格。

d768 NPU独立/共享FFN的gather与grouped四项公式输出失败原样复现，原2e-6/2e-6阈值不变；
原资格输入/参数VJP在6e-6/6e-6内，补充诊断只重算输入VJP。
同NPU两路径的输出及输入VJP阈内，对CPU FP64公式也阈内；CPU32 literal本身有少量超差。
首次观测差异/超差出现在投影、product或down边界，尚未确定指令或累加次序等根因。
不据此追认原资格通过，不把W扩到9B。

中尺度原NPU资格中，baseline推理↔grad有14字段超差（max scaled3.7189），重复VJP阈内；
实际完成step1→2→3→4并保存position512 checkpoint。在保存恢复包前触及20GiB预算，W未开始。
中尺度CPU baseline/W跨模式各124字段超差；W后两次继续更新有392/533字段比较失败，
这些含状态重复表示，并非独立错误事件数。两者各自fresh-process step4→5恢复精确。
P初始推理/VJP对baseline有55/62字段超差；首更新计算后的差异payload触及32GiB预算，
没有完整持久更新记录/恢复。已保存比较未见支持、原始int64、元数据、dtype或非有限错误；
这不消除值、梯度及更新超差。原阈值始终1e-6+0.002|reference|，未自动归因舍入。

独立NPU恢复直接读取上述已有checkpoint，另建明确的空梯度、seed2718参考；
不声称复现旧进程未保存的梯度/RNG。saved data、配置、RNG和next forward精确；
425个参数张量实际从AdamW step4推进到5，真实更新比较在原阈值内但非逐位相同。
最坏embedding梯度maxabs为1.1641532183e-10、scaled为4.115e-5。
parent/child各自peak allocated23.241/reserved24.211GiB；
保守重叠上界为25.372/27.002GiB，低于56GiB。成功恢复bundle依原记录退休，原失败载荷保留。

## 可采取的后续方向

1. 优先让allocation回退只处理请求行，并保留原阈值/face/VJP；减少行不能预先承诺同比加速。
2. FFN把有效行直接交给一次ragged grouped matmul，避免power2多桶、多次plan/调用及空bank跨度；
   复用已知owner顺序，减少sort和逆映射。原始时间继续int64，不能为了设备sort改变时间精度。
3. KV历史改为明确私有的持续bank/窗口布局，减少拼接、重排与state join；先验证淘汰、零接收与梯度再测量。
   带权Attention融合必须保留零接收导数，不能直接用静态mask核替换。
4. 若改参数共享、Full选择频率或区域通信规则，应作为显式模型实例扩展，另做下游任务质量验证。
   当前HB已有固定层次结构，可先利用静态区域/owner信息；不需要因此启动通用图基座。

当前推荐保留gather/rectangular；W/P/WP保留为有明确限制的可复现候选，不能据省显存替换默认训练。
五类记忆绑定、BF16既有超差和所有原证据不被本轮改写；本轮无9B训练、CUDA、多NPU或模型质量结论。

## 资源、复现与限制

29个background.slice服务均终态：23个执行passed、4个failed、2个terminated。
这只是进程执行状态，不能把23个passed当作23项数值资格通过。
失败分别是d768原公式、两个中尺度资格制品预算拒绝，以及CPU Transformer诊断v1参数错误；
两个取消是尚未进入queue的NPU Transformer诊断v1和其依赖long v1。
234个记录的后代PID/start身份均终态；29份冻结快照合计14,056次逐文件对照Git对象/模式通过。
实际legacy systemd cgroup已核对为空或不存在，未遗留本轮模型作业。

CPU服务累计2.385701任务小时，包含1748.899s前置等待，低于12任务小时。
NPU queue累计1.651467卡时（含准入等待），低于8卡时；
NPU service总时长12136.797s中的
6177.350s为准入前依赖等待，不计设备占用。
已记录suite worker分别为CPU 4537.163s、NPU 4728.201device-s；
这些包含构造、warmup、检查、诊断和IO，独立程序未记录的worker全时长保持缺测。
不同时间口径不能相加为“纯计算时间”。普通服务均未超过2小时，大图CPU未超过3小时。

采样进程树RSS最大54.952595GiB，低于每作业192GiB；
10个NPU任务保存了allocator观察，已保存最大allocated/reserved为
34.949521/37.916016GiB，均低于56GiB。
本轮未使用共享占用NPU；CPU最多32线程，NPU主机4线程，实际affinity见JSON。
job.py watchdog在worker启动后移至228–231，worker核组未改，逐run保留调整记录；suite自身监督仍随worker核组。

allocator覆盖限制：grouped probe、local diagnosis、local qualification、原medium NPU qualification、
small NPU qualification没有保存该字段；不能给这些任务补写峰值或完整HBM预算资格。
另两个queue前取消项没有模型分配器记录。独立NPU恢复已将两个文件内的8份观察纳入，
其中嵌入重复只取最大值，不累加；parent-held+child-peak保守上界由恢复资格另行核对。

审计时实验目录保守占用49.197685GiB，低于64GiB；
原失败载荷保留，后续仅写入少量摘要与仓库证据。常规JSON/审阅/源码有独立指纹；
没有追加对所有大型CPU资格.pt载荷的全量重读，保留生成时SHA、大小和已保存失败记录。
固定math108文件及四份用户未跟踪文档原字节复核通过；公共工具链、参考及下游仓库未修改。
原始大记录、完整算子树和恢复载荷在对应artifacts目录，紧凑JSON保留正常标量样本、主要阶段、
资格/恢复结论、资源审计及输入指纹。CPU/NPU不同轨迹、单NUMA诊断、未解释CPU时差、
数值超差和未覆盖后端均不从其他测点外推。


Transformer诊断v1误传--config，CPU在参数检查失败、未构造模型；NPU对应v1及未启动的long v1在queue前终止。
v2使用显式d4352/F15208/L32/head64/window128，失败与取消记录保留，normal吞吐记录不受影响。
所有正式worker都由scripts/launch.py冻结提交后在background.slice运行；普通计时不带同步hook。
原始命令/环境/输入在对应artifacts/ffn-*/launch.json、environment.json、results或breakdown.json；
只读collect_normal.py、collect_inference_stages.py和audit_jobs.py整理已存在记录，不重启测量。


正式复现先核对当时配额、空闲设备和CPU affinity；使用新run名，长作业经launcher冻结。
下面复现中尺度NPU三轮矩阵的原工具身份和负载，queue只准入独占空闲卡：

```bash
python scripts/launch.py \
  --name ffn-repro-medium-npu --revision 2b8851a7 \
  --threads 4 --affinity 224-227 --npu-count 1 --candidate-ids 9 \
  --seconds 7200 --rss-gib 128 --queue-seconds 600 -- \
  python -m benchmarks.execution_suite \
  --config configs/memory-candidates/medium-kv128.json \
  --output-dir '{run}/results' \
  --variants baseline weight_access packing combined --include-transformer \
  --mode train --device npu:0 --batch 16 --length 128 --chunk 32 \
  --threads 4 --repeats 3 --samples 5 --warmup 2 \
  --child-timeout 900 --total-seconds 6600 --observe-first
```

其余准确命令在各run的`launch.json.arguments.command`中，代表入口如下。
此表记录已执行范围，不自动授权新一轮资源使用。

| 事项 | 模块 / 固定工具 | 代表输入与边界 |
|---|---|---|
| CPU中尺度训练 | `benchmarks.execution_suite` / a5853660 | medium-kv128；baseline/W/P+TF，B16、32线程、3×5次更新 |
| CPU大图短推理 | `benchmarks.execution_suite` / a5853660 | 9b-kv128；baseline+固定TF，B8、32线程、3×2次请求、1warm |
| NPU大图短推理 | `benchmarks.execution_suite` / a5853660 | 同B8/p128/ch32/d16，3×5次请求、2warm |
| NPU持续decode | `benchmarks.execution_suite` / 46317a8d | 9b-kv128、固定TF、B8/p128/ch32/d256，3×2次请求、1warm |
| 训练同步细拆 | `benchmarks.training_breakdown` / a5853660 | 中尺度、2warm、1normal、独立stages/operators，恢复同候选锚点 |
| 推理同步细拆 | `benchmarks.inference_breakdown` / 46317a8d | 大图、1warm、1normal、独立stages/operators；TF用显式尺寸参数 |
| 集成资格 | `scripts.check_execution_candidates` | 小图工具05e8338b；中尺度工具a5853660，原阈值、有限制品预算 |
| 独立checkpoint更新 | `scripts.check_checkpoint_update` / 46317a8d | 已有checkpoint、anchor、seed2718、全梯度None新参考 |

公开执行配置在[FFN执行选项](../ffn-execution.md)及`configs/ffn-execution/`。
数据入口、完整标量样本、阶段分区、审阅和资源指纹见[机器可读证据](ffn-execution-devices-20261119.json.gz)。
