# Tide 教学图集

本页 PNG、SVG 按下方“生成与修改”命令在本地重建，默认不随 Git 保存；首次克隆需先生成，再打开图片链接。维护源与历史来源快照继续跟踪，规则见[仓库生成物约定](../../ARTIFACTS.md)。

先用结构图认识节点、区域与消息流，再用 P/S/U/F 简图解释一次区域作用。接着用带权消息图和两个算例讲清局部计算，按需要展开具体记忆与发送规则。

[展示脚本](../presentations/design-introduction/script.md)从 Transformer 的残差展开到显式 MoE，再讨论局部网络与历史。[展示材料](../presentations/design-introduction/README.md)提供离线 HTML、PDF 和备注。

[分步 SVG](design-introduction/)由 [design_introduction.py](src/design_introduction.py)绘制：Transformer 显示 Norm、Attention、FFN 和两次 Add；MoE 沿用[新增教学绑定](../instances/explicit-residual-moe.md)及[核对程序](../examples/explicit_residual_moe_checks.py)的轨迹；HB 仅作后续拓扑说明和独立附例，保留原来的固定布局与参数。下方各局部图也各有自己的数值条件。

## 整体结构与一次区域作用

| 图片 | 用途 | 数学定义 |
|---|---|---|
| 节点、区域与消息流：[PNG](tide-structure.png)／[SVG](tide-structure.svg) | DAG 的发散、汇合、跨区连接与节点局部流程 | [TimedDAG §2—6](../timed-dag-region-selector-learning-note.md#2-固定结构时间节点图端口与区域划分) |
| P/S/U/F 简图：[PNG](PSUF-simple.png)／[SVG](PSUF-simple.svg) | 共同选择、候选状态保留与激活后的发送 | [TimedDAG §5](../timed-dag-region-selector-learning-note.md#5-区域候选集合选择函数选择历史与状态采用) |
| P/S/U/F 详图：[PNG](PSUF-complete.png)／[SVG](PSUF-complete.svg) | 计算快照、下一状态、控制与完整输出 | [共同接口索引](../reference/semantic-interface-index.md#局部函数) |

图中的节点名称只用于对应示例。结构图与 P/S/U/F 图表达正典的一般语义；某个局部架构的附加约束由相应教材定义。

## 带权消息与两个算例

这组图采用[带权消息局部架构](../local-semantics/weighted-message/architecture-learning-note.md)。蓝色表示当前内容与传播，绿色表示记忆，金色表示区域描述与控制。

### 一次接收到发送

先由 Agg 形成接收，再执行 Upd。图中采用 Read$^+$，描述读取新状态；全部候选保留接收更新。区域给出正控制后，Full 计算完整内容，并在每条出边复制同一带权载荷。

Agg 可选来源残差或[同刻局部自注意力](../local-semantics/weighted-message/architecture-learning-note.md#23-同刻消息的局部自注意力)，图中均表示为 $(r^{\mathrm{in}},u)$ 接口。自注意力先在到达消息间联合读取，再逐条加工与汇聚；下方原有数值算例继续采用来源残差。

[PNG](weighted-message-flow.png) · [可编辑 SVG](weighted-message-flow.svg) · [定义：共同架构 §2—4](../local-semantics/weighted-message/architecture-learning-note.md#2-接收聚合)

![带权消息的一次处理](weighted-message-flow.png)

### 区域怎样分配发送强度

沿用教材的三个候选：接收权重与发送上限均为 $(1,1,0)$，评分为 $(2,1,100)$。区域头给出总量 $B=1.4$，二次分配在温度 $2$ 得到 $(0.95,0.45,0)$。总强度为 $1.4$，激活节点数为 $2$。

[PNG](region-allocation.png) · [可编辑 SVG](region-allocation.svg) · [定义与数值：区域分配 §2.5](../local-semantics/weighted-message/region-allocation-learning-note.md#25-三个候选的一次计算)

![区域怎样分配发送强度](region-allocation.png)

### 收到时记住，激活时发送

沿用 KV 教材的两次接收例子。Read$^+$ 和 sum 描述给出长期摘要；硬选择每次激活最高正分节点。节点 $b$ 第一次未激活，仍保留 $+1$；第二次收到 $-1$ 后，两条记忆共同影响它的评分与完整内容。

[PNG](remember-before-send.png) · [可编辑 SVG](remember-before-send.svg) · [定义、参数与数值：KV §5.2](../local-semantics/weighted-message/kv-memory-learning-note.md#52-未激活时保留的记忆)

![收到时记住，激活时发送](remember-before-send.png)

## 记忆模块

四张图按 Upd、Mem、G 的顺序展开。图中描述均采用 Read$^+$；其他快照模式由共同架构定义。循环记忆图展示 state 描述，sum 候选从独立摘要取得描述。图中的 $N_d$ 对应教材的归一化函数 $\mathcal N_d$。

| 图 | 主要问题 | 定义出处 |
|---|---|---|
| KV：[PNG](kv-memory.png)／[SVG](kv-memory.svg) | 带时间的条目怎样保存，怎样用于描述与完整 Attention 读取 | [KV §1](../local-semantics/weighted-message/kv-memory-learning-note.md#1-状态与接收写入)、[§3—4](../local-semantics/weighted-message/kv-memory-learning-note.md#3-记忆描述) |
| 仿射与 Delta：[PNG](affine-delta-memory.png)／[SVG](affine-delta-memory.svg) | 外积叠加与预测误差修正怎样填入同一更新接口 | [基础循环 §1—3](../local-semantics/weighted-message/recurrent-memory/affine-delta-learning-note.md#1-状态与逻辑时间) |
| Gated DeltaNet：[PNG](gated-delta-memory.png)／[SVG](gated-delta-memory.svg) | 短卷积与门怎样形成误差写入；每头 RMS 后怎样调制输出 | [Gated DeltaNet §2](../local-semantics/weighted-message/recurrent-memory/gated-delta-learning-note.md#2-投影与接收更新)、[§3](../local-semantics/weighted-message/recurrent-memory/gated-delta-learning-note.md#3-矩阵读取记忆描述与完整内容) |
| SSD：[PNG](ssd-memory.png)／[SVG](ssd-memory.svg) | 输入步长怎样控制保留与写入；直通项、输出门与全宽 RMS 怎样组合 | [SSD §2](../local-semantics/weighted-message/recurrent-memory/ssd-learning-note.md#2-投影门与写入)、[§3](../local-semantics/weighted-message/recurrent-memory/ssd-learning-note.md#3-矩阵读取记忆描述与完整内容) |

基础循环保存 short、long 两种时间尺度的矩阵；Gated DeltaNet 与 SSD 保存多头矩阵，state 描述的 short 通道使用时间摘要，long 通道使用矩阵查询。各模块的接收门参与记忆更新，区域控制 $c_v$ 决定后续发送。

## 发送规则与逻辑时间

| 图 | 主要问题 | 定义出处与图中条件 |
|---|---|---|
| 发送规则：[PNG](sending-rules.png)／[SVG](sending-rules.svg) | 区域总量、共同阈值、节点上限与 Attention 份额分别起什么作用 | [区域分配 §4.1—4.2](../local-semantics/weighted-message/region-allocation-learning-note.md#41-约束分配)；图中比较六种发送规则，其他候选见教材 |
| 温度与激活数量：[PNG](temperature-support.png)／[SVG](temperature-support.svg) | 同样的总强度怎样对应不同数量的激活节点 | [区域分配 §5.5](../local-semantics/weighted-message/region-allocation-learning-note.md#55-分配与饱和)；节点增益与固定阈值分配，quad，单位上限，$t=(4,3,2,1)$，$k=2$ |
| 固定逻辑时间槽：[PNG](logical-time-slots.png)／[SVG](logical-time-slots.svg) | 缺失接收与弱接收怎样进入短卷积，其他历史怎样保持原坐标 | [短卷积 §1](../local-semantics/weighted-message/recurrent-memory/causal-convolution-learning-note.md#1-缓存卷积与写入)、[§2.2—3](../local-semantics/weighted-message/recurrent-memory/causal-convolution-learning-note.md#22-与通常短卷积的对应)；标量、间距 $2$、三抽头系数 $(1,2,3)$、零偏置 |

Attention 的最终控制按各自公式形成：直接 Attention 满足 $c_v\le\mu_v$；固定总强度和总量缩放 Attention 允许单个分量超过 $\mu_v$。温度图使用指定的 quad 算例，三次总和均为 $2$，激活数量依次为 $4,3,2$。

## 生成与修改

首次克隆可在仓库根目录运行 `python3 tools/build.py`，一并生成图集、展示 HTML 和备注；`--full` 另导出 PDF 与预览。安装方法及 CI 入口见[统一构建说明](../../ARTIFACTS.md#本地生成)。只修改图集时，使用 `python3 tools/build.py --group legacy-figures` 或 `--group local-figures`，会同步更新构建指纹。下方底层命令适合单图或临时预览，不更新统一构建记录；正式核验前再按组构建。

使用 Python 3.10 或更高版本，安装 [Python 依赖](requirements.txt)、Cairo、Noto Sans CJK 和 DejaVu Sans 字体。默认字体路径记录在 [drawing.py](src/drawing.py) 中。在仓库根目录运行：

```bash
python3 -m pip install -r 20-tide-decentralized-neural-network/figures/requirements.txt
python3 20-tide-decentralized-neural-network/figures/render.py
```

也可以只生成一组或一张：

```bash
python3 20-tide-decentralized-neural-network/figures/render.py --group legacy
python3 20-tide-decentralized-neural-network/figures/render.py --group local
python3 20-tide-decentralized-neural-network/figures/render.py --only kv-memory
```

`--output-dir` 可将结果写到其他目录。三张旧图和局部语义图均同时导出 PNG 与可编辑 SVG；旧图沿用原始画布，局部语义图为 2400 × 1350。SVG 保留文字和形状，定义出处见本页及各图来源记录。编辑 SVG 可直接制作演示变体；持续修改宜同步到生成脚本，重新运行会覆盖同名输出。

### 用鼠标修改旧图

用 Inkscape 打开上表的 SVG，按 `T` 使用文字工具修改字词，按 `F1` 使用选择工具移动、缩放对象或改变颜色。箭头的线段和箭头头部已经编组，可以整体移动；节点、标签和相邻连线仍是独立对象，移动节点后需自行调整连线端点。需要一起移动时，可多选后按 `Ctrl+G` 编组。

SVG 使用 Noto Sans CJK SC 字体；编辑机器安装该字体可减少排版差异。手工维护时可另存到 `figures/manual/` 并作为源文件跟踪，避免运行脚本时覆盖。若希望继续由脚本重绘，将文字、坐标或颜色同步回对应的 Python 脚本；采用手工图源时，让展示构建读取该文件。

| 源码 | 维护内容 |
|---|---|
| [src/legacy/](src/legacy/) | 结构图与 P/S/U/F 两图的绘制内容、坐标和配色 |
| [svg_export.py](src/legacy/svg_export.py) | 同步保存旧图的 SVG 文字与图元；PNG 继续使用原 Pillow 绘制 |
| [local_figures.py](src/local_figures.py) | 带权消息流程与两个算例 |
| [memory_figures.py](src/memory_figures.py) | KV、仿射／Delta、Gated DeltaNet、SSD |
| [control_time_figures.py](src/control_time_figures.py) | 发送规则、温度和固定时间槽 |
| [drawing.py](src/drawing.py) | 配色、字体、SVG 文字、形状与 PNG 导出 |
| [catalog.py](src/catalog.py) | 图名注册与定义路径；具体出处及条件与各绘图函数相邻保存 |

完整重绘后，可加 `--record-provenance` 生成本地的[局部图渲染报告](local-render-checks.json)，包含每张图的定义章节、条件、教材与源码哈希、字体、渲染版本及输出哈希。该报告默认忽略；已有[历史核验快照](local-provenance.json)继续留存，正常构建不覆盖。来源和复现核验记录见 [PROVENANCE.md](PROVENANCE.md)。
