# Tide 主干拓扑与规模面板

打开[离线面板](generated/index.html)。页面内嵌图数据、交互代码、MathML 与阶梯图；无需网络。源码和定义入库，HTML、SVG、PNG 及数据报告由统一构建生成。

共用一个入口，按拓扑与时间选择候选，再选择尺寸和参数配置。多跨度 Timed／Settle 在连接视图和规模筛选中分别列项，并可保留同一空间图切换时间。连接视图覆盖三类 HB，规模筛选另包含 FC；FC 连接视图与 LH 探索图尚未接入。三类空间图的范围为：

| 空间图 | 面板尺寸 | 用途 |
|---|---|---|
| 多跨度 HB | 原 5／5-dd4／7／9／13／15／17 层宽图；窄图 5-dd4／7／9／双入口九层／15 层，每区 8／16／32，可选 4 hub | Timed、Timed 与 Settle 同周期的机制对照、Multiscale-Settle |
| 原 HB-Lattice | 5／7／9 层，16，无 hub | 原递归展开收拢，保持 $D=L+2$ |
| Settle 位交换 | 原 7／10／13 层；另有 m 与 R 分开的窄平台，每区 8／16／32，可选 4 hub | 对齐计算的逐轮合作，可切换真实 hub 增强图 |

共 58 个空间配置。含 hub 的配置在每个非末层位置设置 D 区与独立 H 区，末层只有 D。区域图可筛选边族，单节点图完整列出上下游，矩阵按两端实际节点数显示：例如 n=32 时，无 hub 为 32×32，含 hub 的内部位置为 36×36，进入末层为 36×32；公式侧栏随当前选择代入。空间选择改变模型，显示筛选只隐藏连线。

初次阅读可配合[九层连线算例](../../topologies/hb-multiscale.md#31-从位置对应算到节点端点)：选择多跨度 HB 的原 9 层宽图（16+4），查看位置 $(2,1)$ 和节点 $D_{2,1,7}$；端点矩阵选择 $(2,1)\to(3,2)$ 或 $(2,1)\to(4,6)$，即可对照 DD1、DD2 的计算。再选 $H_{2,1,1}$ 查看同例的 HD2 探针。

多跨度图的时间选择同步更新区域边标注、节点时延、端点矩阵与到达表：Timed 单位时延／T=4、Timed 单位时延／与 Settle 同周期、Settle 跨度时延／D=M+1。前两项是 Timed 的两种输入日程。原 HB 与位交换图固定各自 Settle 绑定。层数相位阶梯单独标明是单位时延／T=4 的固定参考，Settle 表则给出对齐时间与宽段潜在来源覆盖。潜在路径不等于实际选择、无损内容覆盖或物理低时延。

规模筛选分别列出 FC-Settle、HB-Lattice-Settle、HB-Exchange-Settle、HB-Multiscale-Timed 和 HB-Multiscale-Settle，支持 n=8／16／32／64／128、d=64 至 768、K／S／X、independent／region_shared／layer_shared，以及两个共享档的适配秩 4／8／16。分配独立选择 quad／Shannon／pool，约束预算只选合法的 1、2、4、8、16、32、64；pool 禁用数值预算；显示总参、预算比例和活跃参数代理，超过 256M 的行也保留。规模筛选不修改上方连接图；有已绘制连线的行可点击“查看”。“按上方连接图筛选”和“查看”均传递空间轮廓与时间绑定，Timed 的同周期设置也随之保留。FC 每层一区，面板标注两个共享档等价；CSV 保留各档标识供筛选，实验无需重复。HB 默认筛选区域共享。

接收配置可选来源残差、两种轻量自注意力（$d_A=64$／128、$p_a=d/8$）和加宽自注意力（$d_A=p_a=d$），也可并列显示全部四档。界面展示所选尺寸公式，每行总参已计入该档全部接收参数；256M 筛选及活跃代理使用这一完整计数。连接图跳转保留接收选择，图的边与时间不由接收尺寸改变。

完整 CSV 共 676,200 行，按三种 allocation_profile 分行；页面去重保留 225,400 份参数配置。profile 带时间后缀，spatial_profile 标识共用的空间轮廓，另列层数、层宽、位置数和主输入周期；agg_profile 标识接收尺寸配置，另列 agg_mode、MLP 宽度及注意力宽度、头数、每头维度。region_profile 标记 encoded_self_attention，另列描述宽度、区域注意力头数及 FFN 宽度；角色约束显式使用 received／interval。参数列以 _M 结尾，单位为百万参数。CSV 保存主周期，面板另可选择 Timed 的同周期对照。内嵌数据与范围判断保留精确计数。字段解释见[比较绑定 §6](../../instances/backbone-comparison.md#6-交互与复算)。候选覆盖不同需求组合；提出新配置或新构造时，可从[主干候选的形成与取舍](../../instances/backbone-design.md)接续。

正式语义入口：

- [局部接收自注意力](../../local-semantics/weighted-message/architecture-learning-note.md#23-同刻消息的局部自注意力)及[四个接收配置](../../instances/hb-multiscale-weighted-message.md#11-接收自注意力的完整替换)。轻量与加宽档使用相同函数，尺寸和准备成本的差异见[比较绑定 §5.1](../../instances/backbone-comparison.md#51-接收自注意力的参数与准备成本)。
- [共同拓扑、全部尺寸、分区与 hub 放置](../../topologies/hb-multiscale.md)。其他配置已有构造定义，暂不都提供界面切换。
- [位交换拓扑与覆盖证明](../../topologies/hb-exchange.md)、[Settle 三图共同绑定](../../instances/hb-settle-weighted-message.md)、[三档整体参数绑定](../../instances/parameter-profiles.md)。
- [统一主干比较绑定与计数口径](../../instances/backbone-comparison.md)、[完整 CSV](generated/capacity.csv)、[生成范围表](generated/capacity-summary.md)。
- [K／S／X 三组主局部绑定](../../instances/hb-multiscale-weighted-message.md)。显示筛选不会隐式切换模型。
- [角色上下界](../../local-semantics/weighted-message/region-allocation-learning-note.md#415-带角色上下界的二次分配)、[设计问题与取舍](../../instances/backbone-design.md)。
- [逐候选核对数据](generated/checks.json)、[9 层区域图](generated/regions-9.svg)、[节点矩阵](generated/node-pair-matrix.svg)、[层数阶梯](generated/depth-ladder.svg)。

## 在另一台机器构建

当前构建使用 Linux 的系统图形库及字体路径；Debian／Ubuntu 或 WSL Ubuntu 可按以下步骤操作。使用 Python 3.10+，先更新仓库，再在仓库根目录运行。首次安装系统依赖：

```bash
sudo apt-get update
sudo apt-get install -y python3 librsvg2-2 libcairo2 fonts-noto-cjk
```

仅构建面板（无需构建其他图集或静态演示 PDF）：

```bash
python3 tools/build.py --group backbone-atlas
python3 tools/check_artifacts.py --require-built --group backbone-atlas
```

成功后用现代浏览器直接打开 `20-tide-decentralized-neural-network/explorers/backbone-atlas/generated/index.html`，无需启动 HTTP 服务。页面内嵌交互所需数据；CSV 下载及正文链接需要保留对应目录。产物不随 Git 同步，拉取后按维护地图检查状态并重建本组。

### 从旧目录升级

2026-11-15 起，面板从 `explorers/hb-timed/` 迁到本目录，构建组从 `hb-topology-atlas` 改为 `backbone-atlas`。Git 更新只迁移维护源；旧机器的 `hb-timed/generated/` 可能仍在。先检查旧目录内容，将旧目录整体移至仓库外保存，再使用上方新命令构建。以下命令从仓库根目录执行，只移动这个旧目录，不删除其中的手工文件或生成物：

```bash
if [ -d 20-tide-decentralized-neural-network/explorers/hb-timed ]; then
  tide_atlas_backup=$(mktemp -d ../tide-atlas-before-move.XXXXXX) &&
    mv 20-tide-decentralized-neural-network/explorers/hb-timed "$tide_atlas_backup/" &&
    printf '旧目录已保存到 %s\n' "$tide_atlas_backup"
fi
```

旧书签请改为本页的新地址；公共构图和计数源的迁移见[路径记录](../../records/history/migration-map.md#2026-11-15-主干面板与参考计算)。置换种子与配置标识保持原值，目录迁移不改变连接或参数数目。

Conda Python 可以调用统一构建器。PNG 渲染由 `tools/native_svg.py` 启动 `/usr/bin/python3 -I -S`（或 `/bin/python3`），只在该子进程中清除 LD_LIBRARY_PATH、LD_PRELOAD、LD_AUDIT、PYTHONHOME、PYTHONPATH，并使用系统 PATH。这样避免系统 Pango／Fontconfig 与 Conda 已加载库混用，调用者环境和其他构建组的 Python 选择不变。依赖预检在相同隔离环境中实际加载图形库，而非只查找库名。

若旧版本出现 `libpangoft2 ... undefined symbol: FcConfigSetDefaultSubstitute`，先更新仓库。临时也可用下面的命令绕过调用者的 Python 和库路径：

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD -u LD_AUDIT -u PYTHONHOME -u PYTHONPATH \
  /usr/bin/python3 tools/build.py --group backbone-atlas
```

## 维护与核对

从仓库根目录生成并核对：

```bash
python3 tools/check_artifacts.py --affected 20-tide-decentralized-neural-network/reference/backbones 20-tide-decentralized-neural-network/explorers/backbone-atlas
python3 tools/build.py --group backbone-atlas
python3 tools/check_artifacts.py --require-built --group backbone-atlas
python3 20-tide-decentralized-neural-network/examples/hb_lh_checks.py
python3 20-tide-decentralized-neural-network/examples/hb_settle_checks.py
python3 20-tide-decentralized-neural-network/examples/hb_compact_checks.py
python3 20-tide-decentralized-neural-network/examples/backbone_capacity_checks.py
python3 20-tide-decentralized-neural-network/examples/incoming_attention_checks.py
python3 20-tide-decentralized-neural-network/examples/role_bounded_allocation_checks.py
python3 20-tide-decentralized-neural-network/explorers/backbone-atlas/check_browser.py
```

构建使用 Python 标准库、系统 Python 及系统 librsvg、Cairo、GObject，字体采用 Noto Sans CJK 或本机已有的 Droid Sans Fallback；不依赖 Python 的 Pillow／CairoSVG。Debian／Ubuntu 可安装 `librsvg2-2 libcairo2 fonts-noto-cjk`，GObject 由依赖带入。浏览器核对另需 Firefox，通过隔离 profile 验证 58 个配置及参数预算筛选、区域跳转、公式映射、矩阵端点数量、末层角色与矩形矩阵、g=1 边界、时间同步及宽窄布局；报告和截图默认写入仓库外临时目录，不进入日常 Git 产物。仓库 CI 在 push／pull request 时构建本组并执行浏览器核对。

核对也逐行比较 CSV 的 M 单位、接收尺寸与页面精确计数，检查四档接收与三种分配、pool 的预算禁用、区域尺寸、Timed／Settle 成对配置、合法预算空值，以及连接图与参数筛选之间的时间绑定、输入日程和接收选择传递。浏览器另核对轻量／加宽的完整计数、维度公式、规模过滤与窄屏显示。

两个带 hub 的构造器默认省去末层 H；`terminal_hubs=True` 可复现此前包含它们的图，面板展示默认图。删除对 group／singleton 保留节点与输出的影响，以及 mixed 的差异，见[共同拓扑 §1](../../topologies/hb-multiscale.md#1-节点位置与区域)。

[公共参考代码](../../reference/backbones/README.md)维护多跨度／Lattice／Exchange 构造、潜在传播与主干参数计数，数学核对程序和本面板共用。`build_atlas.py` 绘制图并生成页面与图数据，`capacity_reports.py` 将计数结果写为 CSV、范围表及页面数据；`capacity.js` 维护规模筛选，`page.html`、`style.css`、`explore.js`、`formulas.js` 维护展示。先核对定义、数值例子、图示与界面，再生成；详见[维护地图](../../../ARTIFACTS.md)。构建不覆盖本 README。
