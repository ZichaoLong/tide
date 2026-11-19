# 图片来源与复现记录

以下保存各批图片最初生成时的来源与核验结论。2026-10-28 起按[仓库约定](../../ARTIFACTS.md)跟踪源码及历史证据，图片在本地重建；下述历史中的“保存图片”描述对应当时的工作。普通重绘报告写入忽略的 `local-render-checks.json`，已有来源快照保持。

## 2026-10-27：恢复已有图片

三份绘图脚本从当前会话 2026-10-12 至 2026-10-13 的原始绘图调用恢复，并按顺序应用后续补丁。恢复保留了绘制内容；仅将固定的临时输出路径改为脚本所在仓库目录，并增加 `--output-dir` 参数。

| 用户保存的原图 | 仓库图片 | 恢复源码 |
|---|---|---|
| `~/tide-structure.png` | [tide-structure.png](tide-structure.png) | [tide_structure.py](src/legacy/tide_structure.py) |
| `~/PSUF-simple.png` | [PSUF-simple.png](PSUF-simple.png) | [psuf_simple.py](src/legacy/psuf_simple.py) |
| `~/PSUF-complete.png` | [PSUF-complete.png](PSUF-complete.png) | [psuf_complete.py](src/legacy/psuf_complete.py) |

三份恢复脚本重新生成的 PNG 均与用户保存的原图逐像素、逐字节相同。原图及恢复前源码的 SHA-256、所用字体与渲染版本见 [legacy-provenance.json](legacy-provenance.json)。绘图源码和 PNG 均保存在仓库中，复现不需要会话记录。

旧图以 Pillow 和 Noto Sans CJK Regular／Bold（TTC 的索引 2）绘制。字体文件默认来自 `/usr/share/fonts/opentype/noto/`。相同源码在其他字体或渲染器版本上可能产生不同的排版或像素；图中数学含义仍以引用教材为准。

## 2026-10-28：旧图的可编辑 SVG

从上述三份绘图脚本新增导出 [tide-structure.svg](tide-structure.svg)、[PSUF-simple.svg](PSUF-simple.svg) 和 [PSUF-complete.svg](PSUF-complete.svg)。共用的 [svg_export.py](src/legacy/svg_export.py) 在执行原 Pillow 绘制时，将同一批文字、形状和连线保存为 SVG 图元；文字保留为可编辑的 `<text>`，每支箭头的线与箭头头部编为一组。

三个 SVG 沿用原图的内容、坐标、尺寸与配色，直接包含矢量对象。文字基线由原字体的度量转换；不同渲染器仍可能产生细微的字形间距和抗锯齿差异。SVG 依赖编辑机器安装 Noto Sans CJK SC 字体。

在仓库根目录执行 `python3 20-tide-decentralized-neural-network/figures/render.py --group legacy` 可同时重绘三组 PNG／SVG。原 [legacy-provenance.json](legacy-provenance.json) 继续保存恢复时的源码及原 PNG 哈希；此次只增加导出接口，原图的绘制内容与数学含义保持。

核验：三份重绘 PNG 的 SHA-256 均与恢复记录中的原图一致；SVG 分别保留 50、30、57 个文字对象，无嵌入位图或外部资源链接。使用 Inkscape 1.4.3 将三张 SVG 导出为与原图相同尺寸的 PNG，逐图并排检查布局、文字、颜色与箭头。文档链接、正典哈希和差异空白检查通过。

## 2026-10-27：局部语义图

新增十张局部语义图，每张同时保存可编辑 SVG、PNG 和 Python 生成函数。定义来自共同局部架构、区域分配、KV 及三种循环记忆与短卷积教材。各图的用途、选定条件、定义章节和逐文件 SHA-256 见 [local-provenance.json](local-provenance.json)；图集中的链接供顺序阅读和查阅。

本批定义对应提交 `6517ac525bce57bd20ea8f22753407d2fc331982` 中的教材内容。生成时使用 Python 3.14.4、Pillow 12.1.1、CairoSVG 2.8.2、Cairo 1.18.4，以及 Noto Sans CJK SC 与 DejaVu Sans。SVG 保留文字对象，字体文件及哈希列在记录中。图中上下标由生成器定位，中文与数学字符分别使用相应字体。

核验结果：

- 十张图均对照定义审读并目视检查；修正了完整内容的激活条件、KV 窗口示意和字体缺字。563 个 SVG 文字片段的全部字符在指定字体中都有对应字形。
- 指定总量算例与三个温度算例，均与现有区域参考程序的标量求解、独立可行面枚举两种结果一致。
- 固定时间槽例子使用现有卷积函数核对，包括零权重与趋弱接收；KV 两次接收例子的摘要、硬选择和残差输出逐项核对。
- 在独立输出目录重绘后，十份 SVG、十份 PNG 均与仓库输出逐字节相同；三份恢复图仍与原图哈希一致。
- 文档检查通过：正典教材哈希保持、局部公式结构及相对链接有效；图中记录的全部定义锚点另行核对。`git diff --check` 通过。

以后修订图中的数学内容时，同步核对所引用的定义与图中条件。完整重绘并审读后，用 `render.py --group local --record-provenance` 生成新的本地渲染报告；需要留存某次核验时，以独立快照登记，保留原有来源记录。
