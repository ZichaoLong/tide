# Tide 展示材料

从 Transformer 的 residual add 出发，展开一个显式 MoE，走完数值计算，再讨论局部网络、历史、接口与成本。共 **14 组画面、19 个停点**；最后两组是 residual 与 HB 附页。主讲到第 12 组结束。

- [离线 HTML](index.html)：方向键／空格推进，N 打开备注与出处，G 打开目录，也支持按钮和滑动。
- [PDF 浏览版](tide-design-introduction.pdf)：19 页，每页对应一个停点。
- [六幅画面预览](preview.png)：Transformer、MoE 映射、完整结果、局部网络、历史和设计反馈。
- [讲者备注](speaker-notes.md)：每个停点的条件、解释与定义链接。

以上文件由维护源生成，默认不随 Git 跟踪；首次克隆按下方命令构建。HTML 内嵌文字、图形和交互，无网络依赖。定义链接指向仓库相对位置，单独打开 HTML 仍能看画面与备注，原文需回到仓库查阅。

## 讲述与数值来源

[研究意图与探索路径](../../memos/design/design-requirements-and-architecture.md)保存研究缘由与机制取舍，[讲述脚本](script.md)保存叙述安排。[显式残差 MoE](../../instances/explicit-residual-moe.md)给出新增教学绑定：每个专家一个节点，选择、贡献合并和残差旁路分别表达。首轮 $y=4$；增加本地激活计数后重复同一输入，第二轮 $y=8$。这是机制核对，不是训练或性能结果。

HB 在第 4 组用于拓扑示意，第 14 组提供独立的[既有数值例子](../../instances/hb-lattice-weighted-message.md#2-三层图上的分叉汇合与记忆)。新例子不沿用 HB 的 $G=u$、KV 写入或 $10/4/8$ 计数。

## 小修改

地址末尾的 `#3.3` 表示第 3 组第 3 个停点，对应本版 PDF 第 6 页。提修改时可使用停点编号或标题，并说明作用范围，例如“3.3，把结果公式放大”或“所有 MoE 图，把残差旁路下移”。内部还为每个停点保存稳定 ID，重排后导出预览不依赖固定页码。

| 想修改什么 | 维护入口 |
|---|---|
| 标题、屏幕说明、备注、引用和顺序 | [content.py](content.py)；查找原句或 `step` 的 ID |
| Transformer、MoE、HB 的文字和图形位置 | [图形源](../../figures/src/design_introduction.py)；分别修改 `transformer`、`moe`、`graph` |
| 页面拼装与示例轨迹接入 | [build.py](build.py) |
| 字体、间距、颜色和布局 | [deck.css](deck.css) |
| 导航、目录、快捷键 | [deck.js](deck.js) |
| 数值函数、时间与检查 | [算例程序](../../examples/explicit_residual_moe_checks.py)，同步修改实例正文 |
| PDF、预览与浏览器检查 | [export.py](export.py) |

持续修改应落在维护源；重新构建会覆盖 HTML、备注、PDF 与生成的 SVG。当前页面是播放器，尚未提供直接改字或拖动并保存的编辑界面。若改用手工 SVG，应另存为源文件、让生成器读取，并按[仓库约定](../../../ARTIFACTS.md)登记。

## 内容与复现

在仓库根目录执行：

```bash
python3 tools/build.py --group presentation-build
python3 tools/build.py --group presentation-export
```

前者生成 HTML、备注和[分步 SVG](../../figures/design-introduction/)，构建时运行数值检查；后者先重建依赖，再导出 PDF、预览和[渲染报告](render-checks.json)。两者记录统一构建指纹。只改文字并预览 HTML 时先运行前者；交付静态版前运行后者。

HTML 构建只需 Python 标准库。导出需要 `playwright-cli`、Chrome、Pillow 与 Noto CJK 字体，临时使用回环地址服务器及独立浏览器配置。画面为 1600 × 900；所有停点分别导出，检查翻页、备注、目录、重看、小屏适配、页面和 SVG 文字边界。版式检查之外还要目视核对箭头、公式与图形层次。

```bash
python3 20-tide-decentralized-neural-network/examples/explicit_residual_moe_checks.py
python3 tools/check_artifacts.py --require-built --group presentation-export
python3 tools/check_links.py
python3 20-tide-decentralized-neural-network/tools/check_documents.py
```

完整依赖与同步流程见[全仓维护约定](../../../ARTIFACTS.md)。底层生成器的单独运行不更新统一指纹；拉取或切换版本后按清单检查本地产物是否过期。
