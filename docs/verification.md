# 研究图验证

## 2026-11-08：语言与结构主线重排

当前变更的生成输入与校验器同本文件所在提交保存；以下仅认证设计记录与离线图，不认证语言模型训练或设备执行。

- 图为55节点/68关系，默认20个当前节点；六个新研究组planned，没有run ID，四个新结果位awaiting。
- 14项研究图约束检查通过，含历史证据显示分区边界、状态与科学支持、来源/范围、关系环及私有URL检查。
- `render_map.py --check`与JS语法检查通过；HTML与当前数据/模板一致。
- Firefox实际打开离线文件：当前/历史/全部切换、视口切换、跨视图查看历史依据、结果筛选、待填提示、搜索、节点/边详情、Esc、重置、缩放通过，无控制台错误。
- 截图目视核对当前联合主线布局；截图在忽略的`artifacts/research-map.png`，不是运行依赖。
- 预算静态核对：60+180+120+60+60=480NPU分钟，与拟议总额8NPU小时一致；所有阶段共享12h墙钟，不重置。计划JSON不可提交。
- 与重排前HEAD c0464fa逐项核对：既有run对象、来源、scope、比较、反证、未知解释和迁移条件保持；历史12个scoped研究结果不升级。
- 当前文档/节点的本地链接存在，Git whitespace检查通过。没有改动运行目录、固定作业源码、依赖安装或Lens服务。

```sh
python scripts/render_map.py --check
python -m unittest discover -s tests -p test_research_map.py -v
node --check web/map.js
node --experimental-websocket tests/browser_smoke.mjs
git diff --check
```

## 2026-11-07：原型验证（历史记录）

下文的节点/结果数量只描述当时固定实现，不能作为当前图统计。

日期：2026-11-07。固定实现：`902b81a0eb1ef8c0542a78af36677f496442df52`。
本记录只认证离线研究图的所列功能，不认证任何模型、任务或设备执行。

- Python标准库检查5项通过：待填结果不能支持结论、实际证据必须有来源/范围、
  删除端点拒绝、前提依赖环拒绝、不安全来源URL拒绝。
- `render_map.py --check`通过：数据关系有效，HTML与生成输入一致。
- Node v20.18.3语法检查通过。
- Firefox 140.10.0esr实际以file方式打开：15节点/18关系渲染、结果筛选、
  待填结果说明、搜索、节点/关系详情、Esc、重置与缩放通过；浏览器控制台无错误。
- 截图经目视核对，默认适配可见完整图。截图为可再生成的忽略产物，不是运行依赖。
- 浏览器检查有40秒时限、私有临时profile和进程组停止；退出0，未发现其遗留Firefox进程。
- 文档相对链接、Git whitespace检查通过。

```sh
python scripts/render_map.py --check
python -m unittest discover -s tests -v
node --check web/map.js
node --experimental-websocket tests/browser_smoke.mjs
```

浏览器检查使用已安装Firefox/Node，不安装系统软件、不启动训练、不打开NPU/GPU。
本图新研究结果为0，两个结果位置为待填；已有基座证据只按固定引用解释。
自动实验结果导入、排期、预算计算和迁移结论审核仍未实现，见ROADMAP。
