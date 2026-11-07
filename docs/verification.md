# 研究图原型验证

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
