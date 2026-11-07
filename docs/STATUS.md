# 当前交接 — 分支与研究图原型（2026-11-07）

用户授权将研究稿从基座移至同一tide仓库的`tide-learning-lab`独立分支，并以图形继续对齐。
基座误放的未推送提交`0510e0d`已撤销，`graph-execution-foundation`恢复为`85d161d`；
该分支文件树没有研究稿、没有新增实验待办。数学仍固定`0770593…`。

本分支建立研究稿、来源选择和可离线浏览的研究图。图中实验均拟议，两个结果位置均待填，
可靠checkpoint尚未选定；唯一已有证据链接基座有限验收，保留综合根失败。
未安装实验环境、下载模型、申请设备或启动训练。当前没有本分支活动作业。

原型实现`902b81a`的[设施验证](verification.md)：5项证据边界检查与真实Firefox交互通过，
生成文件/语法/文档链接通过，浏览器已停止且无遗留；不属于模型实验资格。
研究分支普通推送到同一tide仓库，基座分支不需要推送或强推。

下一步与用户对齐图示结构、初始基线、首组问题、比较和资源档位；训练/推理待统一授权。
本分支不继承基座已有任务的运行授权，不恢复旧矩阵。

```sh
git status --short --branch
python scripts/render_map.py --check
python -m unittest discover -s tests -v
```

浏览`web/index.html`。设计见`docs/research-design.md`，图示见`docs/research-map.md`，
待办仅在ROADMAP；来源声明不是安装锁，实际环境与下游验收后续再建立。
