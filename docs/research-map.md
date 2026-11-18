# 当前研究图

图只载入当前周期的目标、问题、来源和经审阅结果。新周期尚未选定语义与实现，
所以当前节点仅表示目标与待确定事项，没有运行或结果。

旧图与旧倾向不嵌入默认页面，也不通过“显示全部”重新出现。
只有用户明确要求回顾时，才查找指定历史；不自动搬入新图作为前提或推荐。

`research/map.json` 是人工维护的数据，`web/index.html` 是生成的离线页面。
新增研究后，用统一研究组/run ID 关联配置、权威记录和查看工具。
计划、排队、运行、完成、失败、取消和未检查分别标记；作业完成不能自动生成科学支持。
结果应说明比较范围、反证、尚未区分的解释及迁移条件。

修改图数据使用 `scripts.review_io.write_review_json`，再运行：

```sh
python scripts/render_map.py
python scripts/render_map.py --check
python -m unittest discover -s tests -p test_research_map.py -v
python -m unittest discover -s tests -p test_review_io.py -v
```

页面自包含，不请求外网，不包含私有实例信息。
可选浏览器检查：`node --experimental-websocket tests/browser_smoke.mjs`；只使用已安装的 Node 和 Firefox。
