# TIDE Learning Lab

这是 `ZichaoLong/tide` 的独立实验分支，负责模型训练与推理研究、对照和评价。

**长期目标：在同一个模型上，让自然语言 next-token 质量接近合理调参的对照，同时推进数据、容量与依赖深度扩展、低活动和局部交互。**

语义和实现围绕中短期研究问题持续迭代。每批固定所用版本，无需先完成统一架构或通用基座。

当前处于新周期准备阶段：没有活动实验；语义、基座、数据、配方及预算尚未选定。
旧倾向和记录不进入默认上下文，仅在用户明确要求回顾时查找。

- [新线程接续说明](docs/NEW_CYCLE.md)
- [研究章程](docs/RESEARCH_CHARTER.md) · [当前状态](docs/STATUS.md)
- [研究表模板](docs/research-design.md) · [当前待办](docs/ROADMAP.md)
- [当前研究图](web/index.html) · [图的维护](docs/research-map.md)
- [运行入口状态](docs/RUNNING.md)

研究图是自包含的离线页面，无需安装模型环境或访问网络。当前只展示目标和待确定事项，未载入历史结果。
重新生成与检查使用 Python 标准库：

```sh
python scripts/render_map.py
python scripts/render_map.py --check
python -m unittest discover -s tests -p test_research_map.py -v
```

后续执行范围由新周期研究表和当次用户授权确定。
