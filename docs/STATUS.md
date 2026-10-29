# 当前交接 — 2026-10-29

授权分支 `graph-execution-foundation`；数学固定 `0770593838a37c790a568226930567ed60f99b83`。
用户已授权实施新一轮候选配置、拓扑/规模、后端和有限性能设施；可提交与普通推送。
不使用子代理。参考仓库与公共工具链只读，GPU真机延后。构建允许全部或一半可用CPU；
测试可使用空闲CPU/NPU/内存，项目重型计时不与构建重叠。

## 当前工作

启动审计通过：实际路径 `/mi/data2T/zlong/home/llm/graph-execution-foundation`，
起点 `86826fd09bb3546b5084a89b494b4bbdd53d8510`，工作区起初干净；固定快照19+5份通过。
已统一所有构建入口的 `--jobs all|half|N`，删除历史两线程及1/2/4限制。
定向检查 `tests/test_build_resources.py`：7 passed；包括无Torch帮助入口、CPU affinity、half与320显式线程。
随后复用现有公共绑定和参数清单，建设数学配置/执行配置/测量计划的独立身份，
生成遵循core-5的FC/HB/LH启发拓扑和约1B/8B/16B的自然维度用例，补齐比较入口与测试。
每个合法候选可配置，单项与重要组合有检查；不要求遍历笛卡尔积。

## 活跃任务：无

没有启动新长任务。旧任务均终态，未恢复旧队列。最近生产实现是
`04b52153f9785e1489e31c7dd09a16849791d331`，其资格与结果见
[紧凑纤维证据](evidence/core5-compact-fibers-20261029.md)；此前的两线程是真实历史记录。
该证据包括CPU两环境各10、Python/NPU56、SDK7组件检查，以及medium完整窗
2.794→2.470s、内存514.8→376.4MB的一次观察。D64 resident因CANN流资源失败。
原始记录仍在 `artifacts/jobs/core5-compact-fiber-{v2-python,v2-sdk,study}`；
两个资格passed/0，study failed/1；不把它们改为本轮资格。

## 未解除限制与下一步

[ROADMAP](ROADMAP.md)保留D64/D2048/反馈FP32、FP16 GDN/AdamW、默认异步双NPU、
晚注册NPU、resident流资源、容量/正支持及CUDA/x86待验的独立状态和重启条件。
新规模选择不再保持core-3节点数或压缩FFN匹配参数量；旧对照和严格失败保留。

接续：`git status --short --branch`、`python scripts/status.py`、
`python scripts/check_upstream.py`，再看本页及ROADMAP。
当前下一步：实现公共配置与计划身份、core-5自然维度和拓扑生成。
相关长期契约见 [execution-flows](execution-flows.md)，经验见 [performance-lessons](performance-lessons.md)。
