# 当前交接 — 2026-10-30

授权分支 `graph-execution-foundation`；数学固定 `0770593838a37c790a568226930567ed60f99b83`。
本轮覆盖/配置/证据导航修正及分阶段提交、普通推送已授权。无子代理；参考仓库和公共工具链只读；GPU延后。
构建默认全部可用CPU或half，测试使用空闲资源。本轮未新增长性能实验。

## 当前结果

实现固定 `0f4506d097cbae88d4cd7f1c9d31319a2e0a1f4a`，固定提交相关资格已终态：
**73 passed，0 failed，0 skipped**。本页与证据一起提交；无未提交执行器变更、活跃作业或自动重试队列。

- [覆盖修正资格](evidence/core5-coverage-correction-20261030.md)：26种真实FC/HB/LH配置×PyTorch/native，共52项CPU FP64 packed prefill训练；两步独立完整核对和第三窗磁盘换owner精确恢复。其余21项为planning/目录/配置事实/查询检查，不是仓库全矩阵。
- 旧FC测试的depth=1、mixed实际只有KV已更正；新FC至少两层、mixed八层含KV/GDN/SSD，实际拓扑、配置和执行模块都有断言。旧24项通过数保留并缩小解释范围。
- 自然规模窗口与SrcClass/SourceBinding说明已更正；从所有节点/区域配置生成事实，不取首节点代替全图。
- [逐配置机器索引](evidence/core5-index.json)连接52个正确性和26个性能记录（25原测量+1失败checkpoint后续），保留各自源码/模型/工作负载/执行身份；不合并成一次门禁。正确性测试不产生吞吐。
- 已验证无Git/build/artifacts的源码导出可用`python -S`查询全部78条记录，且不改变失败或资格标志。
- 新checkout入口与修改维护映射见[core5-study](core5-study.md)；LH/core-3及本轮正负经验统一在[performance-lessons](performance-lessons.md)。历史仅有汇总的资格仍从[证据目录](evidence/README.md)进入。

## 原结果与未解除边界

[自然规模测量](evidence/core5-natural-study-20261029.md)仍为25配置7项完整通过；
NPU数值/资源失败、FC/HB1B训练尾部严格失败保持，8B/16B仅静态计划。
[大文件checkpoint](evidence/core5-checkpoint-budget-20261029.md)保留实际5.25GB保存/加载通过、原1B下一窗非有限梯度失败。
本次未增加standalone、mixed/resident、NPU/CUDA、FP32/FP16或大尺度/性能资格。
全部数值、异步、resident服务/在途资源、分项预算及目标硬件停止/重启条件由[ROADMAP](ROADMAP.md)维护；
没有具体新方案时，不重跑已停止的同配置大型任务。本轮修正已完成，没有待恢复队列。

## 固定作业与本地产物

| 任务 | 源码身份 | 终态 |
| --- | --- | --- |
| core5-coverage-dev | 0a250fa+归档补丁 | failed/1，52失败/21通过；新增测试变量覆盖错误，未改数学/容差 |
| core5-coverage-dev-v2 | 0a250fa+修正补丁 | passed/0，73项开发检查 |
| core5-coverage-qualified | 0f4506d，clean | passed/0，固定提交73项，119.591s |

目录为`artifacts/jobs/<任务>/`，含status.json、task.log、development.json、tests.xml与源码归档。
服务为`<任务>-20261030.service`；均已终态，成功瞬态服务可被回收，不能用不存在的unit推断通过。
固定树`artifacts/worktrees/core5-coverage-qualified`保持只读；b255的CPU native构建复用前后核验全部源码/二进制hash。
环境为`ascend/dev-workspace-9.1.0`、项目`.local/venv-core5`，Python3.12/Torch2.12；未调用NPU。
服务独立background.slice、Nice10、CPUQuota3200%、MemoryMax128GiB、1200s，ATen/BLAS1线程、native workers2。
原失败归档和日志保留；新checkout复现不依赖这些本地目录，命令和hash在报告及机器记录。

## 接续与查询

```sh
git status --short --branch
python scripts/status.py
python scripts/check_upstream.py
# 以下在Python>=3.10环境，无需Torch或旧运行目录：
python -S scripts/index_core5_study.py docs/evidence/core5-index.json --view coverage --topology fc --family ssd --kind correctness --format tsv
python -S scripts/index_core5_study.py docs/evidence/core5-index.json --view coverage --kind performance --state failed --format tsv
python -S scripts/index_core5_study.py docs/evidence/core5-index.json --view models --case fc-1b-kv --format json
```

查询按实际配置过滤；未出现的组合没有本索引资格，不自动解释为不支持。后续修改先看维护表和对应模块独立测试，
再按新方案做相关资格与有限测量，不将本轮小图资格外推为所有组合已验或低成本优化已耗尽。
