# 当前交接 — 2026-10-30

授权分支 `graph-execution-foundation`；数学固定 `0770593838a37c790a568226930567ed60f99b83`。
用户已授权本轮审视后的修正、分阶段提交和普通推送；不使用子代理。
参考仓库和公共工具链只读，GPU延后。本轮仅修正测试覆盖、配置事实和证据导航，不启动新性能实验。

## 当前工作

起点 `0a250fa905d316eff7afe6781ec7183172cc9c78`；未提交变更均为本轮修正。
已补真实FC层间边、FC混合KV/GDN/SSD以及教材配置断言，计划26种recipe分别用PyTorch与Python-native训练。
已实现完整节点/区域配置摘要、按源码/模型/工作负载/执行身份查询，以及JUnit实际检查提取。
已更正旧报告的KV逻辑时间窗口与来源分类/参数共享说明；旧25性能配置中7通过及原失败不变。
短检查9通过。首轮开发作业52失败/21通过：新增模块集合断言被同名参考结果覆盖；未改执行器/容差，已修正变量。原失败和源码归档保留。修正后开发检查73通过，其中52项为真实recipe训练；尚未固定提交资格。

## 提交与下一资格

开发作业已终态：`core5-coverage-dev` failed/1（52失败/21通过，测试变量重名）；
`core5-coverage-dev-v2` passed/0（73通过）。各自位于`artifacts/jobs/<名称>/`，保留task.log、status.json、development.json、tests.xml及source.tar.gz/hash；旧失败不改写。
实现与相关契约准备提交；下一步在此实现提交的全新只读checkout进行同范围资格，不改变数学、dtype或容差。

计划固定树`artifacts/worktrees/core5-coverage-qualified`，新任务`artifacts/jobs/core5-coverage-qualified`，
服务`core5-coverage-qualified-20261030.service`（尚未启动）。独立background.slice，1200s、CPUQuota=3200%、MemoryMax=128G、Nice=10；
ATen/BLAS线程1、native workers2。无C++改动；再次核验匹配源码和全部binary hash后复用b255 CPU构建。

```sh
git worktree add --detach artifacts/worktrees/core5-coverage-qualified HEAD
# 标记固定树源码只读，再经module与项目Python在独立服务中运行：
python scripts/job.py --output-dir /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/jobs/core5-coverage-qualified -- python scripts/develop.py --output-dir /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/jobs/core5-coverage-qualified --reuse-build --build-dir /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/worktrees/core5-checkpoint-qualified/build/core-cpu -- tests/test_core5_recipe_training.py tests/test_core5_coverage.py tests/test_core5_study.py tests/test_core5_planning.py --junitxml=/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/jobs/core5-coverage-qualified/tests.xml -o junit_family=legacy
systemctl --user show core5-coverage-qualified-20261030.service -p ActiveState -p Result -p ExecMainStatus -p MainPID -p Slice -p ControlGroup
```
停止命令`systemctl --user stop core5-coverage-qualified-20261030.service`。
固定资格通过后从JUnit提取实际52项证据，补索引/导航和经验，单独提交证据并普通推送；不恢复旧队列。

## 既有结果与边界

- [自然规模有限测量](evidence/core5-natural-study-20261029.md)：25配置7完整通过，8B/16B仅静态计划。
- [客户端资格](evidence/core5-study-clients-20261029.md)：保留原计数，附旧FC覆盖退化的纠正说明。
- [checkpoint预算](evidence/core5-checkpoint-budget-20261029.md)：大文件保存/加载通过，原1B下一窗非有限梯度未解决。
- [ROADMAP](ROADMAP.md)保留数值、异步、resident资源与设备待验的停止/重启条件；本轮不扩大这些资格。
- 代码/配置/修改维护映射见[core5-study](core5-study.md)，经验见[performance-lessons](performance-lessons.md)。
