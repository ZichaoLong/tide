# 当前交接 — 2026-10-30

授权分支graph-execution-foundation；数学固定0770593838a37c790a568226930567ed60f99b83。
用户批准多实验仓库协作规范、模板、core-5安装起步示例、定向检查、分阶段提交和普通推送。
无子代理；参考仓库/公共工具链只读；GPU延后。本轮不改基座数学/执行器/容差，不新增性能测量。

## 实现与开发结果

起点ad684b7507e8cc3c9941e419c4bd5ea0c36de8a8。
[协作规范](downstream-contract.md)及模板已完成：固定依赖→请求→基座交付→下游验收/升级，唯一状态及已有授权边界保持。
[core-5示例](../examples/consumer_core5/README.md)已完成：实际公共配置/模块，应用数据/head/loss/SGD/StepLR，
依赖锁字节核验、完整应用原子checkpoint、新进程恢复与独立参考；只示范小型CPU接入。
AGENTS、README、library、development、core5-study和架构导航已接入。

开发第三轮通过：4项定向单元测试、8组仓库外安装消费者、4项错误请求拒绝。
八组为FC PyTorch/native×FP32/FP64，HB mixed PyTorch/native FP64，FC scalar streaming PyTorch/native FP64。
native workers1、autograd多线程true，完整独立两步核对+连续三步对两步保存/新进程恢复一步精确比较通过。
尚未固定实现提交资格，开发通过不称正式资格。

原开发downstream-dev和downstream-dev-v2均failed/1：native workers2 FC/KV FP32独立两步核对通过，
新进程精确恢复梯度/SGD槽不等；仅关闭autograd多线程不足。原输入、归档和失败保留。
单worker是显式不同执行配置，不称双worker问题已解除；没有扩大成基座数值修复任务。

## 作业与下一动作

上述三个任务位于artifacts/jobs/<名称>/，status.json/task.log为外层状态，checks/result.json及各日志为实际检查；
源码归档checks/source.tar.gz保留。服务为<名称>-20261030.service；均已终态，无活跃作业。
downstream-dev-v3 passed/0；其固定开发树artifacts/worktrees/downstream-dev-v3保持只读。

下一步提交本轮实现；再创建artifacts/worktrees/downstream-qualified于该提交，标记源码只读。
资格服务downstream-qualified-20261030.service，独立background.slice、Nice10、CPUQuota3200%、MemoryMax32GiB、900s。
无native构建，复用b255 CPU库前后核验源码和全部binary hash；ATen/BLAS1线程。本机module ascend/dev-workspace-9.1.0及.local/venv-core5。

```sh
python scripts/job.py --output-dir /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/jobs/downstream-qualified -- python scripts/check_downstream_consumer.py --build-dir /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/worktrees/core5-checkpoint-qualified/build/core-cpu --output-dir /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/jobs/downstream-qualified/checks
systemctl --user show downstream-qualified-20261030.service -p ActiveState -p Result -p ExecMainStatus -p MainPID -p Slice -p ControlGroup
```
停止：systemctl --user stop downstream-qualified-20261030.service。当前尚未启动资格；完成后单独提交证据、导航与终态并普通推送。

## 既有边界

上轮[73项资格](evidence/core5-coverage-correction-20261030.md)和[25配置7通过的性能](evidence/core5-natural-study-20261029.md)保持。
其余数值、异步、resident资源与硬件待验由[ROADMAP](ROADMAP.md)维护；不自动恢复旧队列。
