# 当前交接 — 2026-10-30

授权分支graph-execution-foundation；数学固定0770593838a37c790a568226930567ed60f99b83。
本轮多实验仓库协作规范、模板、core-5安装起步示例及定向资格已完成。无子代理；参考仓库/公共工具链只读；GPU延后。
未修改基座数学/执行器/容差，未新增性能测量，也没有改动任何尚未提供的实验仓库。

## 本轮交付

实现提交36113ce70553d847d0a255dedb182e906645816c，起点ad684b7507e8cc3c9941e419c4bd5ea0c36de8a8。
[协作规范](downstream-contract.md)建立固定依赖→请求→基座交付→下游验收/显式升级；
[模板](../templates/downstream/AGENTS.fragment.md)连接接续规则、依赖锁及请求/交付/验收。
[core-5示例](../examples/consumer_core5/README.md)可复制到实验仓库，消费已安装公共接口，
示范应用数据/head/loss/SGD/StepLR、依赖身份、原子应用checkpoint和新进程恢复。

[固定源码证据](evidence/downstream-contract-20261030.md)及其JSON已整理，与实现分开提交：
4项单元、8组仓库外安装消费者、4项错误请求拒绝通过。
八组为FC PyTorch/native×FP32/FP64、HB混合PyTorch/native FP64、FC scalar streaming PyTorch/native FP64。
共同worker1、ATen/BLAS1、autograd多线程true；独立两步核对及连续三步对两步保存/新进程恢复一步逐值比较通过。
这是小型CPU接入资格；真实实验仍需自行验收，不新增大模型、NPU/CUDA、resident或standalone资格。

## 作业终态与保留失败

没有活跃作业。downstream-qualified在只读artifacts/worktrees/downstream-qualified的36113ce运行，passed/0。
2026-10-30 04:04:03–04:06:16 UTC；服务downstream-qualified-20261030.service已结束并回收。
启动时核对PID2538045和独立background.slice，提交Nice10/CPUQuota3200%/MemoryMax32GiB/900秒。
本机module ascend/dev-workspace-9.1.0与.local/venv-core5；未重建native，复用b255 CPU构建并前后核验全部源码/二进制身份。

作业位于artifacts/jobs/downstream-qualified/：status.json/task.log、checks/result.json、checks/tests.xml、
逐项日志和checks/source.tar.gz。审阅证据保存来源/hash/实际配置；新clone可用报告命令重新构建，不依赖旧目录。

开发downstream-dev和downstream-dev-v2保留failed/1：native workers2 FC/KV FP32独立两步通过，
新进程精确恢复仍有45个梯度/SGD张量字段不等；仅关闭autograd多线程不足。
downstream-dev-v3显式worker1开发通过；三轮各自源码归档、原输入/观察和checkpoint均保留。
worker1是另一执行配置，未解除workers2边界；重启条件见[ROADMAP](ROADMAP.md)。

## 接续入口

本轮仅有单独的证据/导航/交接变更，基座源码未再修改。分阶段提交和普通推送已获授权；实际提交/远端状态用下列命令核对。

```sh
git status --short --branch
python scripts/status.py
python scripts/check_upstream.py
```

接到实际实验仓库请求时，读固定版本的协作规范与请求，核对依赖锁/配置/证据/验收条件，在已有授权内执行。
新实验从示例README安装、锁定、运行check.py开始；上游交付和每个实验的验收保持分开。
没有待启动的历史队列。已有[73项资格](evidence/core5-coverage-correction-20261030.md)、
[25配置7通过的性能](evidence/core5-natural-study-20261029.md)及数值/异步/资源/硬件边界保持，待办只见ROADMAP。
