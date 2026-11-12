# 当前交接 — 来源共享与 resident 程序组合（2026-11-12）

数学固定0770593838a37c790a568226930567ed60f99b83，19个快照通过。
分支graph-execution-foundation；允许整体重写、阶段提交/非强推，本对话共享NPU；GPU延后。
mixed-a/b/c仅是可迁移放置预设，不约束软件层级。三份用户TL-REQ-004/005/006 handoff只读、不提交。
P1—P6仍有未完成范围，见[ROADMAP](ROADMAP.md)。

## 最新固定结果

- 1f0e6e43主机[资格/38点成本/静态审阅](evidence/region-tensor-20261112.md)已推送。
  FC16 B8/L128默认native训练CPU1/NPU-host2为1.1854/.2856秒；LH176 B4/L16为1.3900/2.8407秒，直接.7796/.6121秒。
- 41c42759删除resident bank容量树；消息/选择/序列通过，flow反向录入507002原失败保留。
- 042ea369以设备通知调用每区域有界program，统一服务/控制器存储并集与workspace。
  两套resident构建、SDK21进程、Python-owned4进程的组件/整链/训练/新进程恢复通过。
  原预算FC16 B4/L16/d128训练3.944763638秒；LH176 B4/L16/d64训练22.591799125秒。
  两者新owner下一窗精确恢复通过，规模数学not_checked、throughput_qualified=false。
  原raw误记矩形64输入位置，实际staggered输入61；主树修复统计，不改原raw。
- 7fd1dd10修复CPU模拟kernel worker上下文泄漏；固定39项通过。042原29通过/9失败保留。
  投影193行独立FP64 oracle通过原阈值；旧逐行FP32参考自身有1项超差，不改生产tile。

042/7fd的所有resident-services-*作业均已终态，没有仍在运行的规模任务。
详细材料artifacts/resident-services-20261112；作业artifacts/jobs/resident-services-*。
固定source /dev/shm/gef-resident-services-20261112/source；两套resident-{native,sdk}在同根。
core复用1f的/dev/shm/gef-region-tensor-20261112/core-{native,sdk}并核验hash；Python安装317文件匹配042。
SDK首次等待构建前置120秒超时，未用设备；后继sdk-ready单独记录通过。

## 当前集成和下一动作

72e1d22c已固定：source-owner/incidence Aggregate、区域参数视图及实际发送操作数传输去重、resident计数修复。
开发检查96项Python通过；15项native因未指定新构建未运行，不能称C++失败或通过；正式新构建后补验。
扩展Python检查98通过/3原xfail/2新FP32严格超差；同材料FP64全根通过，None/零连接与完整观察一致。
原阈值不变、原失败保留，开发诊断artifacts/source-bank-development-20261112/numerical-diagnostic.json。
lh_direct收尾source-bank；req006_diagnostics整理042/7fd证据；resident_rewrite实现预算预准入及资源统计。
主代理审阅/集成/提交及统一资源调度。源码可与已冻结job并行修改，不修改作业输入。

下一步：完成短检查并固定host实现，建立两套新core及安装包；定向CPU/真实NPU、独立SDK、
安装三步/新进程恢复；LH176三层三模式与FC16有限对照。host C++已变化，不沿用1f二进制作为新资格。
resident新预算/统计有自身后继资格；共享历史/raw布局与程序构造成本继续推进，不能宣称P5完成。
将042/7fd及两原规模结果整理为仓库独立可复现证据，再推送已授权分支。

资源：主盘819GiB、tmpfs279GiB余量；账户325GiB/软300/硬800GiB，6日宽限；暂无磁盘阻塞。
两工具链分别ascend/dev-workspace-9.2.0与独立libtorch-npu/2.12.0-cann9.2.0；不混库。
重新进入：git status --short --branch；python scripts/status.py；python scripts/check_upstream.py。
历史提交：python scripts/resolve_history_commit.py OLD_SHA，见[迁移](history-migration.md)。

新两套core构建：source-bank-build-{native,sdk}.service，固定/dev/shm/gef-source-bank-20261112/source (72e1d22c)。
CPU0–159/160–319，各512GiB、2400秒；artifacts/source-bank-20261112/{native,sdk}-build.sh，
记录artifacts/jobs/source-bank-build-*；构建未完成前不开始依赖资格或计时。

72e1d22c native构建/独立wheel安装已通过；SDK构建仍在运行。开始固定cpu-directed、host-cpu、host-npu资格，
分别CPU0–7/8–15/16–19，各256GiB、900/1500/1800秒；NPU最多准入180秒，原始过程记录同artifacts/source-bank-20261112。

resident后继已实现已知存储/workspace预准入和当前程序/流/通知计数；开发8CPU/8C++ TU语法通过，
主代理已审阅，固定后补构建与设备资源释放/训练恢复检查；CUDA program仅源码、无toolkit或真机资格。
