# 当前交接 — P1—P6继续实施（2026-11-12）

固定数学0770593838a37c790a568226930567ed60f99b83，19份快照通过；当前授权可重写、提交并推送，GPU真机延后。
共享NPU的当前对话持续授权包括性能；每次仍有新PID/start/UID、资源、双采样及组锁。公共/参考环境不改。
用户TL-REQ-004/005/006三份未跟踪handoff只读、不提交。阶段提交后继续，未解除项不冒称完成。

## 源码与下一动作

HEAD838fc30c；远端仍51e93e74。已提交：00bb来源MLP/精确共享及公共resident v2，38ade静态/42点证据，
397bf SDK Model检查显式raw链接，205f公共owner默认Next配置，d00f异宽source梯度的设备gather，838fc数值见证/导航。
所有后继固定构建均保持数值核心00bb，不混用fb83二进制。新公共resident仍不宣称全部规模通过。

本轮待集成源码已稳定：KV sum typed Memory tape/专属VJP、独立Mem/total根；
实验execution-v7的B×L输入/loss、实际完整记录与schema2 archive；standalone BlockCandidate、
请求/生效身份及恢复、outputs/metrics。主代理32项接口检查与公共CLI语法通过；
各开发CPU/逐TU检查保留原日志，不当作真机资格。三份用户handoff保持未跟踪。

下一动作：提交本轮实现并冻结`/dev/shm/gef-p1p6-20261112/resident-v3-source`；
两个background.slice构建服务`gef-p1p6-resident-v3-{native,sdk}-build`各CPU160核/512GiB/2400秒，
复用同C++ hash的00bb核心；SDK补独立standalone包。随后qry/sum独立根、public三AdamW/跨窗/新进程、
实验默认三SGD与精确恢复，再原预算FC16有限测量。脚本在`artifacts/p1p6-20261112/resident-v3-*-build.sh`。
静态审阅明确剩余成本：columns精确Q/K尺寸分组；resident多bank逐输入行复制参数、Read重复历史投影、
单block逐帧raw快照及重复结构根扫描。它们不是core-5所需，不因公共接口已接入而关闭P5。
新block多设备/其他记忆/half尚未迁移，旧general功能单列。

## 已固定结果

[fb83资格](evidence/rope-qualification-20261112.md)：CPU281通过/9原xfail、两ABI、SDK NPU五记忆×三放置、八安装三AdamW/新进程；非空LH176 NPU严格数值失败及None/零投影边界保留。
[42点成本](evidence/rope-performance-20261112.md)与[f475诊断](evidence/lh-static-structure-20261112.md)已审阅入库，不重跑。
显式per-stream队列的一般图六格/四步恢复通过，普通队列EE1017保持；GPU真机仍待验。

00bb新source：Python开发81 passed/5数值xfail；新native核心构建通过，初始CPU104 passed/6 xfailed/3 failed。
[有界诊断](evidence/source-columns-rounding-20261112.md)已确认固定见证：首Aggregate差1.92e-9，逐位同材料FP64全根最大1.71e-13；具名xfail分类不改严格阈值，原failed保留。
SDK核心CPU/五记忆三放置NPU通过；22点来源成本CPU/NPU均终态passed，报告尚待整理。LH差距仍明显，不以小收益结束根因解释。

resident：00bb Python-owned构建通过但首安装构造缺Next失败；205f修复后独立安装三AdamW与fresh-process通过。
397bf SDK构建及CPU Model检查通过；设备flow/session及跨窗恢复通过；public异宽source反向在非连续view的设备copy处失败。
两次有界gdb：第一次捕获厂商内部JSON异常，不用于该根因；第二次准确指向BlockSettleSession::backward/copy。
d00f已改为执行时device index_select，尚待新构建实测。不得把上层作业passed或qry小图覆盖当异宽/sum/规模资格。

## 活跃作业、位置与资源

已终态通过 `gef-p1p6-source-v2-lifecycle-{cpu,npu}`：固定00bb独立安装LH176，CPU64–95/NPU主机56–63，
各256GiB/1800秒，NPU空闲0/5/6/10经队列准入；两个客户端三AdamW/新进程均完成，NPU终态08:17:33 UTC；待固定报告整理。计划及逐格结果在
`artifacts/p1p6-20261112/source-v2-lifecycle-*`，持久终态/log在`artifacts/jobs/p1p6-source-v2-lifecycle-*/`。
先运行 `python scripts/status.py`；历史失败用`--all-jobs`。查看/停止：`systemctl --user show UNIT` / `systemctl --user stop UNIT`，核对完整cgroup。

source-v2-performance的22点已完成，CPU0–31/主机32–39、共享物理9，一次预热一次测量；没有与自身重构建重叠。
原始材料/计划/结果/launcher在 `artifacts/p1p6-20261112/`，不把无插桩计时与profile混用。
冻结源码/构建在 `/dev/shm/gef-p1p6-20261112/`：core-source-v2、core-source-v2-native/sdk/install，
resident-v2-linkfix（397bf）与resident-v2-sdk-linkfix，resident-v2-boundary-fix（205f）与resident-v2-native-fixed。
v3两个构建已准备待提交后启动；以jobs终态为准。后继仍使用background.slice、固定输入、有限预算；不编辑运行中源码。
quota307GiB/软300/硬800，仍有6天grace；shm约295GiB可用，无存储阻塞，不清理用户/参考数据。
