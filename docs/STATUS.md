# 当前交接 — 默认prefill局部合批与部署回归（2026-11-10）

授权分支graph-execution-foundation；固定数学0770593838a37c790a568226930567ed60f99b83。
用户授权持续实施/提交/推送、全部可用CPU及空闲/有余量共享NPU；有限数值超差有界诊断后分列推进。
参考及下游只读，无子代理；GPU延后。未跟踪TL-REQ-003/004用户handoff保持原样、不提交。
当前完成5cb1c9b固定合批证据；后继修复有限resident目录的预算分区。活动作业读取各自冻结源码。

## 已完成及证据边界

默认分量窗口、B×L联合KV准备、整序列VJP、完整query根合并和按需trace已交付。
导航：prefill-regression.md、semantic-map.md、performance-lessons-prefill.md、evidence/README.md。
既有34项CPU/NPU代表运行与精确下一窗已记录于a841860；实际规模独立数值not_checked。

f15c4a5按owner合并KV写伴随的固定任务owner-join已passed：两套新构建、CPU74/NPU37项流程、
两种安装消费者CPU/NPU三步与恢复、独立SDK完整/outputs及新进程恢复通过。原FC16最小返回成本
CPU22.301568/NPU31.023336秒；收益有限。正式小型证据尚待与下一增量汇总。

3d88b0b把cat evidence投影、Read有限值检查和分配分支回读合批。
固定local-batches两套C++构建完成，SDK CTest3项、CPU FP64/FP32 kernel检查及SDK CPU恢复通过；
CPU285项中281通过/4原阈值数值xfail，NPU目标流程68项通过（含CPU辅助）。
两种安装消费者CPU/NPU三步独立检查及新进程恢复通过；原FC16 CPU最小返回18.640405秒。
NPU成本22.902820秒/profile及SDK NPU完整/outputs恢复完成，任务passed；正式证据见prefill-local-batches-20261110.md。

f6092d6驱动显式采用IEEE matmul/conv设置，有限resident代表补静态预算。
CPU相关15项开发检查通过。固定layer-ieee用f15安装/库在NPU7项全部通过，原阈值不变；
同材料FP64见证SSD tensor输出最大差由3.063e-6降到2.414e-8，梯度1.25596e-5降到1.413e-7。
旧vendor-default结果3个数值xfail和原跨设备测试错误保留，不能从此认证标准Transformer/Mamba。

deployment-cpu/single已完成：wide-FC CPU16线程0.332410秒；LH1.051B CPU32线程1.368072秒、
NPU3.397598秒（B2/P8推理）；FP16 query-FC8.137528秒、异构LH88.001672秒（B4/P16训练）。
各自精确下一窗通过，规模独立数值not_checked。原resident两项在窗口前欠预算拒绝，尚无成本。

默认ordinary双NPU FC16停在warmup反向，多分钟无推进；gdb在NPUStream::stream/_index_put_impl_等待。
已停止gef-joint-prefill-deployment-dual，outer cancelled/143；MainPID=0、cgroup及目标PID已消失。
原inner running快照保留，stop-decision及backtrace在artifacts/joint-prefill；不是数值失败。
原同服务LH9B尚未执行；后继同步配置不认证默认异步已修复。

## 当前任务（以实际状态为准）

全部source/驱动已冻结，不可修改活动worktree、包、build或脚本。日志统一为
artifacts/jobs/joint-prefill-NAME/task.log；状态status.json；原始结果artifacts/joint-prefill/。

- local-batches：固定3d88b0b，fixed-local-batches/env-local-batches/build-local-batches及-sdk；
  已passed，320CPU/512GiB/28800秒，NPU准入60秒。全部inner结果已审阅。
- deployment-followup：固定f6092d6驱动，等待local-batches通过；64CPU/640GiB/18000秒。
  新3d库的层链NPU7项和6成本点；原a553 resident/core按静态预算一次重试；
  FC16双卡显式ASCEND_LAUNCH_BLOCKING=1（600秒）及LH9.314B双卡普通推理（3600秒）。
  各项一次预热/测量/精确下一窗。独立组错误不阻断其他组；每次准入最多60秒。
- throughput-sample：固定f609驱动/3d库，等待deployment-followup终态；64CPU/512GiB/26000秒。
  LH1.051B/B32/P16推理CPU32线程与NPU各一次成本/精确下一窗，针对旧B2/P8批小的补充；
  每项1800秒、准入60秒，规模独立数值仍不自动通过。

均已验证在background.slice；构建与正式计时不重叠，小型正确性可用独立卡。
驱动hash见local-batches-driver.json、deployment-followup-driver.json、throughput-sample-driver.json。
最近磁盘802GiB空闲、quota222/300/800GiB；大checkpoint前复查，保留quota安全余量。

## 精确下一步

1. git status --short --branch；python scripts/status.py；python scripts/check_upstream.py。
2. 检查上述状态与每个inner result；不要直接cat大result，按JSON字段读取。日志可能有非UTF8错误文本。
3. local-batches profile完成后用scripts/summarize_query_profile.py写新汇总，再审阅算子/工作量/完整步。
   artifacts/joint-prefill/collect-followup-evidence.py是尚未执行的审阅草稿；不可凭outer通过扩大范围。
4. 写小型固定证据，更新ROADMAP、相关契约/复现导航及prefill经验，替换本交接；按授权推送，不强推。
5. 停止仅限准确的本项目unit：systemctl --user stop gef-joint-prefill-NAME.service；随后核对整个cgroup。

当前唯一待办与停止/重启条件见ROADMAP；下游尚未升级或验收，不能把基座交付等同学习效果。
历史源码用scripts/resolve_history_commit.py OLD_SHA；不恢复旧队列，不修改公共环境。

resident-budget显式重试两项仍拒绝：FC16的optimizer布局仅获预算1/16，SSD反向四份view未计足。
已按源码边界修复，阶段上界使用每样本L×R（DeviceReady并行独立样本，journal自身含B），不再重复乘B。
CPU配置3项通过；新总预算FC16约18.52GB、SSD约6.24GB，真实数值tape/服务限制仍待验证。
下一步固定新驱动，待本项目throughput-sample计时结束后补一次这两条具体修复，旧两轮拒绝保留。
