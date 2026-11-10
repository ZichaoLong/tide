# 当前交接 — 联合prefill部署验证收口（2026-11-10）

授权分支graph-execution-foundation；数学固定0770593838a37c790a568226930567ed60f99b83。
当前HEAD 1f82143；允许持续实施/提交/推送及充分使用空闲CPU/NPU、经身份核对的共享卡；GPU延后。
参考及下游只读，无子代理。未跟踪TL-REQ-003/004用户handoff保持原样，不纳入提交。

## 已完成与边界

默认分量窗口、B×L联合KV准备、整序列VJP、完整query根合并、按需trace及局部合批已交付。
固定3d88b0b：CPU281通过/4保留数值xfail，NPU流程68通过，安装消费者及独立SDK训练/新进程恢复通过。
原FC16最小返回完整步CPU18.640405/NPU22.902820秒。证据：prefill-local-batches-20261110.md。
原34项代表矩阵与精确续接见prefill-representatives-20261110.md；规模独立数值not_checked。

后继已完成而尚待写入正式报告：wide-FC CPU16、LH1.051B CPU32/NPU、FP16 query-FC、异构LH；
显式blocking双NPU FC16及普通双NPU LH9.314B推理/精确下一窗均完成。
层链IEEE NPU小型7项通过，六成本点已完成；vendor默认SSD超差与IEEE/FP64见证分别保留。
默认ordinary双NPU FC16曾在warmup反向挂起；已停且cgroup消失，显式blocking通过不解除默认异步问题。
两轮resident目录预算拒绝保留；1f82143按实际optimizer/反向分区及每样本阶段上界修正了生成器，实际重试待验。

## 中断事实与当前作业

重排前deployment-followup已passed。throughput-sample已完成CPU B32/P16/1.051B计时12.262388秒，
但在checkpoint保存中被取消；原inner running快照及未发布staging文件保留，恢复not_checked。
resident-accounted及layer-final-cpu在依赖等待时取消，未执行各自用例；vendor-policy已passed。
上述旧任务均终态，当前不能恢复为running或改写passed。

新有限服务gef-joint-prefill-closure已启动且cgroup核实在background.slice：固定worktree artifacts/joint-prefill/fixed-resident-accounted（1f82143）；
驱动及hash见artifacts/joint-prefill/closure-driver.json。64CPU/640GiB/12600秒；每次NPU准入60秒。
依次仅补六CPU层链、NPU B32推理、两个修正预算resident、CPU B32无计时重建/保存/公共exact_next。
复用3d88b0b安装/core；resident复用a553安装/core及独立resident库。单项失败继续独立项，不重测已有计时。
启动命令：systemd-run --user --unit=gef-joint-prefill-closure --slice=background.slice --service-type=exec
--working-directory=固定worktree --property=KillMode=control-group --property=RuntimeMaxSec=12600
--property=TimeoutStopSec=30 --property=CPUQuota=6400% --property=MemoryMax=640G --property=Nice=10
/usr/bin/python3 scripts/job.py --output-dir=主仓库/artifacts/jobs/joint-prefill-closure -- bash 主仓库/artifacts/joint-prefill/closure.sh。
日志artifacts/jobs/joint-prefill-closure/task.log；生命周期status.json。活动源码/包/库/驱动不可修改。

## 当前代码增量

已修正内置输出推理的事件预留：仅已审核的ranked-local、PyTorch/Python-native、分量窗口、
无梯度无插件路径按两个最大区域事件块加输出预留；保留完整记录估算，其他分区和路径不降预算。
planning/record_capacity.py与experiment/record_policy.py连接预算、运行和恢复；不是容量豁免或改算法。
恢复诊断保持实际返回路径，并比较下一窗完整raw状态，新增未返回状态错误注入。
开发检查17+29项通过；提交后需固定安装、核对复用3d C++源码未变，并补一次原B32 NPU。
旧B32 NPU在执行前被115339688680字节完整记录估算拒绝；没有计时，不能称实际OOM。
resident修正目录两项仍在construction拒绝KV qry/local-row数值工作区；未有吞吐或恢复，不盲加预算。
closure当前在CPU B32重建恢复；层链CPU六成本点已完成，后续只整理不重测。

## 精确下一步

1. python scripts/status.py；核对closure实际unit、status及每个inner输出，勿直接打印大result/identity。
2. 独立核实CPU B32旧计时与新恢复，resident的真实阶段/预算/服务限制；不把outer通过当数值资格。
3. 完成层链及部署证据、更新ROADMAP/复现导航/性能经验并提交；授权分支推送，不强推。
4. 有限超差诊断与机制检查分列推进，原严格阈值不变；非有限/通信/容量/精确恢复另列处理。
5. 停止仅限准确本项目unit；随后核对MainPID、整个cgroup及后代身份。

最近资源：磁盘空闲813GiB；quota270/300/800GiB；主存available约1.6TiB。新增checkpoint约12GiB可容纳，写前再核对。
下游未升级/验收，不将基座证据替代下游学习结论。唯一待办与重启条件见ROADMAP。
