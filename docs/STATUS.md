# 当前交接 — 2026-10-28

固定上游`0770593838a37c790a568226930567ed60f99b83`，授权分支`graph-execution-foundation`。
37922eb公共训练放置实现与e405b1b能力/文档审计已提交推送。总体goal active，未完成。
用户授权持续实施、私有环境、有限实验及正常提交推送；不使用子代理。
[ROADMAP](ROADMAP.md)是唯一待办，[execution-flows](execution-flows.md)是持续契约。
参考仓库和公共工具链只读；活动冻结源码/launcher不得修改。

## 已完成范围与证据

公共core-5、一般Settle/端口、五记忆及正式发送/评分/证据/key、独立调度、packed/VJP、mixed，
常驻FP32/FP16完整图/训练/恢复从[semantic-map](semantic-map.md)和[能力范围](execution-capabilities.md)进入。
[半精度完整图/训练](evidence/core5-resident-half-training-20261028.md)、
[样本副本/master](evidence/core5-resident-weighted-peers-20261028.md)、
[节点放置前向](evidence/core5-weighted-node-placement-20261028.md)各保留固定源码资格。
前向报告运行库d072c17/检查器1289bc9：CPU8、Python50、SDK两dtype各9双卡配置，
单设备两dtype各21图/20更新；CUDA sm80/90/100构建及安装消费者，没有真机。

[有限性能](evidence/core5-weighted-performance-20261028.md)ceae7bf的24配置22通过，FP32 PDG两项严格失败。
本组小中规模CPU成本最低；medium HB standalone prefill约350ms，streaming456ms，mixed-c约1.41–1.61s，
resident约2.60–2.71s。完整计时含调度/packing/通信/反向/更新/同步，无大型模型或跨卡吞吐结论。
[f72a6cc启动/内存](evidence/core5-weighted-observation-20261028.md)补充已完成，不重跑旧性能矩阵。

## f84a007窗口反向失败与本次修复

原固定源码`f84a00718188b1a19fe5e7da958a816147f629fc`，冻结`artifacts/worktrees/core5-weighted-node-window`。
`core5-weighted-node-window-cuda-qualify`已passed/exit0，构建/loader/help待汇总审阅；无设备结论。
`core5-weighted-node-window-npu-qualify`已failed/exit1，原终态保留：CPU8通过，HB12类独立根通过后，
全节点远端nonaligned在construct-adjoint报aclnnInplaceCopy/CANN361001。
原vendor日志复制保存在该job/vendor-diagnostic；底层报告stream task buffer full。

独立诊断`core5-node-window-failure-diagnose`已passed/exit0，只表示GDB抓栈完成，inferior退出2。
同一原binary重现，栈为CannApi::check→DeviceProgram::finish→WeightedFlowPeers::finish。
记录`artifacts/jobs/JOB/{status.json,task.log}`；两卡物理1/9→逻辑0/1。
对应只读launcher `.local/core5-node-window-failure-diagnose.sh`和`.local/node-window-failure.gdb`。

本次修复在`weighted_flow_peers.cpp`为每个远端节点的VJP/raw/参数归约增加外层嵌套服务，
避免全部节点归约落到peer主程序。次序、全区域函数、根和canonical归约不变；CUDA仍在同一条件图。
已有全远端nonaligned反例、HB/反馈/全区域Attention及跨窗桥将原样复验。
另补公共安装消费者`--node-placed`：peer手算恒等VJP、None optimizer及换放置恢复，
同时检查新构造/恢复重载和实际放置。它不替代完整图/参数发布资格。
SDK两个改动TU syntax-only、CPU静态/入口10项及diff检查通过；设备修复尚待验证。
0155b0f已正常推送；本工作区仅STATUS与后续文档审计（eager FP16/局部组件过时进度措辞）未提交。

## 37922eb公共训练原资格终态

冻结`artifacts/worktrees/core5-weighted-node-training`，源码`37922eb8ccd79f792042bf2aad308fd915f0f432`；
jobs `core5-weighted-node-training-{npu,python,cuda}-qualify`，units均`tide-JOB-20261028.service`。
三项已确认独立background.slice/cgroup。CUDA已passed/exit0；Python已passed/exit0，30项通过。
SDK已failed/exit1：FP32五配置20次更新/三窗与恢复通过，随后refusal检查失败；后续FP16、
node-replicas与原回归未执行。两NPU物理3/13→逻辑0/1，设备约173秒。
日志checks/weighted-node-training-float32.log。原终态保留，不标为整组通过。
检查器把数值溢出的optimizer_finite_error=20误写为21（int64 step overflow）。
独立FP64更新值转存储dtype的溢出前提已通过；当前修复用具名常量并增加实际结果诊断，
尚须固定检查器重新核验实际拒绝码、无提交及后续续接。生产优化器不变。SDK改动TU syntax-only、修改文档本地链接和diff检查通过。
只读launcher `.local/JOB.sh`/`JOB-workload.sh`；记录`artifacts/jobs/JOB/{status.json,task.log}`。
SDK `checks/result.json`、CPU `cpu.xml`；Python `tests.xml`；CUDA build manifest/loader/help。
SDK两dtype各5放置配置/20更新、各2个放置样本副本/6更新，以及原单设备/样本副本回归；
Python新增8项、原训练18项/peers4项。构建1500秒/2worker，queue180秒；SDK设备3600秒、Python1800秒，
整job5340/3540秒、unit5400/3600秒；CUDA job1560秒/unit1620秒。

## 接着执行

1. 本次修复与消费者已提交0155b0f，固定只读checkout `artifacts/worktrees/core5-weighted-node-services`。
2. 已提交并核对live的新jobs `core5-weighted-node-services-{npu,python,cuda}-qualify`；每项独立background.slice、Nice10，
   两build workers、ATen/BLAS1。SDK只复验受影响node-window/node-training/node-replicas、两个单设备窗口和安装消费者；
   Python只复验8个放置训练用例；CUDA三架构构建及安装消费者。不要更改原失败或活动输入。
   已核对三份持久status source=0155b0f且dirty为空；目前running。构建1500秒、queue180秒，
   SDK设备2400秒/安装180秒/job4320秒/unit4380秒；Python设备1200秒/job2940秒/unit3000秒；
   CUDA安装180秒/job1800秒/unit1860秒。launcher .local/JOB.sh和JOB-workload.sh，只读。
   结果artifacts/jobs/JOB/{status.json,task.log}，SDK checks/result.json、single-window.json、consumer-checks.json，
   两standalone的client/result.json与loader，Python tests.xml。
3. 固定检查器修复后，链接已核验运行库进行定向资格；不要修改0155b0f活动源码/检查器。
   该资格含同一硬编码误判，预计需独立记录其终态并补齐未执行项目。
   审阅所有原资格与修复终态，保留失败；分别提交固定源码证据。
4. 最终更新能力/追溯入口、STATUS/ROADMAP并推送；CUDA真机独立待验，不等待硬件。

精确复现命令由冻结workload脚本、build/运行JSON及最终证据记录；当前不恢复旧队列。
状态检查：`python scripts/status.py`；`systemctl --user show UNIT -p ActiveState -p MainPID -p Result -p ExecMainStatus`。
停止自有任务：`systemctl --user stop UNIT`。新服务提交后必须核对live cgroup和持久status。

## 环境与保留限制

cwd`/mi/data2T/zlong/home/llm/graph-execution-foundation`，zlong，磁盘约3.3TiB可用。
现行规范`/opt/software/ascend-locks/dev-workspace-9.1.0/README.md`和`/etc/modulefiles`；旧guide已失效。
Python：source /etc/profile.d/modules.sh；module purge；module load ascend/dev-workspace-9.1.0；source .local/venv-core5/bin/activate。
SDK：module load libtorch-npu/2.10.0-cann9.0.0，Python为$LIBTORCH_ROOT/../../../../bin/python。
CUDA：source /mi/data2T/zlong/gpu-toolchains/activate-cuda.sh，PATH追加项目.local/venv-core5/bin供ninja。
TORCH_DEVICE_BACKEND_AUTOLOAD=0、PYTHONDONTWRITEBYTECODE=1、ACL_OP_INIT_MODE=0、ATen/BLAS1；保留CANN PYTHONPATH。
STATUS用durable_records原子写并回读。e405b1b文档/portability schema与链接检查通过，不认证新设备结果。

严格失败保留：eager FP16 GDN默认AdamW约.001984 strict xfail；默认异步两NPU Python VJP CANN invalid event，
显式launch blocking通过不解除；SDK2.10 FP32 fc/affine第三轮evidence差3.09944e-6/mixed-a同见证；
ceae7bf FP32 PDG性能两项maxabs4.11272e-6/2.71201e-6，独立FP64提升通过不解除。
PDG最小复现保留原fixture/两result/log/CPU consumer与core5-weighted-pdg-rounding-diagnose/diagnosis.json。
前向checker/安装launcher历史失败与取消、672e948检查器bad_alloc均保留，详见报告；不重写为passed。
所有CUDA真机范围待验。

重入：`git status --short --branch`；`python scripts/status.py`；`python scripts/check_upstream.py`。
