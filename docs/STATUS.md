# 当前交接 — 2026-10-28

固定上游`0770593838a37c790a568226930567ed60f99b83`；授权分支`graph-execution-foundation`。
f84a007窗口反向实现已提交推送；本次提交前向独立证据，窗口反向资格仍在运行。
用户授权持续实施、私有环境、有限实验、提交与正常推送；不使用子代理。总体goal active，未完成。
唯一待办[ROADMAP](ROADMAP.md)，持续契约[execution-flows](execution-flows.md)，参考仓库/公共工具链只读。

## 当前完成范围与新证据

公共core-5、一般Settle与端口、五记忆及正式发送/评分/证据/key、独立调度、packed/VJP、mixed、
常驻FP32/FP16完整图/训练与恢复从[semantic-map](semantic-map.md)进入。
[FP16完整图/训练](evidence/core5-resident-half-training-20261028.md)和
[样本副本/master](evidence/core5-resident-weighted-peers-20261028.md)保持各自固定源码资格。

新增[单样本节点放置前向](evidence/core5-weighted-node-placement-20261028.md)：运行库d072c17，
修正检查器1289bc9。CPU8、Python-owned50（放置4/单设备图28/训练18）、SDK两dtype各9双卡配置，
以及单设备各21图/各5配置20训练更新全部通过。完整记录、真实owner、跨放置磁盘切面、空窗、
首阶段共同容量拒绝独立核验。CUDA sm80/90/100构建/loader/help及修正安装消费者通过，无真机。
该前向证据不认证放置后的反向或optimizer。审阅脚本`.local/record-weighted-node-placement.py`。

原始冻结`artifacts/worktrees/core5-weighted-node-placement`；检查器冻结`artifacts/worktrees/core5-weighted-node-check`。
最终jobs `core5-weighted-node-install-{npu,cuda}-v2-qualify`均passed/exit0；Python原任务也passed。
记录`artifacts/jobs/JOB/{status.json,task.log}`，SDK有checks/result.json和single-device-regression/result.json。
原SDKoutput-dir入口失败、错误安装摘要断言、两次预建输出目录的启动失败、两个等待任务取消
保持原终态，详情在报告/JSON。RPATH_REMOVE在已验副本上逐字节复现实际安装库变化，不混淆hash。

[ceae7bf有限性能](evidence/core5-weighted-performance-20261028.md)：24配置22通过，CPU反馈PDG两项严格失败。
小中规模CPU成本最低；medium HB standalone prefill约350ms/streaming456ms，NPU mixed-c约1.41–1.61s、
resident约2.60–2.71s。完整调用含调度、packing、返回、VJP、optimizer与同步；无大模型/跨卡吞吐结论。
[f72a6cc启动/内存补充](evidence/core5-weighted-observation-20261028.md)四新进程与三个消费者通过。
不重跑旧24项性能矩阵；已有局部profile不认证全部尺度。

## 活动资格：f84a007节点窗口反向

源码`f84a00718188b1a19fe5e7da958a816147f629fc`；只读冻结
`artifacts/worktrees/core5-weighted-node-window`，build/{npu,cuda}独立，不编辑活动输入。
jobs `core5-weighted-node-window-{npu,cuda}-qualify`，units `tide-JOB-20261028.service`。
两项已确认live在独立background.slice/cgroup；当前running，尚无终态结论。
launcher `.local/JOB.sh`及`JOB-workload.sh`；SDK设备入口`.local/run-weighted-node-window.py`。
记录`artifacts/jobs/JOB/{status.json,task.log}`；SDK另有checks/result.json与single-window.json。

SDK构建weighted-node-window/window/training，新双卡4组合×12独立根、2个三窗桥（含空窗），
单设备HB/proposal和feedback/old窗口，以及FP32/FP16训练必要回归。CUDA sm80/90/100只构建/loader/help。
构建1500秒/2worker，NPUqueue180秒/设备2400秒，整job4140秒、unit4200秒；Nice10/ATen及BLAS1。
本提交已通过6个实现TU和4个checker TU的SDK syntax-only、CPU静态8项；这些不是设备资格。

实现保持journal逆序、消息祖先、完整region VJP和canonical归约在coordinator；
各节点局部tape/VJP在原owner。边界复制实际raw到公开根设备，快照保留owner及精确元数据；
所有共享参数使用者贡献汇总，反向重放不采用前向状态。预算新增peer主程序持有storage的核算。

## 原工作树未提交与下一步

正在接入公共训练的`node_devices`构造/恢复、manifest及跨设备参数bank发布。
草稿涉及public training header、owner/bindings、Python resident_training、training_optimizer/peer，
新`weighted_training_publish.{h,cpp}`和CMake。canonical FP32 master/slots仍统一，远端接收同一
决策与已更新值；所有设备先提交再等待。不修改活动f84a007输入。
草稿尚未完成定向测试/提交；放置后的FP16训练与checkpoint仍待资格。

接着执行：
1. 核对两个f84a007任务终态/日志；新失败保留最小反例，不把运行中任务写成通过。
2. 完成公共training放置独立检查：两dtype、全部字段梯度、共享参数、SGD/AdamW多步、累计、
   优化器共同拒绝、跨放置恢复；Python adapter与standalone分别验收，必要样本副本回归。
3. 提交训练实现，冻结源码做有界NPU/CUDA资格；分别提交已完成的前向/反向/训练证据。
4. 最终公共接口/能力矩阵/数学追溯审计；CUDA真机独立待验，不阻塞本机工作。

## 环境和保留限制

cwd`/mi/data2T/zlong/home/llm/graph-execution-foundation`，zlong，可用磁盘约3.3TiB。
现行规范`/opt/software/ascend-locks/dev-workspace-9.1.0/README.md`和`/etc/modulefiles`；旧guide已失效。
Python：source /etc/profile.d/modules.sh；module purge；module load ascend/dev-workspace-9.1.0；source .local/venv-core5/bin/activate。
SDK：module load libtorch-npu/2.10.0-cann9.0.0；Python为$LIBTORCH_ROOT/../../../../bin/python。
CUDA：source /mi/data2T/zlong/gpu-toolchains/activate-cuda.sh；PATH追加项目.local/venv-core5/bin供ninja。
TORCH_DEVICE_BACKEND_AUTOLOAD=0、PYTHONDONTWRITEBYTECODE=1、ACL_OP_INIT_MODE=0、ATen/BLAS1，保留CANN PYTHONPATH。
STATUS使用durable_records原子写并回读；长任务固定源码、独立background.slice/有限时长。

严格失败保留：eager FP16 GDN默认AdamW约.001984 strict xfail；默认异步两NPU Python VJP CANN invalid event
（launch blocking通过不解除）；SDK2.10 FP32 fc/affine第三轮evidence差3.09944e-6/mixed-a同见证；
ceae7bf FP32 PDG性能两项maxabs4.11272e-6/2.71201e-6。独立FP64提升通过不解除原失败。
PDG最小复现保留原fixture、性能job两result/log与CPU consumer，及core5-weighted-pdg-rounding-diagnose/diagnosis.json。
原672e948 standalone bad_alloc检查器失败保持；7db0b08修复后master/peer复验通过。所有CUDA真机范围待验。

重入：`git status --short --branch`；`python scripts/status.py`；`python scripts/check_upstream.py`。
任务检查：`systemctl --user show UNIT -p ActiveState -p MainPID -p Result -p ExecMainStatus`，并核对持久status/结果。
停止自有任务：`systemctl --user stop UNIT`；不恢复旧队列。
