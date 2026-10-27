# Core-5 单样本常驻节点放置前向资格 — 2026-10-28

固定运行库源码`d072c1710f33c07736b5dfe1d0f4b841005efc1c`，修正检查器/资格入口源码
`1289bc9e0bb13450c102dded8eafe4d03f45784b`，上游`0770593838a37c790a568226930567ed60f99b83`。
[放置契约](../resident-weighted-placement.md)定义完整纤维、全区域选择与共同采用；
[同名JSON](core5-weighted-node-placement-20261028.json)记录各源码、二进制、loader、日志、
命令、任务终态和安装转换摘要。后续f84a007窗口反向不属于本报告。

| 范围 | 实际结果 |
| --- | --- |
| CPU静态放置定义域 | 8项通过 |
| Python-owned NPU，Torch2.12/CANN9.1 | 50项通过：节点放置4、单设备完整图28、训练18（其中2项静态入口检查） |
| standalone NPU，LibTorch2.10/CANN9.0，节点放置 | FP32/FP16各9配置通过 |
| 同运行库单设备回归 | 两dtype各21图；各5训练配置/20更新与累计/拒绝/恢复通过 |
| CUDA源码/构建 | sm80/90/100完整运行库、三个原始checker构建/loader/help；修正checker安装消费者构建/loader/help通过 |
| CUDA设备运行 | 未验，无真机 |

双卡各配置中，每个样本的节点分布在逻辑0/1；它们不是独立样本副本。五种记忆分别在
FC/prefill执行，异构HB、非对齐TimedDAG、反馈PDG以streaming执行，另有KV qry、match、
全区域Attention、grouped evidence及来源低秩变换。核对完整作用、control、未缩放G、
物理重复边/端口、history、原始状态及pending；零接收与缺失输入分别保留。

检查raw、Full和每个实际参数bank的物理设备。将磁盘checkpoint恢复到交换后的节点放置，
核对后半窗、完整切面及空窗恒等；放置不改变数学graph identity。容量反例先用独立CPU调度
确认首区域两个跨卡节点同时作用、首阶段4条消息/4条pending，然后以容量2核验所有owner
均未采用该阶段；没有把此前已成功的阶段误当成应回滚的初态。

独立CPU Greedy控制调度和完整返回；half使用独立FP32算术/显式FP16存储参考，
没有把CPU路由作为候选输入。该对照证明物理放置组合保持已声明函数，局部函数数学锚点仍沿
[语义索引](../semantic-map.md)导航。单设备训练回归不能认证放置后的反向或优化器发布。

运行库和checker源码比较严格限定`tools/device_online`与build脚本只有
`weighted_flow_peer_check.cpp`不同。原库构建manifest、全部binary/loader摘要已核对；
新checker通过独立CMake消费者链接安装库。安装会移除构建RPATH，不能要求安装前后整个ELF
摘要相同：本次在已核验的私有库副本上执行CMake `file(RPATH_REMOVE)`，结果与实际安装库
逐字节一致；同时核对ldd实际加载该安装库。安装前/后与复现转换摘要分别记录。

失败与取消不改写：原d072c17 SDK资格因未给新checker传output-dir入口失败；357477b后继
在等待期间因发现容量反例前提不充分而取消；原修正CUDA检查器构建完成后因辅助脚本
误要求安装库摘要不变失败，同版NPU等待任务主动取消。两次重提又因预建job.py独占输出目录
在启动阶段失败，捕获traceback和postmortem保留，未执行构建或设备任务。新的v2任务独立通过。

只读源码分别位于`artifacts/worktrees/core5-weighted-node-placement`和
`artifacts/worktrees/core5-weighted-node-check`；最终jobs为
`core5-weighted-node-install-{npu,cuda}-v2-qualify`，原始三构建任务为
`core5-weighted-node-placement-{npu,python,cuda}-qualify`。units均`tide-JOB-20261028.service`，
记录在`artifacts/jobs/JOB/{status.json,task.log}`。终态以持久退出码与实际输出为据，
已回收unit的默认show属性不作退出证据。所有相关任务已终态；未运行旧性能矩阵。

运行时确认独立background.slice/cgroup、Nice10、两构建worker、ATen/BLAS1。
原构建1500秒、设备900秒；修正消费者构建360秒；两类NPU任务queue180秒。
Python物理9/13→逻辑0/1，SDK最终物理1/3→逻辑0/1。记录不提供跨卡吞吐或争用控制结论。

复现可在干净1289bc9源码用匹配core执行`build_device_control.py --checks weighted-node-peer
weighted-flow weighted-training`，然后双设备运行`verify_device_control.py --checks weighted-node-peer
weighted-flow weighted-training`，提供全新输出目录。它会干净重建全部依赖；本次实际将未变运行库
与修正checker分开构建的命令及源码身份保留于JSON。Python门禁为
`test_resident_weighted_node_placement.py`、`test_resident_weighted_flow.py`和
`test_resident_weighted_training.py`。按[开发规范](../development.md)选择匹配环境与独立持久服务。
