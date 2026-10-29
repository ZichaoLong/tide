# 当前交接 — 2026-10-29

授权分支`graph-execution-foundation`；数学固定`0770593838a37c790a568226930567ed60f99b83`，19份+旧5份离线校验通过。
本轮用户要求逐项推进非GPU限制，保留严格标准、有限预算与停止/重启条件。GPU延后，无子代理。
已获修改、有限实验、阶段提交及普通推送授权；参考仓库与公共工具链只读。

## 当前源码与结果

起点121f5b1，生产对照f03e9be；固定诊断bdd7f6c/证据fe5e6e7、eager实现e0d6e27/证据9d1a54c已提交推送。
[隔离证据](evidence/core5-boundary-isolation-20261029.md)和[eager资格](evidence/core5-eager-boundary-20261029.md)
记录原D64/D2048/反馈FP32、FP16 GDN、默认async双NPU的失败/停止条件；未改原输入、容差或optimizer。
固定e0d6e27 CPU资格517 passed/2原strict xfail，无需因本次resident/设施改动重跑这517项。

本提交包含resident KV因子链/完整VJP、实际分配拒绝部分行、反向构造阶段计量，
以及benchmark实际梯度/SGD槽/共享owner解码、跨卡、磁盘恢复、显式静态容量。尚待固定提交资格及性能。
开发记录均终态，原失败保留：
- `core5-boundary-sdk-dev-v2` passed/0：9项SDK组件（两dtype，含单/双卡训练恢复）和6组eager完整训练。
- `core5-boundary-pynative-dev-v3` failed/1：eager6组及原公共KV12项通过，新观测测试误用精确loss比较。
- `core5-boundary-final-dev` failed/1：启动器预建输出目录，workload未启动，已记录systemd postmortem。
- final-dev-v2/v3/v4各failed/1：真实consumer揭示canonical参数/旧buffer、无尾哨兵offset、trainable全别名而slot为物理owner；每次保留冻结快照与日志。
- `core5-boundary-final-dev-v5` passed/0：10项设施及真实NPU梯度/SGD/精确恢复通过；公开实际allocation拒绝反例已在v2/v3通过。
- 早先ArrayRef编译失败和CANN旧缓存unknown file type均保留；新resident目录正式构建通过，未改公共工具链。

记录在`artifacts/jobs/NAME/{status.json,task.log,development.json}`，设备阶段另有device.json/JUnit及完整dirty源码归档。
SDK与Python-owned属于不同Torch/CANN ABI，证据分别记录。

## 活跃任务及下一步

本次开发服务均终态，没有live任务。正在固定上述实现提交，然后建立干净树
`artifacts/worktrees/core5-boundary-final-qualified`，源码hash写入`.local/core5-boundary-final-source`。
正式资格driver`.local/core5-boundary-final-qualify.py {sdk,pynative}`；各自background.slice、
单线程ATen/BLAS、16GiB、有限排队180秒/执行上限，原CPU核心无变更不重跑历史全量。

可复用已完成构建（保留真实dirty-origin manifest）：
- CPU：`artifacts/worktrees/core5-boundary-dev/build/cpu`。
- SDK：`artifacts/worktrees/core5-independent-qualified/build/{npu-core-v2,npu-resident-v2}`。
- Python-owned：同树`build/{npu-python,npu-python-resident-boundary}`。
运行前逐一核对C++/resident source hash、所有binary hash及core依赖，不能把dirty build改记clean。
该共享旧开发树已无live读者；正式任务仍只读新干净树与已冻结二进制。

全部资格终态后运行`.local/core5-boundary-study.py {cpu,npu}`：CPU在先、NPU随后，
一次独立进程/配置，完整训练计时、恢复在计时外，profile另进程。对照固定f03e9be。
CPU中型/D64显式FP64，NPU中型mixed-a/c与resident；D64单/双卡显式blocking及resident。
D64固定计划hash6c8cc36a…：480节点/15区域/2208边/9076975参数；resident仅显式容量queue4096/stages32/cache4，
来自静态Settle上界，不预跑真实轨迹。容量/数值/运行时失败即停，不重复同候选寻找通过。

## 接续与关闭

先`git status --short --branch`、`python scripts/status.py`、`python scripts/check_upstream.py`。
查看确切service、status与预期验收产物；live不能报passed。主项目可编辑，任务读冻结树。
结果进入新固定证据、ROADMAP的逐项表、相关契约和performance-lessons；最后证据单独提交并普通推送。
GPU延期、x86缺目标及所有严格失败继续分别列出。不恢复旧队列或重跑历史全矩阵。
