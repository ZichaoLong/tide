# 当前交接 — 2026-10-29

授权分支`graph-execution-foundation`；上游固定`0770593838a37c790a568226930567ed60f99b83`。
本轮逐项推进非GPU限制，GPU延后，无子代理；已获修改、有限实验、提交与普通推送授权。
参考仓库与公共工具链只读。原输入/容差/optimizer和严格失败保持。

## 已固定的实现与证据

起点121f5b1，对照f03e9be。诊断bdd7f6c/证据fe5e6e7，eager e0d6e27/证据9d1a54c均已提交推送。
当前实现`15c3e65fe3a737324c90bdf26b60c28a34ad2dcf`已提交推送；
resident因子KV、实际allocation部分失败行、反向阶段观测、benchmark实际梯度/SGD物理owner/磁盘恢复/跨卡/显式容量。
[本次固定资格](evidence/core5-resident-boundary-20261029.md)：CPU设施10、公共NPU14、SDK组件9、
两套eager各6组通过；生产eager继承e0d6e27 CPU517 passed/2原FP16 strict xfail，不重复旧全量。

所有开发、正式资格和初始化探针已终态，原失败在报告/原jobs中保留。
Python首次混合CPU/NPU同进程资格failed/1（15 passed/9 failed），原因是CPU autograd后晚注册NPU；
独立NPU资格v2 passed/0，标量early/late新进程探针确认先resolve_device可用、晚注册仍失败。
SDK正式资格passed/0，约713秒设备占用。新CUDA构建/真机与x86未验；GPU按用户延后。
D64/D2048/反馈FP32、FP16 eager GDN、默认异步双NPU的停止/重启条件见[ROADMAP](ROADMAP.md)，未解除。

## 当前活跃任务

仅`core5-boundary-study-cpu-20261029.service`：固定15c3e65干净树
`artifacts/worktrees/core5-boundary-final-qualified`，启动器`.local/core5-boundary-study-cpu.sh`，
driver`.local/core5-boundary-study.py cpu`；输出`artifacts/jobs/core5-boundary-study-cpu/{status.json,task.log,study.json}`。
background.slice、单线程、RuntimeMaxSec2100/MemoryMax32G、进程RLIMIT_AS32GiB。
先确认每配置终态及完整核对，再读计时；运行中不是通过。四个一次完整训练对照后，仅after D64通过才独立profile。
CPU对照基线树`artifacts/worktrees/core5-boundary-baseline`固定f03e9be；新旧C++和binary hash已各自核对。
当前实现提交与正式资格证据分开，性能尚待完成，不把源码优化直接表述为收益。

## 精确接续

先`git status --short --branch`、`python scripts/status.py`、`python scripts/check_upstream.py`。
检查`systemctl --user show core5-boundary-study-cpu-20261029.service`及上述status/study/log。
CPU全部终态后再启动NPU study，禁止两个测量并行：
- unit `core5-boundary-study-npu-20261029.service`，同一冻结树；
- `/bin/bash .local/core5-boundary-study-npu.sh`（脚本在主仓绝对路径），driver同上参数npu；
- background.slice、Nice10、KillMode=control-group、MemoryMax32G、RuntimeMaxSec3300，launcher日志`.local/core5-boundary-study-npu-launch.log`；
- `artifacts/jobs/core5-boundary-study-npu`须不存在，由job.py创建；排队180秒/两卡，设备阶段上限3000秒。

NPU只做一次/配置：前后mixed-a/c中型、当前resident中型，D64 native单/双卡显式blocking及resident。
D64固定plan hash6c8cc36ae97287133d40e67ccc8e137574b80e2df420feff6adfef3395397bb3，
480节点/15区域/2208边/9076975可训练参数，对core-3 Add多47；恢复在计时外。
resident静态queue4096/stages32/cache4、optimizer_bytes1GiB（1/16子分区覆盖36.3MB master），总预算仍5GiB。
其余子预算不变，无真实轨迹预跑。容量/数值/运行时失败即停，不恢复同候选或改输入容差重测。

只读复用构建：CPU在`artifacts/worktrees/core5-boundary-dev/build/cpu`；NPU在
`artifacts/worktrees/core5-independent-qualified/build/{npu-python,npu-python-resident-boundary}`；
SDK另用同树`{npu-core-v2,npu-resident-v2}`。保留dirty-origin归档，formal source/所有binary内容hash另核验。
不能更新这些运行输入。主工作树只有证据/当前交接修改，代码已固定。

完成后审核实际完整结果、梯度/槽/恢复、工作量和分阶段成本，区分通过/严格失败/容量拒绝/超时。
结果进入小型性能证据、ROADMAP逐项表、相关契约和performance-lessons；证据单独提交并普通推送。
不重跑历史全矩阵、不恢复旧队列；继续推进至本轮有限任务终态。
