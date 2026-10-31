# 当前交接 — 三平台矩阵、多卡放置与人类入口（2026-10-31）

授权分支 `graph-execution-foundation`。当前实施固定 `972032fe96cc2ac38044ea6dadcfbfbcc40b6f87`，数学固定 `0770593838a37c790a568226930567ed60f99b83`；本次启动快照校验通过。用户确认旧账户指南文件已不存在；以当前模块/设备探测、项目私有环境、NPU队列和background.slice操作，公共栈与参考仓库保持只读。

## 实施与可复用证据

`tidegraph` 0.3.0 已安装公共配置/无分配plan、共享约束逐卡预算、独立check、三客户端run、持久job、外部恢复、CSV/HTML及可离线查询archive。支持普通/blocking/显式CPU桥，严格/探索/未检查分类，完整/按需返回和packing/profile观察；下游任务包保留独立参数、统一optimizer、源码锁及checkpoint。

[有限矩阵](platform-matrix.md)包含149个明确case，CPU39/NPU55/CUDA55；B128/B512、约1B、A/B/C同材料对照、反馈/非齐延迟/一般端口均入库。尚未运行的case不能称通过。后续主线为实际矩阵、同模型多卡/容量、人类完整演练和六项热点处置，见[ROADMAP](ROADMAP.md)。

上一轮[CUDA推理报告](evidence/core5-cuda-inference-20261031.md)50项中47项完整独立推理/磁盘精确续接通过及3项FP32严格失败直接复用；不重跑旧矩阵、不改原阈值。本轮显式探索可以继续有限数值超差成本，非有限/离散/结构/通信/求解器/容量失败仍拒绝。12方向CUDA无直接P2P和原生原子，多卡resident门禁不解除。

## 本阶段构建与检查

- CPU公共core `de999c2`、standalone `eca070f`：源码hash由构建器核对；standalone构建及3项CTest通过。不能称旧core在972032f重新构建。
- CUDA公共core/Python与standalone resident `de999c2`、standalone消费者 `eca070f`：构建及3项CTest通过。远端所有这些作业终态、无存活后代。
- CPU所选d16 FP32公共full check：完整推理/精确恢复通过，独立VJP一坐标超原阈值，整体失败；训练由独立clients case另验。
- CUDA同材料full check：PyTorch与Python/native在独立根严格失败；standalone完整推理、独立根、训练与恢复通过。原始记录 `gpu3090:runs/platform-check-eca070f-*`。
- NPU PyTorch/native同材料full check：完整推理/精确恢复后，input根一坐标分别差2.8610e-6/2.8312e-6超原阈值，整体失败；不吞掉该失败。
- 972032f报告/静态矩阵定向检查14项通过；CPU已安装doctor实际4×4矩阵前向/VJP通过。前阶段142项开发及40项入口定向检查保持其原范围。

## 活动作业与环境

NPU构建 `gef-platform-npu-build-eca070f-r1` 已passed/exit0、ControlGroup为空，固定检出 `artifacts/platform-matrix/source-eca070f`。
Python-owned resident、SDK2.10/CANN9.0 core/resident/消费者全部完成；消费者3项CTest通过。日志 `artifacts/jobs/platform-npu-build-eca070f-r1/task.log`，状态同目录 `status.json`。查看/停止：

```sh
systemctl --user show gef-platform-npu-build-eca070f-r1 -p ActiveState -p SubState -p ExecMainStatus -p ControlGroup
systemctl --user stop gef-platform-npu-build-eca070f-r1
```

旧首次构建因私有Ninja PATH失败、de999c2 standalone两处声明错误、eca070f源码洁净守卫拒绝均保留日志；后者是wheel生成未跟踪build目录，不是数学失败。已移走生成目录并用全新resident输出重建，未编辑活跃源码。SDK与Python-owned ABI不混用。

本机aarch64/320 affinity CPU，主存与磁盘余量充足。NPU物理4–11最近探测无进程，其他设备被其他项目占用；运行时仍经队列重新检查。Python私有栈Torch2.12/TorchNPU2.12/CANN9.1；SDK独立2.10/CANN9.0。
CUDA四张3090/sm86，Torch2.12.1+cu130/Toolkit13.0，CPU配额56、主存240GiB、数据盘50GiB。最近四卡各free24126MiB且无计算进程、盘free44GiB。远端dev原修改保留，不用于资格。

当前安装 `artifacts/platform-matrix/install-972032f/venv`；本机site `cpu-site-eca070f.json`、`npu-site-eca070f.json` 在同父目录。目标安装 `/root/autodl-tmp/gef-cuda3090/install-platform-972032f`，site仍引用真实旧构建身份。安装、源码与库各有身份，运行期间不修改。

## 下一动作与公开命令

两机重型构建均已终态。GPU首组 `gpu3090:jobs/platform-cuda-coverage-972032f` 已终态（12项小型层级/拓扑覆盖，严格失败保持）。下一活动组 `gpu3090:jobs/platform-cuda-eager-972032f` 为20项clients/batch/packing/modules/placement/precision的显式合并，包含B128/B512与A/B/C，固定安装972032f与eca070f/de999c2真实库身份，单case900s上限。其后顺序执行各平台coverage/clients、batch/packing等组，审阅结果后逐项放行scale/multicard/capacity，profile独立进程。大点每项结束后审阅小记录并按保留规则处理生成文件，避免填满GPU数据盘。

```sh
git status --short --branch
python scripts/status.py
python scripts/check_upstream.py
# 在匹配的已安装环境；所有NEW路径必须新建
tidegraph plan configs/matrices/cpu-batch.json --matrix --site SITE --format json --output NEW_PLAN.json
tidegraph matrix configs/matrices/cpu-batch.json --site SITE --output-dir NEW_JOB --detach
tidegraph job status NEW_JOB
tidegraph job logs NEW_JOB --lines 80
tidegraph results NEW_JOB --format archive --output NEW_REVIEWED.json
tidegraph results NEW_REVIEWED.json --format html --output NEW_REPORT.html
```

gpu3090通过私有 `artifacts/cuda3090/ssh.sh`，无systemd时使用已安装container_job；本机独立background.slice，NPU等待放在服务内部。原始绝对路径仅定位当前材料，干净检出按[人类入口](human-workflow.md)/[开发文档](development.md)重新生成。STATUS唯一当前交接，ROADMAP唯一待办；当前阶段资格证据尚未提交，不能把实施或running写成通过。

接续开发修正：已安装矩阵显示当前case/log和长独立检查阶段；Python eager推理预算排除不存在的第二份设备权重bank（主存/C++/训练预算保留）。47项运行/放置/作业定向检查通过，待提交后安装新版本用于后续容量准入；正在运行的972032f安装未改动。
