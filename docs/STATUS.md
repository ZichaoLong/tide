# 当前交接 — 三平台矩阵、多卡放置与人类入口（2026-10-31）

授权分支 `graph-execution-foundation`；本轮起点实查 `3ddeff17df1a7d99014a275f52c47f069181306f`。
数学固定 `0770593838a37c790a568226930567ed60f99b83`，本轮 `check_upstream.py` 通过。
新任务覆盖 CPU/NPU/GPU 有限矩阵、同模型多卡容量和已安装 CLI 人类流程；允许有限数值超差的显式探索观察，原严格断言与失败不变。
参考仓库只读，允许阶段提交并普通推送。旧“不扩 FP32 / CPU-NPU 仅回归 / 两项优化”任务边界已由本轮授权替代；8B/16B 全扫描仍不在范围。

## 当前实施

公共 `tidegraph` 0.3.0 已实现配置/无分配plan、逐卡预算与共享放置、独立check、三客户端run、持久job、外部恢复及CSV/HTML查询；新增普通/blocking/显式CPU桥、严格/探索/未检查分类、完整/按需返回、独立packing/profile观测。下游可安装task.module例子保留真实参数/梯度/统一optimizer/checkpoint与插件源码锁。
本阶段实施范围与149项有限选点见 [platform-matrix](platform-matrix.md)，CPU39/NPU55/CUDA55；B128/B512、约1B、A/B/C同材料对照、真实反馈/非齐延迟/一般端口已明确。它们仍是计划，不能当作运行通过。

开发检查：142项联合CPU/静态测试通过；随后插件配置副本与参数归属加固的19项定向检查通过。新C++比较/独立根/通信及resident返回源码尚未构建，不冒称真机资格。
先提交本阶段实现，再从该不可变提交分别构建本机CPU/Python-owned NPU/SDK NPU与目标CUDA；完成构建后进行已安装入口演练、有限矩阵及六项热点对照，证据单独提交。

本机项目私有环境已补齐nlohmann_json3.11.2；公共栈未修改。本次预算核对修正：checkpoint新旧两代同时占盘；CPU多窗比较记录以host_results计费，设备当前完整记录以live_results计费，实际峰值仍须校准。
实现已固定 `de999c2fb344fcc40d3c4f39311cadd37d730bca`。正式矩阵尚未启动；构建driver为 `artifacts/platform-matrix/build-local.sh` 和 `build-cuda.sh`，只包装仓库公开构建命令。

## 可复用结果与未解除范围

上一轮 [CUDA 推理证据](evidence/core5-cuda-inference-20261031.md)及同名 JSON 保留：50 项正式中 47 项完整独立推理及磁盘精确续接通过，3 项 FP32 严格失败。
1.010B FP64 三客户端 B1/native B8、2.085B FP64 native 单卡容量点直接复用，不重跑整轮。
原训练、workers2、跨卡精确恢复、NPU resident 服务资源失败均保留；新探索结果不能改写旧严格资格。
12 方向 CUDA 直接 P2P/原生原子限制不解除；普通复制可用于新的 mixed 实验，多卡 resident 仍需原协议硬件门禁。

## 环境、源码和活动作业

当前构建：本机首个 `gef-platform-build-local-de999c2` 在CMake配置前因私有Ninja未加入PATH而失败，exit1，ControlGroup为空；已保留原日志。仅修私有启动器PATH，使用新输出/服务 `gef-platform-build-local-de999c2-r1`，源码不变。目标CUDA构建用 `/root/autodl-tmp/gef-cuda3090/jobs/platform-build-de999c2`，新只读源码 `sources/platform-de999c2`。远端仍在编译不能称通过。本机r1已在standalone编译终态失败：`run_check.cpp`的Owner歧义和`roots.h`缺失ExecutionPlacement声明。保留日志且ControlGroup为空。修复已提交14ce784（两处编译错误及无module任务非有限状态检查）；21项CPU定向测试通过。准备启动独立background.slice服务 `gef-platform-cpu-build-14ce784`，固定检出 `artifacts/platform-matrix/source-14ce784`，重建standalone消费者，公共CPU core复用de999c2且由构建器核对源码hash。日志 `artifacts/jobs/platform-cpu-build-14ce784/task.log`；尚无新构建结果。

本机日志 `artifacts/jobs/platform-build-local-de999c2-r1/task.log`，状态同目录status.json；输出 `artifacts/platform-matrix/builds-de999c2-r1`。查看/停止：`systemctl --user status/stop gef-platform-build-local-de999c2-r1`。远端查看/停止：私有SSH入口调用固定检出的 `scripts/container_job.py status/stop --output-dir /root/autodl-tmp/gef-cuda3090/jobs/platform-build-de999c2`。

远端 35 条旧作业均终态且 remaining_live_pids 为空；本机旧 NPU 作业均终态。
本轮只读审计：`artifacts/platform-matrix/{local-audit,remote-audit}.json`。

- 本机 aarch64，affinity 320 CPU，可用主存约 1.8 TiB、磁盘约 2.55 TB；NPU 4–11 审计时无进程，0–3/12–15 被其他项目占用，不停止或占用它们。每次运行仍须队列重新准入。
- 当前 Python 入口：`module load ascend/dev-workspace-9.1.0`，私有 `.local/venv-core5-inference`，Torch 2.12.0 / torch_npu 2.12.0.post2。现有 SDK 2.10/CANN9.0 必须独立环境构建运行，不能与 2.12/9.1 二进制混用。
- GPU 入口 `artifacts/cuda3090/ssh.sh`，根 `/root/autodl-tmp/gef-cuda3090`；四张 RTX3090/sm86，驱动595.71.05、CUDA13.0.88、Torch2.12.1+cu130，affinity112/配额56，内存240GiB，盘free47,268,773,888字节。四卡审计时各free24126MiB、无计算进程。
- 远端 dev 仍是 cc6b45f 上的独立脏目录，diff SHA256 `3619187d4319ba22ad6a6f6ae0c18451521c3a26b10452d9bbc91c35c05f31a5`；保持原状，不作为资格源码。
- 现存旧 core4960b6c、standalone fd3c72b、resident32edb0f 只保留真实构建身份，不能称为本轮当前源码的新构建。

本机长作业用独立 background.slice 服务及 NPU 队列；远端复用 container_job。固定源码后运行，不编辑活跃输入；重型构建和正式计时不重叠。

## 接续入口

```sh
git status --short --branch
python scripts/status.py
python scripts/check_upstream.py
python -S scripts/index_core5_study.py docs/evidence/core5-cuda-inference-20261031.json --view models --format tsv
```

下一步：提交编译/任务状态检查修复，创建新固定检出并构建CPU消费者；远端de999c2原作业继续，不编辑其源码。用户确认旧账户指南已不存在。已核对当前模块定义、驱动/16逻辑设备及空闲4–11；继续按现有私有栈、队列与background.slice规范操作，不重建旧指南或改公共环境。构建终态后安装wheel并用tidegraph公开入口逐组验收。
STATUS 是唯一当前交接，ROADMAP 是唯一待办；完整终态、命令、构建身份与新证据在后续提交补齐，不能将本页实施说明当作通过。

14ce784 CPU standalone构建/3项CTest和已安装smoke通过；源码不变的de999c2公共core复用，身份分别保留。远端de999c2的core及两个resident库已完成，standalone同两处编译错误失败，remaining_live_pids为空。接续修复相对site入口和standalone resident推理limits解码，增加显式静态FC B128/B512配置；默认拒绝配置保持。

当前固定实现eca070f；40项配置/CLI/矩阵定向检查通过。下一构建：本机 `gef-platform-cpu-build-eca070f`，远端 `jobs/platform-client-eca070f`，均仅重建消费者，复用源码hash核对通过的de999c2公共core/resident库；正式计时等待两机构建终态。

本机NPU构建服务 `gef-platform-npu-build-eca070f`：固定source-eca070f，输出builds-eca070f，Python2.12/CANN9.1与SDK2.10/CANN9.0分开构建；320 affinity线程，超时7200s；日志artifacts/jobs/platform-npu-build-eca070f/task.log。正式计时等待其终态。

GPU eca070f standalone构建及3项CTest通过，container作业终态passed/无遗留；本机对应亦通过。GPU安装入口的doctor/lock准备完毕。接着运行三客户端公开check/full（独立检查，不作为性能）：远端jobs/platform-checks-eca070f，最长1800s，日志同目录task.log。CPU d16 PyTorch full已在独立根比较严格失败（1/256坐标最大3.9339e-6），完整推理/精确恢复部分通过；不放宽阈值，clients组仍分别验小训练。

本机NPU构建终态失败：编译/链接和loader完成后，固定源码脏状态守卫拒绝。原因是同时wheel打包配置化下游示例时，setuptools在该检出生成未跟踪examples/consumer_core5/configured/build；没有已跟踪源码diff，但本次不能认证。保留日志，清点生成目录后移出源码；从干净同提交以新resident输出重建，后续wheel从独立打包副本构造。SDK阶段尚未开始。

NPU新构建服务为gef-platform-npu-build-eca070f-r1，独立完整新resident目录npu-resident-python-r1，重用已完成eca070f core。并行轻量独立检查（非正式性能）服务gef-platform-npu-checks-eca070f，通过空闲队列申请1张4–11内设备，状态artifacts/platform-matrix/npu-check-queue.json；日志artifacts/jobs/platform-npu-checks-eca070f/task.log。
