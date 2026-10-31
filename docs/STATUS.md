# 当前交接 — GPU 规模推理与有限优化（2026-10-31）

授权分支 `graph-execution-foundation`，起点 `cbb0964`；固定数学
`0770593838a37c790a568226930567ed60f99b83`，快照校验通过。本轮允许约1B和安全近显存边界推理，
不重跑旧训练矩阵或8B/16B扫描。用户允许子代理；主代理统一设备、构建、集成和普通推送。
参考仓库只读，远端 `dev` 原cc6b45f未提交修改完整保留。旧失败与多卡resident硬件限制不解除。

## 固定实现与已完成事实

| 源码 | 内容与实际状态 |
| --- | --- |
| eee68c4 | 推理完整独立前向/磁盘新owner精确续接、不构造optimizer；staggered-v1真实不齐输入。GPU有限21项18通过、3项d256 FP32完整keys严格失败；resident推理/训练两dtype四项通过。原core/resident库源码hash匹配并保留32edb0f构建身份 |
| 1c903ae | 实际桶/MM/BMM计数、活动/缓存摘要、观察不能成为吞吐。新core构建通过，standalone编译因work命名歧义失败；失败保留。接续4个profile和两客户端before均通过完整前向；两个正式before还通过磁盘精确续接 |
| 5f9bff1 | 已有批次的detach bool支持/有限性元数据合并取回，CPU/单例保留；CPU相关57项通过 |
| 4960b6c | 首差/独立CPU FP64局部KV诊断、显式单卡CUDA eager FP64、修复standalone限定名。新core/standalone构建通过，相关66项、三客户端GPU选定训练/独立根/共享/optimizer/恢复9项通过；有限测量已终态passed；FP32诊断strict_mismatch作为预期观察保留 |
| 9fce486 | 显式FP64目录与1B/近界静态预算、离线CUDA同步API区间并集；规划15项、观察器11项通过。自动准入恒false，近界须实际1B峰值校准 |

以上提交已普通推送。第二且最后有限优化已提交/推送 `2f17c24`：
eager显式 `return_packing=scalar|packed`，默认scalar；完整返回8MiB组、逐字段独立CPU clone，
不改数学/调度/live continuation/checkpoint，resident packed拒绝。参考仍独立scalar。
还补恢复后的allocator峰值观察。CPU定向检查通过；GPU/standalone构建及实际资格待下一固定作业。

## 现场、活跃作业和日志

GPU入口 `artifacts/cuda3090/ssh.sh`，根 `/root/autodl-tmp/gef-cuda3090`，私有 `venv/bin/python`。
四张RTX3090/sm86；Torch2.12.1+cu130、CUDA13.0.88、GCC11.4；affinity112、CPU配额56、内存240GiB。
启动数据盘约48.2GiB空闲，后续每个大材料前复核。实际12方向直接P2P仍false，不改协议/驱动。
GPU正式计时固定物理0/logical0、intraop/interop/workers=1、FP32 IEEE/关闭TF32；与项目重型构建串行。

已终态容器作业 `inference-d-4960b6c`，来源只读 `sources/4960b6c`，timeout7200，
监督PID43272已退出，exit0、remaining_live_pids为空；进度/内部终态在 `runs/4960b6c-inference/stages.json`。
状态/日志 `jobs/inference-d-4960b6c/{status.json,task.log}`；新构建 `builds/4960b6c-{core,benchmark}`。
不得编辑活跃源码或把running称为通过。检查/停止：

```sh
artifacts/cuda3090/ssh.sh '/root/autodl-tmp/gef-cuda3090/venv/bin/python /root/autodl-tmp/gef-cuda3090/sources/4960b6c/scripts/container_job.py status --output-dir /root/autodl-tmp/gef-cuda3090/jobs/inference-d-4960b6c'
# 同一入口以 stop 替换 status；终态须核对remaining_live_pids为空。
```

已终态：`inference-baseline-cbb0964` passed；`inference-a-eee68c4` failed（18/21单元通过）；
`inference-b-1c903ae` failed（core通过/standalone编译失败）；`inference-b-observe-1c903ae` passed，均无存活后代。
本机 `gef-inference-npu-a-eee68c4` background.slice服务success/exit0、ControlGroup为空，队列分配物理9/logical0；
同FC d8/B8/P4纯PyTorch推理/续接通过，中位559.661ms。私有环境 `.local/venv-core5-inference`，
模块 `ascend/dev-workspace-9.1.0`；不碰其他项目设备0–3。记录 `artifacts/jobs/inference-npu-a-eee68c4`、
输出 `artifacts/cuda3090-inference/npu-a`。

## 下一步

1. 等待并审阅D：FP32首差诊断、FP64三客户端及d64M、同材料优化前后。原FP32失败不因FP64通过解除。
2. 运行已准备 `inference-extra-9fce486.py`：GPU literal kernel、零支持两客户端训练回归，离线解析原8份trace；无重型构建。用container_job新目录 `inference-extra-9fce486`、timeout1200，来源sources/9fce486。
3. 集成完整返回packing，先提交后构建standalone（core hash相同时复用D并记录身份），小型完整推理/训练/恢复及同材料对照；本机NPU少量前后与相关正确性检查串行。
4. 小型FP64通过后按 `planning.inference_budget.fc_inference_budget` 与实际显存/主存/磁盘准入约1B B1/B8，再决定d6144近界。保留至少4GiB GPU、8GiB磁盘余量；不OOM搜索。
   Python/native近界单bank条件点；standalone事务恢复需两bank，近界静态拒绝。resident大width KV Full guard独立拒绝，不盲目调预算。
5. 六项处置、完整实测范围、严格失败与有限收益写入单独审阅证据、索引、performance-lessons，替换本STATUS、更新ROADMAP并普通推送。

A/B审阅输入：`artifacts/cuda3090-inference/stage-{a,b}-raw.json`；六项草稿 `packing-review.md`。
它们是当前整合材料，最终干净仓库复现不能依赖这些目录。命令和配置源已入库。

当前本机下一服务 `gef-inference-npu-d-4960b6c`，固定sources4960b6c纯PyTorch同材料，物理9队列、timeout600；job artifacts/jobs/inference-npu-d-4960b6c，日志与输出artifacts/cuda3090-inference/npu-d*。终态后再启动extra GPU作业；不与构建/正式计时重叠。

extra作业已运行，GPU literal和两客户端zero-support训练检查通过，仍在离线解析8份trace。终态后启动F：来源2f17c24，job inference-f-2f17c24、timeout5400，driver inference-f-2f17c24.py；仅重建standalone benchmark/返回复制检查器，core复用4960b6c且核对hash。随后真实GPU完整返回literal/反馈两窗、三客户端小型训练恢复、FP32同材料前后和FP64曲线至64M。输出runs/2f17c24-inference。

NPU D服务已success/exit0、ControlGroup为空；同卡9同材料独立推理/续接通过，中位527.660ms（A为559.661ms，单次观察）。F本机后续只做相关literal、FP32小图与同材料推理，不复制GPU矩阵。

F原job终态failed：standalone build通过，GPU定向33项和C++FP32/FP64字面复制通过；FP16在test fixture的ATen literal构造报tensor_cpu Half未实现，尚未进入复制。修复只使字面double显式转到目标storage，不改返回/数学；随后新固定提交重新构建checker和benchmark，复验literal并接续剩余训练/测量，原失败保留。

下一固定接续H：fd3c72b（只修复FP16检查材料构造），job inference-h-fd3c72b，timeout5400，driver inference-h-fd3c72b.py；新builds/fd3c72b-benchmark，输出runs/fd3c72b-inference。F33项生产路径GPU检查不重复；新C++literal三dtype及剩余三客户端训练/有限测量在H运行。

大规模有限选点扩为约1B B1三客户端共同材料，加native B8；最大近界仍仅native。显式FP64 payload/scoring和return_packing=packed，原FP32失败不解除。每配置一次正式流程，另仅native B1/B8做无profiler的1窗工作计数。先核对new-owner恢复峰值、磁盘/主存及>=4GiB设备余量；完成后先审阅大checkpoint/fixture哈希和退役dry-run，再生成近界，避免50GiB盘长期堆积。
