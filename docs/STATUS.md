# 当前交接 — 2026-10-28

继续用户授权A—E：已知限制、LH/fractal/core-3经验迁移、参数匹配与性能/正确性设施。
分支graph-execution-foundation；已推送2da3f2c。固定数学0770593838a37c790a568226930567ed60f99b83。
不使用子代理；参考仓库和公共环境只读。阶段提交后继续。

## 当前工作与下一步

本次实现提交：五族参数清单/预算匹配、可扩展FC/HB/DAG/PDG、二进制材料、共享KV低秩投影、
初始化冗余消除、resident共享ready副本与冻结参数bank、发布去重、完整比较错误字段路径。
CPU开发已完成，当前提交固定本增量；下一步干净副本复用同源二进制作固定资格，再进行有限性能对照。
当前新加纯Torch双卡诊断入口用于缩小异步错误范围；它不替代完整图训练资格。

## 活动任务

冻结dirty副本：artifacts/worktrees/core5-performance-dev2，b0321b9加归档修改；严禁编辑该副本。
两个服务均background.slice、Nice10、CPUQuota200%、MemoryMax64G、build2/ATen1。

| unit（均后缀-20261028） | 输出artifacts/jobs/ | 工作/预算 | 当前状态 |
| --- | --- | --- | --- |
| core5-performance-dev2 | core5-performance-dev2 | CPU core/参数bank与相关测试、standalone五族材料/完整训练；2400秒 | passed |
| core5-performance-npu-dev3 | core5-performance-npu-dev3 | SDK core/resident flow/config/训练/双卡；5400秒，队列180/设备2400秒 | running |

检查：`python scripts/status.py`；`systemctl --user show UNIT -p ActiveState -p Result -p ExecMainStatus`；
查看各输出的task.log、status.json、development.json或checks/result.json。停止：`systemctl --user stop UNIT`。
新NPU任务修正PATH：SDK配套Python优先，项目私有ninja随后；不修改公共module。

## 已终态的新结果

- 第一开发core5-performance-dev：CPU363 passed、2 strict xfail；随后standalone缺少ParameterRegistry头导致构建失败，整体failed。dev2已加入头文件修复。
- core5-performance-npu-dev2：配置期找不到ninja，failed；未编译或访问设备。dev3是有明确修复的新任务。
- 固定b0321b9的core5-half-python/core5-half-native均failed/exit2：首步同一conv梯度+3.35e-8/-2.98e-8，AdamW放大后step1差.00198364，原容差保持。
- 固定b0321b9的core5-async-stock failed/exit1：普通Torch拷贝仍在双NPU完整VJP出现invalid event，不能归因于自定义传输函数。
- 所有旧失败保留原终态，不恢复旧队列。源码、原始日志、fixture/witness hashes在各独立输出。

## 已有证据与保留边界

[上一固定资格](evidence/core5-review-fixes-20261028.md)：CPU442项，NPU FP32/FP16单/双卡训练恢复；CUDA sm80/90/100构建/CPU31项，无GPU真机资格。
异步双NPU eager VJP、eager FP16 GDN/AdamW、SDK FP32 affine及FP32反馈PDG严格失败尚未解除。
CUDA真机/x86_64、大规模容量和跨卡吞吐单列。旧性能只覆盖宽4/32，不能外推十亿级参数。
当前待办仅[ROADMAP](ROADMAP.md)，经验仅[performance-lessons](performance-lessons.md)，最终证据单独提交。

纯Torch补充诊断：core5-stock-peer-20261028，固定62073dd只读副本artifacts/worktrees/core5-stock-peer；默认异步/显式blocking各一进程，队列180秒，每进程90秒，总420秒，MemoryMax16G/ATen1。输出artifacts/jobs/core5-stock-peer，已终态failed/exit1：纯Torch默认异步同样invalid event，显式blocking通过；见数值隔离证据。

CUDA构建补验：core5-performance-cuda-dev2-20261028，同一dev2冻结源码，sm80/90/100 core及四resident组件；build2、CPUQuota200%、MemoryMax64G、5400秒、background.slice/Nice10。artifacts/jobs/core5-performance-cuda-dev2与build/cuda-core、cuda-resident。仅构建/CPU核验，无GPU真机结论。

开发终态：dev2的58项、standalone材料5项及24用例×4更新/精确checkpoint通过；主工作树最终参数/材料119项通过（含新增scaled core-3与严格标签、非默认alpha_agg）。尚不替代固定提交资格。NPU/CUDA服务继续读取旧冻结副本，C++与当前实现字节一致。

固定资格：core5-performance-qualify-cpu-20261028，源码2da3f2c只读副本artifacts/worktrees/core5-performance-qualified；复核并复用dev2同C++源码hash二进制，重新构建standalone消费者。相关完整CPU/材料/训练门禁，1200秒、CPUQuota200%、MemoryMax64G、background.slice/Nice10。输出artifacts/jobs/core5-performance-qualify-cpu，已passed：376 passed/2 strict xfail，24图用例×4更新及checkpoint通过。

有限CPU选型：core5-performance-study-cpu-20261028，固定2da3f2c同只读副本；baseline/new original-medium，480节点D64主体参数匹配，D2048四节点训练，480节点D2048约92.62亿参数推理，HB64全上限/稀疏/SSD及独立CPU profile。每配置1warmup+1测量，单进程900秒、总3600秒、MemoryMax192G、CPUQuota200%、ATen1。输出artifacts/jobs/core5-performance-study-cpu/runs。真实比较/容量失败保留；CUDA/NPU有限构建并行产生已知CPU争用，无其他本项目重型计时并行。

当前审查补丁：设备子进程cwd移入独立输出目录，防止NPU kernel_meta污染并行构建的源码身份；保留CUDA原failed终态（编译完成后身份检查拒绝）。新增有界v2→v1无损材料解码及CPU FP32/FP64分歧诊断，14项定向测试、小型native FP64两步诊断通过。随后提交并以新固定源码核验；数学函数/容差未改。
