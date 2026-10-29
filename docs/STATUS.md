# 当前交接 — 2026-10-29

授权分支 `graph-execution-foundation`；数学固定 `0770593838a37c790a568226930567ed60f99b83`。
用户已授权候选配置、自然规模拓扑、CPU/NPU及混合/常驻有限测试；可提交并普通推送。
不使用子代理；参考仓库与公共工具链只读；GPU真机延后。构建用全部或一半可用CPU，
测试用空闲CPU/NPU/内存，项目重型计时与构建分开。

## 当前实现与开发检查

此前公共planning固定提交 `8f51ba6` 的CPU336项通过，见
[evidence/core5-planning-20261029.md](evidence/core5-planning-20261029.md)。
本轮新增有限case/execution目录、单次运行与结果查询；三个client的FP16/评分精度、
线程/packing/peer入口、实际resident梯度/SGD/master导出及standalone磁盘恢复。
导航：[core5-study](core5-study.md)。没有改数学定义、图或checkpoint公共版本。

开发快照 `artifacts/worktrees/core5-benchmark-dev3`（4cf9469加冻结diff）完成：
CPU/SDK各half=160线程重建、loader/help；CPU设施26项、实际standalone CPU6组与NPU8组通过，
包括mixed-a/b/c、resident、FP16、完整参数/梯度/SGD及精确恢复。
主工作区新增24个自然recipe的CPU FP64多步训练/独立调度/磁盘新owner检查通过。
这些是开发证据；不冒充即将固定的新提交资格。Python resident half/peer新参数化待固定资格。
开发v1缺头文件、v2 SDK环境入口、v2 FP16未启用runtime选项的原失败均保留在各job目录。

## 活跃任务与下一动作

开发v3两服务均passed/0，无活跃重型任务、无旧队列恢复。
实现已固定并提交 `372eb87`，只读checkout `artifacts/worktrees/core5-study-v1-qualified`。运行相关CPU资格、CPU/SDK standalone重建与真实消费者、
Python-owned NPU单卡/half/双卡consumer资格。源码与旧core/resident按内容hash核对，可复用相同库，
不继承与之无关的测试结论。新日志在 `artifacts/jobs/core5-study-v1-*`。
随后按有限计划串行比较主架构，结果包含严格失败、实际candidate/Full/消息及合批量。

当前根路径 `/mi/data2T/zlong/home/llm/graph-execution-foundation`，固定教材19+5份校验通过。
本机用`ascend/dev-workspace-9.1.0`及项目venv；SDK另用`libtorch-npu/2.10.0-cann9.0.0`和配套Python PATH。
公共工具链未改。静态可移植性审计1746文件、265项启发式warning/0 error；新入口告警为
测试逻辑设备常量和显式后端allocator adapter，未发现需要隐式fallback的实现。

## 未解除边界

[ROADMAP](ROADMAP.md)保留D64/D2048/反馈FP32、FP16 GDN/AdamW、默认异步双NPU、
晚注册NPU、resident流资源、容量/正支持及CUDA/x86待验的原始状态与重启条件。
新拓扑/自然规模不是旧数值失败的修复。上一生产resident资格 `04b5215` 的medium测量与
D64 CANN流资源失败见[紧凑纤维证据](evidence/core5-compact-fibers-20261029.md)。

接续先 `git status --short --branch`、`python scripts/status.py`、`python scripts/check_upstream.py`。
只维护本页当前交接、ROADMAP待办及相关契约/证据；长期经验见[performance-lessons](performance-lessons.md)。

固定资格服务 `core5-study-v1-{cpu,sdk,python}-20261029.service`，分别独立background.slice，
CPU/SDK half=160构建，CPU16000%/内存256GiB，Python设备CPU1600%/内存64GiB，时限各2400s。
Python请求2卡、SDK请求1卡，队列各最多180s。固定命令/阶段/log在对应job的qualification.json。
计时实验等待这些资格终态，不与构建重叠。

Python资格首轮在构建内容hash检查明确拒绝：旧resident源码hash含旧构建器（两线程默认），
未进入设备检查。保留failed记录；新服务 `core5-study-v1-python-v2-20261029.service` 使用本次
固定源码fresh resident build（half160、256GiB、2400s），再执行单卡/half/双卡消费者检查。

372eb87固定CPU365、standalone CPU6/NPU8通过；Python FP32单/双卡通过，FP16因用旧eager半精度
记录核对resident FP32中间算术而失败。正在接入已有独立storage参考，不放宽dtype检查。
另将resident换owner检查串行关闭/重建，避免同时占用两份设备程序/参数bank。
开发快照 `artifacts/worktrees/core5-storage-reference-dev`，固定patch/hash另存；CPU相关19项通过。
开发服务 `core5-storage-reference-dev-{python,sdk}-20261029.service`：Python2卡检查/CPU800%/64GiB/600s，
SDK半CPU160构建及1卡消费者/256GiB/1200s；新日志 `artifacts/jobs/core5-storage-reference-dev-*`。

精度参考修正开发Python3项passed；SDK编译/loader完成但末尾source-state守卫失败，
终态时源码与起始一致，疑为并发NPU检查在源码cwd创建的短暂scratch，不能冒充构建通过。
后续设备检查统一在job cwd，新的SDK开发数值服务 `core5-storage-reference-dev-sdk-check-20261029.service`
直接检查已编译binary（真实hash记录），正式资格仍须新的干净构建。

精度参考修正开发终态：CPU19项、Python NPU3项、SDK NPU8项通过；包括FP16完整记录dtype、
FP32真实梯度/SGD/master和串行关闭/重建后的精确恢复。即将单独提交该修正并固定资格，
之后开始已声明有限性能。前轮372eb87 CPU365/CPU消费者6/SDK消费者8证据继续按原来源记账。
