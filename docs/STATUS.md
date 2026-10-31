# 当前交接 — 三平台矩阵、多卡与人类入口（2026-10-31）

授权分支 `graph-execution-foundation`；数学固定 `0770593838a37c790a568226930567ed60f99b83`，启动离线快照校验通过。当前实际运行包固定 `1b3e39f`；CUDA首组32项固定972032f。当前开发还补充结果归档v2和简明job状态，不能将这些报告改动冒称旧二进制重新构建。旧账户指南文件确已不存在，按用户确认、当前模块/设备/队列继续；只读参考与公共栈未改动。

## 已交付与本轮终态

公共`tidegraph` 0.3.0提供plan/共享约束逐卡预算、三客户端run/check、持久job、恢复和CSV/HTML/归档。人类入口见[操作手册](human-workflow.md)，配置及149个有限选点见[矩阵](platform-matrix.md)。安装包、共享核心和standalone各保存真实身份。尚未运行的case不称通过。

| 本轮固定任务 | 实际结果 |
| --- | --- |
| GPU coverage，972032f，12项 | 全部执行/离散/精确续接通过；7严格通过、5数值严格失败 |
| GPU eager，972032f，20项 | 全部执行/离散/精确续接通过；7严格通过、12数值探索接纳、LH异构1项严格失败。三客户端小型训练均通过本配置检查；不解除旧独立根失败 |
| CPU eager，1b3e39f，32项 | 全部执行/离散/精确续接通过；15严格通过、12数值探索接纳、5严格失败（HB反馈及LH四项）。含三客户端训练、B128/B512和A/B/C、1/4/16 ATen线程；独立调度workers另列 |
| GPU scale pilot，1b3e39f，99.49M/1.010B B1 | 两项执行/离散/精确续接通过，数值超差explore接纳；1B构造峰值约4.780GB，计时窗约4.322GB，恢复63.03秒；不是严格吞吐 |

GPU B128/B512成本中位2.504/8.605秒，A/B/C为4.191/2.504/1.841秒。CPU B128/B512为1.036/3.889秒，A/B/C为1.659/1.036/1.020秒。均有原FP32数值超差，完整工作量/首差/环境/返回口径须在审阅证据中保持；不提前宣称等价加速。

这些终态外层记录均已核实无后代；CPU unit `tidegraph-45df5b5ac73f449c.service` failed/exit1/ControlGroup为空，失败源于保留的严格case。GPU容器任务分别在目标根`jobs/platform-cuda-{coverage,eager}-972032f`及`jobs/platform-cuda-scale-pilot-1b3e39f`，无存活后代。
本轮审阅证据尚待统一提交。已有小记录本机`artifacts/platform-matrix/cuda-initial-raw-records.json`及两个v1结果投影；CPU原记录在`artifacts/jobs/platform-cpu-eager-1b3e39f/run`。v2归档保留展开配置、实际身份、原逐窗/准备/检查成本、严格诊断和profile；旧v1缺失信息明确标记，不反推。

上轮[CUDA推理](evidence/core5-cuda-inference-20261031.md)47/50完整资格及原FP32失败直接复用。12方向无直接P2P/原生跨卡原子，多卡resident门禁不解除。旧CPU/NPU/FP16/独立根严格失败仍在[ROADMAP](ROADMAP.md)，小训练通过不解除它们。

## 活动作业

- 本机 `gef-platform-npu-eager-1b3e39f.service`：background.slice、KillMode=control-group、RuntimeMaxSec=37800。已等CPU终态后经队列分配**物理11→逻辑npu:0**，32项coverage/clients/batch/packing/modules/placement/precision正在运行，每项900秒上限。固定source/install-1b3e39f。外层`artifacts/jobs/platform-npu-eager-1b3e39f/{status.json,task.log}`；逐项`artifacts/platform-matrix/npu-eager-run-1b3e39f`；队列状态同父目录`npu-eager-queue-1b3e39f.json`。
- gpu3090 `jobs/platform-cuda-observations-1b3e39f`：4项已终态，完整/离散/恢复均检查，数值超差explore接纳；profile实际kernel已记录，不作正式吞吐。下一提交 `jobs/platform-cuda-scale-large-batch-1b3e39f`，1.010B B128/B512，每项1800秒、固定install-platform-1b3e39f，完整返回。静态设备6.424/12.764GB、主存35.552/61.446GB、磁盘10.660/11.923GB，运行前live准入；窗口不拆逻辑batch、不改缓存或容差。

```sh
systemctl --user show gef-platform-npu-eager-1b3e39f -p ActiveState -p SubState -p ExecMainStatus -p ControlGroup
systemctl --user stop gef-platform-npu-eager-1b3e39f
artifacts/platform-matrix/install-1b3e39f/venv/bin/tidegraph results artifacts/platform-matrix/npu-eager-run-1b3e39f --format table
artifacts/cuda3090/ssh.sh '/root/autodl-tmp/gef-cuda3090/install-platform-1b3e39f/venv/bin/tidegraph job status /root/autodl-tmp/gef-cuda3090/jobs/platform-cuda-observations-1b3e39f'
# 同命令将job status换job logs/stop；停止后核实终态和无后代
```

## 环境与构建身份

本机aarch64、320 affinity CPU、约1.8TiB可用主存、2.4TB空盘。NPU16逻辑设备/各64GiB，物理4–11最近空闲，实际运行始终经过队列复核。Python私有Torch2.12/TorchNPU2.12/CANN9.1与SDK2.10/CANN9.0分离；site位于`artifacts/platform-matrix/{cpu,npu}-site-eca070f.json`。
GPU四张3090/sm86，Torch2.12.1+cu130/Toolkit13.0，配额56 CPU/240GiB主存，数据盘50GiB。初始四卡free24126MiB无计算进程、盘free44GiB；大点每次重新核对。远端dev原脏目录保持原样，不作资格源码。

CPU core来自de999c2、CPU standalone来自eca070f；NPU Python/SDK core/resident/standalone来自eca070f；GPU core/resident来自de999c2、standalone来自eca070f。三平台standalone各3项CTest通过；最新NPU构建服务终态passed/ControlGroup空。原Ninja PATH/声明/源码洁净守卫失败保留日志。loader/source hashes核对过，未重新构建的二进制不改来源。全部重型构建终态，不与计时重叠。

当前活动安装不修改。本机`artifacts/platform-matrix/install-1b3e39f/venv`，GPU`/root/autodl-tmp/gef-cuda3090/install-platform-1b3e39f`；旧972032f安装保留供原任务恢复。新归档/status相关13项回归通过（先前报告/CLI16项通过），仅报告和状态变化，不改变计算路径。

## 下一动作

1. 审阅NPU32与GPU4观察终态，保存完整配置、实际实现身份、首差、工作量和逐卡峰值。
2. 继续GPU FC多节点/HB/LH约1B、1B B128/B512、单/2/3/4卡mixed、训练及resident默认/有界配置；每个大点先live准入，按保留规则审阅后退役大checkpoint，避免累积填满50GiB盘。
3. NPU当前组结束后继续CPU独立packing/profile与CPU7规模点、NPU规模/1/2/3/8卡/训练/resident；不得将构造、外层完成或自恢复当独立资格。
4. 按[人类入口](human-workflow.md)完成三平台已安装consumer的base/task check、run、外部新进程恢复、status/log/stop/resume和HTML/CSV演练。九份配置已由安装包生成在`artifacts/platform-matrix/human-configs-1b3e39f`，GPU同名目录已复制，尚不能写成演练通过。
5. 依据新profile处置六项热点、校准真正近界点并记录硬限制；原约2B FP32并非24/64GiB显存近界。单卡/多卡预算含恢复/两代磁盘，不能只加空闲显存。
6. 分阶段提交审阅证据、同步ROADMAP/索引/经验并普通推送，继续至授权范围闭环。

干净检出从开发文档重建环境/库、从跟踪JSON选case；不依赖上述原运行目录。历史身份用`python scripts/resolve_history_commit.py OLD_SHA`定位等价源码，不能当重新资格。STATUS唯一交接，ROADMAP唯一待办。

归档体积审阅：CPU32项展开v2约12MB，新增可直接查询的gzip归档与HTML折叠诊断，保留相同内容和内容摘要；24项相关回归通过。23项前版本检查已在固定de93246检出通过；两机de93246另装报告CLI，未改活动1b3e39f安装。
