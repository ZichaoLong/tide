# 当前交接 — 三平台矩阵、多卡与人类入口（2026-10-31）

授权分支`graph-execution-foundation`，数学固定`0770593838a37c790a568226930567ed60f99b83`，离线快照检查通过。实现最新`bd2ff2e`；当前任务分别冻结在972032f/1b3e39f/bd2ff2e，未编辑活动安装。远端dev原脏目录保持。旧账户指南确已不存在，依据用户确认和实际模块/队列继续，不修改公共栈及只读参考。

## 已完成与证据

[三平台首组审阅证据](evidence/core5-platform-baseline-20261031.md)及同名gzip归档包含102条：98正式流程、4独立GPU观察。43严格通过、38数值探索接纳、17严格失败；全部执行/离散/精确续接通过。三平台B128/B512、A/B/C、真实层级差异和三客户端小型训练均有实际记录。超差成本不称严格合格吞吐，旧失败不解除。

| 终态任务 | 固定源码/结果 |
| --- | --- |
| CPU32 | 1b3e39f；15严格通过、12探索、5严格失败；外层exit1，ControlGroup空 |
| NPU32 | 1b3e39f；14严格通过、12探索、6严格失败；物理11→npu:0；外层exit1，ControlGroup空 |
| GPU coverage12/eager20 | 972032f；合计14严格通过、12探索、6严格失败；无存活后代 |
| GPU scale pilot2 | 1b3e39f；99.49M/1.010B B1执行/离散/恢复通过、数值探索；无存活后代 |
| GPU观察4 | 1b3e39f；A/B/C packing及独立profile，不作吞吐；小isfinite/all及private BMM热区记录在证据 |

固定bd2ff2e相关78项检查通过，日志`artifacts/platform-matrix/qualification-bd2ff2e.log`。新增显式finite-check packing和逐节点活动样例观察，原149项身份不变，新增18项有界对照；总167个跟踪选点。默认scalar、严格容差、独立调度不变。尚未实测的配置不称通过。

公共`tidegraph`0.3.0已实现plan/预算/共享约束放置、check/run、持久job/status/logs/stop/resume和CSV/HTML/gzip归档，见[人类手册](human-workflow.md)、[有限矩阵](platform-matrix.md)。报告v2保留原逐窗、配置、身份、严格诊断，已固定源码相关验证；旧C++库保持原来源。

GPU人类演练`jobs/human-cuda-de93246`已终态且无后代：doctor/安装锁/consumer公式、task related、训练/推理新进程恢复通过；base related的独立参数根原严格超差保留。stop在window2后cancelled/143且无后代，恢复到window8完成；后者metrics/validation=none明确未独立验收。完整argv及退出码在`runs/human-cuda-de93246/walkthrough.json`，尚待取回提交人类证据；CPU/NPU演练待做。

## 活动作业

- 本机`artifacts/jobs/platform-cpu-finite-bd2ff2e`，unit `tidegraph-3bd74d8ef77b47c5.service`，background.slice/控制组隔离，CPU ATen4/interop1/worker1。固定wheel/source bd2ff2e，六项各900秒；B128 scalar/packed、小型训练两策略、B512 packed和单独profile。运行中，无最终收益结论。安装`artifacts/platform-matrix/install-bd2ff2e`的source.json/lock记录来源。
- gpu3090 `jobs/platform-cuda-scale-large-batch-1b3e39f`：固定install-platform-1b3e39f，1.010B FP32 native mixed-c B128/B512、P2、完整返回，每项1800秒。B128执行/离散/精确续接通过、数值超差；B512两测量窗和精确续接已完成，仍在独立检查，不能称通过。静态设备6.424/12.764GB、host35.552/61.446GB、disk10.660/11.923GB；实际B512窗口峰值5.278GB，恢复另计。GPU0使用，GPU1人类小型非计时演练曾短暂重叠，记录背景。

```sh
artifacts/platform-matrix/install-bd2ff2e/venv/bin/tidegraph job status artifacts/jobs/platform-cpu-finite-bd2ff2e
# 上述job status可换job logs或job stop；停止后核实无后代/ControlGroup空
artifacts/cuda3090/ssh.sh '/root/autodl-tmp/gef-cuda3090/install-platform-bd2ff2e/venv/bin/tidegraph job status /root/autodl-tmp/gef-cuda3090/jobs/platform-cuda-scale-large-batch-1b3e39f'
```

## 环境与预算

本机aarch64、320 affinity CPU、约1.8TiB可用主存/2.4TB空盘；NPU16×64GiB，物理4–11为候选，始终经合作队列重新检查并锁定；不打扰其他项目。Python私有Torch2.12/TorchNPU2.12/CANN9.1，standalone SDK2.10/CANN9.0独立ABI。实际site：`artifacts/platform-matrix/{cpu,npu}-site-eca070f.json`。
GPU四张3090/sm86，Torch2.12.1+cu130/Toolkit13，CPU配额56、RAM240GiB、数据盘50GiB；最近空盘31GB。大点每次live准入，先审阅再退役终态大checkpoint，不能无限累积。12方向无直接P2P/原生跨卡原子边界未解除，普通复制不认证resident peer协议。
CPU core de999c2/standalone eca070f；NPU core/resident/standalone eca070f；GPU core/resident de999c2/standalone eca070f。全部重型构建终态，三平台standalone各3个CTest通过；本阶段未改C++，不重复构建、不改旧来源。
报告安装de93246、测量安装1b3e39f保持。bd2ff2e新Python安装已在两机建立、库和loader hash核验；源码固定检出`source-bd2ff2e`，原二进制可复用的源码hash证据各自保存。

## 下一动作

1. 审阅CPU finite与GPU1B B128/B512终态；新finite策略分别在NPU/GPU运行同材料对照和小训练，收益分平台，默认不变。
2. GPU继续FC多节点/HB/LH约1B、同模型2/3/4卡、三通信小训练、训练容量和单卡resident默认/有界点。单项数值失败不阻塞独立可行项；每项预算与内部拒绝保存。
3. NPU32已结束，继续CPU独立packing/profile和CPU规模7项，NPU规模/1/2/3/8卡/训练/resident。按实际峰值校准真正近界点；原约2B FP32不是24/64GiB显存近界，不能冒称。
4. 完成CPU/NPU已安装consumer全流程；取回GPU人类原记录。基座检查与下游任务验收、外层完成与内层失败分列。
5. 六项问题逐平台/规模/多卡重新处置；当前有依据的优化是finite诊断packing。跨节点批、resident布局、反向绑定须有实际profile和预算，不机械照抄后置。
6. 同步唯一ROADMAP、契约/索引/经验，阶段提交证据并普通推送，继续到本轮授权范围闭环。

从干净仓库按development/human-workflow安装，用configs/matrices跟踪材料和site生成新运行；审阅gzip无需旧运行目录。历史身份通过`resolve_history_commit.py`定位等价源码，不当成重新验证。STATUS唯一交接，ROADMAP唯一待办。
