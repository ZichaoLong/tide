# 当前交接 — 三平台矩阵、多卡与人类入口（2026-10-31）

授权分支`graph-execution-foundation`；本次接续审计HEAD=`528025c`、工作区干净；报告实现固定`1b46dc5`，前序证据已推送`528025c`，不回退后续提交。数学固定`0770593838a37c790a568226930567ed60f99b83`，离线校验通过。旧账户指南确实不存在，依用户确认、项目实测site和合作队列继续。远端脏dev已核对并保留，不作资格源码。

## 已审阅结果

- [首组矩阵](evidence/core5-platform-baseline-20261031.md)：98正式流程+4观察，43严格通过、38数值探索、17严格失败；三平台中等模型B128/B512、A/B/C、真实图层级及三客户端有实际记录。
- [规模阶段](evidence/core5-platform-scale-20261031.md)：CPU七项、GPU五项、NPU七项规模正式流程，含约1B/B128/B512及FC16/HB20/LH20；其中18项执行/离散/磁盘精确恢复通过、FP32数值超差仍探索；NPU1B/B512执行完成但下一窗KV keys精确恢复失败、独立CPU检查未执行。GPU另有1B/B128独立profile，不计吞吐。
- [有限值检查packing](evidence/core5-finite-packing-20261031.md)：三平台18条实际记录、完整checkpoint/输出精确同材料比较；B128有限收益，小训练GPU有反增，默认不改。固定bd2ff2e相关78项通过。
- [CUDA同模型多卡](evidence/core5-platform-multicard-20261031.md)：1B的2/3/4卡推理数值探索，三通信小训练严格通过；实际参数/状态/消息/计算位置与共享组审阅通过。
- [CUDA resident与同材料eager](evidence/core5-platform-resident-20261031.md)：默认B128/B512布局拒绝；有界布局实际完成、离散/恢复通过、数值超差。d16同材料eager B128严格通过/B512超差，成本2.154/8.680秒，对resident2.324/9.473秒未见优势。
- [CUDA训练容量](evidence/core5-platform-training-capacity-20261031.md)：1B单卡静态拒绝；双/四卡第三窗原平方loss非有限，前向字段检查有限，未进入该窗反向/恢复。31a3b32仅补拒绝后字段定位；固定相关40项和两小训练严格回归通过。原输入/loss/初始化/容差不改。
- GPU人类演练de93246终态无后代：doctor/lock/consumer/训练推理新进程恢复通过；base related独立根原超差保留。stop cancelled/143后无后代，resume到window8（后者未独立数学检查）。已入库[人类流程与报告资格](evidence/core5-human-workflow-20261031.md)；本机完整流程仍待后继。

## 正在运行与明确依赖

本机后继服务均在`background.slice`，外层running不等于内层通过。性能安装/输入冻结，后继服务有意等待前序终态，不修改活动输入。

| unit | 当前工作/固定来源 | 输出与预算 |
| --- | --- | --- |
| `gef-platform-npu-scale-1b3e39f.service` | 06:52:02Z终态failed/1，控制组空；六项完成、B512下一窗恢复失败 | `artifacts/platform-matrix/npu-scale-run-1b3e39f`；7项各1800秒；外层36000秒含等待 |
| `gef-platform-next-stages-bd2ff2e.service` | 已进入后继；CPU与NPU 1B/B128独立profile已完成、NPU2卡已终态exit0，当前3卡，随后8卡及三通信小训练、1B训练1/2/8卡、resident默认/有界，共17项 | `inputs-b495f10-next`、`next-stages-bd2ff2e.py/.sh`；配置b495f10，安装bd2ff2e；43200秒 |
| `gef-human-local-de93246.service` | 等next-stages；CPU及NPU公共安装/consumer人类流程 | `human-{cpu,npu}-de93246`；安装de93246；57600秒 |
| `gef-platform-final-1e33b4a.service` | 等human-local；CPU/NPU resident同材料eager、小训练诊断回归、有条件一次原NPU非有限定位、NPU12.802B单/2/3/8卡容量 | `inputs-1e33b4a-final`、`capacity-human-followup-1e33b4a.py/.sh`；性能bd2ff2e/诊断31a3b32；86400秒 |

每个本机unit的持久状态/log位于`artifacts/jobs/`下去掉`gef-`和`.service`的同名目录（final目录为`platform-final-1e33b4a`）；next/final内部另有progress.json与逐项log。NPU后继逐case合作队列priority0，候选物理4–11、申请1/2/3/8卡、等待600秒，完成后释放。尚未执行的项没有结果。

GPU单卡容量任务已终态，准备同模型双卡。最新`jobs/platform-cuda-resident-match-bd2ff2e`于06:38:06Z exit0，supervisor/workload/remaining均空。训练三组终态failed且无后代；原始结果/配置/log与失败归档保留。四份已审阅终态checkpoint已退役28.505GB，凭据`checkpoint-retirement-reviewed-ad93848.json`；回收稳定后free43.978GB，超过两代checkpoint42.482GB+1GiB余量。下一任务`jobs/platform-cuda-capacity4561m-c1-bd2ff2e`使用冻结bd2ff2e安装及`inputs-ad93848-capacity/cuda-capacity-4561m-cards-1.json`，1800秒、GPU0、ATen1/interop1/worker1；单卡已于06:53:40Z终态passed、无后代；执行/离散/恢复通过，数值探索，[容量报告](evidence/core5-platform-capacity-20261031.md)保留20.058GB实际峰值及314.967秒恢复成本。07:02Z重核四卡空闲、单卡任务终态无后代；已审阅dry-run `checkpoint-retirement-capacity4561m-c1-528025c.json`，已退役其20.699GB可再生checkpoint；free43.953GB，07:01:12Z启动的`jobs/platform-cuda-capacity4561m-c2-bd2ff2e`已于07:11:16Z exit0且无后代，执行/离散/精确恢复通过、数值探索；逐卡allocated峰值10.299GB，中位0.754613秒；证据已追加容量报告。审阅退役其checkpoint后做3/4卡，每点后独立审阅再退役checkpoint，不将旧产物占盘称模型硬件不足。

```sh
systemctl --user show gef-platform-npu-scale-1b3e39f.service gef-platform-next-stages-bd2ff2e.service gef-human-local-de93246.service gef-platform-final-1e33b4a.service -p ActiveState -p SubState -p ExecMainStatus -p ControlGroup
python scripts/status.py
artifacts/cuda3090/ssh.sh '/root/autodl-tmp/gef-cuda3090/install-platform-bd2ff2e/venv/bin/tidegraph job status /root/autodl-tmp/gef-cuda3090/jobs/platform-cuda-resident-match-bd2ff2e'
```

停止本机作业用`systemctl --user stop`准确unit；GPU用相同安装CLI `job stop`并核对无后代。不要停止其他项目作业。

## 环境与来源

本机aarch64/320 affinity CPU、约1.8TiB可用主存/2.3TiB空盘，16×64GiB级NPU；Python Torch/TorchNPU2.12/CANN9.1，独立LibTorch2.10/CANN9.0另ABI。CPU core de999c2/standalone eca070f；NPU core/resident/standalone eca070f。GPU四张3090/sm86、Torch2.12.1+cu130/Toolkit13、56核CPU配额、240GiB内存和50GiB数据盘；core/resident de999c2、standalone eca070f。真实库hash与安装来源分别保存，本阶段未改C++，不冒称按最新HEAD重建。各平台standalone 3 CTest原通过不重复累计。

本机安装`artifacts/platform-matrix/install-{1b3e39f,de93246,bd2ff2e,31a3b32}`；GPU`/root/autodl-tmp/gef-cuda3090/install-platform-*`。bd2ff2e安装内容hash`afc35286eae8dca7744134dcc966cde82ddf5b30e07720cd0ef491ee30783ff8`；31a3b32 wheel SHA`674c4f3038ed916508c80a0ca57819253498492f80fd83dbecabcbf1159acdce`。不覆盖活动安装。

CUDA12方向无直接P2P/原生跨卡原子边界未解除。4.561B点两代checkpoint预算42.482GB，留余量后准入；NPU12.802B/d7296/FC16单一近界模型由1B实测校准，逐卡6GiB余量、host241.331GB、disk117.232GB。下一宽度静态拒绝不分配。不能由预算或普通复制推断resident跨卡资格。

## 下一动作

1. 完成GPU4.561B单/2/3/4卡有限推理容量；逐项检查/恢复/峰值，必要时有依据停止，不OOM搜索。
2. 监测本机依赖链，审阅NPU scale、多卡/训练/resident/profile、CPU/NPU人类流程及12.802B容量；外层failed不覆盖内层独立通过。
3. 汇总三平台算子形状、桶、实际活动、通信与六类packing问题；更新ROADMAP、performance-lessons及可查询证据。
4. 统一三平台人类演练证据、下游模板/任务接口/T4待验与最终矩阵；阶段提交并普通推送后继续。

从干净仓库按[development](development.md)/[人类手册](human-workflow.md)安装；[188有限配置](platform-matrix.md)与site模板可重建材料，审阅gzip可直接查询，无需旧运行目录。历史提交用`resolve_history_commit.py`定位等价源码，不视为重新验证。

公共HTML结果浏览补齐离线筛选、可比较的关键列与完整Details；报告/逃逸/JS过滤与int64大参数边界10项开发通过，已用私有Firefox无头实际渲染。只改报告，不改变活动安装、模型或实验结果。固定源码10项通过，双机新安装报告生成与浏览器实际过滤通过；无需重复训练/推理。

观测修正固定`c59688c`：补齐CANN实际`aten::matmul`形状，13项冻结解析检查通过；`gef-profile-reanalysis-c59688c.service`于07:04:11Z exit0、控制组空。[CPU/NPU profile证据](evidence/core5-platform-profile-20261031.md)保留旧trace hash和运行来源，未重跑数学/性能。

新静态补充固定`3e4f6b2`，10项冻结配置检查通过：`capacity-sharded`固定NPU38.406B的1/2/3卡及CUDA9.123B双卡，区别分散单卡可容纳模型和真正多卡容量。NPU三卡依赖12.802B实际校准；CUDA两代checkpoint83.896GB超过整个50GiB盘，预期准入拒绝。当前NPU8卡队列因物理9存在其他任务而等待（实际可空闲7卡）；不停止或改写活动任务，按原600秒等待上限处理。

后继准备：`gef-platform-sharded-3e4f6b2.service`将等待final终态，再运行冻结`inputs-3e4f6b2-sharded`，安装仍bd2ff2e；1/2卡仅预期准入拒绝，3卡须原12.802B单卡完成有限推理及精确恢复才运行。各case1800秒、队列600秒，外层86400秒含等待；日志/状态在`artifacts/jobs/platform-sharded-3e4f6b2`，逐项记录在`artifacts/platform-matrix/sharded-capacity-3e4f6b2`。未提交仅本STATUS接续更新。
