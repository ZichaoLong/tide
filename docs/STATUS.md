# 当前交接 — 2026-10-28

固定上游 `0770593838a37c790a568226930567ed60f99b83`（core-5及带权局部定义）。
授权分支 `graph-execution-foundation`；HEAD/origin `ceae7bf`。用户授权持续实施、有限实验、
分阶段提交与正常推送，不用子代理。总体目标未完成；唯一待办 [ROADMAP](ROADMAP.md)，
持续规则 [execution-flows](execution-flows.md)。参考仓库与公共环境只读。

## 当前结论

- 公共表示、五记忆/正式候选、独立调度、packed/VJP、mixed及常驻FP32/FP16图与训练
  已实现；各自固定证据从 [semantic-map](semantic-map.md) 和 [evidence](evidence/README.md) 导航。
- 完整FP16图/训练源码21ed86d：[报告](evidence/core5-resident-half-training-20261028.md)。
  CPU32、Python-owned NPU52、standalone两dtype各21图/15配置/20更新通过；CUDA三架构构建通过。
  后续发现standalone最后master比较借用临时owners vector，原passed状态保留，但该断言无可靠独立证据；
  7db0b08已修正，下面独立复验。Python master与其他完整记录/VJP不受此处影响。
- 跨卡样本副本实现672e948：Python-owned4项/62.50秒通过（物理9/13→0/1）；CUDA构建通过。
  原standalone NPU failed/exit1（物理1/3→0/1）：FP32首用例std::bad_alloc，FP16未运行。
  GDB诊断core5-weighted-peer-bad-alloc-diagnose定位到检查器run()字符串reserve；7db0b08修复两处临时owners引用。
  诊断exit0仅表示抓栈成功，不是数值通过。旧失败保留，不重新标记。
- 样本副本是独立私有状态、梯度归约与共同optimizer提交；**单样本跨卡节点切分仍未实现**。
- 性能消费者3516bef：Python/native/resident入口、独立LibTorch消费者、无轨迹静态fixture和完整计时契约
  见 [weighted-performance](weighted-performance.md)。尚无正式性能选型报告。

## 已审阅的定向资格

冻结checkout `artifacts/worktrees/core5-peer-owner-lifetime`，固定
`7db0b08e33c90b3fc278f2acdbce2ba738297310`，跟踪源码只读，不能移动HEAD。

| job | 最近状态 | build / 范围 |
| --- | --- | --- |
| core5-peer-owner-lifetime-npu-qualify | passed/exit0；物理9/13→0/1，两dtype训练及peer通过 | build/npu；CPU guard4、standalone weighted-training/weighted-peer两dtype |
| core5-peer-owner-lifetime-cuda-qualify | passed/exit0，264步构建 | build/cuda；同两check三架构build/loader/help，无GPU数值结论 |

units为`tide-JOB-20261028.service`；输出`artifacts/jobs/JOB/`，包含status.json、task.log、
control-build.json和NPU checks/result.json/queue-state.json；launcher `.local/JOB{,-workload}.sh`。
每任务1800秒、2workers/BLAS1、Nice10/background.slice。NPU两卡queue180秒/device600秒。
已核验Transient=yes、独立ControlGroup。严格master与全体提交断言已通过；source/binary/loader/log/XML/unit身份审阅完成，证据见
[样本副本与master复验](evidence/core5-resident-weighted-peers-20261028.md)，本提交归档。

## 性能入口最新终态

冻结checkout `artifacts/worktrees/core5-weighted-benchmark`，固定
`3516bef9fef8d1571af0082402ebef71d88d8c32`，跟踪源码只读。

- `core5-weighted-benchmark-cpu-entry`：passed/exit0，待审阅归档。
  build/cpu安装CPU core并构建standalone，KV/DAG prefill训练、SSD/HB streaming训练、
  Python调用native KV/DAG各1warmup+1测量窗，事后完整独立比较通过。仅入口资格，不作选型测量。
  预算1200秒、2workers/BLAS1，launcher `.local/core5-weighted-benchmark-cpu-entry{,-workload}.sh`。
- `core5-weighted-benchmark-python-profile`：failed/exit2，物理9→npu:0。
  `benchmark_weighted.py`在测量前报controlled measurement requires clean immutable source；没有profile/result.json。
  诊断已证实resolve_device后出现临时`?? kernel_meta/`，退出后消失；保留原失败。
  生产库672e948/Python-owned（7db只改C++检查器）；预算600秒/queue180秒/profile360秒。
  launcher `.local/core5-weighted-benchmark-python-profile.sh`。

以上unit/output同一命名规则。CPU入口与profile不作为正式性能结果。

源码拒绝诊断passed/exit0（仅环境诊断）：`core5-weighted-profile-source-diagnose`，同3516bef checkout；
launcher `.local/core5-weighted-profile-source-diagnose.sh` 调用 `.local/diagnose-weighted-profile-source.py`，只打印指定Git身份和NPU初始化前后状态。
预算360秒/queue180秒/probe120秒，一卡；unit/output沿上述规则，不代表性能或数值资格。
主区已将源码检查移到设备初始化前，运行cwd提前切到独立输出目录，相对库路径先解析。
同时补profile阶段耗时、kernel覆盖、copy/scalar耗时（非profile不启用标记）。
短CPU native profile与完整独立记录/参数比较通过；六阶段实际trace和重叠kernel区间手算通过。
修复fea42aa已提交推送；冻结checkout `artifacts/worktrees/core5-weighted-performance`，跟踪文件只读。
`core5-weighted-benchmark-python-profile2`已passed/exit0，同原profile配置与已资格库；新输出/launcher以profile2命名。
预算600秒/queue180秒/profile360秒，Nice10/background.slice，ATen/BLAS1；源码/结果旧失败保留。完整记录与参数事后对照通过；小规模profile阶段记录含backward/finite约1.24秒，kernel覆盖约.135秒。
profile2默认采集周期结束有vendor不完整解析警告；保留记录，不把事件计数当完整硬件工作量。
主区已显式单周期并在导出前prof.step；CPU medium推理profile、完整记录及阶段事件检查通过。
显式周期增量ceae7bf已提交推送，冻结checkout `artifacts/worktrees/core5-weighted-measurement`。
`core5-weighted-medium-profiles`已passed/exit0：KV/HB medium完整训练，resident与纯PyTorch mixed-c各独立进程，串行。
unit/output沿同规则，launcher `.local/core5-weighted-medium-profiles.sh`与同名前缀-workload.py；
每配置timeout360秒，合计workload780秒、queue180秒、service1020秒；一个NPU、ATen/BLAS1、Nice10/background.slice。
resident与mixed-c完整记录/loss/参数对照及完整采集周期通过，无Incorrect schedule警告；物理9→npu:0。
记录profile开销、阶段/实际kernel与事后完整对照；不是正式吞吐。


## 有限选型测量（新任务）

固定脚本源码ceae7bf；静态fixture `artifacts/core5-weighted-measurement-fixtures/` 由同源码导出，
KV/HB small与medium、SSD/HB medium、KV/非对齐DAG medium、KV/反馈PDG medium共5项；没有执行轨迹。
受控清单 `.local/core5-weighted-performance-cases.json` 共24配置（CPU15/NPU9），不是开关笛卡尔积。
问题：完整返回下的CPU streaming/prefill与接口成本；小中规模mixed/resident完整训练是否合适；
SSD、非对齐/反馈与推理是否改变选型。每配置一独立进程、1warmup+3连续测量窗；FP32/SGD、每窗detach。

`core5-weighted-performance-cpu`已failed/exit1：13配置通过，PDG streaming/prefill两项严格数值拒绝。
首个tensor差异maxabs分别4.11272e-6、2.71201e-6，完整门禁未通过，不能报告为等价加速。
原FP32 fixture/日志/测量时间保留，不修改容差或输入。（unit/output同命名规则），源码checkout
`artifacts/worktrees/core5-weighted-measurement`。launcher `.local/core5-weighted-performance-cpu.sh`，
共享只读harness `.local/run-core5-weighted-performance.py`，逐配置记录其hash、CLI、输入与build身份、
load average和当时活动project job。使用3516bef已通过的CPU standalone binary（consumer源码未变）。
单配置120秒、组1500秒、service1560秒、ATen/BLAS1、Nice10/background.slice。
当前7db standalone NPU正确性资格仍在跑，主机争用明确记录；没有其他本项目性能计时。
CPU组已终态，NPU计时组待本次诊断完成后串行启动。NPU standalone入口`core5-weighted-benchmark-npu-entry`准备启动；7db资格与二进制身份已完整审阅。
源码checkout仍ceae7bf，build/npu；匹配SDK core与7db resident，不重编core。
launcher `.local/core5-weighted-benchmark-npu-entry{,-workload}.sh`；安装/build/loader/help后，KV/HB small完整训练1warmup+1窗。
静态fixture由Torch2.12 CPU导出，SDK2.10消费者独立读取JSON；不跨ABI加载Python扩展。
任务720秒/2workers、ATen/BLAS1、queue180秒/device180秒，Nice10/background.slice；仅入口资格，不作正式选型。

诊断`core5-weighted-pdg-rounding-diagnose`已passed/exit0：两dtype四窗口离散结构全相同；FP32差异先见pending/后见raw，最终参数仍过严格比较；同初值提升FP64后两native日程全部完整记录与参数通过，固定ceae7bf checkout；单线程CPU、180秒/服务240秒。
launcher `.local/core5-weighted-pdg-rounding-diagnose.sh`，脚本`.local/diagnose-weighted-pdg-rounding.py`；
原FP32初值及其FP64提升、独立Python时间递归与native两日程、4窗口与SGD；输出diagnosis.json。
诊断退出0只表示完成分析，原PDG严格失败不改写。

## 下一步与检查命令

1. 固定显式profile周期增量，运行medium resident与mixed profile；随后有限正式计时，保留profile2周期警告。
2. 审阅7db0b08两个资格终态，更新half报告master断言复验与新增peer证据。
3. 审阅CPU消费者；NPU组件终态后构建matching standalone benchmark，开展小中规模profiling与有限完整计时。
   项目自身重型计时串行，默认每配置一次独立测量；无全开关笛卡尔积。
4. 继续单样本跨卡节点切分与最终追溯/支持矩阵，不能将样本副本外推此能力。

`python scripts/status.py`（历史审计用`--all-jobs`）；`python scripts/check_upstream.py`。
`systemctl --user show UNIT -p ActiveState -p Result -p ExecMainStatus`；停止`systemctl --user stop UNIT`。
源码不可编辑到活动checkout；实现先提交、固定源码资格、证据另提交。正常推送已授权。

## 保留限制与环境

- eager FP16 GDN默认AdamW严格差约.001984，strict xfail；SGD不解除。
- 默认异步双NPU Python VJP的CANN invalid event；显式launch blocking通过不解除。
- SDK2.10 FP32 fc/affine第三轮evidence差3.09944e-6，mixed-a同见证。
- CUDA所有真机范围待硬件，构建结果不能代替。
- 原half-flow首次目录拒绝、31e017f FP16参考拒绝等旧失败留在独立记录与报告，不改写。

实际cwd `/mi/data2T/zlong/home/llm/graph-execution-foundation`，zlong；最近可用磁盘约3.3TiB。
现行规范 `/opt/software/ascend-locks/dev-workspace-9.1.0/README.md` 与 `/etc/modulefiles`；旧guide失效，用户已授权采用现行规范。
Python：source /etc/profile.d/modules.sh；module purge；module load ascend/dev-workspace-9.1.0；source .local/venv-core5/bin/activate。
SDK：module load libtorch-npu/2.10.0-cann9.0.0；Python为$LIBTORCH_ROOT/../../../../bin/python。
CUDA：source /mi/data2T/zlong/gpu-toolchains/activate-cuda.sh；PATH追加本项目.local/venv-core5/bin供ninja。
TORCH_DEVICE_BACKEND_AUTOLOAD=0、PYTHONDONTWRITEBYTECODE=1、ACL_OP_INIT_MODE=0、ATen/BLAS1；保留CANN PYTHONPATH。
不预建job.py输出目录；service stdout写.local。官方build入口要求新build目录。
STATUS使用scripts/durable_records.py fsync+原子写并回读；只维护当前交接，历史详情见对应报告。
