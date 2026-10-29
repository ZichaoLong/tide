# 当前交接 — 2026-10-29

授权分支`graph-execution-foundation`，上游固定`0770593838a37c790a568226930567ed60f99b83`。
用户授权逐项推进非GPU边界、有限实验、提交与普通推送；GPU延后，不用子代理。
参考仓库与公共工具链只读。原材料、容差、optimizer、默认dtype和严格失败保持。

## 当前完成范围

当前生产实现`04b52153f9785e1489e31c7dd09a16849791d331`（含8c48f49），已提交推送。
resident按物理入边+输入端口界定每节点纤维，消除全局queue的逐节点填充；
全局indices/sentinel、SrcClass多重原子、SourceDomain、int64时间、前向/窗口VJP/peer布局与精确恢复保持。
非法counts和槽越界在写入前拒绝，已有sticky error不覆盖。

[紧凑纤维固定资格/性能](evidence/core5-compact-fibers-20261029.md)：
- CPU元数据两环境各10 passed；公共Python/NPU 56 passed/0 skipped；SDK7个组件/dtype检查全部通过。
- SDK含24组合独立根、两dtype各21完整图、单/双卡两dtype各5组合/20更新/3窗，SGD/AdamW与迁移恢复。
- 新resident均为干净04b5215完整构建；未改core按完整源码/binary/dependency hash复用，保留旧dirty-origin。
- medium同输入/容量/工作量完整核对和磁盘换owner恢复通过；完整窗2.793867→2.469730s，
  peak allocated514803200→376404992 bytes。一次观察、外部争用未隔离，不宣称普遍加速。
- 原D64同参数量匹配/材料/容量在构造时拒绝CANN207005；实际厂商日志为Too many streams/EE1023，
  SQ/CQ观测1984。未进入warmup，无完整训练/恢复/吞吐结果；不把该数值当通用流上限。

前序[15c3e65资格](evidence/core5-resident-boundary-20261029.md)、
[CPU/NPU对照及profile](evidence/core5-boundary-performance-20261029.md)已作为46ecd5a/57250fe推送。
单行VJP、异形字段packing、因子KV、实际分配错误/梯度/SGD槽与磁盘恢复设施已交付。
CPU eager继承e0d6e27的517 passed/2原FP16 strict xfail，本次没有重复该全量。
前序CPU两项完整对照未见收益；NPU mixed-a/c有单次下降但allocator峰值不变，原失败仍保留。

## 活跃任务：无

全部任务已终态，服务/cgroup和持久记录已核对；无自动恢复或等待队列。
- `core5-compact-fiber-v2-python-20261029.service` passed/0；
  `artifacts/jobs/core5-compact-fiber-v2-python/{status,qualification}.json`、metadata.xml/device.xml、build.log/device.log。
- `core5-compact-fiber-v2-sdk-20261029.service` passed/0；
  `artifacts/jobs/core5-compact-fiber-v2-sdk/{status,qualification}.json`与`device/result.json`、逐组件log。
- `core5-compact-fiber-study-20261029.service` failed/1；prepare/medium passed，D64 failed/1，进程97.5秒。
  `artifacts/jobs/core5-compact-fiber-study/{status,study}.json`、medium/ranked64的result.json和独立log。
- 旧8c48f49的`core5-compact-fiber-{python,sdk}-20261029.service`主动取消，status为cancelled/143；
  其内层构建记录未完成，没有设备资格。原记录保留，不改为通过。

固定树`artifacts/worktrees/core5-compact-fiber-qualified-v2`只读；新build/{python,sdk}分别属于Torch2.12/CANN9.1和SDK2.10/CANN9.0。
启动器/driver为`.local/core5-compact-fiber-v2-{python,sdk}.sh`、`.local/core5-compact-fiber-v2-qualify.py`及
`.local/core5-compact-fiber-study.{sh,py}`。它们与旧运行目录是保留证据；仓库报告给出可从干净源码复现的入口。
资格两build线程、设备单线程；performance在全部构建/资格结束后串行。资源/时长和原始hash见报告。
CUDA新构建/真机及x86未验；旧报告不替代本源码资格。

## 停止与重启条件

逐项终态与下一条件统一在[ROADMAP](ROADMAP.md)；实现和测试经验统一在[performance-lessons](performance-lessons.md)。
D64/D2048/反馈FP32、FP16 GDN/default AdamW、默认异步双NPU、晚注册NPU均未解除；
正支持不可表示与真实有限容量继续显式拒绝。自身精确恢复不能替代独立正确性。

新的有依据方向是有界常驻服务分派/复用及显式反向模板重绑；当前没有此候选实现。
先建立嵌套调用/请求完成停止、共享workspace、独立根和跨卡恢复反例，再设计新实现与静态资源准入。
仅增加HBM预算、重跑旧D64或按shape缓存旧adjoint都不能解除当前流资源边界。
保持所有原数值失败的独立重启条件，不重新开启旧队列或9B全图实验。

接续先运行`git status --short --branch`、`python scripts/status.py`、`python scripts/check_upstream.py`，
再读ROADMAP及上述两份新证据。本轮有限推进已经终态，不代表所有实现/数值/平台限制解除。
当前提交仅收录审核后的证据/契约/交接，无未提交生产代码。
