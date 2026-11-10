# 当前交接 — FC16同尺寸归因与当前候选收尾（2026-11-10）

授权分支graph-execution-foundation，数学0770593。用户最新要求：本轮收尾后停下，先对齐，
新授权后才开始下一阶段；不能自动延续剩余优化/大矩阵。无子代理，参考/下游只读。
两份未跟踪TL-REQ-003/004用户handoff保持原样。本轮实现/证据提交及推送仍获授权。

## 已完成

09e0277的可选native阶段计时已构建/验证。原FC16 B8/L128/d128/FP32 minimal完整训练
CPU18.229637/NPU22.998073秒；query/region私有反向是主要耗时，分配求解不是主因。
df80d04独立FC张量/Transformer锚点小检查及有限成本已完成；NPU直接FC训练.315秒、
Transformer16 .046秒。不同函数/投影不冒称完整图等价。真实warm-loaded首token另测。
098c91a原求解区间失败保留；9fd82c3全私有列打包CPU134.121秒、cancelled/143保留。

c20aaa4 gather候选作业已passed/0、PID0，CPU352通过/2原数值xfail，NPU流程206通过
（含CPU辅助检查）；两客户端CPU/NPU三步/新进程精确恢复通过。完整步CPU19.462801、
NPU23.676150秒；消除严重退化但仍无净收益，不保留全查询列打包作为默认。
记录artifacts/jobs/fc16-attribution-gather；固定源码source-gather/build-gather/env-gather。

## 当前收尾候选和有限验证

仅区域保留私有列；ragged查询恢复旧独立叶。查询/区域发布改foreach复制到独立存储和版本。
新CPU定向24通过；旧native仅作开发对照，新C++待冻结构建。初始mutation测试混入的
FP32数值断言差约1.387e-6已留诊断入口；修改后只验证精确隔离，原数值门禁不放宽。
NPU复制探针初始signbit(-0)断言失败，改为host signbit检查实际复制位，不调用NPU signbit作oracle。
原失败/空闲队列超时记录保持；新probe待核对。共享NPU按PID/start身份、HBM/AICore准入，
final-sharing.json限2小时；不是独占计时。

计划本轮最后一次固定候选资格：新Python/native与独立SDK构建（全部320CPU），相关
CPU/NPU检查、安装训练/恢复、FC16训练/推理和阶段对照。命名gef-fc16-attribution-final；
脚本artifacts/fc16-attribution/final.sh，记录artifacts/jobs/fc16-attribution-final。
时限7200秒/MemoryMax400G；构建与重型计时串行。准备完成并启动后须核对记录和PID。
后继只整理本轮证据/经验并提交推送，更新STATUS为等待用户对齐；不自动启动广泛代表/9B/多卡矩阵。

未修改GPU/异步双NPU/resident路径，不解除既有严格数值、容量和硬件边界；全部限制只见ROADMAP。
磁盘余约768GiB，quota281/300/800GiB。尚未向下游写入或升级依赖锁。
