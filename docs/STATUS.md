# 当前交接 — 张量窗口放置迁移（2026-11-11）

已授权继续[ROADMAP P1—P6](ROADMAP.md#当前计划批量张量贯穿与架构验证已授权实施)。
分支graph-execution-foundation；固定数学0770593838a37c790a568226930567ed60f99b83，19快照通过。
当前已推送ee37cdac。两个用户未跟踪TL-REQ-004/005文件原样保留、不提交。

## 已交付与固定证据

- b993f38公共实验张量输入/loss默认，133项固定资格、8安装消费者恢复、18代表训练成本、
  24层链三模式和约1B/9B四点推理，见[evidence](evidence/experiment-columns-20261111.md)。
  原float loss检查器失败与离线typed容差复核分列；三个NPU精确恢复失败不冒称通过。
- 8c72ca94卷积有序转置：固定74项、独立SDK；六个原代表/客户端各两个新进程精确恢复通过，
  deterministic=false。四安装GDN/SSD原示例用true，明确分列，见[修复证据](evidence/column-tap-transpose-20261111.md)。
- 约1.051B推理CPU64/NPU为20.542/9.726秒（481位置）；约9.315B为8.004/3.894秒（61位置）。
  规模独立数值、训练及精确续接未测。4点均completed；实际taskset 64CPU，NPU物理11。
  durable archive为artifacts/columns-20261111/consumer-scale-results.json；90GiB临时checkpoint在shm，非耐重启产物。

## 当前实现与下一动作

mixed-a/b/c列放置开发已完成：Read所需参数与实际bank、区域control/selection保持整列，raw/参数owner不变；
KV按Read sum/qry/Full只读取所需bank字段，评分cast丢失正支持明确拒绝。
Python-owned NPU和standalone SDK两轮重建；首轮207 passed/4原数值xfail，后继24项按原FP32容差通过，
SDK独立streaming、raw、VJP、两SGD窗通过。安装CPU示例显式deterministic=false也完成独立三步/新进程恢复。
新SDK可显式--device/--dtype检查实际五记忆×mixed-a/b/c；安装示例增加放置/评分精度/deterministic身份。
开发pooled_attention误配evidence被配置器正确拒绝，修正测试后通过；未改语义或阈值。

下一步提交该实现，建立qualified-placement固定worktree；hash核验复用placement-dev2/{build,build-sdk}，
固定相关CPU资格并安装wheel，再空闲NPU实际放置/独立训练/精确恢复和少量成本。
新列多payload设备与resident、异步跨卡及旧资源边界仍按ROADMAP推进；GPU继续延后。
全仓审阅未完成，不把7文件/56函数manifest当全部源码已审。

## 可用现场与运行

所有本轮作业已终态，具体查python scripts/status.py。新开发快照placement-dev2/source保持不动。
环境artifacts/column-env.sh，私有env artifacts/additive-work/env；CANN9.2/Torch2.12。
源码/构建/安装/原日志位于artifacts/columns-20261111/，服务记录artifacts/jobs/。
CPU/内存/空闲NPU可用；共享NPU须当前授权。正式计时与重构建错开，实际affinity需验证，
不能仅根据systemd AllowedCPUs声称隔离。当前无存储阻塞，不清理参考仓库或共享工具链。
