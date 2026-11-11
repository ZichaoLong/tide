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

混合列放置实现固定540ccbd8：两套源码/binary hash核验，CPU213通过/4原数值xfail。
NPU安装11项中10项独立三步/新进程恢复通过，SDK五记忆×mixed-a/b/c 15项通过；
Python KV mixed-a原精确恢复失败：72浮点字段不同，离散一致。原column-placement-npu保持failed。
同一前向图重复VJP可复现12个参数/输入根差异；关闭反向多线程或仅取消Aggregate共享的诊断通过。
首个不同中间根为最后区域两个节点共享content；NPU原计算与CPU Read的贡献到达顺序相关。

局部修复：保留Aggregate数值共享，按节点提供只读整列使用槽，反向稳定归并所有使用者，
不改全局并行/确定性开关。CPU公式/版本/零根及相关检查50 passed/9原数值xfail。
新增check_column_adjoint_order.py用于重复同一前向图的VJP和中间边界诊断。
下一步提交修复并固定新源码，核验相同NPU见证、安装训练/新进程恢复，再有限mixed成本。
新列多payload设备与resident、异步跨卡及旧资源边界仍按ROADMAP推进；GPU继续延后。
全仓审阅未完成，不把7文件/56函数manifest当全部源码已审。

## 可用现场与运行

所有本轮设备诊断已终态，具体查python scripts/status.py。placement-dev2/source及qualified-placement保持不动。
环境artifacts/column-env.sh，私有env artifacts/additive-work/env；CANN9.2/Torch2.12。
源码/构建/安装/原日志位于artifacts/columns-20261111/，服务记录artifacts/jobs/。
CPU/内存/空闲NPU可用；共享NPU须当前授权。正式计时与重构建错开，实际affinity需验证，
不能仅根据systemd AllowedCPUs声称隔离。当前无存储阻塞，不清理参考仓库或共享工具链。
