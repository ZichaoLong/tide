# 当前交接 — FC16同尺寸归因与优化（2026-11-10）

授权分支graph-execution-foundation；固定数学0770593。用户授权定位、实现、实验、提交/推送；
GPU延后。参考/下游只读，无子代理；两份未跟踪TL-REQ-003/004用户handoff保持原样。

## 已完成与已知结果

09e0277增加可选native线程排他阶段计时/ATen范围；新build-baseline及CPU/NPU门禁通过。
原FC16 B8/L128/d128/FP32 minimal完整训练步CPU18.229637/NPU22.998073秒。
CPU query/region梯度内部6.75/2.71秒，分配数值+行包装约.061秒；不是纯设备时间。
098c91a独立FC/Transformer同尺寸基线；首轮独立区间求解拒绝保留failed。
df80d04选择最小可行区间距离并中心化，anchors-v2已passed，CPU5/NPU3小检查、各六成本。
NPU直接FC公式训练.315秒，16层Transformer.046秒；前者同参数输出投影、后者不同函数，
完整尺寸尚未独立逐项认证。首token使用真实warm-loaded推理，不从训练时间摊销。

9fd82c3私有列打包：新C++构建、CPU350通过/2原xfail，两客户端独立三步/新进程恢复通过。
但原CPU成本反增至134.121秒（反向126.856秒）；长度桶跨owner间断区间的逐view转置放大
完整大缓冲区归零/累加。已停止gef-fc16-attribution-packed，cancelled/143；PID0/cgroup空。
原CPU测量已完成但诊断pass未完，内层running快照保持，不称整个作业完成；NPU尚未运行。

## 当前改动与下一步

改为真实同base非连续区间的一次index gather及其批量转置；不同base/strided保留正常路径。
Python新反例和相关224项通过，native新候选尚未构建。预备gef-fc16-attribution-gather，
冻结新提交/source-gather、私有env-gather/build-gather，全部320CPU构建；相关门禁后原FC16
CPU/NPU成本/阶段观察与两客户端训练/恢复。时限7200秒，MemoryMax400G。
脚本artifacts/fc16-attribution/gather.sh；记录artifacts/jobs/fc16-attribution-gather/{status.json,task.log}。
检查python scripts/status.py和systemctl --user show gef-fc16-attribution-gather -p ActiveState -p MainPID -p ControlGroup。

后继需独立SDK新构建/检查、小型代表回归和审阅证据；不为既有GPU、异步双NPU、resident容量
或严格数值失败外推通过。精确失败/停止条件仍只在ROADMAP。所有运行输入按原目录/提交保留。
当前quota约280/300/800GiB、磁盘余793GiB；大写入前复核。未向下游写入或变更依赖锁。
