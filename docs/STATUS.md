# 当前交接 — FC16同尺寸归因与优化（2026-11-10）

授权分支graph-execution-foundation；固定数学0770593838a37c790a568226930567ed60f99b83。
用户已授权定位、实现、CPU/NPU实验、提交/推送；GPU延后。参考/下游只读，无子代理。
两份未跟踪TL-REQ-003/004用户handoff保持原样。此前交付与限制从ROADMAP及evidence导航。

## 当前工作

起点c668f7a：原FC16 minimal完整训练步CPU18.640405/NPU22.902820秒；
B8/L128/d128、4区域×4节点、k=2的quad分配，实际Full约50.93%，并非全活动。
用户要求同尺寸Transformer、core-5直接公式、默认图入口三层对照与细分归因。
先添加可选native线程排他阶段计时/ATen范围；不替换数学或调度、不删除检查。
完整B8/L128阶段诊断与小型设备profile分别计量，不把worker时间当设备kernel时间。

## 作业与下一步

gef-fc16-attribution-baseline已终态passed/0；固定09e0277，后台构建和相关小门禁通过。
原FC16正常完整步CPU18.229637/NPU22.998073秒；独立阶段诊断记录已生成。
冻结路径artifacts/fc16-attribution/source-baseline，私有env-baseline/build-baseline。
输出artifacts/jobs/fc16-attribution-baseline/{status.json,task.log}。
脚本artifacts/fc16-attribution/baseline.sh：全部可用CPU构建，相关query根/阶段检查，
随后原FC16 CPU/NPU单次成本和同材料阶段诊断；构建与计时不重叠。
总时限7200秒、MemoryMax400G、TasksMax4096；单卡经站点队列选择，等待上限120秒。
检查：systemctl --user show gef-fc16-attribution-baseline -p ActiveState -p Result -p MainPID -p ControlGroup；
python scripts/status.py；tail -n 40 artifacts/jobs/fc16-attribution-baseline/task.log。
当前独立FC张量基线5项CPU检查通过，含独立标量图、二分求解、FP64差分；准备固定后运行
同尺寸CPU/NPU三层成本及首次实际本地TTFT。新常规路径优化尚未开始。

后续按阶段数据选择块内复制/快照/VJP优化，定向独立检查、完整训练/恢复与代表回归。
原异步双NPU训练、规模resident预算、数值严格失败均保持ROADMAP状态，不能由本轮小检查解除。
磁盘约809GiB可用、quota280/300/800GiB；大写入前复核。不清理旧原始证据或公共环境。
