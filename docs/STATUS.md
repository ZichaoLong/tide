# 当前交接 — FC16归因本轮已完成，等待用户对齐（2026-11-10）

授权分支graph-execution-foundation，固定数学0770593。用户最新要求：本轮后停下，先明确对齐，
取得新授权后再开始后续优化/实验。当前没有待自动启动的作业；不要按历史持续要求恢复队列。
参考/下游只读，无子代理。两份未跟踪TL-REQ-003/004用户handoff保持原样，未升级下游依赖。

## 已交付与入口

实现1f3f62d69f3d89b1cc7661a5a509c6e382e9b51d；本轮其他实现09e0277/098c91a/df80d04/
9fd82c3/c20aaa4均保留真实历史身份。正式结果、来源hash、配置和可重建命令集中于
[固定证据](evidence/fc16-cost-attribution-20261110.md)及同名JSON；
[契约](fc16-cost-attribution.md)、[性能经验](performance-lessons-prefill.md)、[复现导航](reproduction.md)互链。

补充同尺寸Transformer与独立FC公式、实际首token、原生阶段计时。原FC16 B8/L128/d128/
FP32 minimal完整训练CPU18.229637→17.496945秒、NPU22.998073→21.064699秒；
输入位置吞吐58.52/48.61每秒。首tokenCPU4.140645/NPU4.273320秒，CPU较原3.881秒反升。
NPU当前物理3共享，原训练物理11/锚点物理1；单次观察，不承诺独占或统计稳定。
独立FC公式NPU训练.315203秒，Transformer16 .046361秒；不同函数/投影不能充当完整图等价证明。

全查询私有列候选退化（CPU134.121秒）已撤回，取消/数值失败仍保留。
当前只保留区域列、foreach复制到独立输出存储/版本；查询ragged内部叶保持独立。
NPU查询发布阶段1.744→.683秒，区域梯度3.426→2.123秒，通用入口仍远慢于直接公式。

## 最终资格与终态

固定1f3f62d两套新构建：artifacts/fc16-attribution/{build-final,build-final-sdk}，全部320CPU；
source-final只读，env-final为项目私有安装。CPU358通过/2原数值xfail，NPU流程95通过（含CPU辅助）。
纯PyTorch/Python-native各CPU/NPU独立三步训练和新进程精确恢复通过；SDK CPU FP64/NPU FP32的
complete与outputs各通过三步/新进程恢复，complete有独立记录/根，outputs只有自身精确续接资格。
SDK3 CTest及CPU两dtype自定义kernel通过。完整FC16独立数值/学习验收仍not_checked。

gef-fc16-attribution-final已passed/0，MainPID0/control group空；队列completed。
日志/状态artifacts/jobs/fc16-attribution-final/{task.log,status.json}，全部内部结果已审阅归档。
其他本轮：baseline/anchors-v2/gather passed；anchors failed/1，packed cancelled/143。
packed内层CPU测量已完成、profile中断后running快照保持，不能把外层取消改写成整项成功。
空闲NPU探针超时、首次NPU signbit测试oracle失败及后继修正分别保留，不再排队。

## 后续对齐，而非继续执行

唯一待办见ROADMAP的FC16后继。建议先讨论块级查询快照/伴随接口与当前逐字段根的关系，
再定有限回归范围和成本目标。宽FC、176节点HB/LH、sum、1B/9B、多卡/resident本增量规模
回归未做；原提交证据仍有效但不可外推到本增量。GPU/异步双NPU/resident工作区等旧边界未解除。
无新的后台任务、无下游写入；新授权到来先运行git status --short --branch和python scripts/status.py。
提交/远端状态以git status为准；本轮收尾不授权开始下一阶段，后续只停留在讨论/对齐。
