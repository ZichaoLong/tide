# 单行VJP、字段packing与常驻因子KV的有限成本

固定新实现`15c3e65fe3a737324c90bdf26b60c28a34ad2dcf`，对照
`f03e9bee0f069d48f5f1544d3821b7781409a27f`，上游0770593不变。
[审阅JSON](core5-boundary-performance-20261029.json)保存逐配置终态、输入hash、实际工作、
原始结果摘要hash、完整核对、恢复、资源和profile。相关实现资格见[独立报告](core5-resident-boundary-20261029.md)。
本报告发生在紧凑局部纤维改动之前；后续源码不能继承本报告的计时。

## 完整训练对照

每配置一个独立进程、一个warmup窗和一个测量窗，SGD；新源码另做计时外磁盘save/load、
换owner的下一窗精确恢复。计时含输入准备/上传、调度、packing、forward/loss、backward、
finite checks、optimizer、通信、返回/清理和全设备同步；构造、warmup、核对与恢复单列。
CPU study passed/0；NPU study因三项D64严格失败而failed/1，各中型通过项仍分别保留。

| 配置 | f03e9be秒/完整窗 | 15c3e65秒/完整窗 | 独立核对 |
| --- | ---: | ---: | --- |
| LibTorch CPU HB medium FP32 | 0.412401 | 0.432519 | 完整记录、最终参数/梯度/SGD槽通过 |
| LibTorch CPU 匹配D64，显式FP64 | 2.717120 | 2.769739 | 同上 |
| 纯Torch NPU mixed-c medium FP32 | 2.182677 | 1.848464 | 同上 |
| 纯Torch NPU mixed-a medium FP32 | 1.504512 | 1.476871 | 同上 |
| NPU resident medium FP32 | 无本轮同步基线 | 2.793867 | 同上 |

新版五项自身磁盘恢复均逐值通过。原resident基线二进制已被前期开发覆盖，
不伪造同轮前后比较；不把时间变化归因于单个修改。全部为单次观察，外部负载未隔离。
本组没有观察到CPU整体收益；NPU mixed有较短耗时，但不足以宣称稳定加速或大规模胜者。

medium同输入hash `e6e7b7d9f3f12040bb34638de8cd4ffc0d152c5fb452d41f16dae3bd0e836f42`，
候选/Full/消息/输出/pending为128/128/192/32/0。
CPU D64 FP64同材料hash `07a775da9626754a12db7d2daaad4733b81afaa9bd64c71b72be4dd5300ff01c`，
相应工作为479/344/1604/19/0；34次共享准备、445次独立写入覆盖不变，最大私有共享批32、独立批24。
medium NPU mixed-c峰值allocated/reserved前后均39,651,328/44,040,192字节，
mixed-a均37,918,720/41,943,040字节；字段packing没有产生本例allocator峰值下降。

## 独立profile与常驻反向分段

CPU D64 FP64独立profile通过，进程219.03秒、峰值RSS约14.21GiB，不能作普通训练内存或吞吐。
17,697次mm、236次bmm、69,342次copy、12,325次stack与前一profile相同。
1270次SingletonVjp执行替代823次Query和447次Aggregate私有边界；mm+bmm self约0.115s。
inclusive span：forward1.718s、backward/finite2.714s、optimizer0.181s、detach/return/sync0.306s。
减少私有图层次没有在本次完整CPU测量中转化为收益；字段拆装与反向仍值得进一步研究。

NPU mixed-a独立profile通过，进程39.16秒；31,302个kernel事件中AIVEC25,570、
MIX_AIV4,638、AICORE902、AICPU192，vendor matmul606、BMM296。
3,265次`_local_scalar_dense`、2,016次MEMCPY_ASYNC；kernel union132,150us。
forward/backward/return的inclusive span约1.367/0.813/0.216s；
scalar inclusive103,124us、memcpy inclusive1,385us。宿主span相互重叠，不是排他的关键路径归因。
未观察到host fallback标记，设备AICPU另列。小算子、索引、提交和同步成本仍存在，
不能据字段合并或matmul次数宣称低成本优化已做完。

resident medium实际反向计量：构造927,621us、seed6,916us、运行/等待1,094,390us、
参数归约20,064us、bridge0（本例单保留窗）。构造约占完整2.794s的三分之一。
adjoint张量275,214,029、辅助311,144,416、retained230,885,776字节；
测量allocator峰值allocated514,803,200、reserved532,676,608字节。
旧adjoint绑定journal、根和参数地址/版本，不能仅按shape跨窗口复用；
下一候选应是显式模板与重新绑定，并重新核对独立根、版本、跨窗与恢复。

## 参数匹配D64的停止结论

固定计划hash `6c8cc36ae97287133d40e67ccc8e137574b80e2df420feff6adfef3395397bb3`，
480节点/15区域/2208边、宽64、B1/P1、9,076,975可训练参数；core-3 Add主体9,076,928，相差47。
参数量匹配不等于函数或工作量相同。原FP32 seed7材料、原容差不变。

| 配置 | 实际进展 | 结论 |
| --- | --- | --- |
| native单NPU，显式blocking | 测量15.684703s，自身磁盘换owner下一窗恢复精确 | CPU完整记录在node89 keys的1/64坐标失败；不称合格吞吐 |
| native双NPU，显式blocking | 测量16.839709s，自身恢复精确 | 同一严格失败；构造8.115s；没有双卡收益结论 |
| resident单NPU | 构造阶段Aggregate scratch预算拒绝，进程83.56s | 未进入warmup/训练，没有计时或恢复资格 |

两项native的最大绝对差1.40070915e-6、相对差3.81534992e-5，原atol1e-6/rtol1e-5。
恢复精确只证明同一候选可续接，不替代独立CPU正确性。
resident事前静态选定queue4096/stages32/cache4；optimizer_bytes1GiB，使1/16的布局子分区
覆盖36,320,070字节最低需求，总预算仍5GiB，其余分区默认不变。没有CPU真实轨迹预跑。
正整数容量覆盖静态输入/缓存上界，不保证全部局部数值展开和journal符合准入。
已定位旧节点纤维使用全局queue填充；新紧凑候选另行提交与验收，不重试本失败二进制。

## 环境与复现

aarch64、Torch2.12/Torch-NPU2.12.post2、CANN9.1；NPU物理9/13映射逻辑0/1，HF32关闭。
两项D64 native显式`ASCEND_LAUNCH_BLOCKING=1`是独立配置，默认异步EE1017没有解除。
新旧core与resident的全部源码/binary/dependency hash在运行前核验；
构建的dirty-origin仍按[资格报告](core5-resident-boundary-20261029.md)保留，没有冒充干净新构建。
两study串行，background.slice、单线程、32GiB配置预算，CPU上限2100秒、NPU3300秒，
每配置300–420秒；NPU排队180秒。systemd属性不证明独占或硬资源隔离。

从固定提交按[开发](../development.md)构建匹配库，用新目录：

```sh
python scripts/prepare_weighted_workload.py --topology ranked --layers 15 --region-width 32 \
  --width 64 --samples 1 --positions 1 --match-core3 add --output-dir NEW_PLAN
python scripts/benchmark_weighted.py --device cpu --implementation native --native-library CORE \
  --workload NEW_PLAN/workload.json --dtype float64 --steps 1 --checkpoint-check --output-dir NEW_RUN
```

中型使用`--size medium`（默认HB）；NPU选择`python --placement mixed-a/mixed-c`或
`resident --resident-library RESIDENT`，`--device npu:0`。双卡native另加`--peer-devices npu:0 npu:1`。
容量JSON按上表，只改变容量；profile单独新进程加`--profile`，不与吞吐混合。
这些是复现入口，不授权自动重跑已停止配置；新的重启条件见[ROADMAP](../ROADMAP.md)。
