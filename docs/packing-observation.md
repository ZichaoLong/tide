# Eager packing 的独立观测

这是一项可选的主机整数计数，不改变固定数学、数值表达式、调度、返回、状态或checkpoint。
Python `benchmark_weighted.py --profile` 在一个独立采集窗口启用；native通过现有
`reset_work`/`work_metrics`取得相同字段。standalone显式使用 `--observe-work true`，
输出 `kind=work-observation`、`throughput_sample=false`；它仍执行独立完整核对，
计时只是带观测开销的成本，不授予吞吐资格。warmup、独立参考和磁盘恢复不进入计数。
普通测量关闭计数。resident不使用这些eager桶：其实际/物理行和stages仍从`actual_stats`读取。

三个客户端在计时外从完整CPU结果生成`boundary_activity`：每sample的候选、Full、
实际接收原子、逐物理边消息、输出、pending及存储KV owner/cache长度直方图。
只使用已解码的整数和tensor shape，不读timestamp值；原始缓存可保留已过期项，
不能把这一边界分布当作逐查询可见长度，也不会为尚未创建的owner虚构空状态。

计数只消费算法已经形成的长度、shape和容器大小，不新增tensor读值、nonzero、复制或设备同步。
训练包含数值前向和私有VJP准备的前向调用，不代表去重后的逻辑工作，也不统计全部反向算子。
一次完整调用产生多个矩阵调用；计数不能替代设备时间线或端到端成本。

| 字段前缀 | 实际含义 |
| --- | --- |
| `packing/bucket/positive_write_segments` | 每owner正写入长度分布；循环扫描使用这些组，KV则把各组正写入汇成投影批。此项不宣称KV投影按该长度分桶 |
| `packing/bucket/kv_visible_read` / `kv_visible_full` | 每次Read/Full真正使用的可见缓存长度桶，包括空缓存 |
| `packing/bucket/attention_positive` | 真实正权重Attention长度桶；不同head/short/long调用分别计数 |
| `packing/projection/weighted_mm` | `packed::project` / Python `packing.project`实际矩阵乘调用的输入行数，包含合法零行；非全部模型算子的GEMM清单 |
| `packing/projection/private_bmm` | 独立输入/参数行或共享base后的私有映射的实际BMM调用，每次计一次；原独立根边界不变 |
| `packing/projection/kv_write_unprepared_rows` / `kv_write_shared_base_rows` | 同节点KV写准备的输入行组；每组内部有多次矩阵乘，不能与`weighted_mm`相加当总算子数 |
| `packing/skipped/independent_projection_budget` / `independent_projection_singleton` | 8MiB参数打包预算或单行尾组保留原路径的行数；不等于跳过数学作用 |
| `packing/skipped/zero_write_owners` | 原算法正接收数为零、保持原始状态的owner数 |

每类桶导出`calls`、`rows`、`elements=length×rows`累计以及
`length/L/rows/B/calls`直方图；投影导出`calls`、`rows`和`rows/B/calls`。
直方图与累计行数一起判断碎片，最大batch单值不足以证明多数工作已形成矩阵批。
最多保留4096个键，`packing/dropped_updates`明确报告溢出的新键更新，不能把截断分布称为完整。
`packing/schema_version=1`识别匹配的新core；旧库缺失此标记明确拒绝观测请求。

C++计数复用`operator_work.h`启停，细节在`packing_work.h`；工作线程运行期间不得reset或读快照。
Python作用域由`weighted/packing_observation.py`管理，嵌套计数独立；观测器不存入模型对象。
单进程只观察一个执行器。`test_packing_observation.py`用手算不齐KV桶、独立根、
实际dispatcher算子序列及独立C++并发计数见证检查计数口径；真机profile与性能仍分别验收。

研究索引保留profile/观察的完整核对及成本，但不将其标为合格吞吐。
本轮推理若声明`checkpoint_check=true`，还须有实际精确恢复结果；历史未声明的前向证据
保持原范围，不补造恢复资格。运行约定见[完整返回测量](weighted-performance.md)，结果与下一步见[STATUS](STATUS.md)。
