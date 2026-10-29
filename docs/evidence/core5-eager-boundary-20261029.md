# 单行反向、异形传输与实际分配观察资格

固定实现`e0d6e27`（完整hash见[记录](core5-eager-boundary-20261029.json)），上游0770593不变。
本阶段交付公共eager优化与诊断，不将原数值失败写成已修复。后续resident/性能设施另验收。

## 实现与独立验收

Python/C++单行Query/Aggregate直接使用原输入/参数图，轻量一阶边界保存版本并克隆返回；
不再为一行detach/rebind参数及嵌套autograd.grad。多行仍保留独立私有图，None/零和参数别名不变。
传输按device/dtype分组，把实际异形字段flatten/cat，按原shape/offset恢复，8MiB分块；
单个超大字段仍不可拆分。元数据int64单独分组，独立根、空/noncontiguous字段和版本守卫保留。

AllocationFailure补标量数值拒绝；native线程局部有界observer记录实际scalar/batch成功返回，
不重新调用参考、不写文件或回调Python。早退没有算出的残差不得伪造；最小subnormal反例独立验证。
原生完整图启停观察前后逐值相同，连接梯度不变、超预算报告dropped，嵌套观察拒绝。

[对应](../semantic-map.md)和[分配](../weighted-allocation.md)、[Query](../weighted-query-batch.md)、
[packing](../weighted-packed.md)列定义/符号/独立测试。手算KV、局部方向导数、独立字段根与版本检查
验证局部映射；Python/native/standalone独立调度、四更新及checkpoint验证整图组合。

aarch64 CPU、Torch2.12/Python3.12.13：干净e0d6e27资格 **517 passed、2原FP16 strict xfail**，
287.39秒。复用开发二进制前核对完整CPP hash及23个binary hash；manifest保留真实dirty构建来源，
未改写成clean build。相关开发门禁先有177 passed/2 strict xfail与329 passed；正式结果不以它们替代。
本报告不外推GPU/x86_64，也不由CPU推断NPU常驻。

## 原FP32见证与精确恢复

D64原计划hash`6c8cc36ae97287133d40e67ccc8e137574b80e2df420feff6adfef3395397bb3`，
480节点/2208边、9076975可训练主体参数。原SGD与原容差不变，两次更新比较完整记录/离散结构/
参数梯度/参数/动量槽，并各自保存、换owner执行下一窗。

| 原D64 FP32 | 第一步不通过梯度张量 / 参数张量 | 第二步 | 自身逐值恢复 |
| --- | ---: | ---: | --- |
| native prefill vs 独立标量Python | 2322 / 2 | 2712 / 6 | 两条均通过 |
| Python prefill vs 独立标量Python | 2273 / 2 | 2375 / 2 | 两条均通过 |

两步离散结构一致，但完整浮点记录和SGD槽均失败。首个逐值差异是time1/node0的proposal keys，
最大差2.9802322387695312e-8；它本身在容差内，不能把“首次逐值不同”等同首次验收失败。
标量/native实际分配各捕获45行，native19581字节、dropped0；没有分配器拒绝。
因此这次消除私有VJP重建不足以解决原D64跨调度数值问题。

原W32/B4/四位置反馈PDG四窗中，native streaming/prefill均有记录、梯度及SGD槽严格差异，
参数检查通过；reference/streaming/prefill三者各自第三窗保存、第四窗换owner恢复全部逐值一致。
保持严格失败；不修改输入、路由、容差或optimizer，不重复此前九项分段精度探针。
已有显式CPU FP64是另一配置，本阶段未用它替换FP32失败。

## 终态和停止依据

两个background.slice服务各1170秒、单线程ATen/BLAS、MemoryMax=16G；不是独占资源证明。
`core5-boundary-cpu-qualify-20261029` passed/0；`core5-boundary-cpu-diagnostics-20261029` failed/1，
三个诊断进程各strict_mismatch/2，记录保存后正常退出。全部记录/见证保留。

单行VJP作为工程实现通过相关资格，但数值修复候选到此停止；重新推进需要覆盖局部算术、
中间伴随与共享贡献的具体新方案及独立根证据，不能仅重复最后一级汇总变宽。
这些检查没有性能计时，是否带来收益由后续固定源码单次完整训练对照回答。

干净源码按[开发入口](../development.md)构建CPU_CORE，然后按JSON记录的15个pytest文件运行。
原见证使用公共入口：

```sh
python scripts/prepare_weighted_workload.py --topology ranked --layers 15 --region-width 32 \
  --width 64 --samples 1 --positions 1 --match-core3 add --output-dir NEW_PLAN
python scripts/diagnose_weighted_workload.py --workload NEW_PLAN/workload.json --dtype float32 \
  --implementation native --native-library CPU_CORE --steps 2 --checkpoint-check \
  --capture-allocations --output-dir NEW_D64
python scripts/diagnose_weighted_feedback.py --native-library CPU_CORE --dtype float32 --output-dir NEW_FEEDBACK
```

Python D64把implementation改为python即可；长任务按执行契约冻结源码与有限预算。
测试和报告无需依赖旧运行目录，原最小见证保留用于逐值核对。
