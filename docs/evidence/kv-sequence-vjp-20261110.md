# KV整序列反向固定资格（2026-11-10）

实现`413faf6053d9e85042b0c65de1ce72c4a739d385`，数学0770593。
[机器记录](kv-sequence-vjp-20261110.json)保存固定作业、构建身份、严格失败、成本和消费者结果。

未公开中间State根的prepared KV块，把前缀伴随整段回收；完整trace继续保留中间状态的因果根。
适用条件与实现导航见[State VJP](../weighted-state-vjp.md)。三个Read、暖态、共享、稀疏时间、
独立根及完整训练/恢复的CPU FP64/FP32定向检查143通过；NPU FP32 49通过，
另2个原GDN/HB evidence根xfail保留。Python/native安装消费者各在CPU/NPU完成
三步独立训练、原始状态和新进程精确恢复。并非全部图/配置/规模的资格。

有界GDN/HB诊断用原FP32材料逐值提升FP64：evidence_weight梯度对FP64的最大误差，
CPU FP32为3.62990e-6，NPU为2.59307e-7；bias和输入根也更接近FP64。
支持这一个见证的舍入解释，不解除原1e-6/1e-5严格阈值，不把所有有限差异归为舍入。
按用户授权，保留原失败后继续性能工作，不重复阻塞。

原下游固定0439efd的FC16，d128/KV qry/quad，B8/L128、FP32、同初始参数，
一次预热后恢复权重/optimizer再计时一步，完整wall包含准备、前后向、AdamW和同步。
CPU一线程，NPU mixed-c两host线程；Ascend910物理15，共享占用经有界准入记录。

| 返回契约 | CPU wall / forward / backward（秒） | NPU wall / forward / backward（秒） |
| --- | --- | --- |
| 完整事件返回 | 32.564 / 11.887 / 20.175 | 57.795 / 26.871 / 30.107 |
| 最小返回、隐藏中间State根 | 25.407 / 9.273 / 16.026 | 36.841 / 15.689 / 20.963 |

两种返回契约分列，不称同契约加速。最小返回实际joint_state_vjp_events=16384。
CPU候选16384/Full8344/消息25060；NPU候选16384/Full8345/消息25064。
设备之间活动数不同，原任务独立数值/离散/学习均未认证，不能称等价吞吐。
NPU最小返回含预热的peak allocated为4210.89MiB。小profile与吞吐进程独立。

构建复用seq-b冻结overlay，C++ hash b54e39eefab14b5dd8eaef2b4a437326310bf688611211fdabdca01608b4e752
与固定实现相同，原构建manifest仍保留0915781及原二进制身份；未将旧构建身份改写。
新wheel从固定413faf6构建安装。旧seq-a缺头文件失败不改写。

复现：固定源码构建安装后，运行tests/test_kv_sequence_vjp.py、test_kv_prefix_roots.py、
test_weighted_state_vjp.py、test_component_window_weighted.py、test_weighted_block_preparation.py；
NPU设置TIDE_WINDOW_DEVICE=npu:0与独立TIDE_DENSE_NUMERIC_REPORT目录。
安装消费者为examples/consumer_core5/dense_query.py --no-trace；
成本为scripts/benchmark_dense_query.py --trace-records complete|minimal，
数值见证为scripts/diagnose_window_evidence.py。后两项用显式device及新输出目录。
原数据/任务头来自只读下游Git对象；基座验收无需旧运行目录。

本报告不认证大宽度/多节点代表、standalone新反向、resident、多卡、FP16或CUDA。
