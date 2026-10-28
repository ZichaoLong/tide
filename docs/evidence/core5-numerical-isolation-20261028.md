# 数值与异步失败定位

本轮没有放宽容差、修改失败输入或替换默认算法。两项已知限制保持严格失败；
[逐项记录](core5-numerical-isolation-20261028.json)保存源码、设备、结果与原始材料哈希。
固定数学0770593。诊断与功能资格分开。

| 诊断/固定源码 | 结果与能支持的结论 |
| --- | --- |
| eager FP16 GDN，Python/native；b0321b99ba3c4c7cffa091fceb09e8a3a3d50bc9 | 两者都在step1严格失败；不是只发生于native |
| 默认异步双NPU完整delta-PDG，所有传输换普通Torch拷贝；同上 | 完整前向通过，完整根VJP仍invalid event；不是自定义传输特有 |
| 12轮纯Torch逐元素/跨卡to()/双方scalar根；62073ddfbf0f5a8822ba1f1d60feefc12642a09d | 默认异步前向通过、反向EE1017失败；相同程序显式blocking前向/全部梯度通过 |

GDN首步同一坐标`nodes.5.kernel.conv.weight[0,2]`的未缩放梯度约为
+3.35276e-8与-2.98023e-8。AdamW首步的归一化项g/(|g|+epsilon)对这类近零差异敏感，
因此微小梯度差可以产生约1e-3级参数差；第二步参数最大差.001983642578125超过既定容差。
首步完整返回通过不代表优化器轨迹通过，FP32 master也不能恢复已在局部half运算边界损失的信息。
保持具名strict xfail；显式SGD或resident的另一套精度边界按原独立资格使用，不自动替换默认AdamW。

纯Torch程序只调用标准逐元素运算、copy和autograd，无图执行器、局部模块或自定义Function。
这把错误定位到本机Torch-NPU/CANN跨设备自动求导路径所需的更小条件；不能据此进一步断言
是驱动、CANN还是Torch-NPU的具体缺陷。栈为Torch2.12.0+cpu、Torch-NPU2.12.0.post2、
CANN9.1.0、Ascend910_9392；物理1/13映射逻辑0/1。blocking不是异步修复或吞吐证据。
公共工具链未改动；后续厂商修复须同时通过本小反例及完整图独立根、训练与恢复，才能解除限制。

复现（使用新输出目录和本机规定的有界持久服务/双设备队列）：

```sh
python scripts/diagnose_weighted_precision.py --help
python scripts/diagnose_weighted_async.py --device npu:0 --peer-device npu:1 \
  --copy-policy stock --case delta-pdg --output-dir NEW_GRAPH_DIAG
python scripts/diagnose_torch_peer.py --device npu:0 --peer-device npu:1 --output-dir NEW_STOCK_DIAG
ASCEND_LAUNCH_BLOCKING=1 python scripts/diagnose_torch_peer.py \
  --device npu:0 --peer-device npu:1 --output-dir NEW_BLOCKING_DIAG
```

默认异步须先unset ASCEND_LAUNCH_BLOCKING。两个模式各一个新进程、相同程序/初值；不重试取通过。
GDN两个进程各120秒，stock完整图300秒；新纯Torch各90秒、队列180秒，总420秒上限。
均固定只读源码、background.slice、Nice10、ATen/BLAS1。总体诊断job仍failed，内部blocking通过单列。
CUDA真机、x86_64、SDK FP32 affine及FP32反馈PDG的既有边界没有被本报告改变。
