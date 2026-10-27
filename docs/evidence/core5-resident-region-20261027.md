# 常驻完整带权区域组合资格

源码 `ce63aefcceabb83471737a9cdc7a5295b8e26e33`，固定上游 `0770593`。
对应[区域组合契约](../resident-weighted-region.md)；机器可读身份、命令、摘要与运行记录见
[JSON](core5-resident-region-20261027.json)。两项固定源码任务均clean、exit0、inactive。

- CPU：33项，8.69秒。独立公式检查十种发送的context/query/control链及六字段VJP；
  登记入口检查缺席、poison、动态长度、大int64坐标、重复ID和sticky error。
- standalone NPU FP32：14配置，140物理行/112非空帧，包含context/fixed、ind/att、
  quad/bern/ent。实际vendor批量head接设备完整归约和发送，比较全部control/active/history、
  六字段与全部共享参数根，独立零根/None、重放、保存版本反向、容量和失败不提交。
  新单卡队列动态选择物理9→逻辑npu:0；未固定占用卡号。未发现CPU fallback。
- CUDA：sm80/90/100的完整组件、standalone消费者和loader构建通过。没有GPU真机证据。

设备参考是独立标量RegionKernel/autograd；CPU公式为另一个局部链式数学锚点。
共享Alloc/发送数学接口已有各自独立公式/KKT/VJP资格；本项验证其完整组合，
没有将两个调度互比当成对底层所有公式的重新证明。浮点atol=1e-6、rtol=1e-5，
支持、None、ID和history严格相等。描述字段独立偏导检查不代替Describe增益绑定验收。

NPU为aarch64/GCC10.3.1/LibTorch-NPU2.10/CANN9.0/Ascend910_9392，
CPU适配使用配套Python3.11.15；CUDA为Torch2.10+cu128/Toolkit12.8.1。
NPU复用已通过开发tree4a161a96构建，验证器核对整个组件、core源码和二进制摘要与
固定源码一致，原manifest source/dirty保持。CUDA使用新目录构建。
核心670ead3摘要保持，未重跑历史全矩阵。vendor workspace显式64MiB。

这是完整SelStep组件资格。完整图就绪/推进、Agg/五种记忆/Describe、retained训练、
optimizer、精确checkpoint、跨卡、FP16和性能继续按ROADMAP实施；不由本记录代替。
