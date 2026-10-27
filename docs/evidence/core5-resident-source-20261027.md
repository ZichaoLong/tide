# 常驻来源低秩参数链资格

源码 `6f709f8332f179ffce3c315bd3f77e7e7c26cca8`，固定上游 `0770593`。
[契约](../resident-weighted-source.md)规定真实基座/因子返回；[JSON](core5-resident-source-20261027.json)
记录命令、源码和二进制摘要、实际队列及终态。两个任务均clean/exit0/inactive。

- CPU：18项，1.83秒（3项source guard及15项Aggregate公式）。
- standalone NPU FP32：3个bank、d3/H4/r2、三个基座角色和四个低秩因子。
  实际vendor BMM/permute构造有效A/B并接完整Aggregate；独立逐原子公式/autograd
  检查全部载荷和真实参数VJP、共享使用者、None/零、因子/长度重放、保存版本反向。
  非有限前向拒绝、失败不提交、容量拒绝通过。
- CUDA sm80/90/100组件、standalone消费者和loader构建通过；没有GPU真机证据。

原始基座、因子和输入全部改为NaN/非法值后，独立反向程序仍与原实际调用的参考VJP相符。
根存在性与数值零分离；矩阵乘之前屏蔽无连接bank，保留真实零根。
浮点atol=1e-6、rtol=1e-5，None和连接严格比较。此记录证明有效矩阵的导数已传至四因子，
跨bank或不同角色的同一Parameter仍需canonical owner汇总，不能据此认证整图训练。

NPU为aarch64/GCC10.3.1/LibTorch-NPU2.10/CANN9.0/Ascend910_9392；
CPU适配Python3.11.15，动态单卡队列物理9→npu:0，无观察到的CPU fallback。
NPU复用开发treea3ee2b01构建，整个component/core/binary摘要与固定源码相符，原manifest身份保持。
CUDA为Torch2.10+cu128/Toolkit12.8.1的新构建，核心670ead3摘要保持。

首次CPU setup缺kernel登记，生成器报告3 errors，未执行数值检查；
原treebf59ee14和失败复现保留。登记修正后的通过不重标原失败。
完整图、canonical参数owner、retained训练/optimizer/checkpoint、跨卡、FP16和性能继续。
