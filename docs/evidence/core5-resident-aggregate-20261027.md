# 常驻完整带权 Aggregate 资格

源码 `92751c0c4ecc1839828f3e36980821e41520c5d8`，固定上游 `0770593`。
[契约](../resident-weighted-aggregate.md)给出域/编码/反向；[JSON](core5-resident-aggregate-20261027.json)
记录精确命令、源码/二进制摘要、队列与终态。两任务均clean、exit0、inactive。

- CPU：15项，1.3秒。独立逐原子公式/autograd检查同类不同载荷、
  alpha0/0.25、零/负质量、单原子、缺席/零根与完整归一化链式VJP。
- standalone NPU FP32：两配置、24物理行/20非空纤维。实际vendor RMSNorm/双仿射/SiLU，
  4/2/3/1隐藏宽度精确补零，完整载荷及有效A/B/e参数根、共享使用者、零总量分支、
  改写原始输入/参数后的保存版本反向、动态长度重放、容量拒绝与失败不提交通过。
- CUDA sm80/90/100：组件、standalone消费者和loader构建通过；无GPU真机结果。

独立设备参考使用逐原子ATen公式和autograd，不调用候选helper或公共Aggregate实现。
浮点atol=1e-6、rtol=1e-5，None/零、缺席、长度和源owner严格检查。
同一SrcClass每个物理原子分别变换并贡献，未用SourceDomain唯一性代替分类。
(r,u)来自设备实际归约；没有参考递归补写解码结果。

NPU工具链aarch64/GCC10.3.1/LibTorch-NPU2.10/CANN9.0/Ascend910_9392；
CPU适配Python3.11.15。单卡队列动态选择物理9→npu:0，未观察到CPU fallback。
NPU开发tree8f9bc2bf的构建与固定源码整个component/core/binary摘要一致，保留原manifest身份。
CUDA为Torch2.10+cu128/Toolkit12.8.1的新构建。核心670ead3摘要保持。
线程/时间边界在JSON中；本项无吞吐或全图支持结论。

源参数接口在本提交接受有效A/B/e。低秩因子链与跨bank/角色的真实别名owner仍须单独集成，
不能以有效矩阵梯度替代实际参数梯度。五种记忆、Describe、在线图调度、训练/恢复、
跨卡、FP16与性能仍按ROADMAP继续。
