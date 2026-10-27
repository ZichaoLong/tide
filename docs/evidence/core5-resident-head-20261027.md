# Core-5 常驻批量 head 资格 — 2026-10-27

固定实现 `161ccead025f33da2fb9a4f956982a43997a56b4`，固定上游 `0770593`。
[命令、环境、来源与摘要](core5-resident-head-20261027.json)；[ScalarHead/Query接口](../resident-weighted-head.md)。
两项资格任务clean、exit0、已终止。

| 路径 | 已通过范围 |
| --- | --- |
| CPU元数据/归约源码 | 8项，2.05秒；动态int64 owner、安全索引、缺席行NaN、独立零根、共享使用者/未用bank、溢出和sticky error |
| standalone NPU FP32 | ScalarHead I4/H5/O1、Query I3/H3/O3；24行；vendor index_select/permute/bmm/SiLU及其完整VJP、两组动态owner重放、共享参数多个非零贡献、None/零、原bank改写后的真实tape反向、失败不提交和容量拒绝 |
| CUDA构建/loader | sm80/90/100；metadata/reduction kernels、vendor program与standalone检查程序通过；无GPU实跑 |

NPU为独立LibTorch2.10/CANN9.0与动态单张Ascend910_9392。数值参考是逐行独立ATen公式/autograd，
atol1e-6/rtol1e-5，int64/连接关系严格。head输出宽1和向量Query分别检查。
设备按实际owner聚集参数，矩阵乘和激活使用批量厂商算子；所有使用者偏导按owner归约。
两次程序调用复用同一程序并改变owner/输入；反向另建程序，故意改写原input/owner/bank后
从实际保存的版本计算，未重新用参考或原bank补写激活。

两种形状的实际operator workspace高水位都是16,777,728字节，检查声明64MiB上限，
不含调用者bank及独立预算的tape/梯度scratch。连续转置通过显式vendor permute实现并计入scratch。
第一次开发拒绝非连续视图；第二次因原4MiB workspace上限拒绝。均在构造阶段失败，
未计数值通过；原failed与source快照保留，没有静默扩大已提交任务预算。
NPU复用workspace-dev的确切component/core/binary摘要，原build身份保留。

仅验证共同双仿射+SiLU head，不认证完整context/query归约、SelStep、图常驻、retained训练、
optimizer、checkpoint、跨卡、FP16或吞吐。GPU真机待验；CPU适配不替代vendor执行证明。
新checkout按JSON命令重建，无需旧运行目录。资源与有限时长见JSON。
