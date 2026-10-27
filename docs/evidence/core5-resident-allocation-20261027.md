# Core-5 常驻分配组件资格 — 2026-10-27

固定实现 `2814b8799b0b9f7ba251e656f4191a7af4e6c7be`，固定上游 `0770593`。
[命令、环境、来源和摘要](core5-resident-allocation-20261027.json)，[接口/定义对应](../resident-weighted-allocation.md)。
两项资格任务均clean、exit0、已终止。构建与实际设备结果分列：

| 路径 | 结果与边界 |
| --- | --- |
| CPU语义源码适配 | 29项通过，2.67秒；独立quad可行面、80位Decimal熵解、手算与方向差分、独立根/None与零、padding/空区域/严格拒绝 |
| standalone NPU FP32 | quad、bern、ent alpha=1/1.000001/1.5/2；72物理行/60非空帧，完整前向和六输入精确解VJP、同program改变输入/长度重放、失败不提交/sticky error/容量拒绝通过 |
| CUDA构建/loader | sm80/90/100的前向、VJP、错误汇总kernel及standalone检查程序通过；本机无GPU，未运行CUDA张量 |

NPU采用LibTorch2.10/CANN9.0、Ascend910_9392动态单卡；CPU公式检查使用SDK配套Python3.11/Torch2.10。
参考为独立C++ scalar Alloc/autograd，atol1e-6/rtol1e-5，分支及None/零严格。
CANN Exp/Ln、分支、求解与反向均在设备程序中；单个scalar worker写紧凑行，
窗口内求解无主机决定。此报告不声称该算法有吞吐优势。
固定80次前向二分与保存响应点的精确解VJP具有不同身份；没有HST或迭代导数混用。

NPU复用开发build4，完整component/core/binary摘要与固定源码核验一致，原manifest身份保留。
CUDA为固定提交的新构建，Torch2.10+cu128/Toolkit12.8.1。loader闭包未含缺失库、stub或Python运行时。
四次开发失败分别为设备内建名称冲突、PIC链接、CANN搜索路径和bool fixture构造，
原失败与最小源码记录保留；其后通过不改写历史状态。

本组件只验收完整候选区域的Alloc与显式一阶map，不认证完整weighted resident图、
retained training、optimizer、checkpoint、跨卡、FP16载荷或性能。上述仍由ROADMAP推进。
干净源码按JSON命令重建即可，不依赖旧运行目录。资源/时限见JSON，未修改公共工具链。
