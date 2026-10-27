# Weighted公共训练与精确切面资格

固定实现`343429db76c091541ae24ef7b3792ce027e9bea2`，固定上游`0770593`。
[公共契约](../resident-weighted-training.md)、[保留窗口](../resident-weighted-retained.md)、
[完整机器记录](core5-resident-weighted-training-20261028.json)提供定义、独立检查和源码/二进制/loader摘要。

单设备FP32公共训练已通过本次定向资格。standalone NPU的5配置20次更新覆盖FC/KV、HB异构、
非对齐DAG/delta、PDG反馈异构与全部冻结affine；包含SGD、AdamW、部分冻结、完整字段根、
重复输出端口、三窗链与空窗、更新后角色发布及新owner精确恢复。
独立CPU Greedy/autograd先后与候选分别执行，核对所有参数、外部输入、左raw与pending根，
None与连接零严格区分。两组数值梯度累计另用独立求和/SGD核对；容量、token、版本及错误更新拒绝保留。

保存窗口12配置、保留切面8配置×5根模式通过：前向owner关闭后的实际journal反向、连续三窗
跨切面链式VJP、空窗恒等及独立边界根均受检。另有18种布局/54次复制、21个完整推理图案例和
15个正式配置回归。局部数学的手算、独立公式与方向导数沿模块契约追溯；本次调度对照不替代它们。
前向容差rtol=1e-5/atol=1e-6，VJP为3e-5/3e-6；原输入与严格离散支持检查未放宽。

Python-owned NPU通过28项（训练10项、推理18项，89.63秒，无skip），其中八种图/记忆/优化器组合
各三次更新，验证公开字段cotangent、磁盘保存、精确恢复，FC另在新进程读取磁盘并完成后续更新。
预排期曾误估33项；固定源码两个测试文件实际完整收集为28项，没有删选或跳过案例。
standalone与Python-owned分别使用LibTorch-NPU2.10/CANN9.0、Torch-NPU2.12/CANN9.1，均物理9→npu:0。
CPU相关28项通过（无skip；另8项设备测试按明确scope deselect），这不是CPU常驻设备资格。
两个独立安装消费者实际NPU运行通过：手算推理和恒等VJP、无作用参数None、新owner切面恢复。

CUDA完成相同组件、两个公共库及两个安装消费者的构建、loader/help；160个已配置CUDA命令
的架构均为80/90/100。没有GPU真机，不认证CUDA实际前向、训练、恢复、跨卡或吞吐。
三个资格任务均干净固定源码、持久passed/exit0与systemd inactive/success/0相互印证。
NPU复用核对为同源的kernel archives并重建host检查；Python使用全新build；实际binary/loader与安装消费者摘要均重新核验。

开发失败原样保留在机器记录。完整顺序下恢复反向的CANN507011经stream/task定位到
`recurrent_detail::insert`的Bool [2,4]按axis1写入[2,1]。NPU改为已知布局逐行连续厂商copy，
资格使用原完整顺序、输入和容差，诊断跟踪关闭。未证明厂商内部根因。
其余构建、fixture与padding诊断失败也保留，不能因后续成功改写历史状态。

精确恢复比较覆盖present/valid域内全部实际字段、过期但仍保留的raw、int64时间、完整解码、
参数/optimizer与checkpoint；未用journal/padding无数学值契约。standalone公共checkpoint为
完整CPU内存记录；Python额外认证磁盘和新进程恢复，不把二者混写。
本报告不认证weighted resident FP16、peer、性能；原GDN半精度AdamW、默认异步双NPU与
SDK2.10 affine严格失败仍未解除。复现入口见公共契约和机器记录，无需旧运行目录。
