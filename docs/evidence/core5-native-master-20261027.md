# Core-5 standalone master/checkpoint 资格 — 2026-10-27

固定实现 `670ead324744bfb45b006d9ef3a22d1572ed0829`，上游 `0770593`。
[命令、环境、输入源码与日志摘要](core5-native-master-20261027.json)；三项固定任务均
clean/exit0，服务已终止。[接口与精确恢复契约](../native-master-optimizer.md)。

| 入口 | 本次通过范围 |
| --- | --- |
| CPU，Torch2.12 | 24项；独立PyTorch master公式、None/零/alias、SGD/AdamW三步与恢复第四步、有效checksum但不一致master拒绝、独立构造v1字节兼容 |
| CPU standalone完整图 | FP64/FP32各24图；FP16/FP32 scoring显式SGD24图，显式AdamW的FC/KV和反馈PDG/delta两图 |
| standalone NPU，LibTorch2.10/CANN9.0 | FP16/FP32 scoring/mixed-c，显式SGD全部24图及AdamW上述2图；loss_scale16/FP32 masters |

24图为FC、HB、非对齐TimedDAG、反馈/重复边PDG分别组合KV、affine、delta、GDN、SSD和异构。
入口自行构造固定参数/输入，CPU Streaming与候选Greedy独立调度；不传入参考轨迹。
比较完整Read fields、状态、history、逐边/端口值、pending、具名参数与输入梯度及optimizer。
包括保留图分段、三次更新、保存含pending的实际raw切面、加载后第四次更新。
NPU任务还将同一实际设备checkpoint加载到CPU，比较第四步；并非仅编译该分支。
恢复raw/master/slots逐值精确；计算FP16及其FP32观察量atol1e-3/rtol2e-2，整数/None/路由严格。

`TIDENCK1`/`TIDEGCK1`新写schema2，读取配对schema1；Half原值与int64 metadata分开，
量化master与payload须精确一致。加载前拒绝不一致记录且不修改目标。
图v13、Python checkpoint v5不变；没有隐式动态scale、换optimizer或更新失败跳步。

SDK任务动态分配1张Ascend910_9392，loader不含缺失库、stubs或libtorch_python。
launcher的rg不在服务PATH，原警告保留；另行Python核对相同loader文件通过，构建工具也
执行自己的闭包检查。未重跑设备workload。共享目录权限/base-format警告与结果一同保存。
CPU构建复用开发snapshot，其C++及全部binary摘要与固定提交相符，原source字段保留。

本报告不解除GDN默认AdamW的CPU半精度严格差异、SDK FP32 FC/affine严格差异或默认异步两卡错误。
显式SGD通过没有修复其他算法；不据此认证resident、双卡standalone、CUDA真机或完整FP16配置矩阵。

复现：固定实现按[development](../development.md)构建CPU/独立SDK，执行JSON命令并使用新输出目录。
CPU测试四入口无需设备；独立SDK程序不依赖Python运行模型。
任务background.slice/Nice10、2build workers、ATen/BLAS单线程；CPU/构建1800秒、NPU1200秒上限。
