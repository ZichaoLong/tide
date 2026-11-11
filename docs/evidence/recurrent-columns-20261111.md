# 五记忆的 native 张量窗口资格

固定实现 `b860e6ae304696ea91a58a8bc52f3d913462a9fa`；数学 core-5 `0770593`。
[机器记录](recurrent-columns-20261111.json)保留两个任务的终态、原始构建及复用hash、
命令、JUnit和24个安装消费者的逐项结果。前一版KV/位置的证据保持自己的源码身份。

C++ `LocalState/LocalWindow` 增加循环状态类型；affine、delta、GDN、SSD通过
`weighted_column_recurrent_{state,math,query}` 接入共同的区域、Full与消息列接口。
KV仍使用历史bank；循环使用H/Z/W及适用的原始逻辑时间卷积cache，不强制私有形状相同。
时间槽先在int64元数据中匹配，再批量gather浮点值。容量估算包含ragged填充和卷积临时量。
Python/native默认窗口均可处理五记忆及异构组合；完整事件根仍由记录API单列提供。

固定CPU相关检查 **279 passed / 13原具名FP32数值xfail**；另有一般PDG、非对齐
TimedDAG、Settle区域执行回归 **27 passed**。检查包含两种Mem、三Read快照、超大时钟、
共享参数、None/zero、非空旧态独立列根、保留图窗口、空样本和容量拒绝。
旧数值超差与原阈值保留，没有把服务终态通过等同于全部严格数值通过。

安装消费者四族×两客户端×CPU FP32/FP64共16组，NPU FP32共8组：全部三步AdamW
对独立标量调度严格通过，新进程输出/梯度/参数/optimizer槽/raw状态精确恢复通过。
实际都命中 `settle-columns-v1`。NPU空闲物理11、host0–15、ATen2、mixed-c主机调度，
环境为CANN9.2/Torch2.12/TorchNPU2.12.0.post3。CPU池80–111；两个任务使用分离CPU池。
standalone SDK的 `tidegraph-column-check` 实跑CPU FP64，五记忆独立streaming、raw状态、
VJP及两个SGD窗口通过；没有把此CPU结果写成NPU standalone实跑资格。

开发失败保留：dev1的容量测试遗漏pybind的ValueError；随后新进程暴露无卷积cache的
None→Tensor绑定错误，已改用显式optional并在dev2及固定消费者重验通过。
这些是实际修复；原任务失败没有改写。此次未新增循环性能、多卡、resident或FP16资格。

复现：按[开发指南](../development.md)构建匹配库，运行
`pytest tests/test_native_recurrent_columns.py tests/test_recurrent_columns.py tests/test_column_training.py --dtype both`。
安装后从独立cwd运行 `examples/consumer_core5/tensor_window.py --family FAMILY --client CLIENT`
并传dtype/device、匹配native-library及新output-dir；FAMILY取affine/delta/gated_delta/ssd。
standalone使用独立SDK构建的 `tidegraph-column-check`。
这些是局部/小图机制资格，不代替大模型严格训练、所有组合或学习效果验收。
