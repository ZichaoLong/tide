# 已安装带权配置检查接口资格

源码 `ce21a8359146854696af786379b48367756fd456`；固定上游 `0770593`。[契约](../weighted-message.md)；[机器记录](core5-weighted-binding-20261028.json)。

CPU Python/native 45 项（21.78 秒）及 C++ kernel-check FP64/FP32 通过，资格任务 clean、inactive/success/exit0。复用开发库时核验全部 C++ 源码与二进制摘要，原开发 manifest 保留。

检查五记忆与全部字段、来源分类、副本隔离、未安装或 State/Read/Full/Aggregate 配置冲突拒绝；接口读取实际安装的函数对象，不运行数值推理或按 shape/profile 名称猜测配置。原 dev 启动失败保留；dev2 的 45 项与两种 dtype 通过。

此接口资格不认证完整常驻图、设备训练、checkpoint 或性能。
