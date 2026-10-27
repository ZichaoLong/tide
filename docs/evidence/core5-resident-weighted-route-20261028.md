# Weighted 实际消息祖先反向资格

固定实现 `ad8e7b82a20b1e41403ff5274fa034c1c43a8a10`，固定上游 `0770593`。
[组件契约](../resident-weighted-reverse-route.md)；[机器记录](core5-resident-weighted-route-20261028.json)。

CPU 12项（0.76秒）与standalone NPU FP32 12设备案例/dtype拒绝通过，物理9→npu:0。
手算核对重复物理边各一次贡献、私有样本、外部输入和左pending的独立根、连接零及未连接NaN。
大于2^54的相邻发送时间保持不同；缺失/重复/未来祖先、错误边、缺席原子明确拒绝。
已连接非有限值和累加溢出拒绝，错误粘滞，不能导出部分导数。

CUDA Torch2.10/cu128、Toolkit12.8.1的路由kernel/wrapper构建及loader通过。
160个已配置CUDA翻译单元的命令精确sm80/90/100；只声称所选路由目标已构建。
无GPU真机，本报告不认证CUDA实际执行。NPU与CUDA固定资格均clean源码、exit0。

初次两后端开发缺DeviceProgram头声明的编译失败保留。首次CUDA固定资格实际构建成功，
但launcher错误要求重编后二进制摘要不变而failed；新资格在相同固定源码记录真实binary/loader摘要，
原失败不重标。旧manifest与全部component/core摘要核对记录保留；未变目标不冒充重新编译。

组件实现只做实际消息祖先归约。完整窗口/retained训练、optimizer与训练恢复、FP16和peer仍需集成。
CPU同源元数据检查使用独立手算期望，但不另称独立图调度证据。
