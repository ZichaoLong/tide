# Weighted 常驻完整数值阶段 VJP 资格

固定实现 `ebe547115e8a70347c42ae420d669f1c6ea65fba`，固定上游 `0770593`。
[阶段契约](../resident-weighted-stage-vjp.md)；[机器记录](core5-resident-weighted-stage-20261028.json)。

standalone NPU FP32：24配置、342种根模式通过，覆盖五记忆×三Read快照，
qry、区域Attention/match/grouped、节点增益和异宽低秩来源。
raw字段、Describe、Content、control和Full独立根、物理输入及全部canonical共享参数
均比较数值和连接位；连接非有限根明确拒绝。容差保持 `rtol=3e-5, atol=3e-6`。
metadata 5项（0.78秒）及完整前向21图/边界回归通过。物理3映射npu:0。

CPU Greedy独立调度和标量autograd验证链式组合；局部数学独立公式/方向导数证据
由阶段契约链接的各组件提供。同源码局部函数对照不另称数学独立证明。
Read初态/旧态/候选态梯度分别处理；Full只读候选态；使用实际设备pack/order，
未CPU预跑轨迹，未以旧HST替代发送VJP。角色切片恢复原参数shape后归约共享使用者。

CUDA Torch2.10/cu128、Toolkit12.8.1：阶段/完整图/共享库与安装consumer构建、loader通过。
检查159个已配置CUDA单元的命令仅sm80/90/100；未声称所有单元属于所选目标。
没有GPU真机，此项不认证实际CUDA数值、训练或性能。

两任务均clean固定源码、inactive/success/exit0。NPU复用全部源码/core/binary摘要匹配
的构建，保留原开发manifest，不声称重编。2workers、BLAS1；NPU600秒/queue180秒，CUDA1200秒。
开发dev1/2缺测试头文件的失败、dev4遗漏非活动候选独立control参考根的失败均保留。
dev4只修复参考根连接，未改候选、输入或容差；dev5及固定资格通过不重标旧记录。

本报告认证完整数值阶段。跨阶段/跨窗口消息和状态反向、retained训练owner、
optimizer发布、精确训练checkpoint、FP16与peer路径仍需继续；阶段资格不是完整图训练资格。
