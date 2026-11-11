# KV 位置列资格

实现及固定运行 `9fa852cbd3122521f68393600a34acdd87254684`，数学 core-5 `0770593` KV §2–4。
[机器记录](column-positions-20261111.json)包含构建hash、源码、命令、资格和六个消费者记录。
本次不重复上一实现的性能矩阵，也不把该实现未测的性能外推到带位置配置。

Python/C++ 的B×L Read和Full新增相对旋转与距离logits，位置先用int64求差，避免超过
2**54的绝对时钟掩掉小间距；position_origin按固定旋转恒等式抵消。
单个value bank的显式一阶VJP汇总全部查询，以中心值求softmax transpose并精确保留单支持
分支的零logit导数；同一接口仍只声明一阶。旋转临时量纳入workspace拒绝边界。

固定CPU 84 passed、13原具名FP32 numerical xfailed。
新检查含Read0/-/+、一般端口和过期、独立literal softmax/方向导数、非空绑定列初态、
共享参数、普通列根的None/zero、两窗训练、位置checkpoint和exact next window。
同材料FP64和原独立调度检查保持，未放宽阈值或改活动支持。

六个安装消费者CPU FP32/64×Python/native、NPU FP32×Python/native全部独立三步AdamW
严格检查通过，新进程输出/梯度/参数/槽/raw状态精确恢复通过。NPU空闲物理11、host0–15，
匹配CANN9.2/Torch2.12，mixed-c主机推进；GPU、resident、多卡不在本次资格范围。
standalone SDK新构建并实跑CPU FP64 `tidegraph-column-check`，含有/无位置两种配置的
独立streaming、raw状态、VJP及两个SGD窗口；不把CPU executable通过写成NPU SDK实跑通过。

开发阶段两个超大时钟测试从0扫描的进程已停止，改为从对应空cut运行time-major oracle。
另一个Read0测试曾把独立空样本叶误当列根；改成在同列根上解码标量oracle，旧事件根能力
继续由记录接口验证。两项是测试范围/根契约修正，不以容差掩盖实现错误。

复现：按[开发指南](../development.md)安装匹配源码/库，运行
`pytest tests/test_column_positions.py tests/test_column_training.py --dtype both`；
安装示例 `examples/consumer_core5/tensor_window.py --positions`，分别传client/dtype/device及
新的output-dir。实际硬件必须合法分配，长任务冻结源码并使用bounded background.slice。
