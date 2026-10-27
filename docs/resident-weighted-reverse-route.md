# Weighted 消息反向的实际祖先绑定

固定[PDG有限切面](upstream/tide-core-5/positive-delay-graph-finite-cut-learning-note.md)
与[带权局部架构](upstream/tide-core-5/local-semantics/weighted-message/architecture-learning-note.md)
定义消息身份及未缩放载荷 `(c,G)`。本组件接在
[完整数值阶段VJP](resident-weighted-stage-vjp.md)的物理原子梯度之后。

`weighted_reverse_route.h::append_weighted_reverse_route` 消费候选实际 AtomBatch、
节点 packing indices/lengths，以及每个物理载荷的FP32梯度与bool连接位。
`WeightedReverseEvents` 存实际前向的int64时间、发送存在位及每stage/node/sample的Full根；
`WeightedReverseBoundary` 存窗口左侧pending和本窗口外部输入的完整六元坐标、梯度和连接位。
所有缓冲区显式同设备、contiguous、no-grad；浮点宽度为完整 `(c,G)`，不再次乘c。

同窗口边按物理edge→source、发送逻辑时间、sample寻找唯一更早stage的生产者。
重复边各贡献一次，但同一生产者的全部使用者汇总到同一根；样本状态仍私有。
左侧pending和external按全部 `(sample,node,time,kind,source,position)` 匹配边界根。
所有时间与position保持int64；不按接收次数重编时间，不由CPU预跑提供轨迹。

设备串行执行每个stage的消息归约，避免并行scatter竞争。未连接目标不读取其浮点存储，
连接零仍设连接位。身份不合法、祖先缺失/歧义/非早期对应错误41；已连接非有限值或累加溢出
对应错误37。错误粘滞，整个反向结果失效；调用者不得导出部分梯度或重试继续累加。

| 对应 | 实现 | 独立见证 |
| --- | --- | --- |
| 逐边来源、私有样本与跨切面pending | `ascendc/weighted_reverse_route_math.h::route` | 手算重复边的向量和、单独pending/external根 |
| 精确发送时间 | int64生产者查找 | 大于2^54的相邻时间不可合并 |
| None/零梯度 | 每根bool与独立浮点buffer | 未连接NaN不读、连接零保留 |
| 明确拒绝 | sticky error 37/41 | 缺失/重复/未来祖先、非法边、缺席原子、非有限及溢出 |

`tests/test_resident_weighted_reverse_route.py` 在CPU编译同一元数据源，使用独立手算期望；
这不是CPU resident fallback或独立调度证据。
`weighted_reverse_route_check.cpp` 通过真实DeviceProgram执行同一组12案例及dtype拒绝。
本组件不替代完整window reverse、状态前驱、参数归约和训练owner；运行状态从STATUS及证据进入。

```sh
python scripts/build_device_control.py --core-build CORE --build-dir NEW_BUILD \
  --checks weighted-reverse-route --ascendc-soc Ascend910_9392 --jobs 2
python scripts/verify_device_control.py --build-dir NEW_BUILD --output-dir NEW_CHECKS \
  --checks weighted-reverse-route --device npu:0
```

固定来源的组件资格见[2026-10-28报告](evidence/core5-resident-weighted-route-20261028.md)。
