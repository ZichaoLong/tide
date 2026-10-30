# 独立 LibTorch 的 FP32 master 与半精度续接

数学图/局部函数与[带权精度策略](weighted-precision.md)不变。本接口只选择参数更新和
持久化表示；不改变共享参数身份、私有状态、逻辑时间或消息存在性。
公共 `SGD/AdamW(registry, groups, OptimizerPrecision{true, scale})` 显式要求FP16叶参数，
每个canonical owner只有一个FP32 master。其他精度保持原更新路径；不自动改算法。

`backward(FP32 scalar loss)` 乘静态正有限scale，累积到实际half叶梯度；`step()`先将
所有已定义梯度转FP32并除scale，全部有限检查通过后才更新master和FP32 optimizer slots。
None不更新参数/weight decay/状态；连接零保留相应更新语义。随后检查half可表示性，
再发布量化payload。非有限梯度失败不修改master/slots；更新后half溢出要求恢复checkpoint，
不承诺此类失败回滚master。无自动跳步、动态scale、剪裁或隐式detach。

输入必须是声明组内唯一canonical的受支持FP16 trainable leaf；别名按既有registry折叠。
loss scale不改变精确数学导数，但不能恢复已在half叶边界丢失的梯度。
二阶训练、竞争模型效果及跨设备bitwise optimizer轨迹不在这项机制验证内。

`TIDENCK1`和`TIDEGCK1`当前版本由[semantics](semantics.md)统一维护。
schema 2新增Half dtype code 4（二字节原值，int64 code 3仍只用于metadata），以及显式
master策略、loss scale和具名FP32 masters。schema 1仍按旧布局读取，拒绝half tag；
外层/内层schema必须一致。新保存采用schema 2，Python checkpoint v5未变。

加载先验证全部别名、组次序、dtype/shape、精度策略、scale、master inventory和slots；
master量化必须与文件里的实际half参数逐值一致。所有分配/owner转移与完整continuation
校验在发布前完成。恢复保留目标Tensor身份；scale和optimizer slots由文件恢复；
输入grad/retained图不序列化。raw State/History/pending按实际dtype解码，不调用参考递归。

| 契约 | 实现与独立检查 |
| --- | --- |
| 显式更新与None/zero/alias | `optimizer_precision.cpp`；`test_cpp_master_optimizer.py`与独立Python FP32 master比较三步及恢复第四步 |
| 参数/optimizer格式 | `checkpoint_codec.cpp`/`checkpoint_preflight.cpp`；独立构造v1字节、v1非法half tag、精确master/slots恢复、非有限/不一致拒绝 |
| 完整图训练和raw续接 | `weighted_graph_training.cpp`/`graph_checkpoint_codec.cpp`；同CPU输入和量化参数的独立Streaming/Greedy，全部记录/VJP/optimizer及原始int64字段；设备checkpoint转CPU后独立第四步 |

standalone入口新增`--dtype=float16 --scoring-dtype float32 --optimizer sgd|adamw --loss-scale 16`。
half比较atol1e-3/rtol2e-2用于所有浮点观察量；离散与None严格；恢复的raw/master/slots逐值精确。
默认AdamW不会因某个图失败而自动换SGD。开发、固定源码资格与设备范围分别记录于STATUS/证据索引。

[5eadded固定资格](evidence/core5-native-master-20261027.md)分别覆盖CPU24项及
独立SDK实际NPU FP16显式SGD24图/AdamW2图、设备→CPU checkpoint后的第四步。
