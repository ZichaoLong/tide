# 当前有限计划已完成

FFN权重直读W、有效行power2 packing P及组合WP已实现并固定身份；
CPU/NPU资格、继续训练、实际checkpoint恢复、同B8大图短请求和NPU持续decode均已有本轮记录。
事实与限制见[STATUS](STATUS.md)，完整表、阶段解释和复现在
[FFN执行与设备对照](evidence/ffn-execution-devices-20261119.md)。

中尺度NPU W/P/WP和CPU W/P均未提高同batch训练吞吐；省显存及自身恢复通过不替代跨路径数值资格。
原阈值超差、制品预算中止和取消记录保留，W/P/WP未扩至9B。
默认gather/rectangular不变；本轮没有待运行测点，不自动扩大到BF16、9B训练或新batch搜索。
五类记忆绑定、固定Transformer参考和A–D既有证据保持各自身份。

# 后续可选工作

以下需要按具体需求另选有限候选；不是本轮剩余任务，也不启动通用基座路线。

1. **分配回退行压紧。** 先处理真正请求fallback的行，保留原阈值、边界face与VJP；
   中尺度NPU诊断请求324行、执行49152行是定位证据，不能预先承诺相同比例加速。
2. **单次ragged FFN执行。** 有效行直接进入较少的grouped contractions，复用静态owner顺序，
   减少power2多桶、动态排序、空bank跨度与反向组装；放置wrapper的重复plan单独处理。
   先解决/限定原数值失败，再测完整训练和推理，不能从少行或少索引字节推断整体收益。
3. **私有KV与Attention。** 评估持续bank/窗口布局，减少历史拼接/整理和state join；
   保留原始int64、窗口淘汰、正写入、零接收导数和跨块梯度。
   带权Attention融合应保留零权重处的导数，静态mask不自动等价。
4. **参数与数学实例选择。** 共享基座按真实owner复用为较大GEMM的执行机会仍可选；
   当前仅记忆共享保留6.002B独立FFN。若改变FFN共享、Full频率或区域通信规则，
   明确作为新实例和参数配比，需下游任务质量验证。
5. **数值与设备范围。** d768局部FP32公式、继续更新、9B共享区域分组的支持差异及BF16失败均保留。
   针对窄见证继续定位；CPU NUMA间计时差异尚未解释，不能把后来的频率/本地页观察当因果。
   其他Timed、多NPU、CUDA、near-FFN训练恢复等只按实际需求单独资格。

持续FC/HB实例路线可长期演进；通用基座另行明确启动。
数据、任务头、训练配方及模型效果属于下游；本轮没有下游修改、发布或验收。
未来运行遵守[执行契约](execution-flows.md)，共享NPU授权不跨线程继承；
退役前核对依赖、用户改动和下游安装，遵守[artifact retention](artifact-retention.md)。
