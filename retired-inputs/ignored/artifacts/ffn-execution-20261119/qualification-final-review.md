# 最终报告的数值资格与恢复审阅

已核对草稿 `report-draft.md`（SHA256 `9e182cd956974f7cc5738128e523366b461a0d457ecfd7ad818b3d838769862d`）
与既有 JSON/Markdown。没有启动模型、重跑测试或散列大载荷。
总体结论成立；最终 docs 应保留以下范围，并修正计数和恢复措辞。

1. **小图 CPU 的 P/WP 必须分开。** 冻结 `05e8338b`、1,232,383 参数、
   d64/KV32/FFN256、独立 FFN、B2/L8/chunk4、seed1701、4 线程的这一个样本中，
   baseline/W/P 在原阈值内。WP 的初始前向精确，初始输入/部分参数 VJP 已有阈内差异；
   后两次更新只有同一字段 `root.parameters.regional.10.second.bias` 超差。
   草稿“2个字段/3坐标”改成：
   **“同一参数字段在两次更新比较中超差，共 3 次超差坐标出现（独立坐标为 5、6）”**。
   update index1 为坐标6，index2为坐标5、6。四者各自 fresh-process step4→5 恢复精确，
   不能消除 WP 跨路径失败，也不能将 P 的这个小图结论推广到中尺度、共享 FFN 或 NPU。

2. **小图 NPU 的 11 是字段比较次数。** baseline/W 在原阈值内；
   P/WP 各有 11 次字段比较超差（更新 index1 为2次、index2为9次，含重复状态表示），
   不是 11 个不同字段或 11 次更新。四者 inference↔grad-forward 均精确，
   失败来自同模式继续更新比较。四者各自 step4→5 恢复均在阈值内，**只有 W 的实际更新精确**；
   其余三者非逐位相同。该集成样本使用独立 FFN，不覆盖共享 rank8 FFN adapter 的 NPU 集成资格。

3. **保留“979 passed + 1 strict xfailed + 1 warning”。** 不能称980项数值通过。
   strict xfail 是明确保留的 FP32 共享 FFN literal-formula 梯度超差，
   局部 fixture 为 d64/F128/rank4；仅接受已记录的 `up.weight[0,20,27]` 有界差异，
   意外字段、坐标、超出 2e-6 记录上界的差异或 XPASS 会失败。它不代表该公式比较已获资格，
   也不能与集成小图的独立 FFN 或 d768/rank8 诊断混为同一对象。

4. **d768 四项原输出资格仍失败。** independent/region_shared × gather/grouped 的原
   CPU FP32 literal 输出超差精确复现，值阈值仍为 atol=rtol=2e-6。
   原资格的输入及参数 VJP 在 atol=rtol=6e-6 内；**补充诊断只重算输入 VJP，没有重跑参数 VJP**。
   同 NPU 两路径以及相同原数据的 FP64 公式参照下，捕获阶段和输入 VJP 阈内。
   草稿“只定位到…”宜写为“首次观测差异/超差出现在投影、product 或 down 边界，尚未确定根因”；
   不能写成“CPU参照不准，所以原 NPU 资格通过”，也不能自动归因于舍入、具体指令或累加次序。
   范围仅两个 d768/F3072、4节点/8行、seed469/470 的局部样本，没有完整模型、状态或恢复资格。

5. **中尺度原任务的失败保持有效。** 原 NPU baseline 跨模式14字段超差，
   同 owner 重复 VJP 阈内；step1→2→3→4完成，恢复包预留触及20 GiB，W未开始。
   CPU baseline/W 跨模式各124字段超差，W后两次更新各392/533次字段比较失败；
   两者各自 step4→5 恢复精确。CPU P 的55/62是初始推理/VJP对baseline的超差字段数；
   首更新已在内存计算，差异包预留触及32 GiB而未追加完整持久更新记录，恢复未运行。
   草稿目前这些表述准确，应保留。不能把 P 写成“完全没执行更新”，或把缺失结果补写为通过。

6. **独立 NPU 恢复补入以下事实，不能称旧进程无缝续接：**
   冻结工具 `46317a8d` 直接读取原 step4/position512 checkpoint，使用新参考 owner、
   **全部梯度 None、seed2718**；没有恢复原中止进程未保存的梯度/RNG。
   参考进程1905573与新解释器1908840各实际执行 step4→5，425个活动原参数张量。
   saved data/source/config、新梯度-RNG锚点及 next forward 精确；
   实际更新在原阈值内，**非逐位相同**，最大差异在 embedding 梯度：
   max abs1.1641532183e-10、scaled4.1153556253e-5。
   该新参考恢复通过不改变原14项跨模式超差，不资格化中尺度 W。
   parent/child 各自 allocated/reserved 峰值23.241/24.211 GiB；
   parent held + child peak 保守上界25.372/27.002 GiB，低于56 GiB。
   这是进程 allocator 口径，不是全设备硬峰值或正常训练显存。六个记录 PID/start 身份及
   supervisor 终态、实际 systemd cgroup 不存在均已核验。

7. **整体结论保持范围。** 开头“保持模型函数”可改为“保持声明的数学公式和参数归属”，
   避免被理解为已证明所有浮点输出/训练轨迹等价。原阈值不变；
   已保存比较的支持、原始 int64、元数据、dtype、有限性通过，不消除值/VJP/更新超差。
   同候选恢复、正常吞吐和省显存都不能替代跨路径数值资格。
   保留 gather/rectangular 默认、W本轮不扩至9B、不引入模型质量结论的建议与证据一致。

依据：[CPU小图原结果](../ffn-small-cpu-v1/qualification/result.json)、
[CPU WP审阅](cpu-combined-review.json)、[小图NPU审阅](small-npu-review.json)、
[d768审阅](d768-diagnosis-review.md)、
[最终中尺度资格/恢复审阅](medium-qualification-review.md)。
最后一项 JSON SHA256 为
`187ec907d2195e6447260ab0b81e8fb36b1cc1194ebf709a54845262d721bdd3`。
原始证据和此前审阅均未改写；本文件供主代理编写最终 docs。
