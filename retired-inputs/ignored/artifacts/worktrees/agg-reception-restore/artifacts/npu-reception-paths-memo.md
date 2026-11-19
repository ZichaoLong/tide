# NPU Reception 路径有限核对备忘

仅只读汇总三份正式 result.json；wide 已终态。没有启动模型、NPU、重跑或扩大捕获。完整原阈值失败字段、返回差异坐标和输入文件哈希见同目录 `npu-reception-paths-memo.json`。

## 身份与范围

- 三组均为 compact-9 Settle，n=8、h=4、d=128、X、region_shared、FP32、B1/L4、chunk2；有一次真实 anchor 更新，随后三个连续更新，再新进程真实恢复更新。
- baseline 使用单区域、separate QKV、不启用 padding mask，Reception 重算和 SSD prefix 已启用；qkv_mask 只添加 fused QKV 与 padding mask；optimized 再添加 region_batch_size=4 与 SSD doubling。
- 记录源码标签为 `28941994880a277495b315f0fec3213239ef0b49`，三份 `source_dirty=true` 原样保留。实测 Python 包 SHA256 为 `9cd45f321a7d700021d1fb8a0d31a5c55ebde47c8fa248d7866c6a3c3682cf62`；本备忘不以当前 HEAD 替代实测身份。
- 设备映射分别为 light64→物理1、light128→物理7、wide→物理5，进程内均 npu:0；协作队列独占准入，外部任务不受控、共享主机。
- 原阈值统一 `atol=1e-6, rtol=2e-3`。计数是超差张量字段次数，可能重复包含 segment/state，不能当成超差标量数或独立故障数。

## 数值状态

| Profile | baseline | qkv_mask | optimized | 三候选各自新进程恢复更新 |
|---|---|---|---|---|
| light64 | finite_deviation | finite_deviation | finite_deviation | 全部 within_threshold |
| light128 | finite_deviation | finite_deviation | finite_deviation | 全部 within_threshold |
| wide | within_threshold | finite_deviation | finite_deviation | 全部 within_threshold |

共 99 个比较中 39 个 exact；41 个比较包含原阈值超差，合计 1459 次字段超差。9 个候选均无 nonfinite、dtype、support、整数元数据或 None 连接差异；全部显式参数在 VJP 中有梯度。按模式及各次更新单独核对的 candidates/full/positive_messages/candidate_slots 工作计数也一致。
九次恢复均：保存数据和下一前向 exact；实际更新在原阈值内但非 exact；364 个活动参数的 AdamW 计数真实从 4 到 5。恢复结果只说明各自候选的恢复，不消除候选间训练超差。

## 推理与含图前向

| Profile | baseline 超差字段数 / 最大 scaled | qkv_mask | optimized |
|---|---:|---:|---:|
| light64 | 4 / 1.648 | 3 / 3.106 | 2 / 2.413 |
| light128 | 4 / 1.136 | 4 / 1.172 | 2 / 3.084 |
| wide | 0 / 0.731 | 1 / 1.007 | 2 / 1.181 |

这里的失败字段均是 KV keys/values 等浮点状态；未出现活动支持或原始整数时间变化。三组 qkv_mask 的无梯度推理与 baseline exact，但首次带梯度完整 VJP 分别有 3、4、1 个字段超差，全部仍是前向状态字段；初次参数/input 梯度均未超过原阈值。optimized 无梯度推理已有 1、4、2 个字段超差。
同一 owner 的重复 VJP 九组均非 exact、均在原阈值内。baseline/qkv_mask 首个返回差异是 receive.embedding.weight 梯度，optimized 是 describe.encode.bias 梯度；该事实区分重复反向的有限变化与 checkpoint 故障，但不能据此确定具体 NPU 归约算子。

## 跨候选三个更新

| Profile / candidate | 超差字段数：更新1→2→3 | 第3步最大 scaled 所属字段 | 该张量 max_abs |
|---|---|---|---:|
| light64 / qkv_mask | 18 → 49 → 209 | root.parameters.receive.q.left | 0.000116318 |
| light64 / optimized | 22 → 53 → 161 | root.parameters.receive.k.left | 0.000229608 |
| light128 / qkv_mask | 18 → 81 → 141 | root.parameters.receive.k.left | 7.39298e-05 |
| light128 / optimized | 14 → 136 → 179 | root.parameters.receive.q.weight | 0.000163309 |
| wide / qkv_mask | 13 → 62 → 81 | root.parameters.receive.q.left | 0.000184201 |
| wide / optimized | 14 → 58 → 87 | root.parameters.receive.q.left | 0.000256129 |

最大 scaled 行主要落在 receive.q/k 的 private left 或 base weight。部分第2/3次更新还出现参数梯度、输入梯度及更广泛的浮点状态超差；完整字段分类与列表保留在 JSON。支持、整数时间和计数仍一致，不能把这些有限数值差异说成已经观察到不同路由。

## 最早见证与解释边界

三个 qkv_mask 内部见证都返回 `capture_limit: 262,144 captured elements per path`；未建立内部因果起点。optimized 没有第二份内部捕获，是检查器只尝试第一个候选的有限策略，并非内部没有差异。
初始跨候选和 inference↔grad 比较的首个返回浮点差异都是 `segments.0.state.memories.10.K.features`（跨候选前有 `root.forward`）。更新2/3首个返回字段变为 `segments.0.state.memories.0.K.features`。这些是字典树排序中的首个不等字段；memory 10 排在 memory 2 之前，不能据此定位首次发生计算分歧的 region。
已建立：独立 CPU FP64 Reception 公式/VJP/真实零导数检查覆盖选定代数；独立 Decimal AdamW 见证解释了另一个 CPU 小图的 regional.4.second.bias 坐标更新放大。它们支持局部实现语义，但没有重放本次 NPU q/k 最大差异坐标。
未建立：这些 NPU 初始含图前向差异的具体核/表示来源；QKV 融合与静态 mask 的各自贡献；optimized 中合批与 SSD doubling 的各自贡献；全部后续状态/梯度超差的完整传播链。FP32 auto Attention 的梯度相关先验表示是可讨论机制，本记录缺少对应实际调用输入/dispatch 见证，不能直接归因。
结论边界：九个候选执行、连通性、状态元数据与各自恢复检查完成，八个仍标 finite_deviation。不扩大为所有拓扑、批量、长序列、精度、设备或后续训练轨迹的数值等价。

## 原始记录

- light64: `/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/agg-paths-light64-fp32-v1/paths/result.json`；SHA256 `dc8da065ec0e2d1ff2a990ee2e75f3ca29809367d3889ee62125091765703b92`
- light128: `/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/agg-paths-light128-fp32-v1/paths/result.json`；SHA256 `68b7e08e3fdb902d35a97ea17e274fc6dcf5a43efee79a0376482fbdffa44e8f`
- wide: `/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/agg-paths-wide-fp32-v1/paths/result.json`；SHA256 `4abbaa8eeeca98bfedfbe147ed3ef0956cb6f1049d0d932cd2e55a3d6866afdd`
