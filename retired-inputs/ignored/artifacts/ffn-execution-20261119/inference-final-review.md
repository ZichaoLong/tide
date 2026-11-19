# 推理报告最终审阅

结论：新增 prefill/decode 两张互斥阶段表、19.49% 份额和 FFN 乘加对/计时数值均正确。
合入报告前应明确区域分组的表项归属、B8 单次 decode 的单位，并将“矩阵本体用时”改为同步算子诊断计时。
此审阅只读报告和已有 JSON/Markdown；未运行模型、设备、测试或新实验，未修改草稿和原始证据。
训练、恢复和长 decode 其他段落不在本次审阅范围。

## 审阅身份

- 报告审阅快照：`report-draft.md`，21,857 bytes，SHA256 `7769596308b01844a30f2f7a19c1c6fd2dab54d640f6adef35ba9805c8bebc56`。主代理可继续修改草稿，本审阅以此快照及下列证据为准。
- 最终结构化证据：`inference-stage-review.json`，9,730,328 bytes，SHA256 `50e6ca7cac4d9135535966d9e5059952e76aa05dd984187093bf4a993f5a7e27`。
- 最终可读证据：`inference-stage-review.md`，11,539 bytes，SHA256 `3d090fb25470f334694772797ab9a0416b0a22f2c97ba1d2d28256ce5431bae6`。
- 推理工具提交：`46317a8dd73d0611ff0f7b18f4057e97c5baa630`；production tree：`89f2707e71d989340c12de310797a4c7f4c99f64`；实现提交：`05e8338bf62569100d194848689d545f1140afbc`。
- 最终 collector 为 `completed_review`。四个案例均 `completed_finite`、supervisor passed，issues 为空；每例 487 个冻结源码文件核对，共 1,948 个。
- CPU/NPU Transformer 采用 v2。CPU v1 在构造模型前因无效 `--config` 退出 2；NPU v1 在排队前被终止，退出 −15。两者仍保留在 previous_attempts，不能列作模型性能结果。

## 阶段表数值与边界

逐单元格从最终 JSON 重算分组。prefill 最大单元格四舍五入差为 `4.991822242783606e-7` 秒，
decode 为 `4.4279229882704385e-7` 毫秒。每一原始互斥类别恰好归入一行；未舍入合计与 captured 总时间一致。
显示后逐行相加与显示 footer 最多约差 `1e-6` 个相应单位，仅为末位舍入。

| 案例 | prefill 整段秒数 | decode 每次调用均值，毫秒 |
|---|---:|---:|
| CPU Tide | 20.628669131547213 | 1530.3117870353162 |
| CPU Transformer32 | 10.79387991502881 | 1010.6290490366518 |
| NPU Tide | 3.4342331029474735 | 522.2871399018914 |
| NPU Transformer32 | 0.7095328643918037 | 102.98514040187001 |

prefill 是 4 次 chunk32 调用的总计；decode 是 16 次 B8 调用的平均，包含第一次。
normal stable decode 只含后 15 次。release 根 scope 和调用间输出/状态检查不在表中 captured 合计内。
表中余项包含 hook/dispatch、调用者余项与入口等待，metadata 观察另行归类，不能统称 Python 时间。
CPU 同步 helper 为 no-op；NPU observer 强制完成，可能改变排程。这些表不能除以 normal 时长求比例，
也不能据诊断 CPU/NPU 比值声称受控的硬件加速。

表项有一处命名需补全：现“消息收集与接收”实际是
`messages.gather + regions.group + reception`。
建议两表统一改为“消息收集/区域分组/接收”，或在表后明确此映射；数值无需改动。

其他分组已核对：KV 历史为 field gather + history order/compaction；KV Full 包含 Attention read；
区域评分包含其 Attention read；Transformer 行包含所有 transformer.* 及其 Attention read；
入口/观察器/未细分余项恰为 input.projection、observer.metadata、unattributed_scope_remainder。

## 19.49% 的口径

NPU decode 同一阶段趟中，以下互斥项均值相加：

| 项 | 毫秒/次 B8 decode |
|---|---:|
| messages.gather | 21.277031861245632 |
| regions.group | 1.025684643536806 |
| KV_history_field_gather | 36.475545493885875 |
| KV_history_order_including_compaction | 13.19308509118855 |
| Full_query_packing | 15.152027131989598 |
| Full_query_publication | 3.814864903688431 |
| state.join | 7.333143847063184 |
| state.split | 3.513904521241784 |
| 合计 | 101.78528749383986 |

`101.78528749383986 / 522.2871399018914 = 19.488377123924522%`，四舍五入为 19.49%。
同分母下 FFN `169.69998343847692 ms` 占 `32.49170245132861%`，四舍五入为 32.49%。

草稿已正确限制为命名 scope 在该观察窗口的份额，且指出 FFN 内另含权重索引、接收/评分混合组织和计算。
建议把分母写明为“101.785 / 522.287 ms，占该同步 decode 捕获窗口 19.49%”。
这不是“非结构化导致的性能劣化”，不是 normal 时间份额，也不是可回收正常时间的估计。

## FFN 算术、计时及形状

下表属于独立逐算子趟；“每次”均指一个完整 B8 decode 调用，处理 8 个新输入位置。
均值按 16 次调用计算，绝不是每个样本、每个输入位置或每个 gather 事件的均值。

| 量 | Tide | Transformer32 |
|---|---:|---:|
| 16 次可见 FFN 矩阵乘加对 | 415,840,075,776 | 813,283,934,208 |
| 每次乘加对 | 25,990,004,736 | 50,830,245,888 |
| 每次 G 乘加对 | 25.990004736 | 50.830245888 |
| 16 次 FFN BMM/matmul 计时和，秒 | 0.25682899728417397 | 0.5673517361283302 |
| 每次 FFN BMM/matmul 计时和均值，毫秒 | 16.051812330260873 | 35.45948350802064 |

这是乘加对（M×K×N），不是以乘法、加法分别计数的 FLOPs，也不是整个模型的总算术量。
Tide KV128 为 7.268409B，固定 Transformer32 为 8.778371B；FFN 算术量及矩阵形状均不相同。

Tide 的 624 次 float-weight-bank gather 事件横跨全部 16 次 decode，累计计时 `2.4008842408657074 s`；
每次 decode 内 gather 计时总和的平均为 `150.0552650541067 ms`。
这些 gather 累计产生 `207,920,037,888` 逻辑输出字节，含重复物化事件；它不是实际总线流量、常驻权重容量或 allocator 峰值。
150.055 ms 不是单次 gather 事件的均值。
逐算子趟与阶段趟分别重放，不能将这 150.055 ms 加到或从阶段表相减，也不能以它除以阶段 FFN 或 normal 时长推断可回收份额。

草稿所引操作数形状存在：
`[19,8,768] @ [19,768,3072]` 是 Tide 的一个 BMM 例子；
`[8,1,4352] @ [4352,30416]` 是 Transformer 合并 gate/up 投影，后者每次 decode 32 次。
它们是 dispatcher 可见操作数形状，不证明厂商内部 GEMM 分块或核分解。

建议用以下文字替换相关表述：

> 直接可见的 FFN 矩阵乘加对为 Tide 25.990G、Transformer 50.830G，单位为每次 B8 decode 调用（8 个新输入位置）。该同步算子趟中，可见 FFN BMM/matmul 计时和的每次调用均值分别为 16.052/35.459 ms；Tide 的该项计时较低，但两者算术量和形状不同，这不是矩阵核本体时间或同矩阵效率比较。

> 在本次同步算子诊断中，权重 bank gather 是明确观察到的大项，可见 FFN 矩阵计时没有解释全部差距；正常执行中可回收多少仍需干预实测。

保留草稿“尚未证明在 9B 用 W 可回收这些时间”的限制。

## 必须保留的资格范围

四案例同设备 stages/operators 完整输出 SHA 一致；normal 没有保留输出 payload/SHA，不能因此认定 normal 与诊断输出相等。
Tide 最终 44 个命名 `.times` 私有 int64 张量有值哈希，另 20 个 SSD ConvState `.conv.0` 时钟只有 metadata；
浮点私有历史也没有完整逐值哈希，不能称全部私有状态或时钟值相等。

每个 subject 的 CPU/NPU 初始参数和输入 SHA 一致，但双方各 20 次 readout SHA 都不相同（0/20）。
Tide 输出支持/时间 metadata 相同不等于完整活动、状态或浮点轨迹相同；Transformer 未记录 raw output-support 张量，
不能将缺失项 `None == None` 当作观察到的支持一致。
本推理诊断没有跨设备误差阈值/VJP 资格，也没有 checkpoint 恢复；依赖其他恢复作业不赋予它恢复资格。
