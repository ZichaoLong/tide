# FFN 执行候选资格审查

原中尺度 NPU 资格任务以 `execution_failed`、退出码 2 结束：
baseline 的推理与含梯度前向有 14 个字段超出原阈值，
随后在保存恢复参考包前触发 20 GiB 制品预算。
同一 owner 的重复 VJP 在原阈值内；W 尚未执行；原任务的新进程恢复尚未执行。
这些是不同结论，不能用外层执行状态或保留文件的角色标签相互替代。

本审查已完成原中尺度 NPU 任务、完整 CPU 套件、中尺度 CPU 资格及独立
NPU checkpoint 恢复的核验，各自固定身份和结果分节记录。
独立 NPU 恢复的实际 step4→5 更新在原阈值内，但不逐位相同；
它使用新声明的梯度/RNG参考，不改变原任务的数值或执行失败。
普通计时结果不补足资格。

完整结构、指纹和终态复核见 [JSON](medium-qualification-review.json)。
本审查只读现有证据，没有构造模型、运行前向/反向或启动设备任务。

## 固定身份与范围

| 项目 | 完整 CPU 套件 | 原中尺度 NPU 资格 |
|---|---|---|
| 源提交 | `05e8338bf62569100d194848689d545f1140afbc` | `a58536608f7276a771c56703b48e23ea82d2ca65` |
| `src/tide` 树 | `89f2707e71d989340c12de310797a4c7f4c99f64` | 相同 |
| 冻结源检查 | 484 个归档常规文件独立匹配 | 终态审计核对 485 个跟踪文件的 Git blob 与模式 |
| 执行结果 | 979 passed、1 xfailed、1 warning | `execution_failed`，baseline 部分资格证据保留 |
| 硬件/资源 | CPU、4 线程、affinity160–163 | physical NPU9 → logical npu:0，FP32、4 线程、worker affinity224–227 |

NPU 实际构造为 129,025,464 参数，HB Multiscale Settle 5-dd4，
`d=192`、实际 `kv_width=32`、独立 FFN，
B16/L128/chunk32、seed1701、3 次继续更新。
baseline 使用 rectangular FFN packing / gather weight access；
计划中的 W 只将 weight access 改为 grouped，但未执行。
本任务没有 NPU 共享 rank8 FFN adapter 资格。

## 完整 CPU 套件

980 个 JUnit 条目中有 979 个通过、0 failure、0 error、1 个预期失败。
预期失败为
`test_retained_fp32_literal_formula_failure_is_bounded_and_visible`：
保留共享 FFN 的 FP32 独立公式梯度超差，阈值未变。
另有一个 profiler 在周期末清空事件的警告。
不能将这份记录写成“980 项数值通过”。

用时 588.655 秒 / 1800 秒，进程树 RSS 峰值 1.717 GiB / 32 GiB。
当前 v3 [终态审计](../ffn-cpu-suite-v1/terminal-audit.json)
确认全部 65 个记录的 PID/start 身份终态、supervisor 不在、
实际 legacy systemd cgroup 不存在。
服务被回收后的 `Result=success` 不能单独证明实验结果。
本审查采用修正到实际 `/sys/fs/cgroup/systemd` 挂载的 v3 终态记录。

## 原中尺度 NPU 数值证据

原阈值保持 `atol=1e-6, rtol=2e-3`。

| 已完成比较 | 结果 |
|---|---|
| 固定参数/AdamW/私有状态起点 | 精确 |
| 既有梯度与 RNG 起点 | 精确 |
| 推理 vs 含梯度前向 | 14 个字段超差，有限值差异 |
| 同 owner 重复 VJP | 无阈值超差，非逐位相同 |
| VJP 后参数/AdamW/私有状态不变 | 精确 |
| 原始整数时间、支持、元数据、dtype、work | 已完成比较均一致 |
| 非有限值与 None 参数梯度 | 未观察到；两份 VJP 均有输入梯度 |

最大 scaled error 为 3.718924275，位于
`root.segments.2.values`，该字段 max abs 为 1.0848045349e-5。
14 个超差字段包括 segments1/2/3 的输出，以及 KV 41–43 的部分
features/keys/values；完整字段列表和最早观测差异在 JSON 中。
该树顺序下的最早观测字段不是内部因果定位。
没有改变阈值，也没有将差异自动归为舍入。

重复 VJP 的最大差异在 `receive.embedding.weight` 梯度：
max abs 5.8207660913e-11、scaled error 3.0122137571e-5，
处于原阈值内。原制品清单将 `baseline/vjp2.pt` 的角色写成
“retained failed repeated VJP witness”，这是旧脚本按累计检查决定保留，
其中已包含跨模式超差，不能据此断言重复 VJP 失败。
原清单字节保留，解释在本审查修正。

baseline 的三次实际 AdamW 更新完成：step1→2→3→4，
每次记录 425 个活动原参数张量，结果有限。
这是 baseline 单轨继续训练记录；没有完成 baseline/W 的训练比较。

## 原恢复中断与保留证据

CPU mmap 只读检查确认现有公开 checkpoint 为 step4、
`state.position=512`，配置与私有状态一并保存。
固定源码表明 parent 在内存里计算了下一次推理和 step5 更新后，
在 `Artifacts.reserve` 为 `restore-input.pt` 预留空间时拒绝继续。
因此没有写出原进程 step5 的期望结果/梯度-RNG sidecar，
也没有 cache-release 边界、fresh-child 恢复、恢复日志或恢复结果。

保存 checkpoint 与实际恢复后更新是不同事实。
独立恢复入口固定为 `46317a8dd73d0611ff0f7b18f4057e97c5baa630`，
其参考 owner 使用新声明的空梯度与 RNG；
结果必须与原中断任务分开记录，不能恢复未保存的原进程梯度/RNG，
也不能消除这份跨模式超差或资格未完成事实。

8 个保留 .pt 文件均已独立重算大小与 SHA256，全部匹配原清单。
payload 合计 17,236,482,055 字节；
原 result 的 retained bytes 为 17,237,589,198，
当前 payload 加最终 result 为 17,237,589,547 字节。
349 字节差来自 retained bytes 在替换最终 JSON 前计数。
两者均低于 20 GiB；拒绝的是下一份预计载荷。
原 result SHA256：
`19b6f0231eacf5b723cb0d9c72471f03e8f6c295ae54f8d3b617b0ede1ffeb99`。

## 资源与终态

原任务总用时 492.649 秒 / 7200 秒，
包括依赖等待 275.762 秒；进程树 RSS 峰值 11.709 GiB / 192 GiB。
RSS 是采样约束，与设备 HBM 不同。
资格边界尚未到达 parent/child allocator 记录点，不能由参数量或普通性能测量
推断这次资格流程的完整 HBM 峰值。

[原任务终态审计](../ffn-medium-npu-qualification-v1/terminal-audit.json)
与本次只读复核均确认 6 个保存的 PID/start 身份结束、supervisor 不在、
正确的 systemd cgroup 路径不存在。终态审计通过只表示进程生命周期结束。
原记录、日志、checkpoint、输入与冻结源码未改写。

复现参数在
[原 launch](../ffn-medium-npu-qualification-v1/launch.json) 与
[原资格结果](../ffn-medium-npu-qualification-v1/qualification/result.json)。
本审查不建议重复已中止的大矩阵，也不提供普通训练吞吐或常规训练峰值显存结论。

## 中尺度 CPU 资格补充

`ffn-medium-cpu-qualification-v1` 使用相同冻结提交
`a58536608f7276a771c56703b48e23ea82d2ca65`，
同一 129,025,464 参数、B16/L128/chunk32/seed1701 配置，
CPU FP32、32 线程、affinity240–271，原阈值不变。
485 个冻结跟踪文件的 Git blob 与模式已独立核对，无差异。
原结果为 `execution_failed`：P 首个更新差异载荷触发 32 GiB 预留上限。

| 检查 | baseline | W | P |
|---|---:|---:|---:|
| 各自 inference ↔ grad-forward 超差字段 | 124 | 124 | 116 |
| 同 owner 重复 VJP | 精确 | 精确 | 精确 |
| 初始 inference vs baseline | 自身参考 | 精确 | 55 个超差字段 |
| 初始 VJP vs baseline | 自身参考 | 阈值内、非精确 | 62 个超差字段 |
| 已记录继续更新 | step1→2→3→4 | step1→2→3→4 | 无完整持久记录 |
| 新进程实际恢复更新 | step4→5 精确 | step4→5 精确 | 未运行 |

baseline 与 W 的跨模式比较最大 scaled error 均为 18.90320676，
来自 KV36 features；这不是 W 特有的改变。
W 对 baseline 的三次更新比较分别为 0、392、533 个超差字段，
后两次最大 scaled error 为 57.62610445、109.43433932。
字段计数包含分块状态和末状态的重复表示，不能当作互相独立的错误事件数。
已保存比较的原始 int64/discrete、支持、元数据、dtype、work 一致，未观察到非有限值。
这些一致性不消除有限值/VJP/更新超差。

只读 mmap 核对五个已报告的参数/梯度字段，未重新散列整份大 payload：

| 更新索引 | 字段 | 超差元素 | 最大 scaled error |
|---|---|---:|---:|
| 1 | `gradients.describe.readout.value` | 1 | 1.96399226 |
| 2 | `gradients.describe.encode.weight` | 2 | 1.25683511 |
| 2 | `gradients.describe.readout.value` | 1 | 1.93056793 |
| 2 | `parameters.regional.26.second.bias` | 1 | 1.04161457 |
| 2 | `parameters.regional.4.second.bias` | 1 | 1.05822925 |

JSON 保留这些坐标、期望值/实际值、原阈值与来源。
这是有界差异复核，没有作全部传播路径的因果归因。

baseline/W 的恢复由两个不同 PID（1823048、1846496）执行，
与原 owner1800430 分离；saved data/source/config、
梯度-RNG锚点、next forward、实际 step4→5 更新均精确。
每次实际更新记录 425 个活动原参数张量。
这是各自候选局部恢复通过，不能替代 W/baseline 的训练路径等价性。

P 的初始推理/VJP有持久结果。首个更新在内存计算和比较后，
保存差异载荷的 reserve 拒绝发生于 `updates.append` 之前；
因此原结果保留 `state=running, updates=[]`，
但任务已结束。不能写成“没有发生任何更新计算”，也不能从缺失载荷推定
该更新的完整数值或 step 资格。P 尚未运行恢复。

任务用时 880.560 秒 / 7200 秒；
采样进程树 RSS 峰值 54.953 GiB / 192 GiB；
原报告 retained artifacts 为 30.484 GiB / 32 GiB。
19 条制品记录的保留/退休状态与大小一致，2 份成功恢复参考包已按原记录退休。
本次未独立重算这些 CPU 大载荷的 SHA256。
3 个保存的 PID/start 身份和 supervisor 均结束，
实际 systemd cgroup 已不存在；normal timing 不补数值资格。
原结果与恢复证据分别见
[CPU result](../ffn-medium-cpu-qualification-v1/qualification/result.json)、
[baseline restore](../ffn-medium-cpu-qualification-v1/qualification/baseline/restore.json)、
[W restore](../ffn-medium-cpu-qualification-v1/qualification/weight_access/restore.json)。

## 独立 NPU checkpoint 恢复补充

`ffn-medium-npu-recovery-v1` 使用已提交冻结工具
`46317a8dd73d0611ff0f7b18f4057e97c5baa630`；
487 个跟踪文件的 Git blob 与模式已独立核对，全部匹配。
生产 `src/tide` 内容 SHA256 仍为
`492122d8430115f902c3751ff2f88f5fe23e39ca9c7479bf56d2570e1ff54f14`，
与原 NPU 任务相同。physical NPU9 → logical npu:0、FP32、
4 线程、B16/L128/chunk32；模型仍为 129,025,464 参数。
工具的轻量 metadata/mock 检查与这次正式恢复证据是不同层级的验证。

本任务直接读取原公开 checkpoint，校验其原 SHA256；
没有复制或覆盖 checkpoint。输入来自原 anchor 第 4 项（共 5 项），
形状 [16,128,192]，其哈希与原结果完全一致。
参考 owner 从 step4、私有状态 position512 开始，
显式设置全部梯度为 None、RNG seed2718；deterministic algorithms 为 false。
这是新参考 owner，不能重建原中止进程没有保存的梯度/RNG。

参考进程1905573与新解释器1908840分别实际执行 step4→5 更新，
各记录 425 个活动原参数张量；新解释器的 PID/start 身份也包含在外层监督记录中。

| 新解释器检查 | 结果 |
|---|---|
| saved data、source、config | 精确 |
| 新声明的梯度/RNG锚点 | 精确 |
| next inference forward | 精确 |
| 实际 step4→5 更新完整结果 | 原阈值内，非逐位相同 |
| 原始 int64/discrete、支持、metadata、dtype、work | 一致 |
| 非有限值、阈值超差字段 | 无 |

最大差异在 `gradients.receive.embedding.weight`：
max abs 1.1641532183e-10、scaled error 4.1153556253e-5。
该参数本身、AdamW一阶矩/二阶矩还有更小的有限差异。
阈值仍为 atol1e-6、rtol2e-3；没有自动作舍入因果归因。
这份 `within_threshold` 只覆盖此新参考与其局部恢复，
不修复原 NPU 的 14 项跨模式超差，不补足未运行的 W，
也不消除 CPU 训练比较的有限超差。

### 恢复内存与预算

| 进程 allocator 边界 | allocated GiB | reserved GiB |
|---|---:|---:|
| parent 从 checkpoint load 起累计峰值 | 23.241 | 24.211 |
| parent 同步释放 inactive cache 后仍持有 | 2.131 | 2.791 |
| fresh child 从 checkpoint load 起累计峰值 | 23.241 | 24.211 |
| parent held + child peak 保守上界 | 25.372 | 27.002 |

各自峰值与保守重叠均低于 56 GiB。
本审查独立重算 held+peak，与记录一致：
allocated 27,243,293,696、reserved 28,993,126,400 字节。
parent 等待时不进行设备计算。
这些是资格流程各进程的 allocator 计数及保守重叠，
不是全设备实测峰值、不可见运行时工作区总量或正常训练显存。

新参考包为 4,054,966,522 字节（3.776 GiB），
其 SHA256 与 fresh child 报告一致；
成功结果 fsync 后按原策略退休，审查时确认文件已不存在。
新目录最终为 1,080,080 字节，制品上限 10 GiB。
没有为审查重建或重读该大包，原 checkpoint/anchor 保留在原目录。

外层总用时 881.801 秒 / 3600 秒，含依赖等待 771.367 秒；
采样进程树 RSS 峰值 17.816 GiB / 128 GiB。
独立复核确认全部 6 个记录的 PID/start 身份、supervisor 终态，
实际 systemd cgroup 不存在。服务回收后的 `Result=success` 不是唯一依据。

[新恢复 result](../ffn-medium-npu-recovery-v1/recovery/result.json)
SHA256为
`e1cecb06094503e9d12dcee7560487fd2e2b3c3cdf424a933ffdeec9de1a7c5b`。
[独立 child result](../ffn-medium-npu-recovery-v1/recovery/restore.json)
与监督记录 SHA256 匹配。
原中尺度 NPU result 再次重算仍保持原 SHA256，失败记录未改写。
复现入口为冻结源码中的 `python -m scripts.check_checkpoint_update`，
完整绝对输入、新参考 seed、预算与设备参数见
[新 launch](../ffn-medium-npu-recovery-v1/launch.json)；
本轮不再追加模型运行或大载荷读取。
