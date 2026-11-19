from pathlib import Path
import os
import tempfile


def atomic(path, content):
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


page = r'''# Reception FP32 路径、连续更新与恢复证据

三种 incoming self-Attention 在以下 NPU 小图上完成了推理、含梯度前向、重复 VJP、
三个连续 AdamW 更新及各候选自己的新进程恢复更新。9 个候选中 8 个仍为
`finite_deviation`，仅 wide baseline 为 `within_threshold`。原阈值和失败均保留。
九次恢复更新全部在原阈值内；这与跨候选训练超差是不同结论。

本页只整理已经完成的记录，没有新增设备实验。[精简 JSON](reception-numerics-20261118.json)
保留源码、原文件 SHA256、命令、资源、逐项检查、超差分类和恢复事实；完整坐标与字段
列表仍在原始结果。机制及公式见 [Reception 执行说明](../reception-execution.md)，
数值判断遵循[精度契约](../precision.md)。本页不授予 BF16、Timed NPU、长序列、
大 batch、大模型或 CUDA 新资格，也不是任务效果或性能报告。

## 固定身份和负载

数学固定 `4f5b9f20a4030d68ee66ae2d26e34659a5e37234`，108 文件校验通过。
三份 NPU 记录的源码标签均为 `28941994880a277495b315f0fec3213239ef0b49`，
原 `source_dirty=true` 保留。实测 Tide Python 包 SHA256 为
`9cd45f321a7d700021d1fb8a0d31a5c55ebde47c8fa248d7866c6a3c3682cf62`。
本次逐字节复核：21 个已记录 Tide Python 文件的冻结副本、记录哈希及该提交 blob 一致；
主检查器 `scripts/check_training_paths.py` 和所选配置也与该提交一致。
这一核验不覆盖所有辅助脚本或外部工具链，不把 dirty 标签改成 clean。

三组均为 Multiscale `compact-9` Settle，n8+h4、d128、X、quad、region_shared，
FP32、B1/L4、chunk2。各 profile 内先由 baseline 做一次真实 AdamW 更新，生成非空
私有状态及优化器槽位，再从同一固定起点比较。分块间不断开梯度，完整更新后 detach。
随后三个连续更新把 step 1 推进到 step 4，实际新进程恢复更新再推进到 step 5。
364 是参与更新的 **Parameter 张量数**，不是模型的标量参数量。

| Profile | 物理 NPU | 进程内设备 | CPU affinity | Torch threads |
|---|---:|---|---|---:|
| light64 | 1 | npu:0 | 112–115 | 4 |
| light128 | 7 | npu:0 | 116–119 | 4 |
| wide | 5 | npu:0 | 120–123 | 4 |

资源身份为 `exclusive_cooperative_queue; external jobs not controlled`：协作队列独占
准入，共享主机，不控制队列之外的使用者。Torch 字符串为 `2.12.0+cpu`，配合单独的 NPU
扩展；实际运行设备是上述 NPU，不能据版本后缀解读为 CPU 测量。

| 候选 | 相对 baseline 的配置 |
|---|---|
| baseline | region_batch_size=1，separate QKV，padding mask=false；**Reception 重算及 SSD summary prefix 已启用**，SSD scan=block |
| qkv_mask | fused QKV + 结构 padding mask；仍为单区域 |
| optimized | qkv_mask + region_batch_size=4 + SSD scan=doubling |

这是一组组合比较，不能从 optimized 单独推断合批或 SSD doubling 的贡献。
`28941994` 已修复在 inference_mode 下生成静态 mask 后切回训练的缓存错误：缓存 mask
在 `torch.inference_mode(False)` 中构建。该错误曾导致反向异常，与下列有限数值超差
分别处理；本页正式 NPU 记录采用修复后的实现。

## 原阈值和总体结果

主比较及恢复阈值均为 `atol=1e-6, rtol=2e-3`。`scaled` 是逐元素
`abs(actual-reference)/(atol+rtol*abs(reference))` 的最大值；大于 1 即超过原阈值。
表中计数是**超差张量字段出现次数**，分段及最终 state 可能重复包含相关量，
不能解释为错误标量、节点或独立故障数。

| Profile | baseline | qkv_mask | optimized | 三候选各自恢复更新 |
|---|---|---|---|---|
| light64 | finite_deviation | finite_deviation | finite_deviation | 全部 within_threshold |
| light128 | finite_deviation | finite_deviation | finite_deviation | 全部 within_threshold |
| wide | within_threshold | finite_deviation | finite_deviation | 全部 within_threshold |

共 99 项比较，39 项 exact，41 项存在原阈值超差，合计 1459 次字段超差。
所有比较都没有记录到 support、整数时间/索引、dtype、非有限值或 None 连通性差异；
离散/支持哈希一致，已记录的参数 VJP 均连通。另行核对实际存放的工作审计，
各模式及各次更新的 `candidates/full/positive_messages/candidate_slots` 均与 baseline
一致，不能只根据比较器中空的 `work_differences` 作出这个结论。
这些记录未观察到模型活动支持或逻辑时间分歧；执行实现本身则按配置发生了变化。

外层三个作业均为 `passed` / exit 0，表示有限检查任务已完成；各自内部总体仍是
`finite_deviation`。作业完成与数值资格不可互换。

## 推理、含梯度前向和重复反向

下表在同一候选内比较无梯度推理与含梯度前向；各格为超差字段数 / 最大 scaled。

| Profile | baseline | qkv_mask | optimized |
|---|---:|---:|---:|
| light64 | 4 / 1.648 | 3 / 3.106 | 2 / 2.413 |
| light128 | 4 / 1.136 | 4 / 1.172 | 2 / 3.084 |
| wide | 0 / 0.731 | 1 / 1.007 | 2 / 1.181 |

这里超差都在 KV keys/values 等浮点状态，没有支持或时间差异。三个 qkv_mask 的无梯度
推理与 baseline **exact**；第一次完整 VJP 对 baseline 分别有 3、4、1 个超差字段，
仍全部为前向状态，初次输入/参数梯度在原阈值内。optimized 的无梯度推理已分别有
1、4、2 个超差字段。

九个同一 owner 的重复 VJP 均非 exact、均在原阈值内。baseline/qkv_mask 的首个返回
梯度差异在 `receive.embedding.weight`，optimized 在 `describe.encode.bias`。
因此，有限反向差异可以在没有 checkpoint 加载时出现；仅凭字段名不能确定具体
NPU 归约算子或认定其是后续全部超差的原因。

## 三次连续更新

下表相对每个 profile 的 baseline，从固定 anchor 之后的第 1 到第 3 次更新比较。
“最大 scaled 张量的 max_abs”只属于该张量，不是整次比较的全局最大绝对误差。

| Profile / 候选 | 超差字段数：1→2→3 | 第3次最大 scaled 张量 | scaled | 该张量 max_abs |
|---|---|---|---:|---:|
| light64 / qkv_mask | 18→49→209 | receive.q.left | 76.423 | 0.000116318 |
| light64 / optimized | 22→53→161 | receive.k.left | 125.269 | 0.000229608 |
| light128 / qkv_mask | 18→81→141 | receive.k.left | 72.115 | 0.000073930 |
| light128 / optimized | 14→136→179 | receive.q.weight | 83.346 | 0.000163309 |
| wide / qkv_mask | 13→62→81 | receive.q.left | 137.664 | 0.000184201 |
| wide / optimized | 14→58→87 | receive.q.left | 149.808 | 0.000256129 |

这些是更新后的参数字段；最大的 scaled 行主要在接收 Q/K 的私有低秩 left 或基权重。
部分第 2/3 次更新还出现输入梯度、参数梯度及更广泛浮点状态的超差。
字段分类与完整列表的哈希保存在精简 JSON，完整列表及坐标留在原始结果。
不能将这些原阈值失败改称已证明等价，也不能把未见支持分歧改写为已发生不同路由。

九次各候选自己的新进程恢复均满足：保存数据 exact、下一前向 exact；实际更新非 exact
但在原阈值内；配置/源码核对通过，364 个活动 Parameter 的 AdamW step 确实从 4 到 5。
这说明已测候选在所保存状态上的恢复行为，不能消除上表跨候选训练的偏差。

## 有界调查和未建立的归因

三个 qkv_mask 内部见证均达到 `262,144 captured elements per path`，返回
`capture_limit`、`attribution: not established`。检查器只尝试首个跨候选内部见证，
optimized 没有第二份捕获。因此，**尚未建立 NPU 内部最早的因果分歧**。

初始跨候选的首个返回浮点差异为
`root.forward.segments.0.state.memories.10.K.features`；inference↔grad 去掉 `forward`
前缀。第 2/3 次更新首个返回字段移到 memory 0。这是返回字典树按字符串排序后的首个
不等字段，memory 10 排在 memory 2 之前，不能据此定位最早出现误差的区域或算子。

FP32 auto Attention 允许与梯度需求有关的先验表示，是可继续检查的机制；本记录没有
每次真实调用的输入和 dispatch 见证。当前不能将 NPU 差异确定归因于该机制、QKV
融合、静态 mask，或组合中的某个组件，也没有证明全部后续状态/梯度的传播链。
有限调查完成后原样保留超差与边界，不以扩大容差换取通过。

已有 CPU 证据各自只支持其局部结论：

- `6907f3d5` 的 40 项 CPU FP64 Reception 检查通过，覆盖四种接收、独立字面公式、
  输入/参数 VJP、重复/共享 owner、结构删除与真实零质量右导数。它们验证选定小公式，
  不直接验证上述集成 NPU 轨迹。
- `9330df17` 的 CPU FP32 小图，`regional.4.second.bias[20]` 在第 2 次更新的两份
  梯度为 `1.9530908002e-9` 与 `-2.5275292970e-9`；实际参数差 `2.0191073418e-5`，
  独立 Decimal AdamW 预测 `2.0195122629e-5`，差 `4.0492110082e-9`。第 3 次预测差
  `4.0973992671e-9`。这解释该 CPU 坐标的小梯度经 AdamW 归一化/epsilon 放大，
  不解释本页 NPU 接收 Q/K 最大超差坐标；其原超差仍保留。
- 冻结 `b4cf1514` 的首轮 CPU 全套为 715 passed / 11 failed。十个诊断别名泄漏
  已分别修复；另一个 light128 Timed FP32 恢复 exact 失败可由前序 placement 检查
  留下 4 个 Torch threads 复现。两次 4-thread 重放保存值 exact，恢复更新最大绝对差
  `7.4505805969e-9` / `1.8626451492e-9`，在原阈值内；同一 owner 重复反向也非 exact。
  1-thread 重放则均 exact。`1553b301` 只恢复每项测试前后的线程数，保留原 exact 断言，
  14 项定向检查通过；生产线程行为不变。这不是 NPU embedding 梯度重复差异的归因。

## 原始记录与复现入口

仓库根目录下三个作业各保留 `launch.json`、`environment.json`、`queue.json`、
`status.json`、冻结 `source/` 和 `paths/result.json`：

| Profile | 作业目录 | result.json SHA256 |
|---|---|---|
| light64 | artifacts/agg-paths-light64-fp32-v1 | dc8da065ec0e2d1ff2a990ee2e75f3ca29809367d3889ee62125091765703b92 |
| light128 | artifacts/agg-paths-light128-fp32-v1 | 68b7e08e3fdb902d35a97ea17e274fc6dcf5a43efee79a0376482fbdffa44e8f |
| wide | artifacts/agg-paths-wide-fp32-v1 | 4abbaa8eeeca98bfedfbe147ed3ef0956cb6f1049d0d932cd2e55a3d6866afdd |

CPU 原始文件位于仓库根的
`artifacts/worktrees/agg-inference-reception/artifacts/reception-fp32-adamw-witness.json`
及 `artifacts/worktrees/agg-reception-restore/artifacts/restore-threads-b4/report.json`。
相关 XML、源码和检查器哈希均在本页 JSON。工作树/设备映射是本机记录，迁移时应保留
原文件与冻结身份，重新按[执行契约](../execution-flows.md)准入资源。

精简 JSON 的每个 `runs[].command` 保留实际命令。light64 的等价检查入口如下，
需使用上述冻结源码/私有环境及已准入的 NPU，写入新的输出目录；另两个 profile
只替换配置文件名，不能覆盖原结果：

```sh
python -m scripts.check_training_paths \
  --config configs/agg-inference/settle-incoming_light_64.json \
  --device npu:0 --precision fp32 --batch 1 --length 4 --chunk 2 --steps 3 \
  --threads 4 --max-parameters 80000000 --restore-mode new_process \
  --witness-elements 262144 --candidates baseline qkv_mask optimized \
  --candidate-overrides '{"qkv_mask":{"reception_qkv_backend":"fused","reception_mask_padding":true},"optimized":{"region_batch_size":4,"reception_qkv_backend":"fused","reception_mask_padding":true,"ssd_scan_backend":"doubling"}}' \
  --output-dir NEW-reception-paths
```
'''
atomic(Path('docs/evidence/reception-numerics-20261118.md'), page)

p = Path('docs/reception-execution.md')
old = '''The CPU tests include independent literal formulas and every input/parameter
VJP, genuine zero-prior directional derivatives, FP64 multi-update equivalence,
FP32/BF16 real restore/update checks, and inventory/placed-fallback checks.
These scopes are distinct. Early FP32 cross-candidate multi-update checks had
finite optimizer-coordinate deviations without support/time differences; they
are not certified as numerically equivalent by the FP64 or self-restore tests.
Formal device/path qualification and performance recommendations belong to the
current source-scoped evidence. Long experiments follow [execution](execution-flows.md).
'''
new = '''The CPU tests include independent literal formulas and every input/parameter
VJP, genuine zero-prior directional derivatives, FP64 multi-update equivalence,
FP32/BF16 real restore/update checks, and inventory/placed-fallback checks.
These scopes are distinct. The [FP32 Reception path evidence](evidence/reception-numerics-20261118.md)
covers all three incoming profiles on a compact-9 Settle NPU fixture: inference,
grad-enabled forward, repeated VJPs, three continuing updates and nine actual
new-process restored updates. Eight of nine candidates retain `finite_deviation`
under the original thresholds; no recorded support, integer-time or gradient
connectivity difference was observed. All nine candidate-local restored updates
are within threshold, which does not erase cross-candidate training deviations.
The internal NPU witness hit its capture bound, so an earliest causal divergence
was not established. CPU formula and bounded AdamW/thread witnesses support their
own local conclusions only. Performance recommendations retain their separate
source and workload scope. Long experiments follow [execution](execution-flows.md).
'''
content = p.read_text()
assert content.count(old) == 1
atomic(p, content.replace(old, new))

p = Path('docs/instances.md')
old = '''The [Reception execution guide](reception-execution.md) gives the formula,
mask/gradient domain, full-Reception timing and integrated path-check commands.
'''
new = '''The [Reception execution guide](reception-execution.md) gives the formula,
mask/gradient domain, full-Reception timing and integrated path-check commands.
The [FP32 Reception evidence](evidence/reception-numerics-20261118.md) records
three incoming profiles on a small NPU Settle fixture: eight of nine candidates
retain finite cross-mode/training deviations; all nine candidate-local fresh-process
restored updates are within the original threshold. The first internal NPU causal
divergence remains unresolved, and these results do not qualify every scale or mode.
'''
content = p.read_text()
assert content.count(old) == 1
atomic(p, content.replace(old, new))
