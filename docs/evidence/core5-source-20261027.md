# Core-5 来源参数绑定与稳定一阶归约 CPU 资格 — 2026-10-27

固定实现 `503f6092d35dc57f2052c598e15073fde65e834b`，固定上游 `0770593`。
**665 passed / 0 failed / 0 skipped；CPU FP64/FP32、aarch64。**
干净源码、workload exit 0、服务 inactive/success 已核对；[命令/环境/哈希](core5-source-20261027.json)。

[来源绑定契约](../weighted-source-binding.md)给出 region/node/source/adapted 四种 A/B/embedding
别名、私有低秩因子和统一/按类别宽度。共享同一矩阵的组必须同宽；不同组允许异宽。
默认 region 的 graph/profile 和全部初始权重已与冻结 5391017 逐字节核对一致。
Python/C++ 按实际宽度执行，参数别名、物理原子和私有状态互不混同。

独立一维公式及低秩解析导数检查 A/B/e、四因子和输入；双样本 HB/反馈 PDG
以独立调度比较完整记录、独立梯度根、None/连接零、保留图续接。
三步 AdamW 与含 pending checkpoint/换 owner 恢复比较参数及全部优化器槽；
不相容宽度、rank、形状、别名和恢复绑定明确拒绝。
相关 Aggregate、query、循环、评分、区域、State VJP 回归同批通过。
standalone C++ 检查来源低秩、Attention 和 RMS 局部公式/梯度；这不是 standalone 完整图训练资格。

开发 tree bcbaed6b741adca26581b60906f92ffd651b32c6 为 **201/2**：
FP32 node/PDG 的 Python/native 第二次 AdamW 更新中 Full query 参数差 2.7641654e-6，
前向/梯度比较已通过。微小 query 梯度的 Attention 消减误差被 AdamW 放大。
保留原始前向和输入，Attention logit VJP 先收缩 `v_i-v_0` 再减均值，
消除共同值分量；常值/平移不变性、独立解析 Jacobian 和方向差分验证。

第二轮 tree ab6cda80eb576b4e932f7bd2165603e34ca58749 为 **400/2**：
上述训练失败消除，但 Read batch FP32 case 3 的 Python/native 完整 objective
对 cur encoder 的梯度相差 3.2186508178710938e-6。
从 git archive 5391017 提取旧源码，复现完全相同的 Python 失败；关闭区域批量、
改变 Query 投影方向或标量区域头均未消除。共同 RMS 的 VJP 改为归一化坐标收缩
`(bar-n*mean(bar*n))/denominator` 后通过；前向、dtype、输入和既有容差保持。
独立字面 Jacobian、方向差分、原位版本与一阶限制检查支持该数学等价改写。

第三轮开发 tree fc6ca6b13402e3cc8f76372d600ff02d6b95f836 为 **665/0**；
本固定资格复核已提交的实现，排除两个尚未接入的计算共享草稿文件。
原失败的终态、源码和诊断保留于 artifacts/jobs/core5-source{,2}-dev，
artifacts/diagnostics/core5-source-query-rounding 与 core5-query-encoder-rounding；
没有把旧失败重标为通过。

复现：检出固定实现，按 [development](../development.md) 构建，执行 JSON 中的测试列表，
`--dtype both`、新的 basetemp 和当前 TIDE_BUILD_DIR；无需旧运行目录。
Python 3.12.13、Torch 2.12.0+cpu、GCC 10.3.1；ATen/BLAS 单线程、两个 build workers。
unit tide-core5-source-qualify-20261027：background.slice、Nice=10、20 分钟上限。
原始输出 artifacts/jobs/core5-source-qualify；build manifest 已独立保存。
本报告不认证后续计算复用、mixed/resident、CUDA/NPU 真机或性能。
