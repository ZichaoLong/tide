# 静态参数行视图候选

`ModelConfig.parameter_indexing="static_view"` 是显式可选的参数索引策略，默认仍为
`"gather"`。例如连续推理入口可追加：

```sh
--config-overrides '{"parameter_indexing":"static_view"}'
```

该选项不改变参数、活跃选择或矩阵运算，只尝试消除已知完整 owner 顺序下的参数
物化。Reception、KV/SSD prepare 和 Description 由区域静态构造 owner；如果其
实际参数行是连续递增的，`weight[groups]`、bias、NodeVector 及私有 adapter
改用切片视图。共享且只有一个 base 的原 MM 路径保留。fused QKV 的拼接仍是真实
开销，只消除拼接前可以避免的 gather。

对于 `N` 行、输入/输出宽度 `I/O`、adapter rank `R` 的一个有 bias 的共享投影，
非 uniform 的原参数 gather 分配量为

\[
M_{gather}=s\,N\,[OI+O+OR+RI],
\]

其中 `s` 是参数元素字节数。这部分与本次 `B×L` 无关，因而单位置 decode 更容易
暴露其成本。连续视图可以省去这些临时副本；后续 GEMM 仍需要读取权重，计算量不
减少，不能据此把省下的分配量称作实际带宽或预测同等比例端到端加速。

边界由两种不同的 metadata 保证：

- `possible_ids` 只是合法 owner 的 superset，不能推断实际顺序。
- `NodeOwners.from_static` 从同一份有序 host tuple 构造真实 device IDs 和
  `exact_ids`。模型和 placed wrapper 仅在静态构造/静态 global→local 映射时使用。
- 动态 `select_owners` 和不规则 selected packing 明确丢弃 `exact_ids`。Full 仍按
  真实选中节点 gather，绝不把稀疏 Full 改成全节点计算。
- 重复、乱序、非连续参数行继续 gather；同长度或同 superset 都不能触发视图。

forward 不读取 device owner 值到 host。cached IDs 在普通模式下创建，可以跨
inference-mode warmup、训练及重算使用；缓存中没有投影权重或激活。state_dict
加载会清除参数分组的切片计划。checkpoint 保存该配置，恢复时重新构造静态缓存。
普通 autograd 负责 slice 的梯度；训练反向和峰值内存仍需单独实测。

`benchmarks.parameter_gathers.ParameterGatherAudit` 可在独立的非计时调用中记录
真实 Parameter storage 上的 advanced-index 输出字节。它不把路由、cast 或 concat
当作参数 gather，也不测硬件总流量。并发区域使用 caller-thread dispatch 审计时应
显式包在 `serial_region_audit(model)` 中，并保留该串行诊断身份。

小型可枚举见证中，3 行、`I=5/O=7/R=3`、FP32 的上述参数 gather 为 936 字节，
连续视图为 0；相同 superset 下的乱序/重复选择仍为 936。真实 FC/HB 的 Full FFN
依旧动态读取，所以这个局部数字不代表完整模型降峰或吞吐收益。相关独立公式、
VJP、更新、续接和恢复检查位于 `tests/test_parameter_views.py`；硬件结果以冻结
运行记录为准，该候选不是新的默认性能或数值资格声明。

[单卡 FP32 路径证据](evidence/reception-validation-20261118.md#fp32-static_view-路径)
单独改变 gather/views（两者 region batch1）：推理 exact、两次 VJP 原阈值内，
三次更新仍有逐步扩大的有限超差；support、int64 和工作计数一致，各自新进程
恢复更新阈内。内部见证达到 capture_limit，尚未建立全部差异的因果解释。
大参数常驻推理的时间和内存取舍见[推理实测](evidence/reception-inference-20261118.md)，
不能用静态少分配量直接推定吞吐收益。
