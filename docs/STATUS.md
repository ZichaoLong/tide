# 005同类正量链交付完成（2026-11-19）

生产修复：`b4a8d937`；NPU设备求值：`3efcc6b1`。
`src/tide` tree为`6863ff792e3fae1556a80e195c1358a3405e0e0e`。
公式见[positive-chain](positive-chain.md)，详细资格/成本见[本轮证据](evidence/positive-chain-20261119.md)。
固定math `4f5b9f20` 的108文件原字节通过。没有floor、epsilon或原阈值改变。

- b4a冻结CPU全套1182 passed、4原有strict xfailed；3ef最终CPU定向274 passed、3旧xfail。
- 3ef未捕获输出的NPU定向133 passed、3旧xfail、2 warnings，未见主机fallback。
- CPU b4a和NPU 3ef各五个小图（4 FP32、1 BF16）完成训练/续接/新进程实际恢复更新，均within_threshold。
  CPU恢复全exact；NPU三个HB案例更新有约1e-9的有限梯度差异，saved_data/next_forward全exact。
- 固定小HB pool的三组成对成本：CPU完整推理/训练耗时中位比+4.01%/+7.43%，NPU +26.93%/+40.16%。
  初参/输入/工作相同，无Full padding增量。NPU before含frexp/copysign主机回退，after日志未见；
  这是算法和设备实现共同改变的净成本，不外推9B，也不代替质量验证。
- 旧stable-mass的NPU训练/恢复/成本日志原含主机回退，已[更正执行归属](evidence/stable-mass-20261119.md)，
  原数字/失败保留。3ef局部trace有AIVEC、AICPU和HtoD；不是纯AICore或无搬运保证。
  trace导出后解析器失败原样保留，离线审阅不将failed作业改成通过。
- 证据独立提交`f098d3b4`；全部12个服务和148个记录后代已核验终态，无活跃本轮任务。
  13个本轮临时目录已退休，释放约838.3 MB，原记录/失败/trace在压缩证据中保留。
  仓库含Git与主私有环境约151.2 MB；退休后`python -I`隔离导入当前源码，CPU FP32前向/反向/AdamW smoke通过。
  [退休收据](evidence/positive-chain-retirement-20261119.json)记录逐目录大小、保护项、用户hash及主安装检查。

normalize不能保证防溢出：后续可训练投影/偏置无上界，质量也无统一正下界。
具体可表示结果已修；真正最终正发送量低于FP32范围仍严格拒绝。
任意极端Attention、后续共享累加、RMS/SSD及对数质量表示仍有边界，不宣称005全部问题消失。
推荐包含3ef的FP32当前实现，BF16次选；本轮未新增隔离wheel、9B训练、多NPU/CUDA或下游验收。

此前五类记忆、9B/Transformer、FFN/CPU/NPU拆解证据保持各自源身份；默认实例与FFN执行策略保留。
四份用户文档保持未跟踪且原字节。参考/公共/下游未修改，通用基座未启动。
