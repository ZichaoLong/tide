# 四循环与 KV Full 的半精度原始状态资格

实现固定于 `bcef2c49dfdbde277d4115f450e4fec07d7f8dc9`，数学上游固定
`0770593838a37c790a568226930567ed60f99b83`。机器可读的来源、构建、日志摘要和
复现入口在[同名 JSON](core5-resident-half-memory-20261028.json)。
契约分别为[循环记忆](../resident-weighted-recurrent.md)和[KV Full](../resident-weighted-kv-full.md)。

| 验收 | 实际结果 |
| --- | --- |
| CPU 定向测试 | 24 passed，无 skip；pytest testcase 合计 2.596 秒 |
| standalone NPU 循环 | FP32/FP16 各 24 配置；affine/delta/gated_delta/ssd 的 Upd/Mem/Full |
| standalone NPU KV Full | FP32/FP16 各 4 配置、96 重放行、2 heads |
| CUDA 构建/loader/help | 111 个构建步骤完成；所选依赖 34 条 CUDA 编译命令均为 80/90/100 |

原始状态分别实际存储 FP32/FP16，参数、输入、有效快照、局部算术、Mem/Full 返回与
全部一阶伴随为 FP32；时间、计数与 cache stamps 为 int64。独立循环公式先精确拓宽
实际量化的旧 raw，再计算数学函数；KV Full 的独立公式消费实际量化窗口。检查完整
raw/输入/参数 VJP、保存版本和独立输出根，不以输出近似代替梯度或连接检查。

四循环的额外见证覆盖次正规快照、零接收保留原位模式、`2^-25` 正接收仍更新时间/计数与
缓存条目、质量 VJP 为一、有限 FP32 结果溢出 half 时拒绝且不提交 live owner，以及
half cotangent 拒绝。存储 Cast 的声明 VJP 是 FP32 恒等映射。

两个独立任务均从干净固定源码启动，持久状态 `passed/exit0`；审阅时 unit 均
`inactive/success/0`。复核 component/core source、binary、loader、CPU XML 和设备检查
日志摘要。NPU 经队列分配物理 9→逻辑 npu:0，四组合全部通过。CUDA 配置总共生成
160 条 CUDA 命令，实际所选依赖为上述 34 条；两者的架构集合均逐条核验。
资源为 `background.slice`、Nice10、2 build workers、BLAS1，任务上限 1200 秒、
队列上限 180 秒、设备检查上限 360 秒。

这批证据只覆盖局部组件；未运行 Python-owned half 组件，也未重新运行 CPU codec。
codec 由[前期独立资格](core5-resident-half-primitives-20261028.md)负责。
完整 half 图、训练、peer 与性能另行验收；CUDA 真机仍待硬件。
