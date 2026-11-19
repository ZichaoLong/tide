d768/F3072 FFN 有界诊断复核已完成。**原四项公式输出资格仍失败；不放宽阈值，不凭补充 FP64 参照追认通过，W 本轮不扩展到 9B。**

范围为独立参数 seed469 和区域共享/rank8 seed470，4 节点，owner `[3,0,3,1]`，输入 `[4,1,2,768]`。物理 NPU 9 / `npu:0`、Ascend910_9392、FP32、HF32 关闭。原输入和参数哈希匹配，新 NPU 输出也逐路径匹配原输出哈希。

值阈值 `atol=rtol=2e-6`，VJP 阈值 `atol=rtol=6e-6`。下表 scaled 为 `max(|a-b|/(atol+rtol*|b|))`，不超过 1 才在阈内。原失败重放保留 FP32 比较运算；新增诊断用 CPU FP64 计算保留张量的差与比例，阈值未变。

| 参数归属 | 路径 | 原 CPU32 公式输出 max_abs | 原 max_scaled（精确复现） |
|---|---|---:|---:|
| independent | gather | 4.17232513427734e-06 | 1.24008083343506 |
| independent | grouped | 5.24520874023438e-06 | 1.42121064662933 |
| region_shared | gather | 4.41074371337891e-06 | 1.36872589588165 |
| region_shared | grouped | 4.41074371337891e-06 | 1.13177049160004 |

同数据 FP64 独立公式参照下的输出：

| 参数归属 | 路径 | max_abs | max_scaled | 超差元素 |
|---|---|---:|---:|---:|
| independent | cpu_formula_fp32 | 3.99632211e-06 | 1.045988085 | 3 |
| independent | cpu_production_fp32 | 4.23474069e-06 | 1.107650562 | 3 |
| independent | npu_gather | 1.9168041e-06 | 0.561359310 | 0 |
| independent | npu_grouped | 2.44226148e-06 | 0.659679326 | 0 |
| region_shared | cpu_formula_fp32 | 4.36344137e-06 | 1.086993791 | 5 |
| region_shared | cpu_production_fp32 | 4.36344137e-06 | 1.190686346 | 3 |
| region_shared | npu_gather | 2.28919702e-06 | 0.656307578 | 0 |
| region_shared | npu_grouped | 2.03286889e-06 | 0.550652357 | 0 |

两 NPU 路径所有捕获阶段及输入 VJP 对 FP64 均在原阈内。CPU32 独立公式首次观测越界：独立参数在 up 投影（1.008709，1 元素），共享在 down 投影（1.109911，3 元素）；CPU32 production 分别在 product（1.055314，2 元素）和 gate（1.005985，1 元素）。这些是观测边界，不是因果根源证明。

同 NPU grouped 对 gather 的直接比较：

| 参数归属 | 输出 max_abs | 输出 max_scaled | 输入 VJP max_scaled | 全阶段最大 max_scaled |
|---|---:|---:|---:|---:|
| independent | 3.57627869e-06 | 0.810223618 | 0.243213450 | 0.858169862 |
| region_shared | 2.62260437e-06 | 0.851815745 | 0.399944582 | 0.851815745 |

两路径 norm 输入完全相同，首次非零差异在 gate/up 投影；捕获的 product、down、output 与输入 VJP 均在原阈内。这是同设备、同样本的直接证据，范围独立于原 CPU32 公式资格。

共同输入探针把 FP64 norm/product 各舍入一次到 FP32，再给两路径相同局部输入。共同 norm 后各阶段的直接比较最大 scaled 为 0.707576 / 0.747267；共同 product 的 down 直接比较为 0.669420 / 0.556415。相同 product 用 FP64 计算 down 的参照下，gather/grouped 为独立 0.649026/0.497321、共享 0.525330/0.520232。全部阈内，但属于条件局部投影探针。

局部 norm 的 rsqrt 与 sqrt/div 输出最大差均为 2.384186e-7，scaled ≤0.039672；局部输入 VJP ≤0.019860。SiLU-product 原生与 sigmoid 展开输出最大差均为 4.768372e-7，scaled ≤0.068257，局部 VJP <0.004330。这不足以把完整差异归因 norm/SiLU，亦未确定 GEMM 内部累加次序或指令级原因。

原资格全部输入/参数梯度通过并保留；本诊断仅重算输入 VJP。CPU FP64 只作有限诊断，既不覆盖完整模型、私有状态、checkpoint 和恢复，也不构成新训练精度资格。medium 原结果不能覆盖 d768 输出资格限制。

已有 trace 记录 27 个设备 GroupedMatmul 任务（KERNEL_MIX_AIC），int64 sort/cumsum 在设备 AICPU。group0 读原 Parameter bank/view，未观察到 ATen 参数 gather 或布局复制；CANN 分配警告、无参数身份的 SDMA/TensorMove 仍在，不能据此声称供应商内部零搬运。事件时长之和不是正常壁钟占比。

生产源码 SHA256 `492122d8430115f902c3751ff2f88f5fe23e39ca9c7479bf56d2570e1ff54f14` 在原资格与诊断冻结树中重新核对一致。诊断源码/工具提交 `a58536608f7276a771c56703b48e23ea82d2ca65`；原资格提交 `05e8338bf62569100d194848689d545f1140afbc`。

原始诊断：[diagnosis.json](../ffn-local-diagnosis-v1/diagnosis.json)，SHA256 `1f5ba53126f6e8f009a3bab3c88dcdd2cdbb86ce0425d0ee1c14891c9f60bc0a`。保留张量 9,526,443 字节，SHA256 `42586a68d2e3dffbe62443d729f9afc74356347d6acce60cec71b03224ba5fbc`。精确逐阶段数值、所有输入指纹与 trace 引用见同目录 `d768-diagnosis-review.json`。

这里只复核已保存证据，未启动设备实验或修改生产文件。原始结果完成不等于服务/cgroup/后代终态核验；后者由主代理另行记录。
