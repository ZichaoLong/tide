# 来源列 native FP32 有界数值诊断（2026-11-12）

数值实现固定 `00bb5ac2dec61e215966f523063786682caa40c1`，数学固定
`0770593838a37c790a568226930567ed60f99b83`。CPU PyTorch `2.12.0+cpu`，
ATen/BLAS 单线程，新构建 `_tide_native.so` SHA256
`891f2ce77344b9c84d67052e3e0098a493cff2138ce9fa07a520e196f9c400a2`。
原作业 `p1p6-source-v2-native` 的 **104 passed / 6 xfailed / 3 failed** 保持失败终态。
本报告不将 FP32 超差改成通过，也不作 NPU、性能或训练轨迹资格结论。

## 问题与固定材料

`test_settle_columns.test_general_ports_messages_states_and_gradients` 的四节点
general-port 图：同秩独立区域、重复边/输入端口、缺席/零/负输入强度、原始时间
不齐、共享 summary 参数；KV width4/heads2/summary2/window30，alpha_agg=0.2。
测试使用 seed7，三个配置为 `regional_total`、`regional_total_zero`、`regional_threshold`。
独立逐事件 Python 调度与 C++ 列调度各自计算完整返回，不相互补写。

检查包含 Aggregate 内容、Describe 字段、控制、物理边/端口输出、raw 状态、
时间/计数、复合观察 loss 对全部 99 个参数/input 根的 VJP，以及零伴随时的连接。
这是一个复合 loss 的全参数/input 根检查，不冒充每个中间输出独立根的全覆盖。
原严格阈值保持 FP32 `atol=1e-6, rtol=1e-5`；FP64 `atol=1e-10, rtol=1e-8`。

## 首差与高精度结果

三个模式首个观测差异相同：按节点拓扑顺序、行号且 Aggregate 先于 Describe/控制
的观察顺序，在节点0、全局 row2 的 Aggregate content，最大绝对差
`1.9208528101444244e-9`；该行逻辑时间26，质量0.44999998807907104。
一坐标为 native `0.0008071837364695966`、scalar `0.0008071856573224068`。
全部 Aggregate 内容的最大差 `2.9802322387695312e-8`，完整前向仍过原严格阈值。
这定位到来源变换/加权 Aggregate 的数值边界，未声称唯一归因到某条 GEMM 指令。

三个 FP32 配置均只有下面两个梯度字段超差，收集没有截断；离散结构均相同。
表中第一列误差是**全字段最大误差**，不等于超差坐标的最大误差。

| 配置及梯度字段 | native32 / scalar32 全字段差 | native32 / scalar64 | scalar32 / scalar64 | native64 / scalar64 |
| --- | ---: | ---: | ---: | ---: |
| total、threshold：`nodes.0.kernel.base` | 9.5367432e-6 | 1.4819610e-5 | 1.4819610e-5 | 5.6843419e-14 |
| total、threshold：`3.content` | 2.2888184e-5 | 1.9686069e-5 | 1.2099425e-5 | 4.2632564e-14 |
| total_zero：`nodes.2.read_program.evidence_weight` | 3.0040741e-5 | 3.0280017e-5 | 1.1929182e-5 | 8.5265128e-14 |
| total_zero：`nodes.2.aggregate_program.transforms.0.embedding` | 2.9802322e-6 | 2.7772072e-6 | 2.5195120e-7 | 1.7763568e-15 |

例如 base 的唯一超差坐标 `[7,1]` 差 `5.424022674560547e-6`；
native32=-0.367931067943573，scalar32=-0.36792564392089844，
scalar64=-0.3679308774055414。不能只认逐事件 FP32 为无误差真值。
总量零温 evidence 的两个超差坐标为 `[1,4]`、`[1,9]`；其余具名字段各一个。

新的诊断先完成全部 FP32 材料，再逐位提升参数/input，并使用相同 FP32 linspace
观察 cotangent 提升到 FP64。全部99根已逐位核对。FP64 三模式前向/复合 VJP/零 VJP
均过原严格阈值；全梯度最大差分别为 `1.7053026e-13`、`1.1368684e-13`、
`1.7053026e-13`。total/threshold 零伴随保持91连接、8断开；total_zero 为99连接、0断开。

预检发现原 fixture 的 `inputs(FP64, material_dtype=FP32)` 在类型转换后再赋值
负强度 `-0.1`，该单值与真正 FP32 提升差 `1.4901161e-9`。它在 Aggregate 中被
`max(0,m)` 截成零，但不能称逐位同材料；新诊断直接提升完整 FP32 输入消除此差异。
原测试和旧记录未改写。

新来源算子的独立字面 MLP 公式、一阶 VJP、别名/重复使用/None零检查也通过。
结合首差量级、高精度收敛与连接不变，本固定用例支持将这几个差异归为 FP32
批量求值与转置/累加次序的舍入见证；不是仅凭有限性归因，不推广到其他尺度/后端。

## 决定与检查记录

`tests/column_numeric_witnesses.py` 仅将已核实的 Python 同名字段见证扩至 native：
total/threshold 的 base 上界1e-5、total_zero 的 source embedding 上界3.1e-6。
其余已有上界及所有严格 atol/rtol 不变。`Checks` 仍将这些记录为 **xfail**；
未知字段、结构变化或越过具名上界仍失败。

- 冻结00bb的三项 FP32及对应提升测试：**3 passed / 3 xfailed / 17 deselected**。
  FP32完整差异保存在 `artifacts/p1p6-20261112/source-native-dev-strict/`。
- 应用新见证的相关 native 开发检查：**12 passed / 3 xfailed / 51 deselected**，
  含FP32/FP64一般端口、提升及独立来源公式/别名/有序伴随反例。
  JUnit：`artifacts/p1p6-20261112/source-native-witness-dev.xml`。
  数值包与native仍为冻结00bb；见证代码为工作区修订，这不是新提交完整资格。
- 完整有界报告：`artifacts/p1p6-20261112/source-native-rounding.json`。
  保存诊断脚本、fixture、native哈希，以及逐字段/坐标差异、模块实际加载路径。
  本报告与可运行脚本足以重新生成，不依赖保留上述原始运行目录。

## 复现

先按 [development](../development.md) 从固定数值提交构建 native，用包含该诊断脚本的
checkout执行以下命令。若诊断脚本晚于数值提交，脚本路径可以是另一checkout；
`PYTHONPATH` 的三处和 `TIDE_BUILD_DIR` 均指向固定数值源码/构建。脚本记录两者身份。

```bash
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$FIXED_SOURCE/python:$FIXED_SOURCE/tests:$FIXED_SOURCE/scripts"
export TIDE_BUILD_DIR="$FIXED_NATIVE_BUILD"
python scripts/diagnose_source_columns_rounding.py \
  --source 00bb5ac2dec61e215966f523063786682caa40c1 \
  --output /tmp/source-native-rounding.json
```

现有输出会拒绝覆盖；它不重写原失败/阈值，不执行优化器或扩展设备矩阵。
相关源码/数学对应见 [来源绑定](../weighted-source-binding.md)、
[语义导航](../semantic-map.md)与 [张量窗口](../weighted-tensor-windows.md)。
