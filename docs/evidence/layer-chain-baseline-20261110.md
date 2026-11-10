# 同公式层链的独立检查与CPU/NPU成本

默认图入口仍明显慢于限定域的直接张量公式。这个结果确立了后续优化基线，
没有证明通用入口已接近Transformer/Mamba原生性能。两者职责不同，也不能把表中
差距直接写成已通过完整等价检查的加速比。

[审阅数据](layer-chain-baseline-20261110.json)中的最终成本使用核心/安装`3d88b0b`；
NPU驱动`f6092d6`、CPU驱动`1f82143`，相关层链实现相同。
数学固定0770593。定义域、公式、独立性及复现入口见[层链契约](../layer-chain-baseline.md)。

## 检查与有界数值诊断

小型检查B2/L3/d4、两层、两head：CPU FP64/FP32共13项通过（930d1ba驱动、f15c4a5安装/库）；各轮NPU驱动/包/库身份在JSON逐项记录。
独立scalar调度对照张量公式的输出、末态及逐个字段数值导数；默认Python/native
入口另核对输出和参数/输入导数。这里的数值零导数不认证None/连接零身份；
完整记录、共享、optimizer及恢复资格仍见[局部合批](prefill-local-batches-20261110.md)。

首轮NPU 7项中3通过、4失败，其中包含一个跨设备标签比较错误；修正标签检查后
4通过、3个原数值xfail。固定原阈值进一步定位到SSD卷积的实际HF32策略：
vendor-default为matmul关闭HF32、conv开启；显式IEEE同时关闭。完全相同的FP32材料
提升到CPU FP64作见证，SSD tensor输出最大绝对差从3.06344e-6降至2.41431e-8，
输出平方目标梯度最大差从1.25596e-5降至1.41305e-7。

显式IEEE固定门禁7/7通过，新3d安装/库再次7/7通过；原失败和旧阈值保留。
这解释指定见证的算术策略差异，不把全部FP32超差都归因于舍入。
未修改公共环境；实验入口原已采用IEEE，此处让独立基线和测试明确采用同一策略。

## 有限成本

B4/L32/d128、四层、四head、FP32；每层一个节点/区域，单位发送、零左状态。
CPU1线程，NPU主机1线程；图实现为host分量窗口/mixed-c，非resident。
一次预热后重置参数和optimizer，一次完整训练更新；包含输入/传输、前向、反向、
检查、SGD和同步。构造、独立检查另列。主机共享，无独占或统计稳定性声明。

| 记忆 | 实现 | CPU完整步秒 | NPU完整步秒 |
| --- | --- | ---: | ---: |
| KV | 独立张量公式，SDPA | 0.028943 | 0.021841 |
| KV | 纯PyTorch默认图入口 | 1.025452 | 1.456204 |
| KV | Python调用LibTorch默认图入口 | 0.511736 | 0.933841 |
| SSD | 独立张量公式，卷积/因果矩阵 | 0.043775 | 0.032125 |
| SSD | 纯PyTorch默认图入口 | 1.757319 | 3.976994 |
| SSD | Python调用LibTorch默认图入口 | 0.893817 | 2.853459 |

KV native的NPU前向/反向分别0.577612/0.346453秒；SSD分别1.786614/1.048128秒。
KV图运行实际为4区域批、最大B×L=128、4次序列求值、12次私有投影、512个候选和Full；
已经合批的这些统计并不能说明内部快照、索引、VJP和记录装配也只剩固定次数的提交。
此处native仍经Python消费者调用，不能冒充standalone LibTorch的成本。

张量基线省略不影响指定输出的Describe/图记录，SSD因果矩阵有L²代价且限制L≤256。
实测尺度独立数值为not_checked；没有在此零左状态基线上验证checkpoint续接。
标准Transformer/Mamba架构、长序列、其他参数共享/分配、resident、多卡均不能由此外推。

## 复现

在固定提交按[开发入口](../development.md)安装，提供对应native库；NPU经站点准入。
每个实现独立进程、固定线程，不与重构建重叠：

```sh
TIDE_BUILD_DIR=MATCHING_BUILD python -m pytest -q tests/test_layer_tensor_baseline.py --dtype both
python scripts/benchmark_layer_chain.py --family kv --implementation native \
  --native-library MATCHING_BUILD --batch 4 --length 32 --width 128 --heads 4 \
  --depth 4 --threads 1 --precision-policy ieee --device cpu --output-dir NEW_RUN
```

另选`tensor/python`、`ssd`或`npu:0`；NPU测试设置`TIDE_LAYER_DEVICE=npu:0`，并选`--dtype float32`。
`diagnose_layer_rounding.py --precision-policy vendor-default|ieee`重建同材料诊断，
不需要旧运行目录。后续重性能工作依据这些差距与具体profile选择边界，不能用小图
低成本基线的速度替代默认图入口的实际结果。
