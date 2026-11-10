# 同公式层链成本基线

`scripts/layer_tensor_baseline.py`独立实现固定0770593 KV §2/4与SSD §2/3的
整段张量公式；默认图入口通过`scripts/benchmark_layer_chain.py`作同材料有限对照。
它不是默认执行器，也不调用参考轨迹。KV使用厂商SDPA；SSD用深度卷积及因果稠密权重
展开递归，不复用候选affine scan。SSD展开有L²代价，明确限制L<=256。

适用域是每层一个节点/区域的相邻链、每个原始时间位置一条强度恒1的输入、
alpha_agg=0、proportional/k=1/fixed_gain=1；KV无位置旋转/距离偏置，
SSD卷积stride等于Settle的逻辑位置步长。函数入口逐项校验，不能替代任意图或输入执行。
层间payload的c和cG保持区分，此限定域c=1。逻辑时间与卷积槽为int64。

这是指定输出与末态张量的实数函数对照，不是完整记录等价。张量基线省略不影响此投影的
Describe字段和图记录，消去恒零依赖；其None和图实现的连接零只按数值导数比较。
`tests/test_layer_tensor_baseline.py`独立scalar调度检查输出、每个末态张量的数值、
输入与参数的数值偏导。生产路径的None/零、完整返回、共享、恢复仍由原严格门禁负责。
`run(..., start=绝对位置, left=逐节点原始字段)`支持非零左状态和跨窗保留梯度；省略left
表示绑定零初态。`tensor_baseline_state.py`检查字段、B轴、dtype/device、单调int64时间和
窗口之前的快照。KV将旧/新历史各保存一次，使用绝对时间mask；SSD继承H/Z/W，并按实际
时间戳匹配跨窗卷积槽。start可以保留时间间隙，不把它压紧成接收次数。当前一个批次内
各样本使用相同时间槽；任意ragged端口仍由通用基座处理。

`tests/test_tensor_baseline_continuation.py`以独立scalar调度核对非零左态各字段根、共享参数、
三步SGD及momentum的数值投影，另核对跨窗保留图、超过2**53的逻辑时间、有限KV窗口和
卷积时间间隙。返回的原始字段可直接序列化；实际保存model/optimizer/raw state/next position
后由新进程新owner继续两步，要求返回、梯度、状态、参数与optimizer逐位一致。该测试包是
限定投影的checkpoint见证，不是通用图v5 checkpoint：不包含图的pending、history、事件记录
和完整metadata。None/零及optimizer槽结构的跨实现等价仍不在投影资格内。
这里也不声称等同某个标准Transformer/Mamba模型。

每个候选单独进程、一次预热后恢复初始参数/optimizer再测完整一步。包含输入准备/传输、
默认调度与packing、前向、反向、有限检查、SGD及同步；构造另列。张量基线与图入口
返回职责不同，差距表示可继续研究的开销范围，不能报告为已验证的等价实现加速比。
同源码小型正确性资格与实测尺度数值资格分别记录。

跨设备资格显式忽略device标签，保持dtype、shape、int64和原数值阈值。
需要收集全部有限超差时，可给测试设置`TIDE_DENSE_NUMERIC_REPORT=NEW_DIRECTORY`；
完整输出/末态/数值根仍全部检查，有限超差最终标为xfail，非有限/结构错误仍失败。
`scripts/diagnose_layer_rounding.py --device npu:0 --native-library MATCHING_BUILD --output NEW.json`
把同一FP32材料提升到CPU FP64，分别记录scalar、tensor及默认Python/native的输出、末态、
输出平方目标梯度误差。它是有界定位，不改变容差，也不认证层链之外的训练效果。
测试和成本默认显式采用公共实验入口的`ieee`算术策略，关闭HF32/TF32矩阵与卷积缩精度，
记录请求/生效开关。`--precision-policy vendor-default`只供单独诊断，测试对应
`TIDE_LAYER_NUMERICS=vendor-default`；NPU厂商默认允许卷积HF32，不能只按张量FP32标签视为同一算术。

```sh
python scripts/benchmark_layer_chain.py --family kv --implementation tensor \
  --device cpu --batch 4 --length 32 --width 128 --output-dir NEW_TENSOR_RUN
python scripts/benchmark_layer_chain.py --family kv --implementation native \
  --native-library MATCHING_BUILD --device cpu --batch 4 --length 32 --width 128 \
  --output-dir NEW_GRAPH_RUN
```

SSD用`--family ssd`，纯PyTorch用`--implementation python`。NPU通过当前站点准入运行，
长测量用固定源码background.slice服务；CPU/NPU、线程和实际算子实现分列。
下一轮优化可据此判断图开销、局部数值求值和装配各有多少余地；不能从小矩阵或低L
直接外推大型模型、长序列、resident及多卡。

[固定检查与12个成本点](evidence/layer-chain-baseline-20261110.md)记录实际CPU/NPU结果、
原失败和IEEE策略见证；默认图入口仍显著慢于该限定张量投影，不由此宣称目标已达成。
