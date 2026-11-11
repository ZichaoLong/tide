# FP16张量窗口与跨owner后继验证

实现固定 `5816a4c4d5d8ebcec1a22bd84a4069eb5011b97b`，默认实验接入固定
`5959bbc3646a616622d8584d6126869320b03ac8`，独立SDK half入口修正固定
`ce50a2a5b29b87b3dead824d1c377ed47deef010`。数学仍固定core-5 `0770593`。
[机器记录](column-half-20261111.json)保留构建、源码/二进制hash、JUnit、内层结果和真实作业终态；
[具名审阅](column-half-20261111-reviews.json)覆盖16个文件或函数范围，与前份重叠，不是全仓已审。
SDK入口的后继源码/hash单列，不改写前份审阅身份。

## 实现与数值边界

五记忆的张量窗口接收FP16载荷，区域评分仍须显式FP32/FP64。共享KV bank的Attention与位置logits
使用FP32内部点积、softmax、加权收缩及centered一阶VJP；保存原half根以保留版本守卫，返回原dtype。
状态、消息和参数保持所选载荷精度，原始时间/计数保持int64。位置相位沿用原契约。
native能力版本升至3，旧库须重建；workspace按载荷与评分中较宽dtype估算。

默认实验曾被旧record查询的half/peer后备提前改成legacy，绕过新窗口分派；现在让auto窗口按实际能力选择。
内置逐端口loss把half的c和G在乘法前提升FP32，再平方归约，避免先溢出再cast。
新生效配置与旧record身份区分，显式records和历史锁保持原声明。

安装消费者支持 `--dtype float16 --scoring-dtype float32 --loss-scale 16`，使用FP32-master AdamW。
half派生字段按既有 `atol=1e-3, rtol=2e-2` 比较；离散值、None关系与自身恢复仍要求精确。
精度、loss scale和比较策略进入示例checkpoint身份。这里的half通过不能代替FP32原阈值资格。

## CPU与单NPU结果

| 固定源/检查 | 结果与范围 |
| --- | --- |
| 冻结开发快照 | Python-owned与独立SDK构建成功；CPU77 passed，SDK原CPU及16项half通过；不是固定实现资格 |
| 5816a4c4 CPU定向资格 | **188 passed / 6原数值xfail**；SDK原CPU门禁及16项half通过 |
| 5959bbc3 默认调用链扩大回归 | **112 passed / 2原xfail / 2 failed**；失败为旧record KV sum FP32 evidence梯度，见下 |
| 5959bbc3 CPU安装消费者 | 位置KV/SSD × Python/native四项：独立三步master更新、whole/prefix/resumed三个新进程精确恢复通过 |
| 5959bbc3 单NPU普通异步 | 五记忆及位置KV × Python/native **12项实际设备检查通过**；同pytest另含3项CPU辅助，通过不算NPU资格 |
| 5959bbc3 单NPU安装消费者 | 位置KV/SSD × Python/native四项：独立三步master更新及新进程精确恢复通过，mixed-c |
| ce50a2a5 独立SDK | 新构建；CPU与单NPU普通异步各16项half通过，见下述独立范围 |

检查范围有重叠，不能相加成全量矩阵。原6项xfail包含4项KV列FP32见证及2项旧GDN half AdamW见证，
原失败未因新小配置通过而解除。开发初始超大时间探针曾从cut=0遍历空段并被中断；定向检查已从相应
非空逻辑切面开始，覆盖大于2**54的稀疏时钟，不把中断描述为通过。

`test_half_columns`对照独立逐事件调度，检查同秩区域、重复边、共享、完整字段、输入/参数根、
三步master AdamW和raw换owner下一窗。两个字面Attention例检查half点积溢出、非退化权重、
空行/单支持、VJP和原始版本守卫。`test_experiment_columns`另检默认身份/分派、half loss溢出反例、
KV/SSD默认任务master SGD和精确下一窗。

扩大回归的两项 `test_sum_memory_and_other_descriptors[float32-python/native]` 在
`read_program.evidence.encoders.mem.out_bias[0]` 差 `1.5355180948972702e-6`，
超原 `atol=1e-6, rtol=1e-5`。在修改前的固定796baafa、原owner构建上复现完全相同差值。
保持两项失败，不放宽阈值；该对照证明不是这轮half接入新引入的差异，没有单凭有限数值认定其原因。
该作业因pytest失败而未进入安装，后继仅补未运行的安装检查，原作业仍为failed。

## 双NPU：普通异步与显式同步分开

三个设备作业都经空闲卡准入，物理7/11映射为逻辑npu:0/1；host affinity144–159。
未使用共享卡、未绕过锁。先前796baafa的两次准入超时仍是未计算的历史事实。

- 普通异步：首个Python KV完整前向字段比较通过，反向报CANN **EE1017 / invalid event**。
  同pytest进程后续三例受运行时错误影响，不能当作三个独立数学反例。另八个独立安装进程
  （KV/SSD × Python/native × mixed-a/c）均反向失败，strict/recovery为not_checked；SDK首个FP32 peer例同样失败。
- 显式 `ASCEND_LAUNCH_BLOCKING=1`：双owner四项独立根/AdamW/raw恢复通过；
  KV/native/mixed-c、SSD/Python/mixed-c两个安装消费者的独立三步和新进程恢复通过。
  SDK FP32五记忆 × mixed-a/b/c共15项通过。
- 该同步作业末尾SDK half曾因缺少 `allow_npu_float16` 在设备解析阶段拒绝，**没有执行half数值计算**，
  因而外层仍failed。ce50a2a5补显式能力标志后重新构建，CPU、单NPU普通异步、双NPU显式同步
  各16项half全部通过，后继作业passed。

SDK的16项是五记忆 × 三放置及位置KV，用独立C++ streaming对照输出、raw状态、输入/参数VJP和两SGD窗。
它不是FP32-master或新进程checkpoint证据；这些由上表安装消费者单独提供。
CPU上的mixed预设退化为CPU，不算跨设备验证。

普通异步问题没有被张量贯穿修复；同步配置仅隔离数学链路，不作为异步修复或性能结论。
不继续重复相同copy方案；重启需要运行时修复或新的事件所有权方案，先过纯Torch/跨设备反例再验整图。

## 复现与资源

从干净检出按[开发入口](../development.md)构建对应Python-owned或standalone库，不混用SDK与wheel运行时。
CPU定向命令及实际清单在机器记录/JUnit中；最小入口：

```sh
python -m pytest -q tests/test_half_columns.py tests/test_experiment_columns.py --dtype both
python examples/consumer_core5/tensor_window.py --device cpu --dtype float16 --scoring-dtype float32 --loss-scale 16 --family kv --positions --client python --no-deterministic-algorithms --output-dir NEW
```

实际取得空闲卡后，单NPU检查设置 `TIDE_COLUMN_DEVICE=npu:0`；安装命令选 `--device npu:0`。
native客户端另指定 `--client native --native-library MATCHING_BUILD`。
双owner运行 `TIDE_COLUMN_PEERS=npu:0,npu:1 python -m pytest -q tests/test_column_payload_owners.py --dtype float32`；
普通异步与显式launch blocking使用独立进程和输出目录。SDK运行
`tidegraph-column-check --device=npu:0 --dtype=float16`，双owner另设 `TIDE_COLUMN_PEER_DEVICE=npu:1`。

两套构建都用320 workers；固定5816/5959复用经完整C++源码及二进制hash核验的开发构建，保留原manifest身份。
ce50 SDK是新固定源码构建。CPU安装affinity80–95，pytest线程1；NPU安装线程2，SDK intra-op线程1。
没有新的正式性能测量，不从检查用时推算吞吐。当前新列resident、普通异步跨卡、广泛源码审阅及
新half尺度资格仍见[ROADMAP](../ROADMAP.md)；GPU真机继续延后。
