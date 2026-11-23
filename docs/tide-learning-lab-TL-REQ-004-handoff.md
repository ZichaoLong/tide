# TL-REQ-004 接续 issue：TIDE eager 训练的批量执行与反向瓶颈

状态：**待基座线程评审和处理；实验侧定位已完成，基座修正及下游验收尚未开展。**
2026-11-09，用户明确授权实验线程在本目录撰写这一份文档 issue，后续由用户在基座 cwd 线程处理。
本文件是交接入口；唯一权威请求仍保留在提出方 `tide-learning-lab`，不建立另一份独立维护的需求或结果台账。

## 1. 问题和期望

实验重点是同一个模型的自然文本 next-token 质量，以及规模化、实际稀疏和局部结构。
当前 FC16 的 CPU/native 完整训练步明显快于 NPU/native。新 profile 已将问题定位到
Read/Describe 逐行计算、KV 的动态索引/host 回读、细碎字段根及索引反向。
这些开销限制单条轨迹达到百万 token 的训练曝光，不能只用 width128 或总参数量解释。

请求在公共基座内提供一个有明确适用范围、实测改善完整训练步的优化路径。
优先复用现有批量与 resident 能力及 [已有性能经验](performance-lessons.md)，再决定具体修正。
不要求先实现稀疏比例对应的吞吐提升，不要求首先重写整个执行器或完成全平台/全部语义矩阵。
下游继续拥有数据、任务头、loss、配方和应用恢复；不在实验分支复制执行核心。

## 2. 固定来源和材料

- 数学：`0770593838a37c790a568226930567ed60f99b83`。
- 本次被测基座：`74e7bbd62341d2ccd382194a2c52ad4fea033665`，协议 `tide-downstream-contract-v1`。
- 下游报告提交：`0439efd2af31cf10b07ce9e6423b0195f7449968`，已经普通推送至 `tide-learning-lab`。
- 诊断源码：`d90e733cd1b1d2d13f5da578450f17392f4f49a0`（阶段/trace），
  `b4bc62bda4f1424604c4eb0d1736dbe2dea7b735`（宽度、placement及传输隔离）。
- 环境：Torch2.12.0+cpu、TorchNPU2.12.0.post3、CANN9.2，FP32，确定性策略显式开启。
  CPU为1线程，NPU为2个host线程；公共 eager packed greedy 使用1个host scheduler worker。
- native是同一基座提交的 Python-owned NPU 构建，binary SHA256：
  `27ecaa1315d250d9ab2de2a42f2a2487dc4d8225d739c61ffb333b7d66b08289`。

固定公开链接：

- [权威请求 TL-REQ-004](https://github.com/ZichaoLong/tide/blob/0439efd2af31cf10b07ce9e6423b0195f7449968/docs/requests/TL-REQ-004.md)
- [人工审阅的定位报告](https://github.com/ZichaoLong/tide/blob/0439efd2af31cf10b07ce9e6423b0195f7449968/docs/studies/language-execution-diagnosis-v6-results.md)
- [逐运行、实际配置、原始文件hash和Lens对应](https://github.com/ZichaoLong/tide/blob/0439efd2af31cf10b07ce9e6423b0195f7449968/docs/evidence/language-execution-diagnosis-v6.json)
- [有界研究表](https://github.com/ZichaoLong/tide/blob/0439efd2af31cf10b07ce9e6423b0195f7449968/docs/studies/language-execution-diagnosis-v6.md)

本机材料根目录：`/mi/data2T/zlong/home/llm/tide-learning-lab`。
私有固定安装 `.venv-positive-floor-v1`、native构建 `.local/native-build-v6`、
冻结源码 `.local/source-v6-diagnosis` / `.local/source-v6-diagnosis-variants` 均继续保持只读。
新修正使用新的私有安装和构建，避免影响仍运行的语言实验。

## 3. 最小实际见证

模型为 FC16：4区域、每区域4节点、KV/qry、regional_total/quad、温度0.25、width128、
AdamW、任务侧embedding和4096词表全位置head。输入是确定性生成的synthetic token；不需要下载语料或checkpoint即可复现定位。

原形状 B8/L128 的新测量如下。每段同步；不含初次模型构造、输入准备、记录和profile。
core前向包含每次调用内部的public runner构造及完整返回。

| 路径 | core前向 | 反向 | 完整步 |
| --- | ---: | ---: | ---: |
| CPU/native | 26.60秒 | 36.84秒 | 63.61秒 |
| NPU/native mixed-c | 134.93秒 | 129.63秒 | 265.00秒 |

这是一次预热后的一步；原未分段校准 CPU62.6–63.0/NPU232.5–236.6秒另行保留。
共享主机负载、设备和同步边界不同，不将两次观察的差值归因为代码回归。
两后端candidate都是16,384；Full为8,352/8,350，内部消息25,068/25,108，差异如实保留。
这里没有重新认证完整输出、梯度或路由的跨后端严格等价。

B2/L8的小型native NPU见证只有16 token、256 candidate，却产生168,589个kernel记录，
7,308次标量读取、6,875次索引、3,603次索引回写。插桩窗口6.155秒，
profiler Computing0.684秒、Free5.471秒；无插桩步约4.09秒。插桩有额外开销，不能将其比例直接当作正常利用率。
设备AICPU的Index和IndexPutV3各1,170次，共252.56ms；这是设备侧处理单元，不是主机CPU fallback。
Python另有独立设备trace，不能用其耗时占比冒充native数据。

B8/L16，预热后两步均值的有界隔离：

| 设置 | CPU/native | NPU/native |
| --- | ---: | ---: |
| width128 / 5.09M参数 | 6.94秒 | 23.54秒 |
| width256 / 17.64M参数 | 7.98秒 | 26.25秒 |
| width512 / 65.55M参数 | 11.64秒 | 27.45秒 |
| width128，mixed-a | — | 14.77秒 |
| width128，紧凑事件/批量传输且关闭trace | — | 22.88秒 |

增宽确实缩小设备差距，但512以内未反超CPU。mixed-a同时移动Read/control/selection，
改变传输及反向路径；约37%的改善不能全部归给某一个函数。传输选项约2.8%的差异不足以认证稳定收益，返回策略也不同。
两项隔离与默认路径在对应步的candidate/Full/内部消息计数相同，不能据此推出完整离散或梯度等价。
更宽模型只做三次synthetic诊断更新，不是新的语言训练或scaling结论。

## 4. 代码定位与建议修正方向

1. **批量Read/Describe。** [C++](../cpp/src/weighted_describe_batch.cpp)的逐行循环及
   [Python](../python/tidegraph/weighted/read_batch.py)在batch入口内按行计算current、归一化、score/evidence/key。
   小见证中三个典型向量投影各出现256次。原注释说明采用逐行有限算术顺序是为近零VJP差异；
   请求具名批量/融合路径，保留独立参考和原失败，不能通过删除梯度检查换取性能。
2. **KV索引与控制元数据。** [Query VJP](../cpp/src/weighted_query_vjp.cpp)的signature包含私有state shape及正质量分支；
   [KV查询](../cpp/src/weighted_query_kv.cpp)逐行bool索引、nonzero及长度分组；
   [Aggregate VJP](../cpp/src/weighted_aggregate_vjp.cpp)也按请求取质量分支。
   优先批量化必要元数据、复用已知int64可见布局、使用mask/长度或分段算子，减少重复索引和回读。
3. **完整反向和真实算子batch。** Query的ragged分组先于Full query/FFN，
   `max_full_batch`只是调度请求规模，不能证明后续投影也有同样batch。
   小见证仍有大量 `[512,128]×[128,1/2]`；原形状的约1000个Full请求也不能直接视为千行GEMM。
   可将共享投影与FFN从ragged attention分组中分开；优化独立根VJP/分段累积，减少select/stack/scatter和小图。
   需要保留None/连接零及实际参数别名，不能只看前向变快。

这是当前执行路径的瓶颈证据，不是数学错误证明，也没有证明所有反向成本都来自一个类。
现有 [resident训练owner](resident-weighted-training.md)是另一候选，但任务根、VJP、optimizer和恢复边界独立，
不能仅切换preset就认定接入或加速完成。优先提出具体可验的小改动或布局方案，不原样重跑历史大矩阵。

## 5. 基座线程如何接续

先只读打开下游的 `docs/evidence/language-execution-diagnosis-v6.json`；每个run包含来源、配置、分段时间、
算子摘要和原始文件hash。profile路径中的 `*` 替代厂商目录内的私有主机名，实际本地目录唯一。
主要原始材料：

- `runs/v6-diag-native-{cpu,npu}-full/result.json`：原形状阶段测量。
- `runs/v6-diag-native-npu-device-profile/npu-profile/*/ASCEND_PROFILER_OUTPUT/`：native设备trace及CSV。
- `runs/v6-diag-native-{cpu,npu}-small/operators.json`：CPU侧算子调用与shape。
- `runs/v6-diag-python-npu-small/host-functions.json`及其`npu-profile/`：独立Python定位。
- `runs/v6-diag-variants/`：placement/宽度/传输隔离；CPU宽度对照在同级`v6-diag-native-cpu-width*/`。
- `artifacts/jobs/v6-diag-*/job.json`：精确argv、环境脚本hash、固定源码、构建输入和资源。

以下仅打印原始启动身份，不运行模型：

```sh
python - <<'PY'
import json
from pathlib import Path
root = Path('/mi/data2T/zlong/home/llm/tide-learning-lab')
for name in ('v6-diag-native-npu-full', 'v6-diag-native-npu-device-profile', 'v6-diag-variants'):
    record = json.loads((root/'artifacts/jobs'/name/'job.json').read_text())
    print(name)
    print(json.dumps({key: record[key] for key in ('source', 'command', 'resources', 'environment_script', 'inputs')}, indent=2))
PY
```

复现入口为冻结下游源码中的 `scripts/diagnose_campaign_execution.py`，原模型配置
`configs/studies/language/v6/scale-fc16-kv-quad.json`；形状、placement和width覆盖写在实际argv/result中。
按基座线程的任务授权与新预算，先以B2/L8定位，再测B8/L128完整步。
通过现有durable/设备机制分配资源，使用全新输出，不覆盖原run或修改活动实验依赖。
本次已有14个诊断全部完成、10个服务无存活后代；约0.81 CPU核时、0.36 NPU卡时、2.42GiB产物。
这些是已用费用，不是新任务的计算授权。

## 6. 建议验收、交付与边界

- 保持固定数学和原严格阈值 `atol=1e-6, rtol=1e-5`；执行、严格数值、离散语义、恢复、学习和性能分别报告。
  有限数值超差经过有界排查及相关机制检查，可以继续具名探索；保留原失败和未知原因，不默认退回CPU FP64。
- 小型独立检查覆盖输入、共享参数、私有状态和字段根梯度，None/连接零、未使用参数不更新，
  optimizer槽、完整raw/int64状态、正支持/消息存在性和应用新进程恢复。
  非有限、错误分支/共享或恢复错误阻塞受影响路径，不能用学习loss下降替代正确性。
- 完整步比较保留相同模型、head/loss、目标位置、精度和实际工作量；构造、预热、profile、计时、记录分别核算。
  需要优化必要检查的批处理方式，而不是直接删除；不得把关闭完整返回/恢复等不同契约冒称同工作量加速。
- 先交付一个明确范围内的改进，报告源码/构建/安装身份、公共接入参数、实测通过/失败/未检查、复现命令和剩余瓶颈。
  原严格参考保留；新数值路径或返回契约显式命名。基座通过与下游接受是两个事实。
- 下游随后在新私有安装和新固定作业验收，再决定依赖更新及语言训练。
  36h完成4,194,304 token仅纯训练就需约32.4 token/s；当前同形状CPU约16.1、NPU约3.86。
  优先改善可承载的数据曝光，不把理想稀疏加速比例作为学习门槛。

本次只新增这份本地issue文档，不修改基座执行源码、STATUS/ROADMAP、共享环境或既有作业；
保留原TL-REQ-003交接文件。基座接收工作后，在其线程按 [downstream-contract](downstream-contract.md)
登记接受范围及交付；本文件不提前宣布接受、修复或验收，也不自动启动计算。

权威请求快照SHA256：`04b9d022a3efe2a01ae058d571033a77b1c74cae94269f4d5a2be6a451a85ae9`（上述下游提交中的`docs/requests/TL-REQ-004.md`）。
