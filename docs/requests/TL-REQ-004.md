# TL-REQ-004：面向实际语言训练的批量执行与反向

提出方：`tide-learning-lab`。状态：下游已定位并提出需求，尚未修改或验收基座修正。
本请求服务于同一模型的语言质量、规模、实际稀疏与局部结构；不要求先实现与稀疏率成比例的硬件加速。
权威材料为[诊断报告](../studies/language-execution-diagnosis-v6-results.md)、
[测量摘要和原始文件hash](../evidence/language-execution-diagnosis-v6.json)及[有界研究表](../studies/language-execution-diagnosis-v6.md)。

## 固定边界和实际见证

- 数学`0770593838a37c790a568226930567ed60f99b83`，基座`74e7bbd62341d2ccd382194a2c52ad4fea033665`。
- 安装`.venv-positive-floor-v1`，Torch2.12.0+cpu、TorchNPU2.12.0.post3、CANN9.2。
  native为同提交的Python-owned NPU构建，binary SHA256
  `27ecaa1315d250d9ab2de2a42f2a2487dc4d8225d739c61ffb333b7d66b08289`。
- FC16：4区域、每区域4节点、KV/qry、regional_total/quad、temperature0.25、width128；
  公共packed greedy、单host scheduler worker，FP32，AdamW，任务侧embedding、全位置读出和4096词表head。
  NPU默认mixed-c：数值payload/Read/control/selection在NPU，事件推进在host。
- 原形状B8/L128：CPU1同步分段测量63.61秒（core前向26.60、反向36.84）；
  NPU/host2为265.00秒（134.93、129.63）。均为预热后观察，任务读出/head/optimizer不是主要成本。
  原未分段校准CPU62.6–63.0、NPU232.5–236.6秒保留；不同观测时点、同步边界与共享负载不合并成同一次benchmark。
- 小形状B2/L8只有16个token、256个candidate。native NPU的设备profile记录168,589个kernel、
  7,308次`aten::_local_scalar_dense`、6,875次`aten::index`、3,603次`aten::_index_put_impl_`。
  插桩stage6.155秒，profiler Computing0.684秒、Free5.471秒；不把该比例直接称为无插桩利用率。
  相应未插桩步约4.09秒。Python路径另有独立trace，显示同类现象，不用它冒充native耗时占比。
- width128/256/512的B8/L16对应CPU/NPU均已具名测量；仅增宽不能消除该执行开销。
  public mixed-a隔离将NPU完整步从约23.54秒降到14.77秒；不认证跨placement严格数值等价。

这是已有实现的性能/可承载性问题定位，不是数学错误证明。profile、完整步、严格数值、离散、恢复和语言学习分轴。
synthetic输入只用于定位，不产生语言质量或scaling成功结论。

## 需要优先处理的执行路径

1. **Read/Describe实际按行执行。**
   `cpp/src/weighted_describe_batch.cpp:18`及`python/tidegraph/weighted/read_batch.py:23`
   在批量入口内逐行计算current、归一化、score/evidence/key和小投影。小型trace直接出现每候选一次的
   `[128]×[128,32]`、`[259]×[259,32]`及`[32,259]×[259]`调用，各256次。
   原注释说明为保持近零VJP的有限算术顺序而采用逐行方式；这不是未知现象，但已经阻碍下游实质训练。
   请求独立批量/融合实现，同时保留原严格参考和原失败。不能通过删除梯度检查解决，也不应将消除所有
   舍入差异作为明确标记学习探索的前置条件。

2. **KV分组、正支持和可见性处理形成大量动态索引及host取值。**
   `cpp/src/weighted_query_vjp.cpp:19`/`python/tidegraph/weighted/query_vjp.py:28`逐请求建立包含cache shape
   和long/short正质量分支的signature；存在设备布尔值到host的读取。
   `cpp/src/weighted_query_kv.cpp:8`和`:22`及对应`query_batch.py`再按行做bool索引、nonzero、长度分组。
   Aggregate signature也按请求取质量分支（`weighted_aggregate_vjp.cpp:22`）。
   请求批量提取必要控制元数据、复用已知int64可见布局、使用带mask/长度的张量布局或分段算子，
   避免对同一批候选逐行重复启动索引与回读；必须保存零质量分支、正支持及精确时间语义。

3. **声明的Full batch并非最终矩阵batch；独立根反向再次产生碎片化工作。**
   Query VJP的signature包含私有state各slot的shape，ragged历史先拆组，再进入Full的query/FFN等计算。
   `max_full_batch`是调度请求规模，不能认证后续所有矩阵的batch。小见证虽max_full_batch16，
   仍有大量`[512,128]×[128,1/2]`投影。原形状的max_full_batch约1000也不能据此认定已有千行GEMM。
   `weighted_query_vjp.cpp::QueryVjp::apply`及Python `_Query.backward`在独立字段/行根分组中再调用autograd。
   State及索引反向共同构成主要成本；请求将可共享投影从ragged attention分组中分离，
   采用张量化批量VJP/分段累积和明确连接mask，减少逐根select/stack/scatter与大量小图。
   本次没有证明全部剩余反向耗时都属于某一个类，不能只优化前向后声称训练问题已解决。

原始device profile共有177,891个设备任务（含copy等），其中AICPU 2,404项/255.73ms、
AICore 4,681项/10.34ms、vector/mixed任务占大多数。完整路径并非主要时间用于大矩阵计算；
不能仅依据总参数量、NPU算力或packed标签估算训练速度。
这些数字属于该插桩步，不外推成所有配置固定比例，也不把host self时间的嵌套总和误作critical path。
AICPU任务主要是`Index`和`IndexPutV3`，各1,170次，共252.56ms；这里的AICPU是设备侧处理单元，
不能误称为主机CPU fallback。NPU索引反向路径值得单独改善，而非只优化GEMM或关闭确定性策略。

## 有界交付与验收建议

先在现有公共接口内交付一个具名优化路径，优先FC16/KV/quad/FP32；不要求重写全部执行器或全平台矩阵。
基座已有resident训练owner可作为候选方案，但它有独立VJP、optimizer、任务根和恢复边界，
不能把切换preset当作已接入/已加速，也不要求下游复制执行核心。

- 提供实际operator形状/分桶、设备回读、索引、反向以及完整步成本；区分调度request batch和矩阵batch。
  下游复用当前B2/L8定位输入、B8/L128实际训练形状，必要时B32；使用同head、loss、optimizer、
  精度和完整工作量，预热、profile、计时分开。保留原阈值及所有路径失败。
- 小型独立验证包含输入与共享参数梯度、各私有状态/字段根、None与连接零、未使用参数不更新、
  optimizer槽、完整raw/int64状态、区域支持及消息、应用新进程恢复。
  已知非有限/错误分支/共享或恢复错误阻塞相应路径；有限数值超差经有界定位后可单列探索，
  不要求所有正式训练步骤运行CPU参考。
- 不以关闭必要检查、伪造工作计数、替换原问题、减少目标位置或隐藏返回/恢复代价取得速度标签。
  检查允许批量化或明确分层；诊断和工作量观测应有可选择的低开销路径。
- 先证明完整训练步有实质改善并解释剩余瓶颈，再由下游使用新私有安装和新固定作业进行语言学习。
  36h内完成4,194,304 token仅纯训练就需要约32.4 token/s；当前同形状CPU约16.1、NPU约3.86，
  实现效率已经直接限制数据曝光。这个要求与兑现理想稀疏加速比不同。

复现源为下游冻结提交`d90e733cd1b1d2d13f5da578450f17392f4f49a0`（阶段/trace）和
`b4bc62bda4f1424604c4eb0d1736dbe2dea7b735`（具名隔离），入口
`scripts/diagnose_campaign_execution.py`，配置`configs/studies/language/v6/scale-fc16-kv-quad.json`。
每个精确argv、资源、environment hash和构建输入见摘要中列出的`artifacts/jobs/v6-diag-*/job.json`。
现有训练继续使用其原固定源码；本请求不授权本线程改写基座，也没有发送外部消息。
