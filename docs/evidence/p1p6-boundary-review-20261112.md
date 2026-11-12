# P1–P6 生产边界审阅（2026-11-12）

审阅源码固定 `55a3074a441d77720cf00e44433bd702cd09e175`，数学固定 `0770593838a37c790a568226930567ed60f99b83`。
[逐文件/符号与 SHA256](p1p6-boundary-reviews-20261112.json)记录24个实际检查的源码/测试文件，
以及3个文档的具体段落。已有审阅有重叠；此数目不能加到旧清单后宣称全仓已审。
自动2157文件清单仍只是inventory。C++符号范围由人工记录，本轮没有执行Clang符号提取。
新resident代码和后续position-policy修改不在该固定审阅范围；没有新增硬件或性能结论。

## 检查标准与已有覆盖

| 边界 | 已确认的实际行为 | 不能据此推导的结论 |
| --- | --- | --- |
| 默认入口 | Python/native的record入口仍可处理一般PDG/TimedDAG；`run_ports/run_positions`只绑定完整位置Settle。能力判断先于数值执行，失败不触发静默换算法 | 一般PDG不会自动获得整段sequence批次；列API不是任意图/任意中间根接口 |
| 五记忆 | KV、affine、delta、gated_delta、SSD都有typed prepare/Read/Full；真实局部时间及卷积槽保持int64。KV已接跨节点组；循环模块按节点处理整个列窗口 | 循环模块没有跨节点融合不等于缺少batch实现，也不等于已达到最优吞吐 |
| 区域 | Python/C++列函数覆盖十种发送、评分/证据/key与适用正则。真实候选数分组，不用padding改变数学支持。区域仍贯穿描述、控制、Full与下游消息 | 不能把零温小数发送叫硬Top-k；不能把求解器迭代图当精确解映射VJP |
| 调度/根 | 同秩独立区域、非齐端口、零与缺席、重复边均有独立测试入口。ragged区域VJP按实际节点关联及有限数学分支保留None/零 | 显式普通列根不能冒充旧逐事件中间根；公共字段和数学投影有不同资格 |
| standalone | 公共benchmark确实调用C++ WindowExecutor，独立调度、三步SGD、raw checkpoint及新进程续接有实现和定向测试；完整观察保留records | SDK组件通过不能替代真实benchmark运行；任务loss对照不认证全部VJP根 |
| 下游入口 | 内置outputs/metrics任务选择列边界，输入一次按列准备，反向后才解码CPU观察；任意插件仍明确使用records | output-only插件声明本身不授予改写其loss/独立根的权利；插件张量协议仍须明确设计 |

`test_settle_columns.py`与`test_window_execution.py`分开覆盖十种发送的小型整链，
`test_regional_groups.py`检查不同实际候选集合、独立根、零cotangent、别名及版本守卫；
`test_weighted_allocation*`与模块公式测试承担局部数学独立锚点。
这些是代码中可定位的检查范围，不是本轮重新运行全部矩阵的声明。

## 已定位缺陷与有限机会

1. **真实检查设施回归。** 固定源码的`check_query_sdk.py --return-mode outputs`
   默认已命中standalone列路径，却仍要求`query_packed-v2_nodes`和`joint_state_vjp_events`。
   新列路径真实报告的是`column_windows/state_column_calls/region_column_calls`，不能补假计数。
   后继修复按`effective_execution`分支；complete与显式records保持原检查，columns另做三步
   独立scalar Streaming任务loss/work对照及原有精确恢复。独立对照在候选计时之外，
   不把任务投影或统计计数当成完整根证明。修复及实际开发验证由后继提交证据单列。
2. **明确扩展限制。** `positive-floor-v1`是额外数值模式，Python record可用；
   native/standalone仍拒绝。55a已修复standalone手写fixture忽略此字段的问题。
   不能把拒绝报告成支持，也不把它等同固定上游正式候选缺失。
3. **有限性能机会。** native Aggregate仍按目标节点执行来源变换，真实同输入/同参数的
   跨目标计算可进一步复用；不同SrcClass/原子/梯度使用者必须保留。循环模块跨节点合批亦可
   继续按实际收益推进。这些是P5的具体边界，当前没有新计时收益声明。
4. **主机阶段成本。** 零温sorted fill按候选轴N循环，仍同时处理R个实际行；
   entropy求解最多80次有界Newton/bracket及活跃检查。它们没有按B×L逐事件重放，
   但仍有主机驱动/同步与算子成本，不是resident实现。残差/正支持拒绝保持，未静默裁剪。
5. **资格缺口不混同功能。** 本轮已审边界不能认证所有合法组合、尺度、GPU、新resident整链、
   大模型吞吐或普通异步跨NPU。三种stock队列探针即便结果不同，也不能代替完整图训练/恢复。

## 导航漂移

以下问题定位到固定源码；随本报告交付的导航修正不改写历史证据：

- `docs/execution-capabilities.md`的Typed tensor windows行仍称multiple-payload-device migration待做；
  实际已有owner/half实现及固定证据。应分开已实现的host多owner、异步资格和新resident重写。
- 同页当前环境段只写Python Torch2.12/CANN9.1、SDK2.10/CANN9.0；
  当前9.2开发入口和新证据应可直接找到，旧报告原环境身份仍保留。
- `docs/core5-study.md`“当前实测设置与结果”只指向11-10加法结果；
  README“当前core-5性能”只指向10-29自然规模。应补当前张量窗口、混合、多owner及节点/区域报告的入口。
- README把全部真实GPU正确性/常驻/多卡/内存/性能写成待验，与能力页所列10-31 RTX3090固定范围矛盾。
  应表述历史已验范围与本轮新增源码尚待GPU验证，不能删除历史真机事实或外推到新源码。

README、core5-study与execution-capabilities已补当前张量路径/结果导航、已实现多owner范围、
9.2开发入口与历史GPU限定资格。architecture、semantic-map及reproduction连接实际默认入口、
SDK资格、旧计数误判回归与修改时同步项；原报告的失败和源码身份不变。

## P6交付边界与本次未审层

当前治理入口能够完成四条具体路径：配置→实际分派/局部模块；行为→数学与独立检查；
结果→固定源码/命令/材料/失败；修改→契约、消费者、测试与证据维护项。
入口分别由core5-study、architecture、semantic-map、reproduction及evidence索引承担，
不新增并行台账。这是可验收的有限维护能力，不要求把所有历史工具重审后才交付。

该55a审阅没有逐函数重审以下层，不能把24文件覆盖扩展到它们：

- KV跨节点Read/Full组及底层Attention/矩阵VJP实现：本次检查其调用契约和分派；
  `kv_column_groups/column_query_ops`与C++同层源码的具体实现审阅应随其实现提交及定向资格记录。
- 后继common-origin位置算法、新resident sequence/local/flow/owner实现：不属于55a固定字节，
  各自在新实现的审阅、独立公式与整链资格中验证，不能从本报告继承。
- 未改动的一般PDG/SCC调度内部、checkpoint底层codec、optimizer与设备runtime：本次沿用既有
  契约和固定证据，仅核对新入口连接；未来实际改变这些层时补影响范围审阅。
- 旧core-3特化、历史诊断/benchmark及下游应用：保留各自历史证据，未自动纳入本轮重写审阅。

因此当前生产导航与维护入口可独立交付；新重写层的实际实现/资格仍按ROADMAP中的功能验收，
不会转化成无界的“全历史文件尚未重审”阻塞。
