# 完整 weighted 常驻图

固定来源为 [0770593](upstream.json) 的三种图正典与带权局部教材。
公共入口 `tide/resident_weighted.h::WeightedFlow`，CMake 目标 `tide::weighted_resident`；已资格范围为 FP32
独立 C++ 推理 owner，[固定资格](evidence/core5-resident-weighted-flow-20261028.md)分别记录 standalone 与 Python-owned NPU。完整训练、半精度和跨卡各自验收。

## 函数与状态绑定

`node_binding/region_binding` 读取安装函数的真实配置；禁止按 profile 名称或权重
形状猜记忆族、评分或发送函数。`WeightedBanks` 显式记录每个数值角色的注册参数名、
原形状和切片；同一参数的多个角色仍有同一外部身份。私有状态按节点、样本分开。
来源分类保留每个物理原子；SourceDomain 冲突单独拒绝。不同来源隐层宽度以零填充
A/B/e 批量调用；填充不是参数，低秩适配在设备上计算，原始角色仍分别保存。

`weighted_flow_local.cpp` 组合实际 Aggregate、Upd、Mem、Describe、SelStep、Full。
五记忆均接入，各函数配置适用性由原公共校验与局部组件共同承担。Read^0 读取绑定
初态，Read^- 读取旧态，Read^+ 读取候选态；Full 始终读取候选态。发送载荷为 `(c,G)`，
广播逐物理边与逐输出端口返回，强度不重复乘入 G。所有候选保留更新，零接收保留
完整原始状态。参数拷贝与 int64 原始时钟分开，不用有效状态投影代替恢复。

公共FP16源码扩展显式接CPU half参数/边界，队列、pending、raw与发送载荷实际为half，
参数角色bank是实际half参数的FP32精确拓宽。Aggregate、content、Mem、Describe、控制、
Full的未存储G及保存值均为FP32；Update最终raw和每次发送的(c,G)各舍入一次。
这是声明的混合精度算术，不宣称FP16 GEMM或复现eager每算子的half舍入顺序。
局部组件的独立量化公式与全图独立eager调度分别验收；同一浮点容差不能放宽离散支持。
发送前后用FP32检查实际存储值，非有限或正c下溢成零拒绝(error24)，G的数值下溢允许。
控制c与实际payload中的舍入c分别保存；解码逐边/端口返回实际payload，不能重算替代。
raw过期保留、次正规数、int64、None/零及容量失败保持局部契约。完整FP16资格另记。

## 数学到执行的对应

共同定义见 [架构](upstream/tide-core-5/local-semantics/weighted-message/architecture-learning-note.md)、
[有限 cut](upstream/tide-core-5/positive-delay-graph-finite-cut-learning-note.md) 与
[在线作用块](upstream/tide-core-5/timed-dag-chunk-prefill-learning-note.md)。
各局部函数的参数尺寸、值域及指定一阶 VJP 沿本模块引用的局部契约追溯。

| 定义/依赖 | 实际入口与返回 | 独立核验 |
| --- | --- | --- |
| 物理纤维与 SrcClass；所有原子各一次 | `bind_weighted_flow_source`、`append_weighted_node_input`；来源行、真实 payload、参数 owner 与逻辑 slot 分开 | packing 测试；配置检查的重复边、异宽来源、低秩与 origins |
| Agg → Upd → 所选 Mem → Describe | `Impl::append_nodes`；原始 old/proposal 与命名描述各自保留 | 五记忆/三快照完整图；各组件公式与独立字段测试 |
| 完整区域 SelStep → 所有候选保留 → 活动 Full | `append_regions/append_full`；按稳定 node ID 返回 c、活动位及未缩放 G | 15配置全图、严格支持和完整记录比较 |
| 当前实际边界的完整作用块 | `DeviceReady` + `QueueTransaction`；队列、封闭就绪、region/sample/time 计划均在设备上 | 独立 CPU Greedy；反馈、非对齐 DAG、缺席端口和流式/批量 |
| 完整切面与逐边/端口返回 | `checkpoint/decode`；分离 float/int64 原始状态、history、pending、ledger、Event 和各端口 | 分段/新owner、磁盘精确 codec、巨大时间与错误边界 |

`WeightedFlow` 的参数绑定在构造时快照化，C++ 调用者后续修改原模型不改变该 owner；
Python 适配器额外拒绝模型参数版本变化。共享参数身份随快照保留，重复计算与私有状态
仍分开处理；前向不承诺消除跨节点的相同计算。固定图身份和 checkpoint 版本沿用共同契约。

## 在线调度与边界

`DeviceReady` 在当前队列和完整外部 seal 上求封闭候选；`weighted_flow_packing.cpp`
将真实完整纤维和区域按静态节点/样本布局打包。每个区域每样本一轮最多一个时间，
这是局部组件尚未暴露跨时间递归批次时的合法后备。prefill 可同时处理独立完整区域；
没有 CPU 轨迹输入、逐事件主机分支或按测试数据准备的事件表。
局部数值按每节点的sample行执行，缺席/非活动行由组件守卫绑定为无连接的安全填充。
`candidate_updates/full_evaluations`统计实际数学作用，`physical_local_rows/physical_full_rows`
另记提交给局部组件的物理行数；未活动行可能仍占厂商批量算术，稀疏发送不等于全部计算减少。

所有容量与数值检查在 queue/state/history 提交之前。`queue` 限制同时在途原子，
`cache` 限制每私有状态的实际原始缓存，`stages` 限制单窗口作用轮数与诊断 journal，
均不限制累计逻辑时间。NPU将局部数值阶段放入独立的persistent程序，通过设备通知请求、
完成与结束握手；主机只提交整个窗口的程序集合，窗口内没有主机事件推进。
这使完整组合不受单个CANN程序任务缓冲区限制；各局部程序仍受厂商容量约束，
拒绝保留原错误。CUDA将这些阶段追加到同一conditional graph，真机资格独立记录。失败 owner 不再执行；使用先前完整 checkpoint 创建新 owner。
非法外部输入在提交前可重试。总设备张量与 vendor workspace 分别有准入预算。状态与完整窗口 journal 在大额分配前静态预检；
各组件使用剩余额度，按实际唯一存储计入未被执行路径读取的初态与参数 bank。
NPU通知握手保证coordinator与所有局部数值阶段互斥使用算子workspace，因此共享一个按最大需求
分配的arena。发出完成通知后直到下一请求前只执行无workspace的控制任务；不允许并行数值阶段借用此arena。
静态非连续视图用连续 gather/scatter 拷贝，未放宽底层设备算子的连续存储契约。

`weighted_flow_records.cpp` 仅从实际计算张量、整数元数据、选中 Full 和设备消息恢复
完整 Event/Result；解码不执行参考递归或数学函数。结构性广播端口映射在边界恢复。
`checkpoint()` 恢复所有实际 raw 字段、时间戳、presence、history、pending、ledger。
使用既有 Continuation/图身份，图 v13 与 checkpoint v5 不变；磁盘序列化使用现有契约。

## 独立核验与范围

`test_resident_weighted_flow_packing.py` 独立断言重复类别、SourceDomain 冲突、缺席、
同一私有状态多时间拒绝、原始巨大时间、origin 排序、完整区域与提交前错误保留。
`weighted_flow_check.cpp` 独立使用 CPU Greedy 调度与标量局部函数，候选先自行执行；
比较完整返回/状态/控制/逐边/逐端口/history/pending，覆盖五记忆及异构、FC/HB、
非对齐 DAG、PDG 反馈、三种读取快照、streaming/prefill、缺席与零消息、切点恢复。
局部公式的独立数学锚点从各 resident-weighted-* 文档进入。

这些检查不自动认证 retained VJP、optimizer、FP16、peer、吞吐或 CUDA 真机。

Python 适配器为 `tidegraph.weighted.WeightedResident`，使用匹配的 native 与 resident
库，显式接收 `device`、CPU FP32/FP16 模型、batch_size、Continuation 和容量。
`advance(inputs, stop=..., sealed_until=...)` 要求完整 seal；不运行 Python 参考。
`manifest()` 保存请求/生效容量及设备、dtype 和库身份；`checkpoint()` 返回通用
Continuation，可通过原 `tidegraph.checkpoint.save/load` 序列化。参数版本或对象替换
使旧 owner 拒绝执行，用户须从显式切面构造新 owner。
manifest另记raw/payload dtype、compute/scoring/adjoint FP32，不静默替换用户模型dtype。

公开 Python 适配器可独立构建：

```sh
python scripts/build_device_control.py --core-build NPU_CORE --build-dir NEW_BUILD \
  --python-component weighted --ascendc-soc Ascend910_9392 --jobs 2
```

CUDA 架构须为显式计算能力代码（80及以上）；`native/all`、低于80和
`-real/-virtual`后缀提前拒绝。Torch legacy FindCUDA 与 CMake CUDA 使用同一请求，
避免无GPU主机从wheel自动继承额外架构。真实GPU运行仍单独验收。

NPU_CORE 必须是当前 Python/Torch 匹配的 Python-owned 原生库。CUDA 使用显式
`--runtime python --cuda-architectures '80;90;100'` 并省略 Ascend SoC。
`_tide_weighted_resident` 通过独立 `libtide-weighted-training` 与 `libtide-weighted-flow` 提供两种owner；
loader核验native、适配器和两个公共库的摘要，另一条resident支线仍用原入口。旧仅推理build仍按原manifest加载。
这里只是构建命令，实机资格见STATUS。

`weighted_flow_config_check.cpp` 的15配置全图核验涵盖十种发送、两种key、match/grouped、
四种正则分配形态、KV qry/原始时间位置、异宽来源与低秩适配；公共 Python 检查在
`test_resident_weighted_flow.py`，包括三种拓扑域的端口、严格完整记录、参数版本和磁盘恢复。
这些检查的定义不等于已运行通过，仍按固定源码证据记录实际结果。

公共standalone安装为独立`TideWeightedResident`包，导出`tide::weighted_resident`。
只构建weighted目标时，使用`cmake --install DEVICE_BUILD --prefix PREFIX --component weighted`，
并单独安装匹配core；无需构建旧resident库。独立消费者只包含安装头，不引用仓库私有fixture：

```sh
python scripts/build_resident_consumer.py --core-build CORE --resident-build DEVICE_BUILD \
  --output-dir NEW_CLIENT --component weighted
NEW_CLIENT/consumer/weighted-resident-consumer --device=npu:0 --dtype=float32
```

该消费者用`gamma=1`、残差强度为零的单节点手算`(c,G)=(m,z)`，验证公开配置跨库传输与两段新owner续接。
它是安装/调用边界检查，完整图范围仍由上面的独立调度与公式证据承担。
