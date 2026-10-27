# Weighted 公共训练 owner

本页定义公共层契约；单设备FP32的session、完整字段视图、共享参数、两优化器与精确恢复
已取得[343429d固定资格](evidence/core5-resident-weighted-training-20261028.md)。FP16、peer和性能分别验收。
固定数学入口为[共同局部架构](upstream/tide-core-5/local-semantics/weighted-message/architecture-learning-note.md)
及[PDG有限切面](upstream/tide-core-5/positive-delay-graph-finite-cut-learning-note.md)。

公共入口声明在`include/tide/resident_weighted_training.h`。单设备FP32/FP16 owner拥有参数generation、
连续保留窗口、显式一阶backward、SGD/AdamW更新及完整切面导出。loss、任务头和数据游标由调用者拥有。
前向窗口内的就绪、选择和推进仍由设备引擎决定；窗口边界的根打包不是参考预跑。

`WeightedTrainingWindow`以token和[start,stop)定位实际窗口。fields中的每项有浮点 values与bool
present；每个present行是独立数学根。ragged字段另有valid前缀mask，空字段仍可有连接零根。
metadata保存精确int64坐标、时钟、计数、长度和原始状态整数槽，不混入浮点buffer。

| fields键 | 数学返回/快照 | 行域 |
| --- | --- | --- |
| `node.v.content` | 实际Aggregate完整内容 | stage×sample候选 |
| `node.v.control` | c，单独一个浮点坐标 | stage×sample候选 |
| `node.v.full` | 未缩放G | stage×sample活动者 |
| `node.v.describe.NAME` | 实际配置下的Describe字段 | stage×sample候选 |
| `node.v.old.NAME` / `proposal.NAME` | Upd旧态/候选态的命名raw浮点字段 | stage×sample候选；ragged槽由原始length标记 |
| `node.v.final.NAME` | 右切面完整raw字段 | sample已有状态 |
| `outputs` | 每物理输出端口的(c,G) | stage×sample×port |
| `messages` / `fibers` | 实际逐边发送/收到的物理原子 | stage×物理队列槽 |
| `pending` | 右切面所有在途消息 | 物理队列槽 |

这些是有明确mask的窗口布局，不把连续buffer叫作已完成packed优化。
outputs.coordinates为[sample,time,port]，消息坐标保持完整六元身份；布局次序不充当逻辑时间。
Raw字段保持自身shape，ragged字段加capacity轴。metadata中的`node.v.{old,proposal,final}.integer`
保留raw整数表示，不能用effective state重建checkpoint。别名端口可分别种根，各贡献累加一次。

`WeightedCotangents`省略字段表示None。提供的values/connected必须成对、匹配布局/设备，分别为FP32/bool且
无autograd依赖；不能连接不存在的行，连接的ragged padding须为零。未连接毒值不参与VJP。
`weighted_training_view.cpp`只从实际快照创建字段；`weighted_training_roots.cpp`按静态字段/端口绑定
根并区分G和c，不重复乘发送强度。消费者可在另一个autograd上下文计算任务loss的cotangent。

同一generation的所有窗口按前向顺序提交backward，设备逆序计算各窗口并通过实际raw/pending
连接切面。参数别名按canonical owner汇总；`weighted_training_parameters.cpp`使用完整trainable
布局，即使某个owner没有梯度也保留其槽，None由连接位表示。此布局不允许参数值相等就合并身份。
所有实际使用的role bank都从同一master发布，私有状态不参与参数共享。

公共token、step/accumulate/detach、SGD/AdamW发布与checkpoint已接入；FP32资格见上列证据。
保存的局部/窗口数学与独立锚点沿[窗口](resident-weighted-window-vjp.md)和
[保留切面](resident-weighted-retained.md)追溯；公共完整返回/训练/恢复由上列独立资格覆盖。

公共Python入口为`WeightedResidentTraining`，原生为`tide::device_online::WeightedTrainingSession`，
安装目标`tide::weighted_training`。所有方法要求no-grad；任务loss在独立autograd上下文计算。
advance只执行设备窗口并导出实际字段，不强制CPU完整解码；`result()`显式解码，更新后旧journal失效。
`backward`消费同一generation的全部窗口；`accumulate`保留参数梯度并切断状态梯度，最后一批也须显式累积；
`step`按canonical owner更新，再发布每个实际角色bank及CPU参数镜像。None不触发权衰减/step计数，连接零参与更新。
所有冻结/部分冻结参数保持身份。调用者改写原参数、已保存视图（含mask/int64元数据）或待消费梯度均拒绝。
`manifest()`分别记录请求/实际设备、容量和optimizer分组；恢复时实际optimizer来自checkpoint。
原生`optimizer_kind()`/`optimizer_groups()`只读取配置，不导出状态或执行参考计算。

给定已绑定的`binding`、窗口输入`inputs`及完整封闭上界`stop`，消费一个窗口的示例：

```python
with torch.no_grad():
    session = WeightedResidentTraining(
        binding.graph, binding.model, device="npu:0",
        native_library=CORE_BUILD, resident_library=DEVICE_BUILD)
    window = session.advance(inputs, stop=stop, sealed_until=stop)
    output = window.fields["outputs"]
    with torch.enable_grad():
        payload = output.values.detach().clone().requires_grad_()
        # 此loss选择数值投影cG；公开载荷本身仍是(c,G)。
        projection = payload[..., :1] * payload[..., 1:]
        loss = torch.where(output.present[..., None], projection,
                           torch.zeros_like(projection)).square().sum()
        bar, = torch.autograd.grad(loss, payload)
    gradients = session.backward([session.cotangents(window, {"outputs": bar})])
    update = session.step()
    if not update.applied:
        raise RuntimeError(f"optimizer refused: {update.refusal_code}")
    session.save(NEW_CHECKPOINT_PATH)
    session.close()
```

这里只选择output loss；节点字段、逐边消息、pending和原始状态也可独立种根。
多窗训练按前向顺序收集每窗cotangents，一次提交`backward`，再更新参数。

容量准入使用forward/retained/backward/optimizer四个张量预算之和不超过total的保守分区。
retained在advance前按固定journal、raw布局、物理端口数预留，并验证实际窗口视图；反向最多同时持有两个窗口adjoint，
另预留根打包、返回字段、参数部分与跨窗bridge。optimizer预算含完整trainable布局、master/slots、累积及发布表。
预算是owner张量存储的上界，不是厂商allocator峰值；每个程序的operator workspace独立受限。
调用者长期保留已消费的返回张量由调用者预算，不能利用它们重用已消费token。

checkpoint仅在detach或完成step后的完整切面导出，不保存活tape、loss、RNG或数据游标。
原生复用`ResidentTrainingCheckpoint` schema1并用mode=`weighted`区分数学绑定；要求zeta=1。
Python磁盘schema=`tide-weighted-resident-training-v1`，通过现有fsync+排他发布保存CPU张量；
恢复预检所有别名、trainable名称、完整offset、named/master数值和optimizer槽，再构建新owner。
原图identity/checkpoint版本不变，精确恢复只由同栈独立轨迹证据支持。

独立检查`weighted_training_check`包含FC/HB/非对齐DAG/PDG反馈、重复输出端口、全部字段/初态/输入根、
空窗与三窗链、两优化器四次更新、部分/全部冻结、None/零、恢复分支和生命周期拒绝。
`tests/test_resident_weighted_training.py`另核对Python-owned公开调用与磁盘恢复。
这些是验收定义；是否通过仍以固定源码证据为准。

独立安装：先安装匹配core及`--component weighted`，再安装`--component weighted-training`。
`find_package(TideWeightedResident CONFIG REQUIRED COMPONENTS training)`提供`tide::weighted_training`。
只安装weighted推理组件仍可独立消费。公共消费者构建：

```sh
python scripts/build_resident_consumer.py --core-build CORE --resident-build DEVICE_BUILD \
  --output-dir NEW_CLIENT --component weighted-training
NEW_CLIENT/consumer/weighted-training-consumer --device=npu:0 --dtype=float32
```

消费者的单节点手算恒等VJP、无作用参数None和新owner切面恢复不代替完整图训练资格。

FP16源码路径沿用[完整图精度边界](resident-weighted-flow.md)：raw、实际输出/消息/纤维/pending为half，
content/Describe/control与舍入前的Full G为FP32。每个浮点字段的cotangent以及参数、输入、raw伴随
都为FP32，不随公开字段的存储dtype变化；half cotangent明确拒绝。存储Cast的声明VJP为identity。
优化器保留FP32 master及槽，发布时先将master舍入half，再精确拓宽到FP32计算bank；
checkpoint同时保存实际half named参数与FP32 master，恢复要求named等于master舍入后的值。
manifest分列raw/compute/scoring/adjoint/master dtype。此源码扩展的实际资格单列，不沿用FP32通过结论。

半精度全图参考使用原有CPU数学函数和独立调度，只在raw和发送payload边界显式half存储。
`weighted_precision_reference.{h,cpp}`与`tests/weighted_precision_reference.py`是检查辅助，
不进入生产调度或解码。完整记录先核验每类dtype，再对half精确拓宽后的全部字段比较；
没有丢弃raw字段或元数据。训练的`node.v.full`根另读独立公式的舍入前G，不能用实际payload替代。
`test_weighted_storage_reference.py`和`test_resident_weighted_storage_roots.py`的单节点KV手算
独立核对key/summary/weight根、RMS Jacobian、参数外积、FP32输入伴随及无关参数None。
`weighted_training_accumulation_check.cpp`独立用两次已声明detach的梯度之和计算一次SGD结果，
并改写已消费的公共梯度以核对内部累积快照独立。Python FC另外在新进程读取磁盘切面并比较后续一步完整checkpoint。
显式optimizer group须列出参数名称；空组有其字面含义，未提供groups才默认包含全部trainable owner。
每个共享owner只列一次：使用`registry.owners()`的canonical字段；`registry.names()`列出所有别名，
直接把它当optimizer参数列表会正确触发重复owner拒绝。

从干净固定提交复现standalone定向资格（构建环境/持久服务见[开发入口](development.md)）：

```sh
python scripts/build_device_control.py --core-build CORE --build-dir NEW_DEVICE_BUILD \
  --runtime standalone --ascendc-soc SOC --jobs 2 \
  --checks weighted-copy weighted-saved-window weighted-retained weighted-training weighted-flow weighted-flow-config
python scripts/verify_device_control.py --build-dir NEW_DEVICE_BUILD --device npu:0 \
  --output-dir NEW_CHECKS \
  --checks weighted-copy weighted-saved-window weighted-retained weighted-training weighted-flow weighted-flow-config
```

CUDA构建使用匹配CUDA core，改用`--cuda-architectures '80;90;100'`并去掉Ascend SoC。
设备运行命令需实际GPU，构建/loader不代替它。Python-owned使用自己的core：

```sh
python scripts/build_device_control.py --core-build PYTHON_CORE --build-dir NEW_PYTHON_DEVICE \
  --runtime python --python-component weighted --ascendc-soc SOC --jobs 2
TIDE_BUILD_DIR=PYTHON_CORE TIDE_RESIDENT_LIBRARY=NEW_PYTHON_DEVICE TIDE_RESIDENT_DEVICE=npu:0 \
  python -m pytest -q tests/test_resident_weighted_training.py tests/test_resident_weighted_flow.py
```

恢复比较对`present`与ragged `valid`定义的全部逻辑字段逐位精确，含过期但仍保留的raw条目；
未使用的journal行和容量padding没有指定值。另核对完整解码结果及精确参数/optimizer/checkpoint，
不能由一个有效状态投影替代完整raw检查。
