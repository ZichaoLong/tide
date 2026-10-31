# Weighted 公共训练 owner

本页定义公共层契约；单设备FP32的session、完整字段视图、共享参数、两优化器与精确恢复
已取得[269ec4d固定资格](evidence/core5-resident-weighted-training-20261028.md)。FP32/FP16公共完整图、
训练及精确恢复另取得[3416567固定资格](evidence/core5-resident-half-training-20261028.md)；peer和性能分别验收。
[样本副本与共同更新](resident-weighted-peers.md)定义新增跨卡归约、精确optimizer一致性和全体提交边界；其资格另记。
固定数学入口为[共同局部架构](upstream/tide-core-5/local-semantics/weighted-message/architecture-learning-note.md)
及[PDG有限切面](upstream/tide-core-5/positive-delay-graph-finite-cut-learning-note.md)。

公共入口声明在`include/tide/resident_weighted_training.h`。FP32/FP16 owner拥有参数generation、
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
`manifest()`分别记录请求/实际节点放置、设备、容量和optimizer分组；恢复时实际optimizer来自checkpoint。
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
追加`--node-placed`时需要逻辑设备0/1，先在peer上执行同一手算VJP，再在coordinator上
恢复下一窗；它检查安装包的新放置构造/恢复重载与实际返回的放置列表。
这项调用边界检查不代替多节点共享参数发布与完整根资格。

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
  --runtime standalone --ascendc-soc SOC --jobs all \
  --checks weighted-copy weighted-saved-window weighted-retained weighted-training weighted-flow weighted-flow-config
python scripts/verify_device_control.py --build-dir NEW_DEVICE_BUILD --device npu:0 \
  --output-dir NEW_CHECKS \
  --checks weighted-copy weighted-saved-window weighted-retained weighted-training weighted-flow weighted-flow-config
```

CUDA构建使用匹配CUDA core，改用`--cuda-architectures '80;90;100'`并去掉Ascend SoC。
设备运行命令需实际GPU，构建/loader不代替它。Python-owned使用自己的core：

```sh
python scripts/build_device_control.py --core-build PYTHON_CORE --build-dir NEW_PYTHON_DEVICE \
  --runtime python --python-component weighted --ascendc-soc SOC --jobs all
TIDE_BUILD_DIR=PYTHON_CORE TIDE_RESIDENT_LIBRARY=NEW_PYTHON_DEVICE TIDE_RESIDENT_DEVICE=npu:0 \
  python -m pytest -q tests/test_resident_weighted_training.py tests/test_resident_weighted_flow.py
```

恢复比较对`present`与ragged `valid`定义的全部逻辑字段逐位精确，含过期但仍保留的raw条目；
未使用的journal行和容量padding没有指定值。另核对完整解码结果及精确参数/optimizer/checkpoint，
不能由一个有效状态投影替代完整raw检查。

## 单样本节点放置与发布

C++两种构造（初始continuation和training checkpoint）均新增末尾`vector<at::Device> node_devices`
重载，原构造保留；Python `WeightedResidentTraining(..., node_devices=[...])`使用同一原生owner。
None/空C++列表保持所有节点在coordinator；定义域和完整区域语义沿[节点放置](resident-weighted-placement.md)。
独立运行范围见[固定窗口反向/公共训练资格](evidence/core5-weighted-node-training-20261028.md)；旧单设备训练和前向放置报告保持各自范围。

公开窗口字段、cotangent、参数梯度与跨窗桥仍在coordinator；快照保存物理owner，
完整raw状态在边界复制，窗口内的局部反向仍回到原owner。参数共享与私有状态保持分开。
`node_devices()`返回实际构造的放置；Python manifest记录请求/生效列表。放置不进入数学identity，
完整CPU checkpoint可在另一合法放置下恢复；optimizer/master、alias、trainable集合和精确raw解码规则不变。

`WeightedTrainingPublication`按实际bank设备分组。canonical master和SGD/AdamW槽只在coordinator
更新一次；`DeviceOptimizer`先检验所有owner的有限性和各自FP32/FP16可表示性，再共同采用。
各远端程序接收同一错误字和实际master值，只在成功时按绑定payload dtype舍入并发布所有角色bank；
原始私有状态不参与这次发布。CPU参数镜像与generation只在所有设备完成后更新。
20/21拒绝保持master、槽、bank与generation，梯度可显式detach；设备通信/运行异常使owner失效，
必须从先前切面重建，不能返回部分更新成功。

发布前按分组数、完整master副本和描述表预检optimizer预算的四分之一；该选择不宣称最小通信。
各设备程序全部提交后才等待。样本副本共同更新可再把同一canonical决策发布到每个样本的节点放置，
`communication_bytes`亦提前计入这些副本，返回传输量包含两层通信；各owner四分区预算仍生效。

独立检查：`weighted-node-training`两dtype各五组合/20更新，全部字段根、None/零、全部/部分冻结、
重复端口、三窗含空窗、SGD/AdamW、累计与交换节点放置后恢复；额外用FP64算术确认一个
有限梯度会超出目标存储范围，验证共同拒绝及下一窗实际bank仍使用旧参数。
`weighted-node-replicas`把两个独立样本owner各自跨卡放置，核对共享梯度和、完整私有根、恢复及slot不一致拒绝。
Python `test_resident_weighted_node_training.py`另检查四组合×两dtype、独立Python调度/autograd、
三次更新、磁盘恢复到另一放置及KV/FC新进程续接。这些检查在上述固定源码报告中分别记录；不外推全部配置或GPU真机。

## 反向构造与运行的观测

`WeightedGradients.statistics`增加`adjoint_construct_us/root_seed_us/adjoint_run_us/
window_bridge_us/parameter_reduce_us`：当前调用的主机steady-clock微秒，run包括该路径的等待。
构造、实际执行、跨窗桥与参数归约分列；不把这些主机跨度称为设备kernel时间。
数值/optimizer/checkpoint返回契约不变。

当前adjoint绑定实际snapshot的journal、root、参数地址，生命周期结束后关闭。只按shape
缓存整个对象会读旧窗口/参数或保留已经消费的根；可复用的程序模板需要显式重绑buffer与
版本/连接检查，不能直接保存上次adjoint绕过构造。本轮用上述观测量判断其成本。

上述resident增量与实际消费者的[57308e7固定资格](evidence/core5-resident-boundary-20261029.md)
分别记录SDK、Python-owned、原失败和初始化边界。后续[RTX3090单卡固定资格](evidence/core5-cuda-resident-20261031.md)
另验CUDA FP32/FP16组件与公共训练/恢复；多卡硬件条件、设备时间线及性能分别验收。

[57308e7有限成本证据](evidence/core5-boundary-performance-20261029.md)已取得实际构造/执行/归约分段；
它不授权跨窗口复用绑定旧地址的adjoint。优化器canonical布局的1/16子分区和总预算须分别准入。

[紧凑纤维资格/成本](evidence/core5-compact-fibers-20261029.md)覆盖相同布局的窗口VJP与单/双卡恢复。
medium实测反向构造仍约1.057秒，执行/等待约0.700秒；未缓存旧adjoint。
较大D64在前向owner构造的CANN流资源边界失败，不能继承小图的完整训练/恢复资格。
