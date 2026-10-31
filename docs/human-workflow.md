# 使用已安装的 tidegraph 开展实验

入口是 `tidegraph`，运行时不导入仓库的 `scripts/`、`tests/` 或内部 benchmark。
数学固定为 `0770593838a37c790a568226930567ed60f99b83`。
`plan`、帮助和结果查询不导入 Torch、不分配模型、不执行参考轨迹。
CPU、Python/native 和独立 C++ 客户端保留各自实现身份；统一命令不把 C++ 运行时变成 Python 核心。

## 1. 安装与锁定

在目标机器已有、匹配硬件的 Torch 环境中建立项目私有环境。NPU 先按
[development](development.md) 激活当前模块栈；3090 使用已选私有环境，不能混用其他架构的二进制。
下面的 `SOURCE` 指向固定的干净检出，`PREFIX` 指向新的项目私有安装目录。
创建环境与构建属于准备工作，正式计时开始前完成。

```sh
python -m venv --system-site-packages "$PREFIX/venv"
"$PREFIX/venv/bin/python" -m pip wheel --no-deps --no-build-isolation "$SOURCE" --wheel-dir "$PREFIX/wheels"
"$PREFIX/venv/bin/python" -m pip install --no-deps "$PREFIX"/wheels/tidegraph-*.whl
"$PREFIX/venv/bin/python" -m pip freeze --all > "$PREFIX/requirements.lock.txt"
"$PREFIX/venv/bin/tidegraph" --help
"$PREFIX/venv/bin/tidegraph" doctor --device cpu --probe metadata
```

该示例有意复用已匹配的 Torch，不自动下载安装另一个公共工具链。
`--no-build-isolation` 需要环境已有 setuptools>=68；缺少时仅补充到私有环境。
wheel、依赖锁和源码提交共同交付。不要在长任务运行时更新其安装目录。

复制一个[站点模板](../configs/sites/)到自己的配置目录，并填写实际构建路径。
纯 PyTorch 可以保持 `paths={}`；Python/native 填 `native_library`；resident 再填
`resident_library`；standalone 填 `standalone_binary`。每个路径附带原构建 manifest。
原构建来源与当前 Python 包哈希分别保存；加载旧库不会成为“在新提交重新构建”的证据。
构建过程见 [development](development.md) 和 [device-control](device-control.md)，构建只使用固定源码。

```sh
tidegraph doctor --site site.json --device cuda:0 --probe operator --dtype float32
tidegraph lock create --site site.json --wheel /path/to/exact/tidegraph.whl --output foundation.lock.json
tidegraph lock verify foundation.lock.json --site site.json
```

`doctor metadata` 只报告环境元数据；`operator` 只检查所选设备的一个小型矩阵前向和梯度，
不认证完整图、训练、resident 或跨卡。显式设备不可用时失败，不改用 CPU。
`lock` 重新计算安装包全部层级文件、所选构建产物与 manifest 的哈希。
Torch/TorchNPU/NumPy 记录版本和 distribution RECORD 元数据；这不是每次重新哈希所有第三方二进制。
`--wheel` 还验证 wheel 内的包文件与已安装文件相同。依赖身份通过与数学资格通过是不同事实。

## 2. 配置、预算与独立检查

将以下内容保存为 `run.json`。JSON 中说明使用 `description` 字段，不需要 YAML 插件。
这是可直接运行的小型输入；较大模型先修改配置并查看逐卡预算。

```json
{
  "schema": "tide-experiment-v1",
  "name": "fc-kv-small",
  "description": "真实不齐样例；完整诊断返回；两个计时窗和独立精确续接",
  "model": {
    "recipe": {"topology": "fc", "family": "kv", "width": 8,
               "depth": 2, "region_width": 2, "history": 4}
  },
  "task": {
    "samples": 2, "positions": 3, "seed": 7, "input_pattern": "staggered-v1",
    "objective": {"kind": "weighted-output-square", "scale": 0.001},
    "optimizer": {"kind": "sgd", "lr": 0.001, "momentum": 0.3, "weight_decay": 0.01}
  },
  "execution": {
    "family": "settle", "client": "pytorch", "device": "cpu", "placement": "cpu",
    "dtype": "float32", "scoring_dtype": "float32", "schedule": "prefill",
    "packing": "auto", "return_packing": "scalar", "return_mode": "complete",
    "workers": 1, "intraop_threads": 4, "interop_threads": 1
  },
  "measurement": {
    "mode": "inference", "warmup": 1, "steps": 2,
    "validation": "strict", "checkpoint_check": true, "timeout_seconds": 1800
  },
  "resources": {"max_materialize_bytes": 2147483648, "checkpoint_max_bytes": 2147483648}
}
```

```sh
tidegraph plan run.json --site site.json
tidegraph plan run.json --site site.json --format json --output resolved.json
tidegraph check run.json --site site.json --scope smoke --output-dir checks/smoke-new
```

模型、任务、执行、站点和测量分别展开并保存请求值与生效值。
`model_id` 不随 batch/设备改变；`workload_id` 包括输入与原始逻辑时间；任务、执行视图、放置与完整配置各有身份。
`model` 也接受 `definition`（公共 `resolve_model` 参数）、`resolved`（已校验的公共模型）或
`path`（上述模型 JSON）。这些入口支持公共 Config 已有的合法组合，不受有限 benchmark 列表限制。
`recipe.memory_overrides`、`recipe.region_overrides` 复用公共校验器；异构模型可逐节点/区域写 definition。
HB 的 `region_width=1` 需要显式 `offsets:[0]`；默认 `[0,1]` 需要至少两个区域内节点。

`execution.family` 选择 `settle`、`timed-dag`、`pdg` 或 `auto` 的合法执行视图，不修改物理图。
在 `model.transforms` 中加 `{"kind":"non-aligned","edge_index":0,"extra_delay":1}` 或
`{"kind":"feedback","delay":2}` 会产生新的物理模型和身份，必须选择其合法层级。
一般端口、重复边、同秩区域可直接写公共 definition；时间仍是 `origin[port]+stride*global_position`。

`plan` 列出逐节点/区域/共享参数到设备映射、不可拆共享组及逐卡参数/状态/KV/梯度/optimizer/
返回/恢复预留。未知 workspace、journal 和运行时峰值保持未知；`automatic_admission=false`。
默认累计窗口数包括 warmup、计时窗和一次恢复检查，不能拿单窗预算当完整实验预算。
Python训练的host预算另含调用者、边界和下一窗同时保留的参数/梯度/optimizer快照；
`training_snapshot_extra_bytes`明确列出原四份bank/两份optimizer估算之外的保留量。
这部分在主存，不随卡数分摊；不是额外设备参数，也不改变数值、窗口或checkpoint字节。
现场还须检查当前显存、主存、磁盘与其他任务，经过有限实测校准后才能做容量点；不能反复 OOM 找边界。

`check` 是独立的基座验收入口：smoke 只做小型所选路径检查；related 做声明的相关检查；
full 只以结果内列出的完整资格范围为准，缺少硬件/构建不能通过跳过检查获得完整资格。
三个名字均不意味着所有配置/客户端/平台已通过。详见 [verification](verification.md)。

## 3. 单次推理、训练与恢复

```sh
tidegraph run run.json --site site.json --output-dir runs/first
tidegraph run run.json --site site.json --resume /path/from/result/checkpoint --output-dir runs/resumed
tidegraph results runs --format table
```

训练将 `measurement.mode` 改为 `training`，其他定义不变。
内置诊断 loss 是 `(mass*content).square().sum()*0.001`；SGD 配方与旧工作负载相同。
这检验一阶训练机制，不评价模型效果。下游任务使用下面的公开扩展点。
恢复会校验完整 checkpoint、配置、任务进度及安装身份；始终写入新目录，失败作业不被覆盖。
同栈精确续接与跨设备 checkpoint 可移植性分别验收。
Python/native 路径每个完整窗口保存一次持久 checkpoint，只有新指针 fsync 后才退役上一个 generation；
standalone 也在每个完成窗口原子发布原生 checkpoint 和 latest 指针。
`checkpoint_check=false` 只关闭额外 exact-next 验证，不关闭保存，所以 plan 始终预算持久 checkpoint。
Python/native运行可显式设置`measurement.checkpoint_interval=N`（正整数，省略仍每窗1）；
按绝对`next_window`的N倍数发布，并始终发布本次最终窗口。它不合并计算/训练窗口，不改变loss或optimizer边界。
中断时只能恢复最近已发布边界，尚未保存的窗口需重算；严格检查与最终exact-next保持。
plan按本次将发布的次数预算1或2代；resume在现场按起始窗口重新计算，不能沿用不适用的单代预算。
默认原两代保守预算保留；standalone当前每窗发布，此新增间隔选项会明确拒绝非1值。
[4.561B间隔对照](../configs/checkpoint-cost/cuda-disk-periodic.json)与[9.123B单/双卡](../configs/checkpoint-cost/cuda-capacity-periodic.json)
明确使用不同持久化策略，不撤销旧每窗准入拒绝，不将少保存的工作流成本当窗口算子加速。

保存时间在每窗 `checkpoint_seconds` 或 standalone 对应字段单列，不混入窗口算子计时。
Python/native与resident公共运行现在记录`checkpoint_cost`，拆分权重搬到主存、编码、序列化、
文件fsync、hash、原子发布与旧代清理；`checkpoint.cost`拆分精确恢复的快照、重建/读取、复制和比较。
这些是嵌套的墙钟span，不能相加，也不增加设备同步；未被对应客户端插桩的内部成本不臆测。
固定安装内容hash改变后仍须使用原安装恢复旧实验，观测字段不放宽依赖身份检查。

内存文件系统可作为**易失的实验存储**：配置`resources.storage_medium="tmpfs"`后，
`tidegraph plan`把两代checkpoint/原生材料的未来文件字节同时计入主存；磁盘reserve只算空闲余量。
`run`自动检测实际输出挂载并联合准入，显式请求与挂载不符会拒绝；已有shm文件已减少现场主存，不重复计费。
`/dev/shm`额度与容器RAM不能相加。fsync/原子发布只保证完整文件可见，tmpfs在容器或宿主重启后丢失，
需要跨重启保留的checkpoint必须另经校验复制到持久存储。结果记录挂载类型和cgroup使用量，
不把tmpfs恢复通过称为磁盘耐久资格。默认每窗保存及完整检查不变。
`measurement.observation="checkpoint"`声明单独成本观察（1计时窗，warmup另列），不计入合格吞吐。
同材料例子为[磁盘](../configs/checkpoint-cost/cuda-disk.json)和[tmpfs](../configs/checkpoint-cost/cuda-tmpfs.json)：
先分别`plan`核对预算，再串行`job start --output-dir`到实际磁盘或`/dev/shm`的新目录。
两者保留完整返回与exact-next，但`validation=none`表示不重复独立CPU参考，不能新增数学资格。
本机大文件周转示例为[NPU12.802B tmpfs](../configs/checkpoint-cost/npu-tmpfs.json)，须在既有任务终态、相同单卡原流程完成并精确恢复后单独准入；不移动正在运行的作业。其原磁盘结果是不同Python观测版本，跨版本成本仅作观察，并说明相同窗口0/1和线程/返回口径。
保存和恢复成本查询`results --format json`中的`details.metrics[].checkpoint_cost`和`details.checkpoint.cost`。


CPU 配置明确区分 ATen/BLAS 的 `intraop_threads`、interop 和调度 `workers`。
已安装Python eager客户端可显式选择`execution.finite_check_packing=packed`，
把兼容的有限值诊断输入按device/dtype合为最多8MiB组；默认`scalar`保留逐张量检查。
它独立于数值算子packing和完整返回packing，不改精度、阈值、求导连接或非有限值拒绝。
大/非连续/特殊张量保留逐张量检查，plan另预留每卡及主机32MiB临时空间。
该策略不改变C++核心；standalone/resident及旧study脚本仍用其原检查入口，明确拒绝此新覆盖项。
启用前后使用同材料独立比较与训练/恢复回归；profile中的`isfinite`累计时间不是排他归因或收益保证。
GPU/NPU 将 `device` 改为显式 `cuda:N`/`npu:N`，`placement` 选 `mixed-a/b/c` 或合法 resident；
不要把 CPU placement 留在 accelerator payload 上。纯 PyTorch 保持独立主机调度，resident 使用实际公共设备库。
standalone 将 `client` 设为 `libtorch` 并提供真实 C++ 程序路径，Python 只负责配置与启动。
NPU 站点中 `site.environment` 对应 Python 导出适配器；不同 SDK 的纯 C++ 进程使用独立的
`site.standalone_environment`（同一环境变量白名单）。不要把 standalone SDK 的库路径加载进 Python。
前台 CLI 在所需运行环境不同时先重新执行进程，再导入 Torch，使动态加载器实际使用请求的库路径。

结果将执行状态、数值、离散控制/路由、完整记录、checkpoint 和精确续接分列。
`validation=strict` 保留严格阈值；`explore` 仅接纳已完成、有限且离散语义相同的数值超差，
仍记录严格失败。非有限、非法配置、求解器/通信/容量错误均不能被探索模式吞掉。
非有限检查仅在拒绝后逐字段定位，错误包含forward/loss、gradient、updated parameter或task阶段及字段/参数名；
这个额外诊断成本属于失败流程，不改变正常接受条件、精度或数值容差。
`none` 表示未做独立验收，不能显示为通过；普通训练不要求每一步都运行 CPU 参考。

`return_mode=complete` 保留完整诊断返回，适合独立状态/控制/消息/pending 检查。
`outputs` 或 `metrics` 是实际实验的独立返回契约；设备状态可以留在设备上，checkpoint 仍完整保存。
这两种模式要求显式 `validation=none`，先通过独立 `check` 检查所需配置。
所有结果保存 timing_contract；不能把少返回字段的耗时与旧完整返回直接算等价加速比。

## 4. 同模型多卡与有限矩阵

例如单个三卡请求：

```json
{
  "client": "python-native", "device": "cuda:0", "peer_devices": ["cuda:0","cuda:1","cuda:2"],
  "placement": "mixed-c", "placement_strategy": "balanced-shared-groups",
  "communication": "ordinary", "dtype": "float32", "scoring_dtype": "float32"
}
```

这是 `execution` 的内容；其余字段沿用 `run.json`。卡数无二的幂限制。
`balanced-shared-groups` 固定首节点到主卡，其余不可拆共享组按参数+旧缓冲字节降序、
负载和设备顺序稳定选卡；`region-round-robin` 按区域分配；`explicit` 搭配完整 `node_devices` 列表或节点ID映射。
共享参数不允许通过复制为独立参数逃避同卡约束。单组过大明确拒绝，张量切分属于另一个方案。
`ordinary`、`blocking`、`cpu_bridge` 是独立的通信请求，运行记录实际方式/成本；不把普通复制称为直接 P2P。
多卡 resident 继续受双向 P2P/原子协议门禁约束。

保存 `matrix.json`，每一行都明确列出：

```json
{
  "schema": "tide-experiment-matrix-v1", "name": "fixed-model-batch",
  "cases": [
    {"name": "b2", "config": "run.json"},
    {"name": "b8", "config": "run.json", "overrides": {"task": {"samples": 8}}},
    {"name": "train-b2", "config": "run.json", "overrides": {"measurement": {"mode": "training"}}}
  ]
}
```

```sh
tidegraph plan matrix.json --matrix --site site.json
tidegraph matrix matrix.json --site site.json --output-dir runs/matrix-new --detach
```

矩阵逐配置独立进程、有限超时、顺序运行，避免线程/运行时设置跨配置泄漏。
每一项保留独立失败；矩阵外层结束不认证内部结果。profile 应另开配置/任务，重型构建与正式计时不重叠。

## 5. 持久任务、状态、日志与停止

```sh
tidegraph job start run.json --site site.json --output-dir jobs/example
tidegraph job status jobs/example
tidegraph job logs jobs/example --lines 80
tidegraph job stop jobs/example
tidegraph job resume jobs/example --checkpoint /path/to/complete/checkpoint --output-dir jobs/resumed
```

本机站点用 systemd 用户服务 `background.slice`、`KillMode=control-group`，不依附 focus.service。
无 systemd 的目标容器用同一已安装 subreaper：有限超时、PID启动身份、所有后代追踪和 TERM/KILL 清理。
SSH/提交进程退出不终止任务；这不是跨重启服务。`stop` 必须观察终态和无存活后代。
监督器丢失时标为 unconfirmed，不能仅凭进程消失推断通过。
任务目录保存只读 resolved.json、安装/构建身份、status.json、task.log、supervisor.log 和 run/结果。
`job status`显示矩阵已完成项数、当前case及其构造/测量/精确恢复/独立检查阶段；`job logs`读取当前case的原日志，结束后读最后一项。外层作业状态不覆盖这些独立验收结果。
后台启动前后都核验所读安装内容，运行期间必须保持这些路径不变。

## 6. 查询、导出与下游协作

```sh
tidegraph results runs jobs --platform cuda --client python-native --dtype float32 --format table
tidegraph results runs --graph timed-dag --topology hb --params-min 1000000 --format csv --output results.csv
tidegraph results runs jobs --format html --output results.html
tidegraph results runs --exploratory true --format json
tidegraph results runs --format archive --output reviewed-results.json
tidegraph results reviewed-results.json --batch 512 --format html --output batch512.html
tidegraph results runs --format archive --output reviewed-results.json.gz
tidegraph results reviewed-results.json.gz --batch 512 --format table
```

HTML 自包含，可直接用浏览器打开；没有服务、账号或外部CDN。
页面内可以按平台、图、拓扑、参数范围、batch、精度、客户端、放置、任务、状态及packing筛选；
默认将模型、工作负载、执行、独立检查和恢复分列，完整原记录在Details中展开。
过滤只改变显示，不改变归档、资格或成本口径；关闭JavaScript时仍显示全部记录，CLI筛选继续可用。
大诊断字段默认折叠，点击Details展开；`.json.gz`归档保留相同内容并可直接查询，避免重复配置膨胀长期证据。
archive v2保留各项身份、严格/探索/恢复结论、资源拒绝、计时、工作量及内存的可查询投影，
并在details保留完整展开配置、原逐窗/warmup成本、构造/检查成本、严格差异与安装/构建来源。
重复的逐文件哈希表压缩为表的摘要和文件数，包/构建的整体哈希和来源不变。
旧v1归档仍可查询，但其缺失诊断明确标为unrecorded，不从汇总反推原窗口或重新验证。
不再依赖原运行目录；记录原result的hash和投影内容hash。它不是重新验收，也不保证原checkpoint仍在线。
子进程若在恢复或检查中异常退出，报告通过同一config_id关联其持久supervisor记录，
显示`process-failure`和退出码/最后阶段；已完成窗口成本保留，未检查项不补写通过。
原始实验快照不改，监督状态及其hash进入Details；整个矩阵的外层状态不能替代单case状态。
支持平台、图、拓扑、参数范围、batch、dtype、客户端、放置、通信、返回方式和执行/资格状态筛选。
每秒样例是每测量窗的逻辑batch/耗时，不是 token/s；真正候选、Full、消息、时间批和内存保留在记录中。
计时图表或速度结论必须先核对返回、工作量、精度、控制和资源争用是否可比。

下游保留数据、任务头、loss 和训练配方；公共 `task.plugin` 指定已安装模块：
`{"module":"my_project.tide_task","factory":"create_task","options":{}}`。
工厂接口和必需方法见 [experiment/tasks.py](../python/tidegraph/experiment/tasks.py)，
输入构造不能读取参考动作轨迹；候选与独立参考使用不同任务实例。
resident 训练另需明确的 output-field VJP 入口。checkpoint 必须保存插件的 task state。
先运行基座检查，再运行下游小型接入验收，最后提交配置驱动的任务。

按 [downstream-contract](downstream-contract.md) 和[模板](../templates/downstream/)保存固定源码/安装锁、
配置、接入检查与验收条件。遇到问题提交 request；接收基座 implementation/evidence 交付后，
下游在自己的真实任务复验、记录 acceptance，之后显式更新依赖锁。
基座通过和下游通过是两个事实；旧实验继续保留旧依赖身份。

## 7. 未来七张 T4

[t4-seven-pending.json](../configs/sites/t4-seven-pending.json) 只是站点模板。
新机器分别检查 driver、CUDA runtime、Toolkit、Torch/LibTorch、sm_75 编译支持、实际算子/精度与每向通信。
可为新站点选择版本，不修改当前3090或NPU公共环境，也不复用sm_86产物。
显式构建标准ATen eager客户端后，按 CPU→单卡→双卡→实际七卡、推理→训练→恢复逐项验收。
当前 CUDA resident 协议要求 CC>=8.0，T4 的 sm_75 不满足这一静态门禁；更换协议需另行具体设计和验证。
未接触 T4 的构建、运行、峰值、通信和性能均为待验，模板或预算不是资格。

迁移时在**新T4站点**按以下顺序操作，所有输出存入新的项目目录：

1. 查看`nvidia-smi`中的实际driver、设备/进程/空闲内存，另运行`nvcc --version`和
   `nvcc --list-gpu-code`。driver显示的CUDA能力、Torch自带runtime和本地Toolkit分别记录。
   在选定私有环境执行下列查询；Torch必须包含实际T4可用的代码，环境变量不能补出缺失架构。
2. 选择相互兼容的driver/runtime/Toolkit/Torch或LibTorch后锁定版本。仅在该新站点安装或选择，
   不因此更改当前3090/NPU公共环境。CUDA扩展以7.5为目标；本仓库eager核心使用ATen，
   resident自己的条件控制核仍拒绝sm75，不能用移除架构检查取得资格。
3. 从固定源码构建eager核心及独立客户端，安装包、填写T4 site的实际路径，再执行doctor/lock。
   以下命令是长构建的workload，须由该站点持久服务启动；不与正式计时重叠。

```sh
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_arch_list()); print([(i,torch.cuda.get_device_name(i),torch.cuda.get_device_capability(i)) for i in range(torch.cuda.device_count())])'
TORCH_CUDA_ARCH_LIST=7.5 python scripts/build.py --backend cuda --build-dir build/t4-eager --jobs all
python scripts/build_weighted_benchmark.py --core-build build/t4-eager --output-dir build/t4-client --jobs all
tidegraph doctor --site t4-site.json --device cuda:0 --probe operator --dtype float32
tidegraph lock create --site t4-site.json --wheel /path/to/t4-install.whl --output t4-installation.lock.json
```

4. 先用小模型分别验三客户端的完整推理、声明一阶训练、独立根、optimizer和新进程恢复。
   FP16另配，不从3090的FP16通过外推；大规模前保存逐卡预算和实际峰值。
5. 同一16节点/8共享组模型从1→2→3→7卡逐次准入，显式写
   `peer_devices=["cuda:0",...,"cuda:6"]`、`placement_strategy="balanced-shared-groups"`。
   每对方向的实际复制、反向及请求通信方式独立核验；P2P查询和普通复制分别记录。
   任一不可拆共享组超过单卡安全容量时停止该放置，提出张量切分方案后再扩展。
6. 按下游锁定流程复验应用任务，只有实际接受后才更新依赖；提交T4自己的结果/失败、
   build manifest、安装锁与配置。跨设备加载与同栈精确恢复分别检查，模板不预填通过。

checkpoint恢复的精确CPU值比较可先使用`torch.equal`确认完全相等；不相等时仍执行原严格断言并保留首差诊断。
有限值、dtype/shape/设备、None/零和int64检查保持，跨设备容差比较不进入该捷径；它不是跳过独立验证或改用hash验数值。
