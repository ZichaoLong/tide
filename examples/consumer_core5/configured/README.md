# 配置驱动的下游任务示例

本目录可以复制到实验仓库并独立安装。任务包拥有数据生成、可学习Linear头、目标与MSE、SGD配方；
已安装的`tidegraph`负责图执行、统一参数/optimizer、独立验收、持久任务和报告。
运行代码只导入公共包接口，不导入基座`scripts/`、测试或内部benchmark。

这是小型机制示例，不评价任务效果。默认FP32、FC/KV、B2/P3、真实错开输入，原始逻辑时间与物理端口身份保留。
输入的不同样例/窗口使用独立CPU RNG数据；同一外部根到多个端口仍共享根，不能把端口复制当独立输入。
新任务的输入、head和loss与旧诊断工作负载不同；它通过不解除旧FP32训练失败。
本目录的配置与实现不是CPU/NPU/CUDA真机通过证据，资格应引用相应固定源码的实际结果。

## 固定安装和依赖

先按基座[操作手册](../../../docs/human-workflow.md)在私有、匹配Torch的环境安装固定`tidegraph` wheel，
并填写`site.json`。然后在复制后的本目录构建和安装任务包：

```sh
python -m pip wheel --no-deps --no-build-isolation . --wheel-dir wheels
python -m pip install --no-deps wheels/tidegraph_consumer_core5_example-0.1.0-py3-none-any.whl
python -m tidegraph_consumer_core5.configure --help
tidegraph lock create --site site.json --wheel /path/to/exact/tidegraph.whl --output installation.lock.json
tidegraph lock verify installation.lock.json --site site.json
sha256sum installation.lock.json wheels/tidegraph_consumer_core5_example-0.1.0-py3-none-any.whl
```

使用[来源锁模板](../../../templates/downstream/foundation.lock.example.json)记录完整基座/下游提交、数学/协议、
实际wheel与CLI安装锁SHA256。CLI安装锁不推断Git来源，来源关系由固定checkout的构建记录证明。
任务插件的整个顶层包文件hash另进入运行/checkpoint身份；自己的额外依赖仍须固定并保留环境锁。
不要在活动作业中更换安装包。复制示例到下游后，将相对文档链接换成所锁定基座提交的链接。

## 生成可读JSON，先预算再运行

生成器调用公共无分配校验器，不执行图、不打开GPU/NPU，也不需要旧运行目录。
不覆盖已有文件；生成后可以编辑完整JSON，公共合法模型/执行配置不受示例的默认值限制。
`--model model.json`接受公共模型对象，如`{"recipe":{...}}`、`{"definition":{...}}`或`{"resolved":{...}}`。

```sh
python -m tidegraph_consumer_core5.configure --device cpu --task base --output base-check.json
python -m tidegraph_consumer_core5.configure --device cpu --mode inference --output inference.json
python -m tidegraph_consumer_core5.configure --device cpu --mode training --output training.json
tidegraph plan training.json --site site.json --format json --output resolved.json
```

模型、输入/任务、执行、站点和测量是分开的配置节。任务配置包含：

```json
{
  "plugin": {
    "module": "tidegraph_consumer_core5.task",
    "factory": "create_task",
    "options": {"data_seed": 8, "head_seed": 9, "input_scale": 0.2, "target_scale": 0.05, "loss_scale": 0.001}
  }
}
```

这是`task`节的一部分；生成文件还含samples、positions、seed、input_pattern与optimizer。
插件使用自己的loss，`task.objective`保留基础诊断配置，不冒称本任务仍是旧平方loss工作负载。
预算先覆盖图的逐卡参数/状态/缓存/返回/恢复；外部head和额外任务workspace需要下游另加预算并实测，
静态plan不会导入任意插件或假称知道它的内存。默认head仅`width+1`个参数，示例的64MiB资源预算不是基座上限。

## 三平台推理、训练和恢复

CPU、GPU和NPU的操作相同，只更改显式设备。NPU须先激活本站匹配的私有模块环境，
GPU须使用对应Torch/CUDA环境；实际空闲卡由操作人确认，不运行到其他项目占用的卡。

| 平台 | 配置生成示例 | 站点要求 |
| --- | --- | --- |
| CPU | `python -m tidegraph_consumer_core5.configure --device cpu --mode training --output cpu-training.json` | 固定ATen线程与调度worker；默认均为1 |
| CUDA | `python -m tidegraph_consumer_core5.configure --device cuda:0 --mode training --output cuda-training.json` | 当前空闲卡、匹配Torch/CUDA；显式mixed-c、IEEE FP32 |
| NPU | `python -m tidegraph_consumer_core5.configure --device npu:4 --mode training --output npu-training.json` | 先确认逻辑设备编号及空闲资源，早注册匹配torch_npu |

将`--mode training`改为`--mode inference`生成独立推理配置；推理不会运行optimizer。
以下示例以生成的CPU文件为准，在目标平台改用对应文件与`site.json`：

```sh
tidegraph doctor --site site.json --device cpu --probe operator --dtype float32
python -m tidegraph_consumer_core5.check --device cpu
tidegraph check base-check.json --site site.json --scope related --output-dir checks/base-new
tidegraph check cpu-training.json --site site.json --scope related --output-dir checks/task-new
tidegraph run cpu-training.json --site site.json --output-dir runs/train-first
tidegraph run cpu-training.json --site site.json --resume runs/train-first/checkpoints --output-dir runs/train-resumed
tidegraph run inference.json --site site.json --output-dir runs/infer-first
tidegraph run inference.json --site site.json --resume runs/infer-first/checkpoints --output-dir runs/infer-resumed
tidegraph results runs --format table
tidegraph results runs --format csv --output results.csv
tidegraph results runs --format html --output results.html
```

生成GPU/NPU配置时也为同一设备生成`--task base`文件；不能把CPU基座检查冒称目标设备验收。
`check`模块是下游独立的小公式/输入根/数据RNG检查，明确不检查图/optimizer；公共`check --scope related`
再验证完整图记录、head loss独立根、共享参数贡献、统一optimizer与精确续接。
`run --resume`是新进程实际恢复演练。每次输出使用新目录，旧失败和已保存checkpoint不会覆盖。
查看结果时分别核对execution、strict数值/离散、checkpoint/exact_resume；未检查不记通过。

纯PyTorch默认保留独立调度。Python/native加`--client python-native`生成配置，并在站点填写匹配的`native_library`；
它使用同一任务插件和统一owner，但必须独立验证。默认workers=1，旧workers2跨进程精确恢复失败仍保留。
FP64可以通过`--dtype float64`选择并单列；本任务示例显式拒绝FP16，基础`--task base`可选择已实现FP16代表再独立检查。
resident暂不注册外部`task.module`与统一head optimizer，明确拒绝此扩展；不能静默遗漏head。
standalone不能执行Python插件。基础诊断示例可用`--task base --client libtorch`，真实下游C++任务使用公共C++库，
独立保存自己的源码/任务实现/构建身份，不把Python包装器视为standalone任务实现。

## 长任务、有限矩阵和返回口径

```sh
tidegraph job start cpu-training.json --site site.json --output-dir jobs/train-new
tidegraph job status jobs/train-new
tidegraph job logs jobs/train-new --lines 80
tidegraph job stop jobs/train-new
tidegraph job resume jobs/train-new --checkpoint jobs/train-new/run/checkpoints --output-dir jobs/train-resumed
```

`stop`后核实终态及无存活后代。恢复路径从实际结果读取，若任务尚未发布完整checkpoint则不能恢复。
本站使用background.slice；无systemd容器使用已安装容器监督器，配置/环境/代码在作业期间冻结。
有限矩阵沿用`tide-experiment-matrix-v1`，每个case引用生成的JSON并显式列出overrides，详见操作手册。
不让示例自动进行规模扫描或隐式追加设备实验。

默认`return_mode=complete`用于完整状态/控制/消息/pending独立检查。
实际实验可另存配置，把return_mode设为`outputs`或`metrics`，并明确`measurement.validation=none`；
先前独立check结果仍单独保留。完整checkpoint仍保存，按需返回与完整返回不混算加速比。
修改loss、目标、输入、head、dtype或配方均形成新工作负载；探索性数值超差不是原严格失败的修复。

## 任务扩展和升级

工厂`create_task(task=..., plan=..., options=...)`不得执行图或读取参考轨迹。
必需方法是inputs、loss、optimizer、state_dict、load_state_dict。
可选`module`由运行器在optimizer创建前放到首个图参数的设备/dtype；调用optimizer时传入的model包含唯一
`graph.*`和`task.*`owner。旧无module任务参数名不变。任务内部共享参数保留别名，图与任务不能偷偷共用同一个Parameter。
head若读取其他设备输出，必须显式传输；本例使用公共copy_rows并遵守所选通信策略。

任务module权重/buffers和全部optimizer槽由统一checkpoint记录；task.state保存CPU数据RNG及下一窗口编号。
增加scheduler、数据加载器或随机增强时，由任务保存其全部可序列化状态，加载前先验证。
公共运行器检查参数归属、非有限值、代码身份和完整恢复；不会推断未声明的额外应用状态。
本例head输出用于任务loss；图返回仍遵循公共返回契约，应用需要导出其他预测时应明确自己的输出协议。

升级遵循[协作规范](../../../docs/downstream-contract.md)：固定候选安装→来源/安装锁→基座检查→
下游小型接入与实际目标配置→记录acceptance→显式切换选定锁。失败保留旧安装和旧运行，基座交付不自动成为下游验收。
需求、交付、验收使用[模板](../../../templates/downstream/)，带配置、状态分类、精确命令和有限预算。
跨版本/跨设备迁移不是同栈精确续接，须单独验证或另声明为权重初始化的新运行。

未来7×T4先按[站点模板](../../../configs/sites/t4-seven-pending.json)及操作手册检查driver/runtime/Toolkit/
Torch/LibTorch和sm_75，并重新构建。先1→2→3→7张卡核对真实传输、独立梯度和同模型逐卡容量，不能复用3090 sm_86产物。
当前CUDA resident协议的CC>=8门禁不满足T4；尚无T4真机资格，相关构建、算子、精度、通信和性能全部待验。
