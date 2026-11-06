# CANN 9.2.0 项目迁移：构建、运行与保留的严格失败

本次完成日常开发/新实验入口迁移，保留旧9.1环境、固定实验和历史资格。
没有修改公共软件栈、数学定义、输入、容差或优化器配方。
[JSON](cann92-project-migration-20261106.json)记录构建hash、逐客户端结果、命令、失败、队列和终态。
小型训练/恢复通过不解除独立综合根失败，也不升级历史规模或resident资格。

## 固定身份与实际构建

- 数学：`0770593838a37c790a568226930567ed60f99b83`，19份core-5快照核验通过。
- 入口实现：`f1480a002284a3c5fb5cae56b85f0b469492b5cd`。
- 实际wheel、Python/native及独立SDK core/standalone客户端构建：
  `9df0ecf8a5f8a98e4144f45161cffd5054b04002`，干净固定检出，jobs=320。
- 恢复驱动修复：`60e19755b04f09d676f0636dd32a5cabc19e6560`。
  它复用了经配置/安装内容/二进制身份核对的9df0ecf结果，没有声称在新提交重建C++。
- Python：3.12.13，Torch2.12.0+cpu，TorchNPU2.12.0.post3，公共dev-workspace/CANN9.2.0。
  独立C++：`libtorch-npu/2.12.0-cann9.2.0`，aarch64/ABI1，匹配独立LibTorch2.12 CPU分发。
  SDK库hash、CMake配置hash、全部构建产物hash在JSON中。

standalone显式`--libtorch-prefix`入口不导入Python Torch；本机系统Python实际为3.9.9。
初次构建驱动使用3.11才有的`hashlib.file_digest`失败，改为分块SHA256后完成。
独立客户端loader检查没有`libpython`、`libtorch_python`、未解析库或构建桩库。
CANN编译运行时可使用配套Python工具，不改变C++核心和调度的独立性。

## 小型验收结果

同一跟踪配置：[ascend-migration.json](../../configs/checks/ascend-migration.json)。
SettleGraph/FC/KV，四节点、两区域，生成器width8，8190可训练/8202去重总参数，
B2、每样例3个输入位置、逻辑窗12、stride4，真实staggered输入。
FP32 payload/scoring，ATen，prefill，packing/完整return packing均packed，worker/ATen/BLAS各1。
CPU用CPU放置；NPU用mixed-c，队列两次空闲检查后物理卡1→逻辑npu:0。

训练使用原公共默认weighted-output-square（scale0.001）和SGD
（lr0.001、momentum0.3、weight_decay0.01）；没有修改loss制造通过。
完整推理含状态/控制/history/消息/pending及换owner精确下一窗。
训练比较完整前向、当前声明loss、梯度、参数、optimizer槽；另比较连续三步与
两步checkpoint后新进程再走一步。Python逐值核对完整state/task和输出；
C++完整原生checkpoint字节一致，且前后进程均独立数值核对通过。

| 客户端 / 位置 | 完整推理与自身恢复 | 声明训练与新进程精确恢复 | 独立综合根门禁 |
| --- | --- | --- | --- |
| PyTorch / CPU | 通过 | 通过 | 所列八组根通过 |
| Python/native / CPU | 通过 | 通过 | 严格失败，见下 |
| standalone LibTorch / CPU | 通过 | 通过 | 所列16个整窗/切面根案例通过 |
| PyTorch / NPU mixed-c | 通过 | 通过 | 严格失败，见下 |
| Python/native / NPU mixed-c | 通过 | 通过 | 严格失败，见下 |
| standalone LibTorch / NPU mixed-c | 通过 | 通过 | 所列16个整窗/切面根案例通过 |

各客户端独立根遵守自己的公开验收定义；C++根目标为逐字段mean-square，
Python综合目标由`compare.objective`定义，不能把C++通过冒称Python同一根通过。
失败均在`experiment/checks.py:eager_roots`的首个`all`根，
`input.0.1.0.1`第4坐标（9个中1个），原atol1e-6/rtol1e-5不变：

| 范围 | 最大绝对差 | 最大相对差 |
| --- | --- | --- |
| CPU/native | 1.7881393432617188e-6 | 2.4992439648485743e-5 |
| NPU/PyTorch | 2.86102294921875e-6 | 3.9987302443478256e-5 |
| NPU/native | 2.8312206268310547e-6 | 3.957078661187552e-5 |

NPU/PyTorch与[原9.1人类入口](core5-human-workflow-20261031.md#本机cpunpu完整实操)
的坐标和绝对差相同；不能归结为新CANN专属回归。CPU/native和NPU/native本次仍明确记录失败，
没有用其他客户端、训练loss检查或精确恢复替代这些根。
按用户的小范围迁移要求停止扩展数值工程；重启须定位该根的累加/归一化首差并保持原材料。

## 环境与设施检查

- 9df0ecf固定CPU公式/native梯度/optimizer/checkpoint相关pytest：52通过，0失败/0跳过。
- 60e1975固定环境/SDK/复用身份设施：9通过，0失败/0跳过；与前组部分重叠，不累计成独立数学配置。
- 新私有LM环境：Transformers4.57.1、Accelerate1.11.0、PEFT0.17.1，CPU随机小模型/cache/
  optimizer/恢复/队列22项通过。没有下载或重新运行公开权重，也没有新的LM NPU吞吐资格。
- 日常`.local/venv-cann9.2.0`继承9.2公共dev，editable安装；其包内容hash与资格wheel相同。
  pip check、CLI metadata doctor与配置预览通过。资格wheel环境单独保留，不随日常源码编辑变化。
- 实际尝试用9.2模块启动旧9.1 venv，身份守卫在操作前拒绝；旧pyvenv/ready hash未变。

新LM默认路径为`EXPERIMENT_ROOT/env/transformers-cann9.2.0`，不覆盖旧`env/transformers`。
`TIDE_ASCEND_MODULE`和`TIDE_LM_ENV`可显式成对选择历史基线/环境，运行前仍需通过身份守卫。

## 原失败、作业与范围

`gef-cann92-migration-a`在SDK元数据阶段失败；`-b`完成构建及所有有限检查，
保留数值失败及恢复测试的workload_id拒绝。测试驱动曾把check内部展开材料的checkpoint
用于原recipe，现有身份守卫正确拒绝；修复采用同一run配置的prefix/whole/resumed，
没有放宽checkpoint身份。`gef-cann92-recovery-c`只补缺失流程，复用已核对的原结果。

三服务全部终态exit1，MainPID=0、ControlGroup为空，无匹配的任务后代。
b/c的整体严格失败由上表根失败保留，不能改写为passed。
队列已经释放卡锁，没有排队或运行作业。全部记录在JSON中，后台提交本身不是验收结果。
项目新增材料约0.4GiB，账户约180GiB，软/硬配额300/800GiB；没有必要使用tmpfs。
用户unit的CPU/memory controller未提供独立强制限额，实际约束来自编译并行、测试线程和有界输入。

resident、跨卡、FP16、大batch、规模训练/性能、完整PDG/TimedDAG/局部模块矩阵均未重验。
本次不是逐算子placement profile；日志的allocator padding/base-format/TypedStorage提示保留，
不据此宣称每个算子都在NPU。旧GPU已释放，未连接或重租。

## 不依赖本轮运行目录的复现

从干净固定检出按[development](../development.md)准备公共模块和有界background.slice服务，
workload为（`NEW_OUTPUT`必须不存在；此已知严格门禁预期保留上述根失败）：

```sh
bash scripts/qualify_ascend_migration.sh /absolute/NEW_OUTPUT \
  /absolute/use-local-ascend/scripts/run_on_free_npu.py all
```

它从跟踪配置生成全部材料、创建新venv、两套构建与wheel，再独立验证。
不需要旧LM根、旧源码包或本轮artifact。迁移后日常入口：

```sh
module load ascend/dev-workspace-9.2.0
source .local/venv-cann9.2.0/bin/activate
python scripts/ascend_environment.py --require-venv
tidegraph plan configs/checks/ascend-migration.json
```

本次机器上的`.local/site-cann9.2.0.json`指向实际通过的构建；换机器或清理后按development重建，
显式更新site路径/依赖锁，不假设它是已跟踪的可移植配置。历史9.1结果继续按原固定环境复现。
