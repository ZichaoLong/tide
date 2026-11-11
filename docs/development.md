# 从干净源码开始开发

本页使用新路径，不要求任何旧 `TASK`、构建、运行目录或 LH 快照。
先读 [AGENTS](../AGENTS.md)、[STATUS](STATUS.md)、[执行契约](execution-flows.md)。
当前采用的教材与实现域从 [语义索引](semantic-map.md) 进入。
日常实现使用[仓库开发技能](../.agents/skills/develop-tide-foundation/SKILL.md)，
性能问题使用[优化技能](../.agents/skills/optimize-tide-execution/SKILL.md)，接口和导航治理使用
[维护技能](../.agents/skills/maintain-tide-foundation/SKILL.md)。它们引用本页命令与现有契约，
不要求一次加载全部技能，也不是构建、运行或测试依赖。

新检出使用`git clone --single-branch --branch graph-execution-foundation git@github.com:ZichaoLong/tide.git`。
旧工作区迁移、旧提交号查询和归档按需读取见[历史迁移](history-migration.md)；不要将旧分支合回新基座。

## 无设备的接续检查

在仓库根目录：

```sh
git status --short --branch
python scripts/status.py
python scripts/check_upstream.py
```

第三条核验当前 Git 内的固定教材快照，不读取上游尚未审核的工作区修改。
如需对比原 Git 对象，加 `--upstream-repo /path/to/ObsidianVault`。
新 clone 没有本地 job 记录是正常状态，不代表旧实验未运行，也不要求找回它们。

默认接续只依赖Git源码/文档和声明的外部工具链；本地环境、预编译库和旧运行目录不保证存在。
按下面的步骤重建，已知问题从[复现导航](reproduction.md)进入；[产物规则](artifact-retention.md)说明历史材料边界。

新的 B×L 公共客户端见 [tensor-window-v1](weighted-tensor-windows.md)。安装当前包后可从任意 cwd
运行 `python /path/to/examples/consumer_core5/tensor_window.py --client python --dtype float64 --output-dir NEW_DIR`，
验证三步 AdamW、raw continuation 和新进程精确续接；`--positions` 同时检查 KV 旋转/距离项；
`--placement mixed-a|mixed-b|mixed-c`指定Read/control/selection位置，`--scoring-dtype`指定评分精度。
示例默认开启deterministic测试策略；检验生产默认重复性时显式加`--no-deterministic-algorithms`，
该策略进入示例checkpoint身份，不修改基座默认。`tidegraph-column-check --device=npu:0 --dtype=float32`
使用独立SDK实际检查五记忆×三放置，默认无参数仍运行CPU FP64及显式FP32评分。
`--family affine|delta|gated_delta|ssd`检查四循环、适用卷积和对应raw恢复。native 增加
`--client native --native-library /path/to/matching/build`；真实 NPU 在合法分配后选
`--device npu:0 --dtype float32`。该小消费者的资格不能代替全部局部组合/规模。

## Python CPU 起步

使用项目自己的 Python/Torch 环境。不要覆盖其他项目或公共厂商环境。
本机默认 `/usr/bin/python` 不是可用的 Torch 环境；选择下方已核验的配套 Python，
或使用自己的 CPU Torch 环境后，再运行本节命令。
`pyproject.toml` 声明 Torch 和 test 依赖；旧资格仅认证报告中的实际版本组合。

```sh
python -m pip install --no-deps -e .
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m pytest -q tests/test_semantics.py --dtype both
```

这组是手算、消息结构、输入 seal 与 Emit VJP 的纯 Python 检查，不需要 native 库。
它是起步检查，不能代替完整 CPU/设备资格。相关模块用 `rg` 从语义索引定位。

## 构建 LibTorch CPU 与相关开发测试

```sh
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python scripts/build.py --backend cpu --build-dir build/cpu --jobs all
TIDE_BUILD_DIR="$PWD/build/cpu" python -m pytest -q tests/test_semantics.py tests/test_native_settle.py --dtype both
```

`build.py` 使用当前 Python 对应 Torch 的 CMake/ABI，并生成新的 build manifest。
公共独立 C++ 不依赖 Python 调度；纯 LibTorch 配置入口：

```sh
cmake -S . -B build/cpu-standalone -DCMAKE_PREFIX_PATH=/path/to/libtorch \
  -DTIDE_BACKEND=CPU -DTIDE_PYTHON_BINDINGS=OFF
cmake --build build/cpu-standalone --parallel "$(python scripts/build_resources.py)"
ctest --test-dir build/cpu-standalone --output-on-failure
```

`/path/to/libtorch` 必须是目标机器实际匹配的 SDK。长构建/测试放独立持久服务；
本页命令是 workload，不替代本机服务/资源规则。不要让运行任务读取正在编辑的源码。

完整门禁、原始 LH 额外门禁和固定源码资格流程见 [verification](verification.md)。
原始 LH 需要另行提供源码快照；没有该输入时不声称完成 LH 对照。
当前实现没有修改时，不为接续重复全量资格或旧性能矩阵。

## NPU / GPU

按 [accelerators](accelerators.md) 和 [device-control](device-control.md) 选择
CPU、CUDA、Python-owned NPU 或 standalone NPU 构建；它们不能混用二进制。
常驻库另建，并核验与 core 对应的 manifest。缺少显式后端、库或 dtype 必须失败。
standalone weighted benchmark另需nlohmann JSON头文件；不再依赖退役的LH快照内vendor目录。
在项目私有环境安装`nlohmann_json==3.11.2`，或给`build_weighted_benchmark.py --json-include INCLUDE_DIR`。
构建记录实际版本、完整头文件树hash及CMake include目录；公共C++运行时不依赖这个Python包。
CUDA已取得[RTX3090单卡eager及有限双卡实测](evidence/core5-cuda-eager-20261031.md)；三客户端分别验收，standalone原数值失败及native双卡精确恢复失败保留。
单卡resident用`verify_core5_resident_single.py`的明确库存和零跳过门禁；完整三卡门禁仍独立保留。
同源码构建、单卡正确性、设备时间线、跨卡和性能均是不同证据，不由编译成功推断。

本机新开发入口采用 CANN 9.2.0：`ascend/dev-workspace-9.2.0`，
Python3.12、Torch2.12.0+cpu、TorchNPU2.12.0.post3；公共规范及其实际位置由
`module help ascend/dev-workspace-9.2.0`查询。公共环境只读。
旧实验继续使用自己的固定源码、9.1模块和venv，不能只换module后沿用旧venv/二进制。

私有环境目录由调用者选择，可在仓库外；下例用`TIDE_VENV`传入位置，不要求特定目录名。
先初始化站点的module命令，再执行：

```sh
: "${TIDE_VENV:?请先设置你选择的私有环境目录}"
export TIDE_ASCEND_MODULE=ascend/dev-workspace-9.2.0
bash scripts/prepare_local_ascend.sh "$TIDE_VENV"
module purge
module load "$TIDE_ASCEND_MODULE"
source "$TIDE_VENV/bin/activate"
python scripts/ascend_environment.py --require-venv
python -m pip install --no-build-isolation --no-deps -e .
python scripts/build.py --backend npu --npu-runtime python --build-dir build/cann92-python --jobs all
```

`prepare_local_ascend.sh`核对venv的真实base prefix、Torch/TorchNPU版本和模块约束身份，
拒绝把旧环境或公共prefix当作新私有环境。增量包用`-c "$ASCEND_CONSTRAINTS_FILE"`安装。
venv通过`--system-site-packages`复用公共Python依赖；激活仅选择私有安装入口，CANN路径仍由module提供。
editable安装指向当前检出，源码修改立即影响使用者，适合本项目开发；下游应按自己的依赖锁安装固定wheel。
运行接口通过环境参数及site配置接收路径，不从本账户skills或本次实测目录推导依赖。
账号通用增量的发现/选择规则由站点文档或`use-local-ascend`维护；仓库仍提供完整的无AI安装说明。
历史证据中的实测路径用于溯源，不作为下一次安装的固定位置。
NPU Python-owned构建也可执行CPU张量；这不等于另做了一套CPU standalone资格。

独立C++使用另一个shell和已发布SDK，不加载Python TorchNPU库：

```sh
module purge
module load libtorch-npu/2.12.0-cann9.2.0
# Ninja须在PATH；可以只提供已安装工具的bin，不加载另一套NPU运行库。
python3 scripts/build.py --backend npu --npu-runtime standalone \
  --libtorch-prefix "$LIBTORCH_ROOT" --cxx11-abi 1 --build-dir build/cann92-sdk --jobs all
```

显式`--libtorch-prefix`从SDK的CMake元数据读取版本、hash实际库，构建驱动不导入Python Torch。
ABI必须与SDK发布记录匹配，链接及实际设备执行另行核验。CANN算子编译可能使用其Python运行时；
这不使C++核心依赖Python调度。旧2.10/CANN9.0及9.1证据继续按原身份保留，不自动升级。

迁移的固定源码小门禁为`scripts/qualify_ascend_migration.sh NEW_OUTPUT NPU_QUEUE_HELPER [all|half|N]`，
在干净固定检出通过`job.py`和有界background.slice服务运行。
它新建私有环境和两个独立构建，安装wheel，使用
[小配置](../configs/checks/ascend-migration.json)核验三客户端CPU/NPU的FP32完整推理、
独立根、SGD/槽、换owner恢复及新进程续接；另做CPU公式/梯度回归和私有LM小型随机模型检查。
队列最多等30分钟、每个客户端流程最多15分钟；构建并行由实测CPU/内存决定。
不下载权重、不重跑历史矩阵；resident、多卡、FP16、大batch和性能不在迁移小门禁范围。
[2026-11-06迁移证据](evidence/cann92-project-migration-20261106.md)记录三客户端CPU/NPU小型完整推理、
声明训练与新进程恢复通过，以及仍存在的三处独立综合根严格失败。当前状态见STATUS，不能由入口存在推断通过。
若已完成构建但迁移测试驱动需修复，`check_ascend_migration.py --build-root BUILD_ROOT
--output-dir NEW --device cpu|npu:0 --continue-from PREVIOUS_CHECKS`显式核对旧配置、已安装包及二进制身份后，
复用原related/whole结果，只补缺失训练续接；原NumericMismatch仍使整组严格失败。
check内部展开材料与原recipe可能具有不同workload_id；外部恢复须使用同一run配置保存的前缀，
不能放松checkpoint身份守卫。

NPU 验证启动器添加项目 `PYTHONPATH` 时须保留 module 已提供的路径；直接覆盖会使
CANN 初始化找不到其 `tbe` Python 包，即使 Torch/NPU 导入和构建已通过。
这是任务环境错误，应修正私有启动器并保留原失败；不要据此改数学实现或安装另一套公共包。
长任务使用独立 `background.slice` 服务、固定源码与有限时长；本机操作方法可参考
`run-durable-user-task`，项目启动器和记录仍由仓库维护。构建和测试可使用全部可用CPU、
内存与空闲NPU，集中加速或并行独立任务由实际占用决定。构建默认 `--jobs all`，
`--jobs half` 或正整数只是可选配置。计时前固定并记录ATen/BLAS/调度线程，避开
争用同一资源的重型构建/测试。共享NPU须有当前prompt明确授权，范围按该prompt限定，
不从历史记录推定；规则见[执行契约](execution-flows.md)。实际任务见STATUS。

## 下游接入与反馈

多实验仓库遵守[协作规范](downstream-contract.md)，模板和core-5安装消费者从该页导航。
修改示例时运行`tests/test_downstream_consumer.py`，再在固定源码的独立服务中运行
`python scripts/check_downstream_consumer.py --build-dir MATCHING_CPU_BUILD --output-dir NEW_OUTPUT`。
此入口复用并核验CPU构建、构建wheel安装到私有环境、复制消费者到源码外，验证实际公共调用与应用恢复。
它是定向安装检查，不是全仓库或设备资格。修改规范时同步模板、AGENTS与导航；正式资格仍先提交实现。

## 实现与验证次序

先声明语义差异/特例定义域，建立独立参考和小反例，随后实现、跑相关回归。
新一轮完整资格使用干净固定源码，先提交实现，再记录资格证据。
性能问题先小中规模 profiling，再有限对照；不把模型效果训练加入基座必交付。
每次交接只更新 STATUS/ROADMAP 和相关契约，避免生成多个相互竞争的任务清单。

CANN生成的合并对象不按普通C++对象缓存复用：2026-10-29的任务私有增量包装器
复用旧resident目录后，在CANN9.1链接阶段出现`ld.lld: unknown file type`，模型尚未运行。
改用正式`build_device_control.py`和新的resident build目录后完成构建。旧失败保留；
该例不要求改公共工具链，也不说明数学实现失败。core与resident分别核对源码/二进制hash，
跨Torch/CANN环境仍不得混用。

本机Torch2.12/Torch-NPU2.12在同进程先运行CPU autograd、随后才注册NPU时，
实际出现`device_ready_queues_`内部断言。首次固定资格的CPU设施10项与前向5项通过，
随后9项设备反向在该初始化边界失败。当前门禁将CPU与NPU消费者放入独立进程；
NPU应用应在第一次autograd之前调用`resolve_device("npu:0")`注册并选择设备。
不要把进程初始化失败当作分配算法或数值容差问题；晚注册路径未获资格，旧失败保留。独立两个新进程的标量探针已复现late失败、early通过；
具体程序和退出码随本次固定证据保存。

外部Dense/MoE的固定资产、私有增量环境、安装CLI与有限后台矩阵见[external-lm](external-lm.md)；模型通过与core-5资格分列。
