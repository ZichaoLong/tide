# 从干净源码开始开发

本页使用新路径，不要求任何旧 `TASK`、构建、运行目录或 LH 快照。
先读 [AGENTS](../AGENTS.md)、[STATUS](STATUS.md)、[执行契约](execution-flows.md)。
当前采用的教材与实现域从 [语义索引](semantic-map.md) 进入。

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
CUDA 实机门禁仍未完成；不要将本机编译结果写成 GPU 运行通过。

本机截至 2026-10-27 的实际规范来源是
`/opt/software/ascend-locks/dev-workspace-9.1.0/README.md` 和对应 `/etc/modulefiles`。
旧 `/home/zlong/ASCEND_CODING_AGENT_GUIDE.md` 已不存在，不按失效路径加载环境。
当前公共入口为 `ascend/dev-workspace-9.1.0`（CANN 9.1、Torch 2.12、Python 3.12）。
项目私有增量使用模块提供的 constraints，不修改公共环境：

```sh
module purge
module load ascend/dev-workspace-9.1.0
python -m venv --system-site-packages .local/venv-core5  # 首次
source .local/venv-core5/bin/activate
python -m pip install -c "$ASCEND_CONSTRAINTS_FILE" pytest ninja
python -m pip install --no-build-isolation --no-deps -e .
```

新环境的最小 CPU 链路与后续各设备证据分开；环境存在不代表完整矩阵通过。
旧独立 LibTorch-NPU 2.10/CANN9.0 与私有 CUDA2.10/cu128 是不同工具链，不能混用二进制。
使用独立 SDK 时，CANN 还需配套 Python 的运行时入口：把该 SDK 配套 Python 的 `bin`
置于任务 `PATH` 首位，并核对 `python3-config --prefix` 与该 Python 的 `sys.prefix` 一致。
本机 `libtorch-npu/2.10.0-cann9.0.0` 模块本身不设置 Python PATH；不能让 9.1 私有 venv
或系统 `python3-config` 混入该任务。只调整任务环境，不修改公共 module 或软件栈。
该 SDK 任务还须能找到 Ninja；`build.py` 明确选择 Ninja，缺少它会在配置阶段失败。
可以从项目私有工具目录提供 Ninja，但 SDK 配套 Python/bin 必须在 PATH 最前，
且不激活另一套 Python 环境。分别核对 `command -v ninja`、`python3-config --prefix`
和实际 SDK Python；构建工具可复用不表示 Torch/CANN 二进制可混用。
NPU 验证启动器添加项目 `PYTHONPATH` 时须保留 module 已提供的路径；直接覆盖会使
CANN 初始化找不到其 `tbe` Python 包，即使 Torch/NPU 导入和构建已通过。
这是任务环境错误，应修正私有启动器并保留原失败；不要据此改数学实现或安装另一套公共包。
长任务继续使用 `run-durable-user-task` 的独立 `background.slice` 服务，
固定源码与有限时长。构建默认使用进程可用的全部 CPU，`--jobs half` 使用一半，
也可显式指定正整数；实际资源约束造成的降低须说明原因。测试使用空闲 CPU/NPU/内存，
计时前固定并记录 ATen/BLAS/调度线程，项目重型构建与性能计时分开。实际任务见 STATUS。

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
