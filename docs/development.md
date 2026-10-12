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
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python scripts/build.py --backend cpu --build-dir build/cpu --jobs 2
TIDE_BUILD_DIR="$PWD/build/cpu" python -m pytest -q tests/test_semantics.py tests/test_native_settle.py --dtype both
```

`build.py` 使用当前 Python 对应 Torch 的 CMake/ABI，并生成新的 build manifest。
公共独立 C++ 不依赖 Python 调度；纯 LibTorch 配置入口：

```sh
cmake -S . -B build/cpu-standalone -DCMAKE_PREFIX_PATH=/path/to/libtorch \
  -DTIDE_BACKEND=CPU -DTIDE_PYTHON_BINDINGS=OFF
cmake --build build/cpu-standalone --parallel 2
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

本机截至 2026-10-12 的定位提示（迁移时不依赖这些路径）：

- NPU：公共 module `libtorch-npu/2.10.0-cann9.0.0`；配套 Python 位于
  `/opt/miniconda/envs/ascend900-train-full-torch-npu-2.10.0-py311/bin/python`。
- CUDA：私有 module `torch-cuda/2.10.0-cu128`，工具链在
  `/mi/data2T/zlong/gpu-toolchains`；本机没有 GPU。
- CANN 使用 environment modules 暴露的 `/opt` 版本；不改共享驱动或厂商库。
- 长任务用本机 `run-durable-user-task` 规范的独立 `background.slice` 服务。
  设备使用按用户授权和本机 skill 检查健康/余量；不停止其他用户的进程。
  cgroup 配置是请求，不能据此宣称内核已完全执行资源隔离。

这些环境没有随旧实验产物清理。未来启动前核对实际模块和设备，别沿用旧任务队列。

## 实现与验证次序

先声明语义差异/特例定义域，建立独立参考和小反例，随后实现、跑相关回归。
新一轮完整资格使用干净固定源码，先提交实现，再记录资格证据。
性能问题先小中规模 profiling，再有限对照；不把模型效果训练加入基座必交付。
每次交接只更新 STATUS/ROADMAP 和相关契约，避免生成多个相互竞争的任务清单。
