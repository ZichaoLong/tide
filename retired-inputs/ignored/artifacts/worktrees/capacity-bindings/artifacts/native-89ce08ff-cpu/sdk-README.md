# LibTorch-NPU 2.12 / CANN 9.2.0（站点构建 r1）

此为官方 Ascend/pytorch 的独立 `build_libtorch_npu.py` 源码构建，不是从 Python wheel 提取后端库，也不是华为发布的独立二进制 SDK。保持现有驱动 25.3.rc1。公共 CPU LibTorch 2.12 头文件/库/CMake 已独立复制到 `/opt/software/libtorch/2.12.0-cpu-aarch64-abi1`；运行不需要 Python 环境或解释器。

```bash
module load libtorch-npu/2.12.0-cann9.2.0
cmake -S /opt/software/libtorch-npu/2.12.0-cann9.2.0-aarch64-abi1/tests -B ./sdk-test-build -DCMAKE_BUILD_TYPE=Release
cmake --build ./sdk-test-build --parallel
./sdk-test-build/sdk_acceptance cpu ./cpu-first
./sdk-test-build/sdk_acceptance cpu ./cpu-resumed ./cpu-first/checkpoint.pt ./cpu-first/expected.pt
# 先选择真正空闲的物理 NPU，设置 ASCEND_RT_VISIBLE_DEVICES；程序使用逻辑 npu:0。
./sdk-test-build/sdk_acceptance npu:0 ./npu-first
./sdk-test-build/sdk_acceptance npu:0 ./npu-resumed ./npu-first/checkpoint.pt ./npu-first/expected.pt
```

每个输出目录必须新建。Python TorchNPU 与 standalone SDK 使用独立终端/模块环境；不能把两种 libtorch_npu 混进同一进程。Python 原生扩展通常链接当前 Python 的 Torch，由 torch_npu 注册后端，不加载本 SDK。

`qualification.json` 和 `tests/evidence/` 保存普通用户下实际 CPU/NPU FP32、三步梯度/AdamW 槽位及新进程精确恢复结果，不代表所有算子/模型/精度/多卡已通过。C++核心没有 LibPython、libtorch_python 或构建桩库的运行依赖。

构建来源为官方 post3 提交 `6b88d20d143ef15243c44f3582f37c480e786f2e`。`build-manifest.json` 保存冻结源文件和 CANN 公开头文件 hash、160 编译线程与资源约束。调整仅涉及打包：私有 torchgen、禁用桩库 RPATH、使用匹配 CANN 9.2 公开 ACL/super_kernel 头文件；可选 Tensorpipe RPC 未构建。GitCode 固定 Tensorpipe 提交未能从 GitHub 镜像取得，不替换为其他提交。HCCL 不由 RPC 缺席推断不可用，需按客户端单独验证。

构建工具和源码在 recipes/、src/；不得在已发布前缀原地重建。管理员复建应选新修订目录、固定提交、重新生成 manifest 并重新验收。NNAL/ATB 专用调用和全部分布式能力不在本次小型资格范围内。

本版包含一个站点构建修复：官方 post3 的 ProcessGroupHCCL 引用了 NPUPluggableAllocator，但 standalone CMake 未编入其实现，导致消费者链接失败。`recipes/standalone-pluggable-allocator.patch` 仅补编已有且带 BUILD_LIBTORCH 守卫的源文件，不替换算法或引入 Python。原失败保留在 tests/evidence/initial-consumer-link-failure.log，修复后重新链接、loader 和 CPU/NPU 三步完整验收通过。从官方源码复建时，在匹配头文件/隔离 torchgen 等打包步骤后、首次构建前应用该 patch。此修复不代表 C++ HCCL 已经真机认证。
