# LH a10fdb1：17.27B Attention 复现包

本包用于在另一台 **64 位小端 Linux CPU** 上重建本次测试。
包含固定图数据、nlohmann/json 3.11.2 头文件、CPU CMake、配置脚本、
可选的诊断入口和日志汇总脚本。Python 仅使用标准库；不用运行 LH 的
Python 解释器，也不用重新安装图生成依赖。LibTorch 和 LH 源码由目标机器提供。

默认是 `a10fdb1` 的 Attention 配置，将八处 emitD/receiveD 从 512 改成 2048。
其他模型设置保持原样：四头 Attention、allsoftmax、SiLU/RMSNorm、clear=true、
两次 body step、vocab=50304。batch=512、selectnum=1、100 次 think。
图来自原 Graph.py/BaseUtils.py，seed=7；固定图字节保证边数与本次运行一致。

| 核对项 | 应得到的值 |
| --- | --- |
| 每个静态图层级的节点数 | [0,1,7,224] |
| 静态节点 | 232 = 224 底层节点 + 8 hub |
| inputA / outputA / ioA / oiA 边数 | 984 / 984 / 232 / 8 |
| 参数量 | 17,269,426,339 |
| 模块 | inet、onet、Pronounce 全部 Attention |
| 原始随机策略 | 不额外固定模型和 token 随机种子 |

inet 和 onet 各自持有状态和参数；232 不是两者合计的实例数。
图生成时写入 cfg.json 的绝对 graph_node_info_dir 已改成
`../test/graph-data`，数值数据没有改变。**运行目录必须是 cpp 下的 build 目录**。

## 1. 解压并建立独立、干净的 LH checkout

下面假定现有 LH 在 `~/llm/lh`，复现包归档在当前目录。
新目录名可以自选；不要使用已存在的复现实验目录。

```bash
tar -xzf lh-a10-wide-repro-kit.tar.gz -C "$HOME"
lh_repro_packet="$HOME/lh-a10-wide-repro-kit"
cd "$lh_repro_packet"
sha256sum -c SHA256SUMS

git clone --no-hardlinks --no-checkout "$HOME/llm/lh" "$HOME/llm/lh-a10-wide-repro"
git -C "$HOME/llm/lh-a10-wide-repro" checkout --detach a10fdb1883fccd63ec21e36cc9cffa294c63c9e2
git -C "$HOME/llm/lh-a10-wide-repro" status --short

python3 "$lh_repro_packet/configure.py" --lh-root "$HOME/llm/lh-a10-wide-repro"
```

最后一条命令检查 revision、干净工作树和包内哈希，再安装配置。
它保留原 test-cortexnet.cpp 的计算循环，另生成 test-cortexnet-info.cpp。
`Connectome/cpp/repro/configured.json` 记录最终参数和文件哈希；
`configuration.patch` 记录已跟踪文件的修改。原有 `~/llm/lh` 工作树不受影响。

## 2. 编译 CPU 版本

需要 CMake >= 3.18、支持 C++17/OpenMP 的编译器，以及 CPU LibTorch。
优先使用原机器的原有环境，并保留 configure.log 中的版本和路径。
本次服务器使用 Torch 2.10.0+cpu / GCC 10.3.1；旧机器的版本和 BLAS 可能不同。

如果使用某个 Python 环境附带的 LibTorch，先激活该环境，确保下面 `python`
指向它，再运行：

```bash
cd "$HOME/llm/lh-a10-wide-repro/Connectome/cpp"
set -o pipefail
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$(TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -c 'import torch; print(torch.utils.cmake_prefix_path)')" \
  2>&1 | tee configure.log
cmake --build build --parallel 2 2>&1 | tee build.log
```

如果使用单独下载的 LibTorch，将 CMAKE_PREFIX_PATH 直接设为它的根目录，
例如 `-DCMAKE_PREFIX_PATH=/opt/libtorch`。需要在目标 CPU 上重新编译，
不要复制本服务器的 aarch64 二进制到 Intel 机器。

## 3. 运行带诊断信息的测试

下面以 56 线程为例。它不绑定具体 CPU，也不设置内存上限。
如需固定核心，先用 `lscpu -e=CPU,CORE,SOCKET,NODE` 查看编号，再在命令前
加适合该机器的 `taskset -c ...`。不要直接复制本服务器的 CPU160–319 编号。

```bash
cd "$HOME/llm/lh-a10-wide-repro/Connectome/cpp/build"
set -o pipefail
OMP_NUM_THREADS=56 MKL_NUM_THREADS=56 OPENBLAS_NUM_THREADS=1 \
  ./test-cortexnet-info-nograd --device cpu --dtype float32 2>&1 | tee nograd.log
lh_repro_exit=${PIPESTATUS[0]}
python3 "$lh_repro_packet/summarize.py" --log nograd.log \
  --configured ../repro/configured.json --native-exit-code "$lh_repro_exit" --output nograd-summary.json
```

随后运行开启 autograd 的前向：

```bash
OMP_NUM_THREADS=56 MKL_NUM_THREADS=56 OPENBLAS_NUM_THREADS=1 \
  ./test-cortexnet-info-grad --device cpu --dtype float32 2>&1 | tee grad-forward.log
lh_repro_exit=${PIPESTATUS[0]}
python3 "$lh_repro_packet/summarize.py" --log grad-forward.log \
  --configured ../repro/configured.json --native-exit-code "$lh_repro_exit" --output grad-forward-summary.json
```

`grad` **只表示开启 autograd 的前向**，没有 backward、optimizer 或每步 detach。
保留原实现和原有的有限 100 步状态延续。不要用这项测试认证 LH 的反向正确性。
诊断打印不会额外生成 token、clone 参数或计算 logits 归约。

输出前缀：

- `LH_ENV`：Torch/编译器/CPU 架构、ATen/interop 线程、BLAS/OpenMP 构建信息、线程环境变量。
- `LH_MODEL`：完整模型配置、总参数与各子模块参数、四块图边数、dtype、grad 模式、初始 KV 容量。
- `LH_STEP`：每步外层 think 耗时、摊销 ms/token、logits 的 requires_grad 和 shape、RSS、实际线程与 CPU 亲和性。
- `LH_DONE`：外层 think 的汇总。RSS 峰值是截至打印时的进程峰值，不包含随后模型销毁的潜在峰值。
- 原有 `Think took ... ms`、`t: ...` 仍原样保留；summarize.py 用这组整数毫秒计时作为主指标。

额外诊断发生在原始 Think 计时之外，但输出、资源采样和第二层计时仍可能影响
缓存或调度。因此另提供 **没有新增诊断的原始入口**，用于核对打印的影响：

```bash
OMP_NUM_THREADS=56 MKL_NUM_THREADS=56 OPENBLAS_NUM_THREADS=1 \
  ./test-cortexnet-nograd 2>&1 | tee original-nograd.log
```

另一个原始入口是 `test-cortexnet-grad`。两者按原程序接口启动，无命令行参数。
计时窗口均包含冷启动第一步；汇总另列 steps4–99、steps4–11、steps80–99。
`ms/token = sum(Think毫秒)/(batch×完成步数)`，不是单条序列的下一 token 延迟。
不同进程未固定权重/token RNG，不能把两次运行当作同一数值轨迹的严格对照。

## 可选的小规模构建检查

在另一个新的 a10fdb1 checkout 上先配置：

```bash
python3 "$lh_repro_packet/configure.py" --lh-root /path/to/fresh-lh-smoke \
  --width 64 --batch 4 --steps 4
```

按上述方式编译和运行，两种模式都应完成 4 步，参数量为 23,133,347。
这只验证入口、数据路径和当前 LibTorch 是否能运行，不能替代 17.27B 的测量。

反馈时可带上两份 summary.json、原始日志、configure.log、build.log、
`repro/configured.json`，以及 `lscpu`、`free -h` 的输出。
