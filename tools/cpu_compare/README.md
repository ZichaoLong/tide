# LH / PDG 固定图 CPU 一键对照

这是源码包：固定图、两边的 C++ 源码和 nlohmann/json 头文件均已携带。
不用安装 LH Python 解释器或图生成依赖，也不需要另外找 LH/Tide checkout。
Python 脚本负责构建和记录，模型计算全部运行在独立 C++/LibTorch 进程中。

需要 64 位小端 Linux、Python 3.8+、CMake 3.18+、C++17/OpenMP 编译器和
CPU LibTorch。本机验证栈是 Torch 2.10.0+cpu / GCC 10.3.1 / aarch64；
Intel x86_64 需在目标机器重新编译，优先使用相同 Torch 版本。
包内没有二进制、模型权重或本机绝对输入路径。

默认：17,269,426,339 参数，D2048、B512、V50304、FP32/no_grad，
12 tokens，前4个 warmup，报告索引4–11。四头 Attention、all-softmax、
SiLU/RMS、clear、两次 body step/token；seed7，固定外部 token IDs。
权重独立初始化，两边使用相同四块静态图；不宣称函数完全相同。

可选 `--memory add` 将两侧 cortex 和 readout 都实例化为 tick-decayed
Add + all-softmax；D2048 固定图有9,468,020,899个参数（除以1024³约8.818）。
保留相同连接、SiLU/RMS、clear和两次body step。PDG 使用正典的
`lh-add-repeat-v1` 与 source-aware Aggregate；LH 保留原始 Add 内核。
`--mode grad-forward` 只开启autograd前向，无backward/optimizer/detach，默认关闭计数。
LH 的 grad-forward 仍拒绝 work-count/operator-profile；PDG 可显式开启这些
诊断计数和计时，但正式性能对照应关闭它们。

对照历史关闭RAII timer的测试，可用：

```bash
python run_lh.py --device cpu --threads 56 --memory add --lh-timer outer --work-count 0 --output-dir runs/lh-add
python run_pdg.py --device cpu --threads 56 --memory add --phase-profile 0 --work-count 0 --output-dir runs/pdg-add
```

`--lh-timer outer` 在独立准备副本中关闭原RAII timer，使用已有的外层steady-clock
区间；原始默认仍解析`Think`整数毫秒。`--phase-profile 0`关闭PDG调度阶段计时。
两者的构造、数值检查及日志不在主计时内。加`--mode grad-forward`可测同窗口
开启autograd的前向；显式保留`--work-count 0`。Add拒绝非默认的Attention专用选项。

## 两条正式测试命令

解压后，在目标 Torch Python 环境中执行：

```bash
tar -xzf cpu-attention-compare.tar.gz
cd cpu-attention-compare
sha256sum -c SHA256SUMS

python run_lh.py --device cpu --threads 56 --output-dir runs/lh-wide
python run_pdg.py --device cpu --threads 56 --output-dir runs/pdg-wide
```

PDG 默认 `--attention-packing exact`，按精确 KV/query 形状分桶。
可使用 `--attention-packing single`，将同一节点的真实 query 合为一个批次，
对 KV 补齐并按 sample 遮罩；仍保留 QKV 批处理、node 并行和其他设置：

```bash
python run_pdg.py --device cpu --threads 56 --attention-packing single --output-dir runs/pdg-single
```

该选项仅改变 built-in same-fiber Attention 的执行策略。缓存管理、逐事件
pooling 和训练语义保持原有方式；LH 入口不接受此选项。
`--smoke --attention-packing single` 可先检查新策略。

PDG 另有五个独立实验选项，默认保留此前路径：`--fiber-pooling event|csr`、
`--fiber-cache cloned|owned`、`--defer-state-release 0|1`、
`--projection-layout input|linear`、`--attention-layout event|head`。分别测试节点批量 CSR pooling、复用已有的
样本 KV 分配、并行清理旧状态容器、投影权重布局，以及 attention 临时张量的 head 排列。
这些选项不改变图和参数量；性能需按机器与配置分别测量。

还有两个默认关闭的独立选项：`--packed-sources 0|1` 将来源载荷批量缩放后在
Aggregate 与 Attention 间复用；`--batch-next 0|1` 批量采用/清空状态。
两者保留完整来源、比较快照与下一状态语义，可分别启用做对照：

```bash
python run_pdg.py --device cpu --threads 56 --packed-sources 1 --batch-next 1 --output-dir runs/pdg-transport
```

`op/fiber_scale_elements` 记录实际来源缩放，`op/fiber_reused_elements` 记录复用量；
逻辑事件数与这些实际操作计数分开报告。先加 `--smoke` 检查目标机。

每条命令独立完成：校验包 → 准备专用源码目录 → CMake 构建 → 运行 → 汇总。
运行目录必须是新目录；重测请换名字。编译默认2个作业，可用 `--jobs 4`。
默认不设置内存上限、不绑定特定 CPU 编号；`--threads` 默认最多56个可用逻辑 CPU。
需要固定核时，在两条命令前使用目标机合适的 `taskset -c ...`；
脚本继承并记录 CPU affinity。LH 和 PDG 应顺序运行，使用相同 affinity。

`--threads 56` 的含义：LH 的 ATen/OpenMP 为56；PDG 的 node/head worker
分别为56，分阶段运行，ATen/OpenMP/BLAS 请求1。实际 BLAS 池大小会打印；
OpenMP OpenBLAS 可能忽略 `OPENBLAS_NUM_THREADS=1` 而跟随 OpenMP，不能只看环境变量。
LH 使用 OpenMP 默认等待策略，PDG 设置 PASSIVE，与本次测量入口一致。

## 首次在目标机运行

建议先用同一入口完成小规模检查：

```bash
python run_lh.py --device cpu --threads 4 --smoke --output-dir runs/lh-smoke
python run_pdg.py --device cpu --threads 4 --smoke --output-dir runs/pdg-smoke
```

`--smoke` 是 D16/B4/V257/6 tokens/warmup2；PDG 同时执行小规模完整状态对齐检查。
小规模通过表示目标机的构建和入口可用，性能对照仍应运行上面的正式配置。
`--width/--batch/--steps/--warmup/--vocab` 可显式覆盖默认值；它们会完整记录。
本次 LH 入口是 FP32，PDG 还允许 `--dtype float64` 做单独的小规模验证。

使用独立 LibTorch、Python 中没有安装 torch 时，给两条命令都加：

```text
--torch-prefix /path/to/libtorch --libtorch-label torch-2.10.0-cpu
```

该路径只用于目标机器的 CMake 配置，不需要 Python Torch。
工具不会自动安装或替换任何环境。默认 native 超时3600秒，configure/build
各自超时3600秒；慢机器可显式增大 `--timeout-seconds` / `--build-timeout-seconds`。
`--memory-gib 0` 表示没有额外地址空间上限；可显式设置其他值。

## 打印和记录

终端打印 CONFIG、LibTorch 来源、构建阶段和日志路径、实际线程池、每 token
耗时、QKV 行数、attention 分组调用数，以及测量窗口的均值/中位数/离散程度、
吞吐、矩阵 GFLOPs、padding 比和进程峰值 RSS。长阶段每30秒打印存活信息。

每个 run 目录包含：

- `run.json` / `summary.json`：配置、源码与输入身份、成功/失败、测量窗口、资源；
- `metrics.jsonl`：逐 token 原始标量，包括 warmup；
- `stdout.log`：完整 native 输出；`configure.log` / `build.log`：构建及编译器信息；
- `host.json`：CPU、内存、affinity、环境及 CMake 信息；
- `prepared.json` / `source/`：本次实际使用的配置源码、哈希和目标机构建结果。

`ms/sample-token` 除以了 batch；不是单条序列的一次迭代延迟。构造不计入
测量窗口，KV/history 跨 token 保留。LH 默认主指标来自原 Think 整数毫秒计时（outer选项使用外层高精度计时），
PDG 主指标来自 native cursor+head；两边计数和统计输出发生在主计时之外，
计数更新本身仍在计时中。`--work-count 0` 可单独检查计数开销。

两边可加 `--operator-profile 1` 输出 QKV、KV 构建/搬运、attention、pooling、
输出投影和 Emit 的细分计时，默认关闭。`detail/*_worker_seconds` 是各调用线程
互不重叠的累计耗时，包含线程被调度暂停的时间；不是端到端耗时，不能与
`profile/*` 相加。LH 需要新生成且标明支持此功能的源码包。性能结论应使用
关闭细分计时的配对测试，并单独检查计时开销。

Trackio 默认为 best-effort；未安装时仍完成完整本地记录，不自动安装。
可以使用 `--tracking off` 关闭投影。比较时先检查 summary 的 completed 状态，
然后反馈两份 run.json、summary.json、metrics.jsonl、host.json 及日志即可，
不必复制 source/build。中断、编译失败和超时均保留失败记录并返回非零退出码。

本源码包来自仓库的 `tools/cpu_compare/`；导出入口是
`scripts/export_cpu_compare.py --lh-prepared PATH --topology PATH --output-dir NEW`。
复现不依赖旧的 commit ID 操作指令；包内 manifest 已记录精确来源和全部文件哈希。
## PDG 的可选批量 autograd 路径

`--full-autograd replay|batched` 与 `--aggregate-autograd replay|batched` 是两个
独立选项，默认均为 `replay`。前者批量计算 Full 的仿射 VJP，后者批量计算内置
Aggregate 的来源缩放、贡献和归约 VJP，保留各个公开输出的独立梯度依赖。
状态与 Read 的 replay 仍然保留；两项都可用于 Add 或 Attention workload。

例如开启两项优化，测量 Add 的 grad-forward：

```bash
python run_pdg.py --device cpu --threads 56 --memory add --mode grad-forward \
  --full-autograd batched --aggregate-autograd batched \
  --phase-profile 0 --operator-profile 0 --work-count 0 \
  --output-dir runs/pdg-add-grad-batched
```

固定 Full 为 `batched`、只切换 Aggregate，可单独衡量 Aggregate 的收益。
小规模 `--smoke` 会先验证 scalar/batched 的梯度根；目标机器先做该检查。
这些选项覆盖声明的 CPU FP32/FP64 一阶训练语义，保留 None/connected-zero、
共享参数和规范化梯度；不支持任意自定义模块或高阶微分。在 nograd 下仍使用
原有数值批处理。此入口只计 grad-forward，不执行 backward 或 optimizer。
详情见仓库的 `docs/full-batched-autograd.md` 与
`docs/aggregate-batched-autograd.md`。

另开诊断 run，可使用 `--phase-profile 1 --operator-profile 1 --work-count 1`。
Aggregate/state/Read replay 的 worker 累计计时可能在线程之间重叠，不能相加后
当作端到端延迟，也不能与关闭诊断的正式结果混用。
