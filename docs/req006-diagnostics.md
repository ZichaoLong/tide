# 重复诊断 TL-REQ-006 的直接实现

入口是 [diagnose_req006.py](../scripts/diagnose_req006.py)。它执行下游固定提交
`6e7bd49eb5e35c15445826cac3a9b6ddda75aff4` 的实际直接实现，帮助判断局部计算、
节点/区域装配、反向、检查和 optimizer 的成本。它不运行基座默认执行器，也不认证基座
完整独立根、一般图调度或 continuation。权威请求为
[TL-REQ-006@dc943cc7](https://github.com/ZichaoLong/tide/blob/dc943cc7ce773e3dac4d62736ab524953573decb/docs/requests/TL-REQ-006.md)。

四份原始配置按固定报告 `89959587` 的 `config_sha256` 核对，原始字节以 gzip 保留于
[配置清单](../examples/diagnostics/req006/manifest.json)。重现不读取旧 `runs/`、活动配置、
checkpoint 或安装的 `tide_lab`。调用者提供含固定提交的下游 Git 仓库；脚本仅 `git archive`
其 Python 源码到新的输出目录，从那里导入并记录每文件 hash，结束时复核未变。
下游 Git 可独立检出固定版本；不需把其历史合并到基座开发分支。

| case | 原配置 | 原物理批 / 全局批 |
| --- | --- | --- |
| hb88-pool-s11 | 88节点、22区域，每区4；d128/FFN512；KV qry、pooled | 16 / 64 |
| hb184-quad-s11 | 184节点、46区域，每区4；d128/FFN512；regional_total/context/quad | 8 / 64 |
| tree-r64-h2-q3-s11 | 640主体节点及21边界节点；d64/FFN256；quad、活动退火 | 4 / 64 |
| tf-hb184-s11 | Transformer512×16，FFN2048、4 heads | 64 / 64 |

均为 L256、FP32。R64 的边界区域、主体区域和退火配置以完整配置为准；`k` 是发送强度
预算，不是硬 Top-k 数量。`--progress-tokens` 必填，将活动退火与学习率固定到明确进度。
脚本使用固定构造器/seed 的**初始权重及合成 token**，指定进度不代表恢复了该训练步。
这些新输入产生的工作量和性能不能冒充原学习轨迹或历史采样窗口。

此次下游 `SmallTransformer/CausalBlock` 明确计算 `QKᵀ → softmax → AV`；不是本仓库
FC 对照中的融合 SDPA Transformer。两者具有独立身份。

## 独立测量、trace 和计算图统计

在已选择的项目 Torch 环境执行；以下是 workload 命令。原规模长任务仍须使用
[有界持久作业](execution-flows.md#7-提交资源与交接)，先固定基座源码并完成设备准入。
本工具不自行分配/抢占设备，不修改下游仓库或其活动任务。

```sh
python scripts/diagnose_req006.py \
  --downstream-repo /path/to/tide-learning-lab \
  --case hb88-pool-s11 --progress-tokens 0 \
  --device cpu --threads 32 --mode training --action measure \
  --output-dir artifacts/req006-measure-new

python scripts/diagnose_req006.py \
  --downstream-repo /path/to/tide-learning-lab \
  --case hb88-pool-s11 --progress-tokens 0 \
  --device npu:0 --threads 2 --mode training --action profile \
  --output-dir artifacts/req006-profile-new
```

`--mode nograd|grad-forward|training`分别运行无梯度前向、构图前向及完整真实更新。
前向保留模型原有观测，按 microbatch 消费后释放输出/计算图，不做 loss 或反向。
训练直接调用固定下游 `logical_update`，保持有效目标总量归一化、每个 microbatch 的
反向及观测、一次 clip/AdamW 边界。全局批64不等于每个算子 B64。

每次 warmup 后和重复测量前恢复同一模型、optimizer 与 CPU/设备 RNG；原始 CPU
optimizer 槽也复制，避免 step counter 污染基准初态。输入在 CPU 一次生成，物理分批
不改变输入内容。`--microbatch/--global-batch/--length`只用于明确的形状诊断，覆盖值
与原配置同时记录，不能代替原形状成本。`--config`接受额外小型直接/Transformer配置。

默认初态的 AdamW 槽为空，完整更新计时包含首次槽创建。研究已有槽的更新时显式加
`--prime-updates 1`：先在同一固定进度做一次真实合成更新，再冻结其权重/槽/RNG，后续
warmup和计时均从该初态恢复。prime数量进入材料身份，不作为原训练checkpoint或学习证据。

需要跨进程配对 measure/profile/census 时，可导出并复用**同一份 prime 后初态**。
只使用相同 seed 和 prime 次数不能保证两个设备进程更新后的权重逐位相同。
`--initial-state-out`保存模型参数/缓冲、完整 optimizer 槽和 CPU/设备 RNG；
`--initial-state-in`先校验材料，再恢复并核对完整状态 hash，**不会重新执行 prime**。
载入时仍提供原来的 `--prime-updates`，它声明保存材料的来源，并非要求再次更新。

```sh
python scripts/diagnose_req006.py \
  --downstream-repo /path/to/tide-learning-lab \
  --case hb88-pool-s11 --progress-tokens 0 --prime-updates 1 \
  --device cpu --threads 32 --mode training --action measure \
  --initial-state-out artifacts/req006-paired-initial.pt \
  --output-dir artifacts/req006-paired-measure

for action in profile census; do
  python scripts/diagnose_req006.py \
    --downstream-repo /path/to/tide-learning-lab \
    --case hb88-pool-s11 --progress-tokens 0 --prime-updates 1 \
    --device cpu --threads 32 --mode training --action "$action" \
    --initial-state-in artifacts/req006-paired-initial.pt \
    --output-dir "artifacts/req006-paired-$action"
done
```

两个参数互斥。校验绑定完整有效配置、实际输入字节及形状、输入/模型 seed、进度、dtype、
prime 次数、固定下游源码及关键数值运行时（包括实际线程、精度设置及设备类型/型号）。
action、mode、物理卡号和输出目录可不同；CPU/NPU 或线程数不同则不能声称此种精确配对。
`result.json`记录文件/材料/权重 hash、生产源码、载入方向及 `executed_prime_updates`。
同一初态不保证后续非确定设备计算逐位相同，最终模型/optimizer hash仍独立记录。

artifact 使用 `weights_only=True`载入，身份和内部内容 hash在恢复 live owner 前检查。
保存先写同目录 staging并 fsync，再原子发布且**不覆盖已有文件或符号链接**；并发发布者
已占用目标时拒绝。本地文件保存的实际字节受磁盘余量影响；开始前检查空间，失败清理 staging。
目录 fsync若在发布后失败，完整新文件可能已存在，命令仍失败，不能据文件存在推断成功。
初态 I/O/hash不进入步骤计时；此文件不是下游训练checkpoint，也不认证基座 continuation。
轻量复现可用`--config tests/fixtures/req006-tiny-transformer.json`替换`--case`：
实际为width4/一层、global batch2/microbatch1/L3。它保留原配置的其余元数据以固定字节身份，
其中旧recipe/calibration摘要不代表这个小模型的测量或效果。

默认一次 warmup、一次测量；`--iterations`可显式增加独立初态重复。profile/census
每进程只跑一次受观察步骤，分别启动新进程、新输出目录；插桩结果不报告吞吐。

- `measure`：无 profiler 或阶段标签。分别报告包含分批/上传/目标计数的完整区间、
  固定下游原有更新区间；分子为有效非 ignore 目标，另列输入位置数。没有生成 token 或 TTFT 声明。
- `profile`：标记 aggregate、memory/KV、region/allocation、NodeLinear、loss、反向、
  clip/optimizer 等实际范围；导出 CPU/设备 trace、矩阵形状、索引/装配、标量/同步、
  反向引擎调用及 kernel 类型。独立线程范围、嵌套 ATen/矩阵调用和 device kernels
  分开；inclusive 时间不能相加成 wall-time 分摊。AICPU 不等于 host fallback。
- `census`：逐 microbatch 在反向前登记实际 logits 和 loss，复用公共 autograd census；统计
  可见节点、边、保存张量和真实反向执行。分配的自定义 Function 内部求解无 autograd 图，
  不将一个 Function 当作一个设备 kernel，也不推断未暴露的 opaque native 内部图。

记录参数/输入/初态及最终模型与 optimizer hash、原始配置、实际运行时、线程/affinity、
可见设备环境和内存峰值。CPU RSS 是含构造/warmup 的进程高水位；设备 allocator 峰值在
恢复后重置，包含活跃模型。输入构造、恢复、预热、哈希、trace导出和文件I/O从步骤计时
排除，诊断作业自身总耗时另记，不称完整学习作业吞吐。

## 预算、结果和维护

`result.json`按完整记录原子更新；错误保存为 failed，原始 trace 保留。
profile 默认声明8GiB trace空间，开始前核对可用空间，导出后检查大小；该检查不是
文件系统写入硬配额，调用者仍须提供服务时限/内存和存储余量。超预算不称成功。
CPU trace允许没有设备 kernel；请求加速器但未记录真实 kernel 明确失败。

时间线指向静态线索，因果判断仍需同材料候选对照。当前固定源码已有 `[node,B×L,d]`
bmm、自定义分配VJP和可选 matmul 聚合；节点/区域间 select/stack、参数高级索引、
28轮 FP32 求根与3轮 quad 校正、观测及 optimizer 是待测边界，不能直接写成已证实热点。

脚本职责分为[材料身份](../scripts/req006_material.py)、[初态配对](../scripts/req006_initial.py)、
[执行与恢复](../scripts/req006_runtime.py)、
[插桩与trace汇总](../scripts/req006_profile.py)。
[定向检查](../tests/test_req006_diagnostics.py)覆盖固定 Git 字节、配置hash、带槽恢复、
分批归一化/计时分母和设备/主机计数区分。它们不替代实际四代表性能、严格数值或下游验收。
[初态配对检查](../tests/test_req006_initial_state.py)覆盖含随机输入/Dropout的精确下一次
AdamW更新、身份/内容拒绝、加载字节预算和失败/并发写入不覆盖已有材料。
新证据由基座[证据索引](evidence/README.md)导航，当前待办仍只维护在 ROADMAP。

`diagnose_scalar_floor.py`将实际浮点Scalar与同dtype/device的0D Tensor阈值分开调查，
以独立分段公式验证阈值两侧、相等点、signed zero及次正规值，再独立测量/profile。
默认保留请求的Python float；`--bound-representation dtype`显式先舍入为输入dtype的精确
可表示常量，原请求、实际参数、输出dtype阈值均记录。它用于区分Scalar提升/比较边界与
算子成本，不改变生产模型的正支持、下游clamp或原失败。不得用`maximum`代替：相等点
的输入导数不同。任一公式失败均保留；对不同常量表示的成功结果不能覆盖原来的失败。
