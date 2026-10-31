# 当前交接 — 2026-10-31

当前授权分支 `graph-execution-foundation`；允许阶段提交、普通推送。GPU真机本轮有界实施、资格、profile、有限对照与容量定位均已终态。
没有活动作业；严格失败与硬件限制仍保留，不称全部CUDA范围合格。数学固定
`0770593838a37c790a568226930567ed60f99b83`，19份core-5及旧core-3快照校验通过。
启动实现 `a066756fe23260e45d61afa158f87883d426f0d5`；保留所有后续源码，没有回退或覆盖用户修改。
只用新分支，不引入旧历史；历史身份与映射见 [history-migration](history-migration.md)。无子代理，参考仓库只读。

## 已交付与实际资格

- 实现：有超时/整组停止的容器subreaper；CUDA13条件graph API修复；明确单卡resident库存；IEEE策略与peer独立根；
  按首字段次序packing；CUDA控制所有权/增量预算；standalone显式JSON头依赖。
  主要提交 cc6b45f、c962961、7fd0576、1fe1e37、31d0afd、c368b45、6cacad0、8a34f66、32edb0f。
- [eager证据](evidence/core5-cuda-eager-20261031.md)：三客户端真实构建/加载/执行；PyTorch、Python native单卡两dtype各19/19，
  standalone FP32 23/24、FP16 22/24；KV mixed-a/b三客户端6/6。完整根/更新/恢复与CPU handoff分别记录。
- [32edb0f单卡resident与补验](evidence/core5-cuda-resident-20261031.md)：core、standalone/Python resident、standalone benchmark四阶段新构建通过，
  56线程/SM86；56组件/dtype+53公共项零跳过通过。scalar FP32六代表×三客户端18/18、standalone双卡6/6、native FP16六代表6/6。
  CPU仅小型公式与相关回归（包括110项放置和C++两dtype核），未开展完整x86 CPU资格或扫描。
- [时间线与有限成本](evidence/core5-cuda-finite-20261031.md)：固定实现 `32edb0f2b035ef2b732b6f27cf577f9e98058eb6`，目录a864fef；
  两个完整resident客户端FC-small trace各68540个graph-node活动/16次顶层launch，实际ready/packing/选择/路由/提交及反向/optimizer可见。
  原37项18通过/19失败，另2个小型FC scalar完整通过；39项中20有完整成本资格。小图GPU未见对同材料CPU收益，不外推规模。
- 时间线观察器c96ff85及修订601a9e8已提交；7项独立SQLite见证通过，固定修订实际读取三份trace约6秒。
  原无索引查询55秒超时/停止记录保留，没有重跑GPU profile。观察器不自动授予正确性/常驻/吞吐资格。

## 严格失败与硬件边界

当前限制与重启条件只维护在 [ROADMAP](ROADMAP.md)。本轮没有更改原容差、输入、epsilon、loss scale或strict xfail。
standalone FP32 SSD/AdamW、FP16反馈两例、mixed-a精确恢复及native双卡GDN精确恢复均未解除。
FP16两例在CPU复现同值；SSD显式scalar通过仍不算packed修复。
中型FC/HB/LH及HB-small resident有完整keys/values等严格差异。LH medium原queue64在checkpoint后容量拒绝；
静态物理边/端口推导的显式queue452越过容量，同栈新owner下一窗逐值通过，但独立第0窗node12 keys仍失败。
容量诊断不是合格吞吐，原37项结果没有重标。

全部12方向直接P2P/原生跨卡原子不支持，enable217。普通复制72单元、standalone通用owner36例/72更新/144连接窗、
显式Python CPU桥delta-pdg完整通过；不认证直接P2P、NCCL或多卡resident，AccumulateGrad流警告保留。
完整三卡门禁不跳过、不放宽；未改宿主驱动、IOMMU、PCIe或通信协议。

## 目标现场与终态记录

SSH通过项目私有 `artifacts/cuda3090/ssh.sh` / 认证CONNECT代理，凭据未写入证据，无全局配置修改。
目标根 `/root/autodl-tmp/gef-cuda3090`；私有venv为Python3.12.3/Torch2.12.1+cu130、pytest8.4.2/Ninja1.13.0、nlohmann_json3.11.2。
Toolkit13.0.88/GCC11.4/CMake3.22.1/Nsight2025.3.1，四张RTX3090/24GiB/SM86，CPU affinity112/配额56，内存240GiB。
构建使用56线程；计时ATen/BLAS/interop/native worker均1。源 `sources/32edb0f` 干净只读，构建 `builds/32edb0f-{core,resident-cpp,resident-python,benchmark}`。
`dev` 是基于cc6b45f的独立dirty开发目录，不作为资格来源，未覆盖/清理：旧placement、training/checker、CUDA控制、benchmark修改及四个新增CUDA/JSON脚本仍保留。
必要时先用 `git -C /root/autodl-tmp/gef-cuda3090/dev status --short` 审阅；正式来源仍是只读32edb0f。任务自身环境/源码/产物均不依赖旧aarch64二进制。

最近作业（均终止并确认无存活后代）：
- build-all-32edb0f、resident-single-32edb0f、profiles-32edb0f：passed。
- eager-regression-32edb0f、half-boundary-32edb0f、half-packing-32edb0f：failed，原断言保留。
- finite-32edb0f：failed/exit1，37项18/19，物理0，10800秒上限，实际约555秒。
- lh-queue-32edb0f：failed/exit1，queue452，900秒上限，实际约131秒；容量/精确恢复与独立数值分列。
- finite-scalar-32edb0f：passed/exit0，两项，600秒上限，实际约10秒。

所有目标日志 `jobs/NAME/task.log`、状态 `jobs/NAME/status.json`，输出 `runs/NAME`；原材料当前保留，但不是后续开发依赖。
初次86-real/API构建和resident-inventory-c368b45失败仍保留原身份；后续通过不能改写旧失败。
SSH断连继续与stop后无后代已实际验证。本机没有长作业。最终目标四卡均1MiB/0%利用率，数据盘剩约48.2GiB；未改公共环境/宿主配置。

## 接续入口

先查询已审阅结果和ROADMAP停止条件；不要重复旧全矩阵、相同失败候选或扩为1B/8B/16B扫描。
下一数值工作须提出覆盖实际首个差异及独立根的新算术/累加方案；性能候选优先实际局部行合批或显式反向模板重绑，
须独立验证journal/根/版本/跨窗，不能按shape复用旧owner。完整x86 CPU资格仍后置；多卡resident仍受硬件限制。

```sh
git status --short --branch
python scripts/status.py
python scripts/check_upstream.py
python -S scripts/index_core5_study.py docs/evidence/core5-cuda-finite-20261031.json --state failed --format json
python -m unittest discover -s tests -p test_cuda_timeline.py -v
```

无旧运行目录的私有环境/重建、完整单卡资格、逐配置复现和profile命令见上述三份CUDA报告及 [development](development.md)。
有限配置 `benchmarks/core5-cuda-finite-v1.json` / `run_core5_study.py`，显式资源覆盖与实际SHA在审阅JSON中；只读参考不变。
本次证据提交同时更新能力声明、索引、复现入口和正负经验；本机实现修改均已提交，远端dev临时dirty树保留；提交/推送与本机干净状态以Git为准。
