# 当前交接 — FC4/FC16三层调查完成（2026-11-11）

用户授权先完整比较默认Tide、直接FC与Transformer，查明增量工作及架构影响。
本轮诊断、有限CPU/NPU实验及证据/导航检查已完成；没有修改生产计算路径。
暂不扩大LH/GPU/多卡或启动生产架构重写，后续先汇报对齐。
数学固定`0770593838a37c790a568226930567ed60f99b83`，19份core-5快照离线核验通过。

## 当前结果与入口

- [完整报告及复现](evidence/fc-three-layer-comparison-20261111.md)、[可查询JSON](evidence/fc-three-layer-comparison-20261111.json)：103个成本记录、8组投影比较、12个独立profile；三类模式、计时/数值/硬件证据分列。
- FC16/B8/L128/d128完整训练CPU1默认/直接/Transformer16为12.414/1.414/0.766秒；单NPU为16.525/0.745/0.0427秒。FC4结果同样显示明显默认路径差距。
- 已定位快照布局、Read/Full私有图、逐快照伴随和汇总开销；单省根包装不能闭合差距。外层4阶段仍有822次layout；NPU小profile默认82351 kernels、直接16224、Transformer16为2298。
- 四维SDPA的NPU Full仍走数学展开，未获得Transformer的融合路径；没有提升为默认实现。
- 主设施固定16项检查通过，最终扩展组13项通过（有重叠）。完整FP32任务投影仍严格超差；同一FP32材料提升FP64后值/控制/支持/状态通过，梯度两字段各1坐标仍严格失败。原失败/阈值保留，无非有限。
- 这是成本调查及指定投影检查，不增加all-root、多步optimizer、精确恢复、resident、规模学习或下游验收资格。现有限制/下一设计边界仅见[ROADMAP](ROADMAP.md)。

## 固定源码与终态作业

| unit | 固定输入 | 状态、资源和记录 |
| --- | --- | --- |
| `gef-fccompare-cpu` | `15d391f367c450effd25996b264c223bb8b075db` | passed/0；CPU80–111，ATen1/另测32；128GiB/5400秒；`artifacts/jobs/fccompare-cpu/` |
| `gef-fccompare-npu` | 同上 | passed/0；host0–7/ATen2，空闲physical11；128GiB/5400秒；`artifacts/jobs/fccompare-npu/` |
| `gef-fccompare-triage` | `b0edf96a8b4d86d461f83329559b14cd604ab3e2` | passed/0；CPU160–167；128GiB/1200秒；`artifacts/jobs/fccompare-triage/` |
| `gef-fccompare-sdpa` | `fd243f41417b3f2fd8b49156a577335de0588010` | passed/0；CPU80–111或NPU host0–7，空闲physical11；128GiB/1800秒；`artifacts/jobs/fccompare-sdpa/` |

四unit均inactive/MainPID0，无活跃任务，不恢复旧队列。外层passed不代表内部严格比较通过。
复用`e6f7bd5`的`artifacts/additive-work/build-native`与`env`；C++至测量提交无差异，未重建。
终态收集器固定`96500b0`，结果hash、吞吐算术、初始参数恢复和530条本地文件链接已核对。
主launch/命令/结果在`artifacts/fc-comparison-20261111/`；大型临时profile与固定源码在
`/dev/shm/zlong-gef-fc-20261111-15d391f/`。永久证据包含精确命令、配置、结果摘要与hash；
重现从固定Git源码生成材料，不依赖这些旧目录。最终13项JUnit为`final-diagnostics-qualification.xml`。
查看终态：`systemctl --user show gef-fccompare-cpu gef-fccompare-npu gef-fccompare-triage gef-fccompare-sdpa -p MainPID -p ActiveState -p Result`。

## 资源与接续

账户约292GiB/软300GiB/硬800GiB；收尾文件系统约900GiB可用，shm400GiB中使用约1.3GiB。
CPU/NPU主实验在不同CPU池并行，无重型构建重叠，未用共享NPU；磁盘没有阻塞或降低并行。
shm仅保存可重建临时材料，持久小证据已保留。若存储实质阻塞后继，及时汇报，不静默缩小范围。
共享NPU仍须当前prompt授权；不修改公共环境或参考仓库。

接续先运行`git status --short --branch`、`python scripts/status.py`、`python scripts/check_upstream.py`。
阅读上述报告、[成本契约](fc16-cost-attribution.md)、[架构](architecture.md)、
[经验入口](performance-lessons.md)与[ROADMAP](ROADMAP.md)。现有小型完整资格保持原提交身份。
用户未跟踪的`docs/tide-learning-lab-TL-REQ-003-handoff.md`与`TL-REQ-004-handoff.md`原样保留，不纳入提交。
