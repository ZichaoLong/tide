# 当前交接 — 2026-10-12

本仓库已完成 **tide-core-3 的本机交付**，实现交付提交为 `0edcd2b`。
当前整理任务：精简导航、保存固定教材快照、补语义对应、退役旧实验产物，
验证从干净源码接续。没有运行或排队的实验；历史 CPU Attention 任务已取消。
新版教材及特例等待用户审核，本轮不采用新语义、不启动新性能矩阵。

当前维护清单服务 `gef-reentry-retire-plan02-20261012.service`；日志与状态在
`/mi/data2T/zlong/gef-maintenance-20261012/retire02/`。首个清单过程超时，未执行删除；
已优化扫描并缩减大记录。新清单审阅后才执行删除。

## 新线程入口

1. 在目标仓库运行 `git status --short --branch`、`python scripts/status.py`。
2. 阅读 [执行契约](execution-flows.md) 与 [唯一待办表](ROADMAP.md)。
3. 用 [语义索引](semantic-map.md) 定位固定教材、接口、实现和测试，按需读模块文档。
4. 按 [开发入口](development.md) 新建构建/输出目录；先运行 `python scripts/check_upstream.py`。
5. 用户提供新版教材后，先比较固定版本的语义差异及具体特例需求，再按授权开发。

本机路径 `/home/zlong/llm/graph-execution-foundation` 实际指向
`/var/tmp/zlong-graph-execution-foundation/repository`；分支 `graph-execution-foundation`。
始终显式指定仓库 cwd。ObsidianVault、lh、fractal-latcarf 是只读参考。
旧报告中的外部路径是历史来源，不是启动依赖；见 [产物保留规则](artifact-retention.md)。

## 已交付与证据边界

| 部分 | 当前结论 |
| --- | --- |
| 图与模块 | 三类图、独立调度、局部程序、续接、checkpoint、声明的一阶训练；具体表示与模块域见语义索引 |
| CPU | 最后 G 阶段 composite 9650 项接受（9648 原通过＋2 同源码资源准入修复），654 可选跳过；范围与源码见证据 |
| NPU | 公共模块 236 正例＋2 CSR 拒绝；常驻 133 组件调用和 547 公共用例；FP32/FP16、训练、多卡与续接按声明范围通过 |
| CUDA | eager/常驻源码、独立条件图/P2P 后端和本机编译已交付；未在真实 GPU 上运行 |
| 性能 | 有限选型已收尾；中等规模 Add CPU 接近且略快于 resident、快于 mixed；不外推成全尺度排名 |

最终来源：[本机资格](evidence/batched-vjp-cuda-local-20261010.md)、
[CPU 后续](evidence/add-batch-followup-20261010.md)、
[NPU 公共模块](evidence/npu-module-batching-20261011.md)、
[常驻 composite](evidence/resident-composite-20261011.md)、
[完整选型](evidence/native-selection-20261011.md)。历史通过不认证后续新源码。

保留限制：严格近似平票 [路由失败](evidence/original-add-route-witness-20261004.md)，
Attention resident 选型案例的反向缓存预算拒绝，以及自定义/非默认模块的后备成本。
未放宽离散对照、None/零梯度或容量拒绝。性能观测可能受共享资源争用影响。

## 剩余事项

- GPU 真机的完整正确性、实际常驻、多卡、内存校准与有限性能验证；
  [目标机入口](device-control.md) 已有，不由本机编译代替。
- 新硬件/环境版本只在实测后认定支持。
- 新教材、新特例和进一步性能优化属于下一次明确任务；旧实验默认不复核、不重启。

## 工作方式

用户确认的执行契约优先于通用实验 skill。保持独立参考、有限验证范围和必要记录；
不做模型效果搜索，不自动增加性能重复或排队等待。不使用子代理。
长构建/测试使用独立 `background.slice` user service、固定源码和有限资源；
共享设备使用须遵守用户授权、健康与显存检查，不操作其他人的进程。
本机工具链定位见 [开发入口](development.md)，先核对实际环境。
提交前更新本交接；历史来源留在 evidence/Git，不重新累积任务日记。
