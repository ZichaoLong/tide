# 本轮产物收束与清理后隔离安装，2026-11-18

[机器记录](roadmap-retirement-20261118.json.gz)保存逐路径清单、来源核验、原终态、
删除量、保留项与完整安装观察。本次只处理当前线程创建的本项目产物，
未删除公共环境、参考仓库、用户文件、主环境、注册工作树或下游安装。
[本轮交接](../STATUS.md)与[保留规则](../artifact-retention.md)分别管理事实与方法。

## 审查与实际删除

69 个 launcher 作业均为终态；删除前逐个检查 MainPID=0、空 cgroup 和已记录后代的
PID/start 身份，并检查没有活动进程以项目 artifacts 子目录为 cwd。
冻结归档逐文件与其 Git 提交核对，没有未识别的额外源文件或不匹配。
清单中没有注册工作树；主/下游工作树的直接私有 venv pth/direct_url 元数据未引用候选路径。
所有路径限定于本项目 artifacts，不跟随符号链接删除外部目标。

| 删除类型 | 路径数 | allocated bytes |
|---|---:|---:|
| `project_only_installed_smoke_or_build` | 2 | 19,718,144 |
| `rebuildable_git_archive` | 69 | 329,404,416 |
| `rebuildable_tensor_or_checkpoint` | 47 | 6,459,613,184 |
| 合计 | 118 | 6,808,735,744 |

主清理实际释放 **6.341 GiB**；随后 artifacts 的
快照为 845,516,800 bytes。账户额度仍为300 GiB soft /700 GiB hard，
清理后记录使用约39,736 MiB；账户和全文件系统用量也会受其他工作影响。
主清理前 artifacts 清单为 7,654,203,392 allocated bytes。
清单之后写入的小记录会产生少量新增量，不能用文件系统空闲量差推导本次释放量。

保留全部小结果、配置/输入身份、命令、日志、原始失败与 stop/superseded 记录，
以及主 `.venv-instances`、Git 历史和固定 wheel。
六个未决数值/支持见证共 143,290,368 allocated bytes，
路径和 SHA256 在 JSON；四份原用户未跟踪文档的原 SHA256 均复核一致。
旧绝对路径和 hash 是历史身份，不保证已退休源码或张量在线。

## 清理后的独立安装

所有118个主清理候选路径不存在后，使用已保留的 wheel 新建私有 venv：
源码 `0992b467584c1f603314ed04d8efe8596a14f576`，wheel SHA256
`1b3349526257e87daf1177d88c5a97ea30484b402f88080d00a72b4f521eb0e4`。
示例仅从该提交的 `examples/train.py` 提取；复用已固定的安装检查 harness。
从 `/`、无 `PYTHONPATH`、Python `-I`、vendor autoload=0 运行。
17 个包文件与 wheel 一致，15 个已加载 Tide 模块均来自新 venv。
Torch 实际版本 `2.12.0+cpu`，4线程、CPU224–227，未使用 NPU。
环境共享公开依赖的只读安装，Tide 包本身安装在新 venv 内。
完整建环境/安装/检查过程 7.471 秒，逐条命令和 stdout 在 JSON/保留日志。
这验证退休路径不再是安装运行的必要依赖。

FP32/BF16 均执行推理、三次 AdamW 更新、checkpoint、恢复后前向和真实后续更新。
保存/恢复的继续前向精确一致；后续优化器容差仍有以下有限失败：

| 精度 | 参数最大绝对差 | 参数超差 | optimizer 观察 |
|---|---:|---|---|
| FP32 | 1.38133764e-5 | `regional.1.second.bias` | moments 在原阈值内 |
| BF16 | 1.66445971e-5 | `regional.1.second.bias` | 首个报告的 exp_avg 差异：39/16384 元素超差，最大绝对差9.76566225e-5 |

阈值仍为 atol1e-6/rtol0.002。外层 smoke `passed` 表示机制、隔离和输入路径检查完成，
不是所有 optimizer 容差通过。原安装记录的FP32两项偏差及BF16在阈值内的那次观察不变；
本次使用同 wheel、Torch路径、初始推理参数/输入 hash、训练seed、线程和affinity，
仍不能保证后续梯度/optimizer轨迹逐位一致。没有为更换标签而重复运行。
[已完成的有限调查](roadmap-numerics-followup-20261118.md)保留同 owner 反向变动和
选定 AdamW 放大的独立见证；这些见证不构成本次每一项偏差的单独因果证明。

## 最终留存与复现

本记录先在 `4ddaec87` 固定安装检查结果，再移除本次独有的临时 venv、临时目录和两个 checkpoint，共 4 个路径、71,618,560 allocated bytes。小结果、驱动、输入、日志及固定 wheel 保留。

最新 snapshot：artifacts 845,565,952 allocated bytes，
本项目目录（包括保留 Git 对象）1,009,971,200 bytes；
测量时间见 JSON，随后发布文档会再增加少量字节。原始终态和失败没有被改成成功，
没有活动本轮 Tide service。没有发现当前账户以 Tide/项目命名的顶层 `/dev/shm` 条目；
未清理任何其他 tmpfs 内容。

从[开发入口](../development.md)可重建主私有环境；
[原安装报告](roadmap-installed-20261118.md)给出从干净固定源码重建 wheel 与检查脚本的方法。
各实验按自己的 launch/source/config/command 在新路径重建，遵守当次资源准入和
[执行契约](../execution-flows.md)。清理驱动、安装驱动、精确命令及源码输入身份均留在
`artifacts/retirement-roadmap-20261118/`，无需恢复已删除的绝对目录。
