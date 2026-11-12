# 当前交接 — 相邻张量阶段与 resident KV bank（2026-11-12）

数学固定 `0770593838a37c790a568226930567ed60f99b83`，19 份快照通过。
授权分支 `graph-execution-foundation`；允许整体重写、分阶段提交与非强推；本对话共享 NPU 已授权，GPU 延后。
**本轮完成正在进行的两条边界重写及必要验证后，停下向用户汇报、重新对齐。不要自动展开下一轮。**
这是最新任务边界；P1—P6总体未完成，见 [ROADMAP](ROADMAP.md)。

## 已确认原则，不是每次重入的新任务

固定数学、公共基座与代表性能是锚点；旧源码/接口可替换，历史兼容不构成性能或架构约束。
mixed-a/b/c 仅为可迁移的性能对照预设，不强迫核心分层；重入时遵守，不重复启动同一调查。
三份用户 `docs/tide-learning-lab-TL-REQ-004/005/006-handoff.md` 只读、不提交。

## 最近固定结果

- `72e1d22c` 来源 bank/incidence 与区域实际传输复用：两套 core 构建；CPU 325 passed / 7 原 xfail / 2 新 FP32 严格失败。
  同材料 FP64 全根通过，None/零与完整观察一致；阈值未改，原失败保留。
  SDK 7 CPU / 15 NPU 组合和 12 安装三步/新进程恢复通过。
  LH176 B4/L16/d64 默认 native 训练 CPU1/NPU-host2 为 1.405757/2.902243 秒；FC16 B8/L128/d128 NPU .291365 秒。
  32 点成本未显示 LH 主要差距缩小。[报告](evidence/source-bank-20261112.md)及 JSON 已集成。
- `08a349ce` resident 已知存储/workspace 预准入、实际资源 inventory：两构建、8 CPU、6 SDK / 4 Python NPU 进程通过；无新性能。
  [报告](evidence/resident-resources-20261112.md)及 JSON 已集成。
- `9ae62d0d` 增量来源容量规划：两构建、75 CPU / 7 standalone CPU、NPU 安装三步/新进程恢复通过。
  [报告](evidence/source-planning-20261112.md)及 JSON 已集成；未重跑成本矩阵。
- `042ea369` resident 有界设备 program 组合及 `7fd1dd10` CPU harness 修复已有
  [固定报告](evidence/resident-services-20261112.md)。原预算 FC16 B4/L16/d128 / LH176 B4/L16/d64 完整训练 3.9448/22.5918 秒，
  新 owner 下一窗精确恢复通过；实际 staggered 61 输入位置，规模数学 not_checked，非严格等价吞吐。
- `72e1d22c` 三层 CPU profile/census 已终态通过：直接外图 11192 节点；公共 native 外图 6069、含已注册私有图 15138；
  默认 native 外图 8853、native 合计 17922。计数宇宙重叠不可相加，profile 不是无插桩吞吐。

## 当前改动与本轮剩余

- `lh_direct`：主机 Read/Describe→区域的 actual-row FieldBlock；发布一次、节点独立根视图、连续借用及整块伴随。
  最终 Python 定向91项通过；扩大153通过/3原xfail/2既有FP32严格超差；3个C++ TU语法通过。
  主代理审阅补空列/stride反例及导航；现固定host源码，接着两套新构建、定向资格与有限成本。
- `resident_rewrite`：唯一 initial/write bank + old/proposal 快照索引、raw 解码与块伴随；当前尚未接完生产消费者。
  先完成相邻阶段及原小型整链/训练/恢复，不新增规模扫描。新增 C++/kernel 需进入两套构建。
- `req006_diagnostics`：仅收尾已产生的证据和 mixed 预设定位审阅，不把已确认原则变成新改造任务。
- 主代理统一集成、提交、资源调度与有限成本对照；完成后停下报告 P1—P6实际进度和下一轮取舍。
  主树中的代码是本轮协作修改，不回退；不要将未完成的 resident 草稿混入 host 可资格提交。

## 作业和资源

重入时所有旧构建/资格/成本/profile 作业均终态，无在途长作业。
最近 `source-bank-profile`、`source-planning-cpu-directed`、`source-planning-host-npu` passed；
详细日志在 `artifacts/jobs/对应名称`，运行材料在 `artifacts/source-{bank,planning}-20261112`。
旧固定源码/构建分别在 `/dev/shm/gef-source-{bank,planning}-20261112`；不修改旧作业输入，不冒称旧二进制验证新源码。
两工具链 `ascend/dev-workspace-9.2.0` 与 `libtorch-npu/2.12.0-cann9.2.0` 独立。
私有 Python 环境 `artifacts/additive-work/env`，公开环境只读。
当前主盘余 813 GiB、tmpfs 276 GiB；账户 331/软300/硬800 GiB、6日宽限，无当前容量阻塞。
可用全部 CPU/内存及合法准入 NPU；两构建分 CPU0–159/160–319，正式计时与重构建错开。

重入：`git status --short --branch`；`python scripts/status.py`；`python scripts/check_upstream.py`。
历史提交：`python scripts/resolve_history_commit.py OLD_SHA`，见 [迁移](history-migration.md)。
