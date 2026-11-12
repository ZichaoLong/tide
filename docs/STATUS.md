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

- **c6afc20c** 主机Read/Describe→区域字段owner：[资格与20点成本](evidence/read-region-owner-20261112.md)。
  两套新core构建、SDK7CPU/15NPU组合、12个CPU/NPU安装三AdamW/新进程恢复通过；
  CPU245 passed / 7原xfail / 2原FP32严格失败，原坐标及差8.8512897e-6保持，阈值不改。
  LH176 B4/L16/d64默认native训练CPU1/NPU-host2为1.419332/2.471804秒，直接.749660/.621882秒；
  FC16 B8/L128/d128默认native训练CPU/NPU为1.156975/.301235秒。主要差距未闭合，不继续同一局部优化循环。
- `72e1d22c`来源共享与`9ae62d0d`增量规划：[原成本/独立profile](evidence/source-bank-20261112.md)、
  [后继定向资格](evidence/source-planning-20261112.md)已归档；外图与私有图计数不得相加。
- `08a349ce` resident资源预准入/inventory：[两套构建与设备资格](evidence/resident-resources-20261112.md)通过，无新性能。
- `042ea369`有界program组合及`7fd1dd10`CPU harness修复见[固定报告](evidence/resident-services-20261112.md)。
  原预算FC16 B4/L16/d128与LH176 B4/L16/d64完整训练3.9448/22.5918秒、实际61输入位置；
  新owner下一窗精确恢复通过，规模数学not_checked，非严格等价吞吐。

## 当前改动与本轮剩余

主机边界已完成固定资格与有限成本。resident的唯一initial/write bank、old/proposal索引、
Mem/Full及块伴随已接入默认局部消费者；公共old/proposal字段按shape提供独立根，
显式observe从实际bank解码。契约与安装消费者同步迁移，见[bank](resident-kv-bank.md)。
开发CPU独立bank/pool/read/root检查29项、公共实验/资源定向26项通过；正式固定构建与设备资格尚待执行。
不以开发检查冒称resident整链通过。本轮不扩新记忆/多卡/规模或主机raw-bank下一阶段。
完成下列有限资格、证据及提交推送后，停下汇报并等待用户重新对齐授权。

## 作业和资源

`read-region-owner-*`两构建、CPU定向、SDK设备、两个安装资格和两个成本任务均已终态。
CPU定向原状态failed/exit1保留，其余passed。暂无在途长任务；即将启动本轮两套resident固定构建。
记录`artifacts/jobs/read-region-owner-*`与`artifacts/read-region-owner-20261112`；
固定source/core/install在`/dev/shm/gef-read-region-owner-20261112`，不修改旧作业输入。
两工具链`ascend/dev-workspace-9.2.0`与`libtorch-npu/2.12.0-cann9.2.0`独立；私有Python环境`artifacts/additive-work/env`。
可用全部CPU/内存及合法准入NPU；正式计时与重构建错开。当前主盘余807GiB、tmpfs276GiB；账户337GiB/软300/硬800GiB、6日宽限，无本轮写入阻塞。

重入：`git status --short --branch`；`python scripts/status.py`；`python scripts/check_upstream.py`。
历史提交：`python scripts/resolve_history_commit.py OLD_SHA`，见[迁移](history-migration.md)。

本轮新材料：`artifacts/resident-kv-owner-20261112/plan.json`及launch/qualification脚本。
固定checkout目标`/dev/shm/gef-resident-kv-owner-20261112/source`；构建目标resident-native/resident-sdk；
复用c6两套core前由构建器逐一核对cpp源码SHA及二进制SHA。
构建分别CPU0–159/160–319、512GiB、2400秒，background.slice；其后独立CPU门禁及NPU小型组件/全根/三步/新进程恢复。
下一命令（固定提交和worktree后）：
`python artifacts/resident-kv-owner-20261112/launch.py native-build 0-159 2400`；
`python artifacts/resident-kv-owner-20261112/launch.py sdk-build 160-319 2400`。
单元/日志：`resident-kv-owner-*` / `artifacts/jobs/resident-kv-owner-*/{status.json,task.log}`。
