# 2026-10-31 跟踪内容清理

状态：完成。已按用户授权仅保留`.git`与Git跟踪内容；
远端未提交源码包稍后分析，不重新运行已审阅实验，不访问已释放GPU。
起始提交`49cf37aedc144f611b3ec7847fba94e197b238ea`，授权分支`graph-execution-foundation`。
固定清理来源`0b07e7212a6303901de5b9c673901cde60370286`。
固定数学`0770593838a37c790a568226930567ed60f99b83`在删除前后均校验通过。

| 实测项目 | 清理前 | 清理后/释放 |
| --- | ---: | ---: |
| 仓库allocated占用 | 300,211,847,168 bytes（279.594GiB） | 80,121,856 bytes（76.410MiB） |
| 仓库磁盘释放 | — | 300,131,725,312 bytes（279.519GiB） |
| 6个项目tmpfs释放 | — | 17,223,258,112 bytes（16.040GiB） |
| 登记worktree | 27 | 1（主工作区） |
| 用户quota占用 | 约312GiB | 32422MiB（约31.7GiB）；软限400GiB/硬限800GiB |

按`du -sx -B1`计算allocated blocks，包含移除worktree登记后的Git元数据回收；
清理后数值取最终证据提交前，后续Git对象小幅增加。完整清单与原始数值见[同名JSON](artifact-cleanup-tracked-only-20261031.json)。

## 保全与删除范围

- 26个附属worktree均干净、基线均为当前分支祖先；10个无Git源码快照逐文件对照固定提交，无额外差异。
- 144个项目service无活动MainPID且ControlGroup空；可读本用户进程路径无产物使用者。受权限限制的`/proc`字段未伪称全部可读。
- 环境、构建、原始日志/trace/张量/checkpoint与副本退役；6个明确项目tmpfs目录单独列入清单。公共环境、参考仓库、其他项目及`.git`保留。
- 九份platform/finite-packing归档的204条结果继续保留，严格失败不改写。
  [终态归档](retired-jobs-20261031.json.gz)新增41份外层状态、5份补充记录及144个unit观察；外层passed不是内部全部通过。
- B512原checkpoint及两次大tmpfs的字节退役；hash只保留身份。固定配置/输入生成器/有界诊断见[复现导航](../reproduction.md)，新见证不等于原字节复验。

## 仓库外源码保全

本机目录`~/.local/state/graph-execution-foundation/preserved-source/20261031-cuda3090/`，权限0700。
主包`retirement-dev-20261031.tar.gz`已移入并校验：5,452,397 bytes，
SHA256 `6b89b411611378a619149d69870c35e3f587db0baf1ed70fd2a21a005e2fe31e`。
它保存原远端16项跟踪修改及4项未跟踪源码；本次不提取分析、不自动合并。
两份`development.patch`/`cpp-development.patch`另存`additional-source-deltas.tar.gz`（4,435 bytes），
同目录manifest/JSON保留文件hash，包SHA256为`bfb957c5e319a5e3071f6871fe1675d4cd5e2de40c0dfe75ae0aa02f312ed125`。
删除后两个保全包再次校验相同。保全不构成新源码资格，也不声称差异均已合并。
这些小包是本机待分析材料，不纳入公共Git依赖；新clone正常开发不依赖它们。

## 有界执行与核验

操作目录`~/.local/state/graph-execution-foundation/cleanup-20261031/`保存脚本、固定清单、状态和日志，
不位于待删目录内。准备记录先提交为0b07e72，只读plan审阅后按该固定来源执行apply。
服务`gef-artifact-cleanup-49cf37a-20261031.service`在`background.slice`运行，
1核/1GiB内存、1800秒超时、KillMode=control-group、停止宽限30秒。
计划与执行校验HEAD、脚本hash、跟踪文件字节/权限、目标inode/device和作业状态，
worktree先用Git正常移除，再删除清单目标；不跟随符号链接。

实际15:57:32Z开始、15:58:25Z完成，44个步骤终态passed/exit0。
2,838个跟踪文件的SHA256/权限未变，`git status`干净、`git clean -ndx`无输出，
仅主worktree、18个清单目标全消失。原PID3065003与完整控制组均已消失；
瞬态unit已卸载（LoadState=not-found），不把卸载后的默认ExecMainStatus当作退出码证据。
退出码依据已fsync的status，日志末行`Cleanup verified complete`。

清理后公共Python3.12.13通过源码CLI查询9份归档，204条唯一run完整保留：
CPU48/NPU78/CUDA78；69 strict-checks-passed、113 strict-failure、20 not-established、2 process-failure。
归档schema与内容digest均由公共查询器核验，没有导入Torch；这些计数不是性能全通过。
命令如下，原环境/构建/运行目录均不需要：

```sh
python -B scripts/status.py
python -B scripts/check_upstream.py
PYTHONPATH=python /opt/miniconda/envs/ascend910-dev-workspace-torch-npu-2.12.0-py312-r1/bin/python -B -m tidegraph results docs/evidence/core5-platform-*-20261031.json.gz docs/evidence/core5-finite-packing-20261031.json.gz --format table
```

其他机器使用Python>=3.10；构建/实际计算从[development](../development.md)与[人类手册](../human-workflow.md)重建，
必须重新生成构建身份，不冒用已退役二进制。本操作没有新构建、训练、设备或性能资格。
