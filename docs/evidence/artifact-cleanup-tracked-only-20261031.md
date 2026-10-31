# 2026-10-31 跟踪内容清理

状态：准备记录，删除尚未执行。用户授权本仓库仅保留`.git`与Git跟踪内容；
远端未提交源码包稍后分析，不重新运行已审阅实验，不访问已释放GPU。
起始提交`49cf37aedc144f611b3ec7847fba94e197b238ea`，授权分支`graph-execution-foundation`。
固定数学`0770593838a37c790a568226930567ed60f99b83`校验通过。

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
同目录manifest/JSON保留文件hash。保全不构成新源码资格，也不声称差异均已合并。
这些小包是本机待分析材料，不纳入公共Git依赖；新clone正常开发不依赖它们。

## 有界执行与核验

操作目录`~/.local/state/graph-execution-foundation/cleanup-20261031/`保存脚本、固定清单、状态和日志，
不位于待删目录内。先提交本记录，再运行只读plan并审阅，最后按固定源执行apply。
服务`gef-artifact-cleanup-49cf37a-20261031.service`在`background.slice`运行，
1核/1GiB内存、1800秒超时、KillMode=control-group、停止宽限30秒。
计划与执行校验HEAD、脚本hash、跟踪文件字节/权限、目标inode/device和作业状态，
worktree先用Git正常移除，再删除清单目标；不跟随符号链接。

终态须确认跟踪字节/权限未变、无ignored/untracked剩余、仅主worktree、tmpfs目标已删，
并检查quota、保全包hash、MainPID及ControlGroup。尚未执行的这些检查不记通过。
本操作没有新构建、训练、设备或性能资格；完成结果另行提交。
