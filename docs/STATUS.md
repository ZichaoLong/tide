# 当前交接 — 2026-10-31

当前任务：core-5 CUDA 真机开发与资格；授权分支 `graph-execution-foundation`，允许阶段提交和普通推送。
启动 HEAD `a066756fe23260e45d61afa158f87883d426f0d5`，工作树干净，无旧活动作业。
数学固定 `0770593838a37c790a568226930567ed60f99b83`，19 份 core-5 快照及旧 core-3 快照校验通过。
历史治理已完成；只取新分支，原证据身份/映射见 [history-migration](history-migration.md)。

## 现场与范围

- 经本机认证 HTTP CONNECT 连接 `ssh gpu3090`；凭据只在私有配置/内存，无全局配置修改。
- 复查：x86_64、Python3.12.3/Torch2.12.1+cu130、Toolkit13.0.88、GCC11.4、CMake3.22.1、Nsight2025.3.1。
- 四张 RTX3090/24GiB/SM86 均空闲；12方向 Torch P2P 查询全部 false。多卡 resident 硬件条件不足，其他路径继续。
- CPU affinity112、配额56核；内存240GiB、当前约1.1GiB；数据盘50GiB基本空闲。初始构建56线程，显式sm_86。
- 非登录 SSH 没有 `python3` PATH；使用 `/root/miniconda3/bin/python`。首次探测因此退出127，随后绝对路径探测通过。
- GPU是主线，先单卡FP32完整训练/推理/恢复，三客户端分别验；CPU仅小型公式与相关回归。
  resident/FP16/可用多卡及有限性能随后；不提高原容差，不把复制成功称直接P2P。

## 当前作业与下一步

实现提交 `c962961`：CUDA eager IEEE政策、分配置三客户端门禁、完整同栈精确恢复；容器subreaper清理独立组与孤儿。
开发检查：CPU公式92、检查器CPU9、容器6通过；三个客户端CUDA首个KV通过；C++新增精确恢复CPU FP64/FP32、CUDA FP32通过。
初始CUDA core `cc6b45f` 构建209目标成功，56线程实际约73秒。开发检查不冒充完整GPU资格。

目标工作根 `/root/autodl-tmp/gef-cuda3090`。准备冻结 `sources/c962961`，以该提交重新构建 `builds/c962961-core`，
作业 `jobs/core-build-c962961`，超时3600秒/56线程/sm86。构建期间先启动独立 `jobs/eager-pytorch-fp32`：全19例mixed-c，3600秒，结果 `runs/c962961-pytorch-fp32`。
构建通过后启动native/standalone的固定全例，并单列KV mixed-a/b；正确性与构建可重叠，正式性能不重叠。
私有环境 `venv` 使用Torch2.12.1+cu130，额外pytest8.4.2/Ninja1.13.0；原setup作业passed。
原始日志/持久状态位于 `jobs/NAME/task.log`、`status.json`，结果目录 `runs/`；没有覆盖原运行。
断连探针实际重连确认父子存活，stop后cancelled/无存活成员；后续subreaper的独立session/孤儿探针还须目标复验。
本机新增单卡core-5 resident入口：39组件+3公共测试文件，注册/无peer清单与CLI校验通过，真机尚待运行；原三卡完整门禁保持。
活动源码只读，独立开发目录 `dev` 保留已验证补丁，不能当正式来源。

```sh
git status --short --branch
python scripts/status.py
python scripts/check_upstream.py
# 目标机（通过本机认证CONNECT，凭据不进仓库）：
/root/autodl-tmp/gef-cuda3090/venv/bin/python /root/autodl-tmp/gef-cuda3090/sources/c962961/scripts/container_job.py status --output-dir /root/autodl-tmp/gef-cuda3090/jobs/core-build-c962961
```

正式CUDA资格后单独提交证据；首个KV不代表全部五记忆/配置/拓扑通过。接着建立单卡core-5 resident明确门禁并构建，
不修改现有完整多卡门禁的三卡/P2P要求。旧CPU/NPU严格失败保持 [ROADMAP](ROADMAP.md) 的停止条件。
