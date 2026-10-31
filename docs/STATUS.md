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

容器启动器实现已提交/推送 `cc6b45f`，本机5项实际生命周期测试通过。
目标源码 `/root/autodl-tmp/gef-cuda3090/sources/cc6b45f` 已只读冻结；新分支bundle克隆，不含old引用。
目标 `jobs/disconnect-probe` 已在断连后确认父子进程存活，停止后cancelled/无存活组成员；`jobs/setup` 已passed，pytest8.4.2/Ninja1.13.0安装在私有venv，Torch保持2.12.1+cu130。
工作根 `/root/autodl-tmp/gef-cuda3090`；环境 `venv` 复用现有Torch。构建将用56线程/sm86。
原始审计本机 `artifacts/cuda3090/startup-system.json`；目标状态/日志为各 `jobs/NAME/status.json`、`task.log`。
`jobs/core-build` 已passed（209目标，实际约73秒；上限3600秒，56线程）；工作命令 `bash /root/autodl-tmp/gef-cuda3090/core-build.sh`，cwd为上述固定源码，构建目录 `builds/cc6b45f-core`。
开发检查：目标CPU公式92通过，纯PyTorch CUDA mixed-c/KV完整检查通过，检查器CPU回归9通过；均不是完整矩阵。
构建现场发现Ninja子进程组；subreaper修复及6项进程测试通过，覆盖另建session、孤儿与忽略TERM。
Python/native/standalone首个CUDA KV四更新通过；C++新增精确续接已通过小型CPU FP64/FP32，CUDA定向正在完成。
本机未提交：上述启动器修复、CUDA检查显式IEEE策略、workers/packing记录、完整同设备恢复精确检查及本交接；活跃源码快照未修改。

```sh
git status --short --branch
python scripts/status.py
python scripts/check_upstream.py
python -m unittest discover -s tests -p test_container_job.py -v
```

接着在目标机建立私有环境、固定源码，验证 `scripts/container_job.py start/status/stop` 的断连存活与组清理，
再以56线程构建CUDA core。已有完整多卡resident门禁保留，单卡入口需要独立建立，不能跳过得到完整通过。
旧CPU/NPU严格失败及停止条件保留于 [ROADMAP](ROADMAP.md)，不自动重跑旧矩阵。
