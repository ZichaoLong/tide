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

无活动长作业。正在提交容器无systemd时的有界进程组启动器及独立生命周期测试。
目标工作根计划 `/root/autodl-tmp/gef-cuda3090`，私有venv复用匹配Torch，补pytest/Ninja；仅传新分支bundle。
源码冻结在独立checkout后启动，环境/构建/日志放该根下；不编辑活跃任务的输入。
原始启动审计暂存本机 `artifacts/cuda3090/startup-system.json`，最终审阅证据另行提交。

```sh
git status --short --branch
python scripts/status.py
python scripts/check_upstream.py
python -m unittest discover -s tests -p test_container_job.py -v
```

接着在目标机建立私有环境、固定源码，验证 `scripts/container_job.py start/status/stop` 的断连存活与组清理，
再以56线程构建CUDA core。已有完整多卡resident门禁保留，单卡入口需要独立建立，不能跳过得到完整通过。
旧CPU/NPU严格失败及停止条件保留于 [ROADMAP](ROADMAP.md)，不自动重跑旧矩阵。
