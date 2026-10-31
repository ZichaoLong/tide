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

目标根 `/root/autodl-tmp/gef-cuda3090`；私有 `venv` 使用原Torch2.12.1+cu130，另装pytest8.4.2/Ninja1.13.0。
固定源码 `sources/c962961` 的CUDA core `builds/c962961-core`：209目标/56线程/sm86通过。
小型CPU公式92、检查器CPU9通过；目标容器/架构14通过。断连/stop探针已确认无残留进程；无systemd，使用subreaper启动器。
数学快照本次接续再次校验通过。无本机长作业。

- `jobs/eager-pytorch-fp32` 已passed，`runs/c962961-pytorch-fp32` 全19例mixed-c通过。
- `jobs/eager-native-cpp-fp32` 仍running（物理卡1，7200秒）：native19例全通过；standalone目前fc/ssd原严格tensor比较失败，其余继续。原日志保留，待定位字段/阶段，不能称整体通过。
- `jobs/resident-build-7fd0576` failed：86-real被既有整数架构守卫拒绝，未编译。
- `jobs/resident-build86-7fd0576` failed：CUDA13的cudaGraphAddNode增加edge-data参数。旧两次失败均保留。工作树已作版本兼容修复，零依赖WHILE/SWITCH图语义不变；目标dev两源文件定向编译及8项架构检查通过。
- 已提交 `jobs/eager-mixed-ab-fp32`：固定c962961、物理卡0、三客户端KV mixed-a/b、1800秒。
- 已提交 `jobs/eager-fp16`：固定c962961、物理卡2、三客户端完整mixed-c、7200秒；保持AdamW/原容差/输入，失败逐配置记录。

CUDA13兼容提交 `1fe1e37` 已推送；`jobs/resident-build-1fe1e37` 正在冻结源码构建（56线程/86，standalone39组件和weighted Python适配，3600秒）。
下一步：核对构建终态，
再独立执行单卡门禁及时间线。原三卡/P2P完整门禁保持。C++ fc/ssd失败单独定位，不阻塞其他通过域。
新peer探针开发已在物理2/3验证12个传输/流/根单元；尚未提交或正式资格，不是完整多卡模型通过。
正式资格之后另提交审阅证据。CPU/NPU旧严格失败保持ROADMAP停止条件。

所有目标日志为 `jobs/NAME/task.log`、`status.json`；输出为 `runs/NAME`，不会覆盖旧运行。
活动源码只读；`dev` 为dirty开发，不能作为正式来源。未提交内容：standalone训练失败字段/数值诊断、peer探针及本交接。

```sh
git status --short --branch
python scripts/status.py
python scripts/check_upstream.py
# 本机任务私有SSH包装（认证只在内存；不修改全局配置）
artifacts/cuda3090/ssh.sh '/root/autodl-tmp/gef-cuda3090/venv/bin/python /root/autodl-tmp/gef-cuda3090/sources/c962961/scripts/container_job.py status --output-dir /root/autodl-tmp/gef-cuda3090/jobs/eager-native-cpp-fp32'
```
