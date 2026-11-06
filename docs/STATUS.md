# 当前交接 — CANN 9.2.0项目迁移（2026-11-06）

用户已授权把日常开发与新实验入口迁到公共CANN9.2.0，并重建本项目native/standalone，
仅做小型CPU/NPU完整推理、训练、独立梯度根与恢复检查。GPU已释放，不连接。
公共9.2.1 vLLM为独立服务栈，本次不采用，不修改公共工具链。

启动HEAD `b221fda4f07c3dd3ad5b9ade9107adedd7e02139`，授权分支
`graph-execution-foundation`，启动工作区干净、无本项目活动服务。
固定数学 `0770593838a37c790a568226930567ed60f99b83`及19份快照核验通过。
截至启动账户约179GiB（软300GiB、硬800GiB），磁盘空闲1.3TiB、主存可用1.8TiB、CPU affinity320。
本轮新增环境/构建预算10GiB，保留旧LM资产/环境；不需要tmpfs或新权重下载。

## 当前实现与下一步

实现已提交`f1480a002284a3c5fb5cae56b85f0b469492b5cd`，开发设施25项通过。
固定检出`.local/cann92-source-a`（跟踪文件只读），已终态failed
`gef-cann92-migration-a.service`；总时限10800秒，构建jobs=all（affinity320）、测试ATen/BLAS=1。
结果`artifacts/cann92-20261106-a`，job记录`artifacts/jobs/cann92-migration-a`；
监督日志`artifacts/cann92-migration-a-supervisor.log`，任务日志为job目录下`task.log`。

```sh
systemctl --user show gef-cann92-migration-a.service -p ActiveState -p Result -p MainPID -p ControlGroup
cat artifacts/jobs/cann92-migration-a/status.json
tail -60 artifacts/jobs/cann92-migration-a/task.log
# 停止完整控制组，随后核实MainPID/ControlGroup及后代为空：
systemctl --user stop gef-cann92-migration-a.service
# 从任意干净固定检出重新验证，必须选择全新的输出目录：
# bash scripts/qualify_ascend_migration.sh NEW_OUTPUT /absolute/use-local-ascend/scripts/run_on_free_npu.py all
```

a作业于08:40:39 UTC退出1，MainPID=0/ControlGroup为空。Python/native129项构建完成，
SDK元数据在系统Python3.9.9因hashlib.file_digest不可用而失败，尚未编译SDK或执行设备检查。
已修为分块SHA256，7项回归及实际系统Python读取SDK身份/不导入Torch通过；准备新固定提交后继。
旧失败保留，不改公共Python。正式证据须读取所有内部结果，外层passed不能替代它们。
门禁通过公共check/run对三客户端分别执行CPU/NPU小型SettleGraph/FC/KV、B2、T3、d8、FP32，
含完整状态/控制/消息/pending、独立根、两步训练/SGD槽、换owner和新进程精确续接。
另跑有限CPU FP64/FP32公式/native回归和独立新LM环境随机小模型设施。
不把这次迁移认证成resident、多卡、FP16、大batch、规模模型或性能资格。

长任务使用独立background.slice、固定源码、完整日志、有限时长；NPU合作队列一张空闲卡，
逻辑设备npu:0。CPU/memory控制器未细分到用户unit，不将systemd预算宣称为实际强制限额。
本项目重型构建不与性能计时重叠，本轮无性能计时。

```sh
git status --short --branch
python scripts/status.py
python scripts/check_upstream.py
systemctl --user list-units 'gef-*' --state=running --no-pager
quota -vs
```

## 保留的旧范围

[LM15项终态](evidence/llm-isolation-terminal-20261105.md)与
[B512手动入口资格](evidence/llm-capacity-entry-20261105.md)保持原版本/环境身份；不自动重启。
旧根`/mi/data2T/zlong/home/llm/gef-lm-experiments/20261102`，
`env/transformers`继承9.1、TorchNPU post2；固定source及材料不修改。
此前公共维护中b221fda的纯PyTorch小型9.2消费者通过，不是本项目native或standalone重建证据。

原core-5九份归档204条结果及严格失败从[证据索引](evidence/README.md)、[ROADMAP](ROADMAP.md)进入。
[GPU释放](evidence/core5-cuda-release-20261031.json)及[清理](evidence/artifact-cleanup-tracked-only-20261031.md)继续有效。
远端未提交源码保全包仍在仓库外`~/.local/state/graph-execution-foundation/preserved-source/20261031-cuda3090/`，本次不分析或运行。
