# 当前交接 — LM隔离后继已在后台运行（2026-11-04）

实现固定`694eb605b6769f8d3f208710a920dbfbf504abf4`，检出`source/isolation-h`。
非editable wheel已安装，固定设施资格**125项通过**（含目录隔离、超时后代清理、原二进制恢复）；
开发38项为重叠范围，不累计成模型资格。数学0770593838a37c790a568226930567ed60f99b83校验通过。
用户已授权的15项后继已建立后台依赖队列；当前运行/等待状态不表示模型通过。
[本次启动证据](evidence/llm-isolation-launch-20261104.md)/[JSON](evidence/llm-isolation-launch-20261104.json)保留固定观测。

## 活动作业与下一步

外部根`lm_root=/mi/data2T/zlong/home/llm/gef-lm-experiments/20261102`。
计划在`recovery/plans-20261104-h/{all,compute,llama,placement,budgets,launch}.json`；
旧llama build固定为同目录`previous-llama-build.json`。各job都有status.json/task.log/supervisor.log。

| job目录 | systemd unit（加.service） | 工作和有限边界 |
| --- | --- | --- |
| jobs/qualify-h | gef-lm-qualify-20261104-h | 已通过、后代为空；1800秒 |
| jobs/assets-moe-h | gef-lm-assets-moe-20261104-h | 续传Qwen MoE；30小时 |
| jobs/llama-h | gef-lm-llama-20261104-h | 构建/小回归已通过，两个B32等待计算锁；24小时 |
| jobs/dispatch-h | gef-lm-dispatch-20261104-h | Qwen7B已提交，等待MoE后另提交CPU/NPU；依赖30小时 |
| jobs/placement-h | gef-lm-placement-20261104-h | 等待MoE，四NPU小检查及B32/B128；42小时 |
| jobs/cpu-small-20261104-h | gef-lm-cpu-small-20261104-h | Qwen7B CPU B8，已提交；24小时 |
| jobs/npu-small-20261104-h | gef-lm-npu-small-20261104-h | Qwen7B NPU B8，已提交；24小时 |

MoE就绪或下载失败终态后，dispatcher将自动另建`jobs/{cpu,npu}-moe-20261104-h`；
unit为`gef-lm-{cpu,npu}-moe-20261104-h.service`。失败资产只生成依赖失败，不运行残缺模型。
勿再提交相同清单，勿修改/删除固定源码、私有环境、活动partial、GGUF或build。
主工作区可继续开发；所有重型构建和正式计时共用`compute.lock`，下载可同时进行。

下一步仅读取进度与终态：检查每case执行、数值/离散、恢复和成本，单独归档。
任务结束前不自动重启或延长预算；保留旧[21项终态](evidence/llm-restart-terminal-20261104.md)及原严格失败。

```sh
git status --short --branch
python -B scripts/status.py
lm_root=/mi/data2T/zlong/home/llm/gef-lm-experiments/20261102
systemctl --user list-units 'gef-lm-*' --all
cat "$lm_root/recovery/20261104-h/dispatch.json"
cat "$lm_root/recovery/llama-20261104-h/status.json"
cat "$lm_root/recovery/placement-20261104-h/status.json"
tail -n 25 "$lm_root/jobs/assets-moe-h/task.log"
"$lm_root/env/transformers/bin/python" -I -m tidegraph lm results "$lm_root/runs" --format csv
# 停止任一具体服务（不要通配误停其他项目），随后检查其job状态、实际PID和ControlGroup：
# systemctl --user stop gef-lm-llama-20261104-h.service
# systemctl --user show gef-lm-llama-20261104-h.service -p MainPID -p ControlGroup
quota -vs
```

## 范围、资源与保留事实

`lm retry-matrix --round isolation`可从干净仓库重建15项：compute10（Qwen7B2、QwenMoE8）、
四NPU placement3、llama B32两项。前13项配置不变；B32仅显式延长1200→7200秒并增加阶段日志。
新旧llama B2/L16/O4完整输出/状态/恢复小回归均已通过，不代替独立数值资格。
原数值/token差异不解除；已完成B1/OLMoE B512不重跑，成本点未查项目仍not_checked。

模型/恢复子进程使用独立`process/work`和`restore-process/work`，绝对输入/输出路径；
保持源码clean guard。旧`source/restart-g/fusion_result.json`保留，新源码当前干净。
旧binary/shared libraries及GGUF保留；checkpoint使用原build记录，不随新ready指针切换。

下载仅MoE，c28单流、32MiB Range、I/O120秒、每文件24次临时重试。已验4/16片及第5片partial
原地续传，Range/hash错误仍失败。NPU用普通空闲卡两次采样/整组锁，未续签旧共享策略。
主存可用约1.7TiB、账户约136GiB；项目160/账户300GiB停止新增大写入、800GiB硬限保持。
实际name=systemd cgroup已验证位于background.slice；v1 memory控制器仍是共享user service，
MemoryMax是请求，不当作已证明逐job硬隔离。无需扩盘/tmpfs，不连接已释放gpu3090。

公共Torch2.12/TorchNPU2.12/CANN9.1只读；私有Transformers4.57.1/Accelerate1.11.0/PEFT0.17.1。
旧实现79197ad的118项资格及历史模型结果保留原身份，不能升级为本次重新验证。
llama上游2cbca389f515df455412dafa07e2b1359cc72a6d保持，新binary SHA256为
87377b5274eb8f93e2dc949dcfdb8ce6819afd6b13bf704a22bedb3db9922027；旧恢复身份仍可用。
唯一待办见[ROADMAP](ROADMAP.md)，公共入口见[external-lm](external-lm.md)。

## 原core-5终态继续保留

上一轮三平台/多卡/人类入口已收尾。九份归档204条唯一run：CPU48/NPU78/CUDA78，
69 strict-checks-passed、113 strict-failure、20 not-established、2 process-failure；不是全部性能通过。
按[证据索引](evidence/README.md)、[原八卡终态](evidence/core5-platform-eight-followup-20261031.md)及[ROADMAP](ROADMAP.md)定位原失败与重启条件。
固定launcher/配置d3de18f、实际Python安装4485697和NPU库eca070f的旧身份不因本次调查升级。
GPU释放见[审阅](evidence/core5-cuda-release-20261031.json)，不默认重租或连接旧实例。

[彻底清理](evidence/artifact-cleanup-tracked-only-20261031.md)已移除旧环境/构建/原始checkpoint和26个附属worktree，
原204条小型结果、原失败和41份终态记录仍在Git；旧材料不作为接续依赖。
远端dirty-dev源码包及早期patch仍待用户后续安排分析，位于仓库外：
`~/.local/state/graph-execution-foundation/preserved-source/20261031-cuda3090/`。
`retirement-dev-20261031.tar.gz`为5,452,397 bytes，SHA256
`6b89b411611378a619149d69870c35e3f587db0baf1ed70fd2a21a005e2fe31e`；本次未打开、修改或运行。
