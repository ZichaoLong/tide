# 当前交接 — Dense/MoE失败重启已后台提交（2026-11-04）

用户明确授权重启失败作业。**北京时间10:10观测：6个service运行/等待，固定资格已通过；
不是21项模型全部通过。** [重启证据](evidence/llm-restart-20261104.md)/[JSON](evidence/llm-restart-20261104.json)
保存启动观测；后续以各job/case持久记录为准。旧[57项终态](evidence/llm-terminal-audit-20261104.md)与所有失败保留。

## 固定来源与已验事实

实现79197ad557b33bfa82c1914a02038f8944351247；固定检出`source/restart-g`。
数学0770593838a37c790a568226930567ed60f99b83本次校验通过。
新私有wheel安装、118项相关设施检查通过；`jobs/qualify-g`已终态、后代为空。
llama.cpp固定2cbca389f515df455412dafa07e2b1359cc72a6d源码直连下载及完整C++构建通过，
archive/binary/共享库/编译器身份见`engines/llama/ready.json`。GGUF转换等待compute锁，模型运行尚待验证。
公共Torch2.12/TorchNPU2.12/CANN9.1只读；Transformers4.57.1/Accelerate1.11.0/PEFT0.17.1不升级。
转换另用新私有`env/llama-convert`，不混称公共基线变更。

所有外部路径以下列根为准：
`/mi/data2T/zlong/home/llm/gef-lm-experiments/20261102`。

## 活动服务与后续自动提交

| unit（均为.service） | job目录 | 本次观测 |
| --- | --- | --- |
| gef-lm-assets-small-20261104-g | jobs/assets-small-g | Qwen7B从原partial增长；OLMoE完整资产复用 |
| gef-lm-assets-moe-20261104-g | jobs/assets-moe-g | 等small终态，再续Qwen30B-A3B已验4片和第5片partial |
| gef-lm-llama-20261104-g | jobs/llama-g | 完整构建已过，等待compute锁做GGUF转换、2小检查和4成本点 |
| gef-lm-placement-20261104-g | jobs/placement-g | OLMoE双NPU B512等待整组设备；后续Qwen MoE四设备3项 |
| gef-lm-dispatch-20261104-g | jobs/dispatch-g | ready已提交；等待两模型资产后分别提交后续CPU/NPU队列 |
| gef-lm-cpu-ready-20261104-g | jobs/cpu-ready-20261104-g | OLMoE CPU B512已进入计算，整流程7200秒上限 |

后续自动创建`gef-lm-{cpu,npu}-{small,moe}-20261104-g.service`，对应同名去掉`gef-lm-`的jobs目录。
这些4个服务在本次观测尚未提交，不能称running。模型资产记录中qwenmoe旧failed仍属前次attempt；
新服务正在等待依赖，尚未开始写新asset-state。所有已启动supervisor实际cgroup均在background.slice，
与当前focus.service分离。主工作区可继续开发，不编辑活动固定源码、模型partial、环境或站点适配文件。

## 选择、资源与停止条件

`lm retry-matrix`从跟踪生成器重建21项：compute11、llama6、placement4。
19个未执行依赖点保持原配置；OLMoE CPU B512只改1200→7200秒整流程上限。
原单NPU B512估算63.080GiB仍拒绝，另列双NPU显式跨层，每卡36.828/36.444GiB；
保持B512、输入512/输出128、BF16、prefill_chunk64、完整KV和Attention可见性。
双卡小检查执行/精确恢复复用原身份；其严格数值/token失败不解除。已完成曲线与数值失败不重跑。

模型走试测可用的c28，8MiB Range、每文件最多10次临时错误重试；两个模型流串行。
llama源码用试测可用的直连，旧5MiB失败partial另名保留。源码下载/构建/转换和正式计时由compute.lock协调，
构建实际affinity320线程，模型计时16 ATen线程。所有hash/Range/非有限/通信/精确恢复失败仍明确失败。

NPU沿用两次采样/整组锁。根据用户既有低占用共享授权，仅物理9允许与PID3334312/start2131067677/UID1010共享：
已观测约5.5GiB HBM、0%利用率；阈值HBM≤8192MiB、AICore≤5%、空闲≥53248MiB，授权策略48小时过期。
新占用进程、预算越界或过期拒绝，未列入的卡仍须无进程。`site/sharing-20261104-g.json`和
`site/npu-20261104-g.py`按hash固定，逐case保存sharing-audit.jsonl，不称共享观察为独占计时。
双/四卡仍需其余设备合格，不停止其他项目。

small下载30小时、moe/llama/placement服务42小时；模型依赖等待最多30小时。
设备/compute锁等待各最多6小时，自动计算service24小时、队列23.5小时。
到期写失败/未执行，不无限重试。CPU B512是7200秒，其他case仍原1200秒。
当前账户约106GiB，实验根约55GiB加新源码/构建；预计剩余权重/GGUF/构建约70GiB。
项目160GiB、账户300GiB停止新增大写入，硬配额800GiB。主存约1.6TiB可用，计算请求384GiB；
v1 controller共用user service，不声称逐unit硬内存隔离。无gpu3090或tmpfs依赖。

## 接续命令

```sh
git status --short --branch
python -B scripts/status.py
systemctl --user list-units 'gef-lm-*-20261104-g.service' --all
quota -vs
lm_root=/mi/data2T/zlong/home/llm/gef-lm-experiments/20261102
# 生成计划或读取结果均不启动模型：
"$lm_root/env/transformers/bin/python" -I -m tidegraph lm retry-matrix --group all
"$lm_root/env/transformers/bin/python" -I -m tidegraph lm results "$lm_root/runs" --format csv
# 观察一个服务：
tail -n 40 "$lm_root/jobs/llama-g/task.log"
cat "$lm_root/recovery/20261104-g/dispatch.json"
# 只在用户要求停止本轮时：先停dispatcher避免继续提交，再停本轮unit。
# systemctl --user stop gef-lm-dispatch-20261104-g.service
# systemctl --user stop 'gef-lm-*-20261104-g.service'
```

每job含status.json/task.log；每case含case.json、allocation/phase、process/status及run/result或check，
有checkpoint时另验restore-process/resume.json。停止后核对真实ControlGroup、PID/start及remaining_live_pids，
外层exit0不能代替内部检查。生成材料在`recovery/plans-20261104-g`；dispatcher只选择未完成项。
下次先审阅本轮终态和内部结果，不直接重跑同一attempt。公共复现入口见[external-lm](external-lm.md)。

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
