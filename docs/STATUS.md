# 当前交接 — Dense/MoE首次后台矩阵已终态（2026-11-03）

用户本次询问状态。已只读核查全部作业、下载、逐case/check/resume及实际进程，未启动新下载或计算。
**当前没有活跃LM作业或遗留后代**；最后一项于2026-11-02 20:33:26北京时间结束。
[终态审阅](evidence/llm-terminal-audit-20261103.md)/[80项JSON](evidence/llm-terminal-audit-20261103.json)是本轮结果入口。
旧[启动证据](evidence/llm-background-execution-20261102.md)保持原观测身份，不能把当时running当当前状态。

## 已完成与严格边界

- 80项清单均已处理：43项执行完成、32项下载依赖失败未执行、4项CPU超时、1项NPU静态预算拒绝。
  其中71测量/观察为36完成、30依赖失败、4超时、1拒绝；9独立小检查为7完成、2依赖失败。
  四个计算队列exit0只表示清单处理结束，不代表内部全部通过。
- 7项已执行独立小检查的严格数值均未通过；Qwen1.5B单/双NPU各有2个token不同。
  CPU FP32、NPU BF16及不同attention/缓存日程的比较范围保留；大点未逐项独立检查，吞吐属于探索性观察。
- Qwen0.5B/1.5B/3B已有CPU/NPU真实推理；1.5B的NPU B1/8/32/128/512均完成。
  BF16、输入512/输出128下，NPU B128整批6.339s、2584.46输出token/s；B512为24.924s、2629.45 token/s，
  allocated峰值分别5.591/13.668GiB。CPU B128为580.328s；CPU B512在1200秒流程上限终止。
- CPU4、NPU5个训练配置各完成3次计时更新；覆盖0.5B全参、1.5B全参与LoRA。
  两平台各3项新进程精确恢复共6项全部通过（1.5B推理、0.5B全参、1.5B LoRA）；不是独立数值通过或训练效果资格。
- 4个CPU超时：1.5B B512/L512与B8/L8192推理、0.5B全参L2048的micro1/4训练。
  NPU0.5B全参L2048/micro4估算约73.23GiB，超过当时安全可用约54.73GiB，分配前拒绝；
  不是实际OOM或已经证明的硬件上限。micro1已运行，峰值15.853GiB，可供后续预算校准。

## 下载与剩余事项

Qwen0.5B、1.5B、3B全部hash通过。OLMoE、Qwen7B、Qwen30B-A3B均因响应body提前结束而失败；
分别保留约1.276/0.453/1.468GiB partial。OLMoE一次显式恢复仍失败。**MoE尚未实际运行模型**。
这些失败只影响依赖项；此前完整资产和实测不重跑。10个优化引擎/TP、EP对照仍未适配和排队。

下一步优先调查长响应断流，考虑固定Range分段及有界续传；完成资产后只补依赖失败/未跑case。
再审阅数值/token差异、校准保守训练预算和CPU时限覆盖，继续有价值的引擎对照。
使用整份queue加新attempt会重跑已完成项；后继应先提供case筛选或从原case.json提取单点配置。
唯一待办在[ROADMAP](ROADMAP.md)，不把本次终态称为全部实测计划完成。

## 固定源码、环境和材料

数学固定`0770593838a37c790a568226930567ed60f99b83`，本次校验通过。
资产bootstrap为`3360eafd3f5f94dc993d5d8cefdc428186d50b02`；
CPU/后批/已安装wheel为`6bbc7e2240c2a20958c1bb5206030f4d2ade2ac4`（固定设施81项通过）；
首批NPU为`15dfabe55ea8f0608cf6fc795513c6487eb664c4`（锁等待后继，固定相关19项通过）。
这些资格不因主分支后续文档提交更换身份。公共Torch2.12/TorchNPU2.12/CANN9.1只读，
私有Transformers4.57.1/Accelerate1.11.0/PEFT0.17.1未升级；gpu3090未连接。

外部根：`/mi/data2T/zlong/home/llm/gef-lm-experiments/20261102`。
`source/`固定检出、`env/transformers`私有环境、`models/`资产、`runs/`逐项结果、`jobs/`状态/日志。
四份有效队列为`runs/cpu-b`、`runs/npu-c`、`runs/cpu-stage2-b`、`runs/npu-stage2-b`。
旧启动失败、取消、下载失败保留原终态，不改写。仓库`artifacts/jobs/lm-*`是外部记录链接。
原始文件尚未退役；模型partial和必要复现不得因外层队列完成而擅自清理。

```sh
git status --short --branch
python -B scripts/status.py --all-jobs
systemctl --user list-units 'gef-lm-*' --all
quota -vs
# 不导入Torch、不启动模型：
/mi/data2T/zlong/home/llm/gef-lm-experiments/20261102/env/transformers/bin/python -I -m tidegraph lm results /mi/data2T/zlong/home/llm/gef-lm-experiments/20261102/runs
```

完整CLI、配置/预算、检查/运行/恢复与报告见[external-lm](external-lm.md)。
主仓库可继续编辑开发；后续作业仍须新固定源码和独立输出，不自动采用主工作区变更。

## 资源事实

2026-11-03账户61,280MiB，约59.8GiB；实验根du约25GiB，硬配额800GiB。
项目80/160GiB、账户300GiB停止线保持；当前缺口不是磁盘容量。
静态主存预算384GiB；systemd MemoryMax虽请求同值，但v1 memory controller仍共用user service，
不能声称逐unit硬隔离。生命周期background.slice与focus.service隔离、超时及后代停止已有实际证据。
本次没有重新分配NPU或修改公共环境。

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
