# 当前交接 — Dense/MoE后继全部终态（2026-11-04）

本次为状态审阅，未启动新下载、构建或模型执行。最后一项于2026-11-03 21:29:46北京时间结束。
**当前无活跃LM作业或项目遗留进程**：11个后继service均终态、记录后代为空，原PID/start身份已退出，
排除检查自身祖先后无进程命令行引用实验根。主工作区可继续开发。
[本次终态报告](evidence/llm-terminal-audit-20261104.md)/[57项JSON](evidence/llm-terminal-audit-20261104.json)
是本轮结果入口；[11-03启动观测](evidence/llm-continuation-20261103.md)保持原时间身份，不继续称running。

## 完成、失败与待续

- 57个唯一case与跟踪生成器完整配置逐项一致：36执行完成、19依赖失败未执行、1CPU超时、1NPU预算拒绝。
  26完成测量/观察含2个profile，另10个检查/诊断；外层queue exit0不是全部内项通过。
- 原4个CPU超时点均在明确7200秒后继中完成。Qwen1.5B BF16、16线程、B512、输入512/输出128：
  正式生成2344.508秒（约39.1分钟）、27.953输出token/s，warmup837.722秒另列，RSS峰值25.966GiB。
  B8/L8192正式生成1259.049秒；0.5B L2048/globalB4全参micro1/4各3更新中位155.515/534.987秒。
- OLMoE完整模型hash通过并已实际运行：CPU/单NPU B1/8/32/128、L128/512/2048、变长B128、LoRA及profile。
  B128输入512/输出128为CPU635.409秒/25.785 token/s、单NPU93.415秒/175.389 token/s。
  Q/V LoRA globalB4/micro1/L512三更新中位CPU49.616秒、NPU4.284秒；不是全专家训练或模型效果认证。
- Qwen1.5B与OLMoE双NPU跨层B1/B32完成，成本比相应单NPU更高；显式复制不是TP/EP。
  新4个磁盘/新进程精确续接全部通过：OLMoE CPU/单NPU/双NPU B1、Qwen1.5B双NPU B1。
- 严格边界：OLMoE CPU FP32独立检查36字段中5失败但token相同；单/双NPU BF16各36字段失败、各4个token不同。
  CPU FP32/eager训练锚点1160字段全通过，NPU对应307字段失败。大点独立检查未执行，成本是探索观察。
  原阈值、旧失败、4个因素诊断结论保持；自身精确续接不代替独立数值或路由检查。
- NPU micro1显存诊断完成：实际96次融合Attention与96次对应反向，单instrumented更新allocated峰值12.151GiB。
  该见证证明实际融合路径，不能直接外推micro4容量；原73.23GiB静态拒绝仍保留。
- OLMoE B512 CPU在1200秒整流程上限超时，无完整正式计时；单NPU静态估算63.080GiB超过当时安全54.725GiB，
  分配前拒绝，未实际OOM。后继须新时限/显式多设备或有依据的预算方案，不能缩batch或填补推算吞吐。

## 下载和引擎阻碍

- Qwen0.5B/1.5B/3B及OLMoE完整资产可复用。Qwen7B在首片保留2.930GiB partial；
  Qwen30B-A3B已验证前4/16权重分片，完整文件14.899GiB，第5片保留1.578GiB partial。
  后两者均在第7次临时错误后耗尽6次重试预算（最后是读超时），依赖点未执行。
- llama.cpp固定归档下载在5MiB处read timeout，未进入CMake编译。流程phase=building包含准备过程，
  不误报编译器/C++失败。4个成本点及2小检查未跑，尚无GGUF/BF16 KV实际资格。
- vLLM-Ascend A3原NNAL/原生栈配套边界保持；本次无新增vLLM或TP/EP实际结果。Qwen MoE四设备点也因资产失败未跑。

下一步先补两模型和llama源码下载；网络重启方案应利用现有partial/已验分片、评估更短Range/单流和有界重试，
不重新下载完整资产或重跑完成曲线。随后补llama四点、Qwen未跑点及OLMoE B512独立时限/容量方案。
数值/token差异按具体见证继续定位，不放宽容差。唯一待办在[ROADMAP](ROADMAP.md)。状态查询本身不重启作业。

## 固定来源、环境、材料与命令

数学0770593838a37c790a568226930567ed60f99b83，本次校验通过。
下载固定8b45cb42325224e38f945405988f5efe0bd83e3e（source/recovery-d、固定相关27项）；
后继472d94ffdcb23f6dc270626cd7089945d2a3b833（source/runner-e、已安装wheel、固定94项）；
引擎/放置983da722038814d24c03659c7187ec769b28b263（source/engines-f、固定48项、C++语法通过，完整构建未执行）。
94/48有重叠，不累计成模型资格。引擎作业实际用固定PYTHONPATH，不冒称最新wheel安装。
公共Torch2.12/TorchNPU2.12/CANN9.1只读，私有Transformers4.57.1/Accelerate1.11.0/PEFT0.17.1未升级。

外部根`/mi/data2T/zlong/home/llm/gef-lm-experiments/20261102`。
`jobs/assets-small-d,assets-moe-d,llama-f`失败；dispatcher、六个cpu/npu ready/small/moe队列及placement-f外层处理结束。
每job含status.json/task.log，逐case含process/status.json、run/result.json或check.json及resume.json。
`runs/*20261103-*`是本次57项；旧`cpu-b,npu-c,cpu-stage2-b,npu-stage2-b`是[首次80项终态](evidence/llm-terminal-audit-20261103.md)，
保留原失败和6次原精确续接，不覆盖或重复计数。所有原始必要材料和partial尚未退役。

```sh
git status --short --branch
python -B scripts/status.py --all-jobs
systemctl --user list-units 'gef-lm-*' --all
quota -vs
lm_root=/mi/data2T/zlong/home/llm/gef-lm-experiments/20261102
# 只读现有结果，查询不启动模型：
"$lm_root/env/transformers/bin/python" -I -m tidegraph lm results "$lm_root/runs" --format csv
# 生成后继配置/预算不分配模型；不可直接整轮重跑：
PYTHONPATH="$lm_root/source/engines-f/python" "$lm_root/env/transformers/bin/python" -B -m tidegraph lm recovery-matrix
PYTHONPATH="$lm_root/source/engines-f/python" "$lm_root/env/transformers/bin/python" -B -m tidegraph lm engine-matrix
```

新提交/新attempt只选择需要修复或未跑项，失败原记录保留；不要再使用已存在的attempt。
干净仓库配置、安装、单次运行与恢复见[external-lm](external-lm.md)，不依赖旧终态目录才能生成配置。
当前没有活动固定源码；未来长任务仍须新固定检出与独立持久服务。

## 资源

本次账户约106GiB，实验根du约55GiB，800GiB硬配额；project160/account300GiB停止新增大写入线保持。
当前主要缺口是网络与有限时限/容量，不是磁盘。无扩盘、tmpfs或gpu3090依赖。
systemd MemoryMax=384GiB是请求，v1 controller共用user service，不称逐unit硬隔离。
本次仅读取状态与小结果，未占用新NPU或修改公共环境。

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
