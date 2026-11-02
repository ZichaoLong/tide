# 当前交接 — Dense/MoE实测准备完成，待用户授权（2026-11-02）

本轮按用户最新要求完成环境只读检查、三路线有界测速，以及Dense/MoE具体有限计划。
没有下载模型权重、导入Torch/模型、安装引擎、执行训练/推理或提交后台任务；没有活动项目作业。
用户要求先审阅计划再授权执行。下一步按[准备报告与待授权有限表](evidence/llm-experiment-preparation-20261102.md)等待统一授权。
不恢复上一轮core-5矩阵，不依赖已释放的gpu3090。

## 本次准备

准备审阅基点`5d0e99e37cddf9db982fc9ad5aae13cd33ed0b54`；本次仅修改文档/数据。
[报告](evidence/llm-experiment-preparation-20261102.md)和[JSON](evidence/llm-experiment-preparation-20261102.json)记录：

- 9次顺序网络样本，共约31.33MiB公开非模型payload，读取后丢弃。直连/relay失败或超时；c28两类完整样本约1.06MiB/s。wheel SHA256与PyPI相同。一次权重HEAD成功、body为0，不是权重下载资格。
- 实际公共Torch/TorchNPU2.12/CANN9.1未改；Transformers等增量依赖尚未安装。vLLM-Ascend0.30声明2.10，普通vLLM0.30声明2.13；需专用配套与私有环境，未强装或升级公共栈。
- 首批dense/OLMoE 32个CPU/NPU主推理点，后批3B/7B和Qwen MoE 12点；B128/B512明确保留。另设真实变长、实现/放置、训练、校准/profile上限，总计最多81个流程，不是已运行配置。
- 首批项目磁盘80GiB、全部阶段160GiB；账户300GiB暂停新大写入，保留400GiB软限/800GiB硬限余量。临时tmpfs32GiB与每任务主存384GiB联合预算。大模型不是仅按激活参数预算。
- background.slice在线，尚未创建任何新unit。将按依赖分开下载/环境/CPU/NPU任务，单配置20分钟、设备等待6小时、计算unit24小时、大资产下载unit30小时；具体实现与任务名均待授权后生成。

## 既有公开调查

授权分支`graph-execution-foundation`，调查重入源码`37b567da553e988a35c65a9746a2c466aaa5ace0`；
此次只改文档/数据，无运行实现或数学变更。固定数学仍为`0770593838a37c790a568226930567ed60f99b83`，离线校验通过。
最新提交及是否有未提交修改，以`git status --short --branch`为准。

- [调查报告](evidence/qwen-throughput-survey-20261102.md)、[JSON](evidence/qwen-throughput-survey-20261102.json)、[CSV](evidence/qwen-throughput-survey-20261102.csv)：83条外部记录，CPU15/NPU13/CUDA55（52推理、3训练）；30份来源，CSV按指标展开为178行。
- 公布模型大小、精度、引擎、硬件、batch定义、长度、原数字/位置、文档hash、缺失条件与计时范围。全体标为外部未在本机验证；不是83条合格run，不混入core-5的204条原结果。
- 区分Qwen3总token速度、纯decode、服务输出吞吐、固定batch延迟和训练s/it。NPU dummy权重、原问题与后继、当前脚本/原运行锁、CPU量化/平台差异分别保留。
- 核对两套固定模型config及结构源码，给出14项权重+KV静态下界、参数共享计数区别、CPU/NPU B128/B512与训练的有限后续表。静态预算不是准入/峰值，也没有新增本机吞吐。
- 已核验83条身份/来源、30份引用hash、178个CSV指标对应、五对CPU统计、NPU token/耗时比值、14项预算算术及固定数学校验。仅文档数据检查，不称模型正确性、optimizer或恢复验收。

公开资料通过现有`use-netpath-network`的c28入口读取；没有修改全局网络、SSH或公共软件。
参考仓库保持只读；本次没有使用子代理。报告/数据约0.23MiB，不依赖临时网页缓存。

## 下一步边界

实测需用户对本次具体计划统一授权：先CPU/NPU Qwen2.5-0.5/1.5B和OLMoE，再进入3B/7B及双芯片Qwen MoE。
新增外部LM入口尚未实现；公共`tidegraph`不能直接把现有core-5配置替换成Qwen模型名。
实现/量化/输出契约分开；原公开资料和网络准备都不认证本机模型正确性、恢复或吞吐。
缺少本机结果、训练绝对吞吐和真实大batch曲线的事项只维护在[ROADMAP](ROADMAP.md)。
不能直接将core-5样例窗/s与Qwen token/s相除，不以相近参数量推断同函数、工作量或质量。

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

## 资源与干净检出接续

本次准备观察账户占用32438MiB，配额400GiB软限/800GiB硬限；用`quota -vs`核对，不能用整盘df代替。
未来tmpfs必须联合主存预算。公共TorchNPU2.12/CANN9.1和独立LibTorch2.10/CANN9.0 SDK分开，均未修改。
模型实验环境兼容性尚未验证；不要把资料中的安装命令当本机已部署事实。

```sh
git status --short --branch
python -B scripts/status.py
python -B scripts/check_upstream.py
quota -vs
# 无需Torch、联网或原运行目录即可查看新调查：
python -B -m json.tool docs/evidence/qwen-throughput-survey-20261102.json > /dev/null
# 具体按平台/指标查询命令见调查报告；CSV可直接在表格软件筛选。
```

需要重建core-5时按[development](development.md)、[人类手册](human-workflow.md)和[复现入口](reproduction.md)。
此时无应自动恢复的历史作业；Dense/MoE实测仍停在用户授权前，不自动下载或排队。
