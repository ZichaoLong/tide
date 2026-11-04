# 当前交接 — Dense/MoE重启后全部终态（2026-11-04）

**本批后台全部结束，当前无活动项目进程。** 最后一项于北京时间15:37:11结束。
本次状态审阅未启动新下载/模型/构建。7个job（含资格）记录后代为空，实际PID/start均退出，
系统无active gef-lm unit，排除检查自身祖先后无实验根相关进程。
[本次终态](evidence/llm-restart-terminal-20261104.md)/[21项JSON](evidence/llm-restart-terminal-20261104.json)
是当前结果入口；[启动记录](evidence/llm-restart-20261104.md)保持原10:10时间身份。

## 完成与缺口

- 21项：6执行完成（4成本/2小检查）、2超时、3资产依赖未跑、10调度失败未提交。
  外层exit0不是全部通过，未创建的job/case不能写成运行完成。
- OLMoE CPU B512完成：BF16、16线程、输入512/输出128，正式1962.182秒、33.400输出token/s；
  另warmup728.533秒，RSS66.136GiB。原1200秒超时保留，新7200秒后继完成。
- OLMoE双NPU B512完成：相同工作负载183.145秒、357.837 token/s；物理5/9，卡9共享争用。
  allocated峰值31.738/31.175GiB，reserved56.908/55.705GiB；显式跨层复制，不是TP/EP。
  两大点独立数值/离散/checkpoint均not_checked，成本为探索观察；原单卡预算拒绝不解除。
- llama.cpp完整构建及两份GGUF已完成；Qwen1.5B/OLMoE B1正式51.258/37.278秒、2.497/3.434 token/s，
  两次新C++进程完整序列化状态/下一logits精确续接通过。B2小检查两者均数值失败，Qwen token不同、
  OLMoE token相同；独立KV未查。两B32均触及1200秒整流程时限，无完整吞吐。
- Qwen7B完整14.196GiB、4分片和全部资产发布hash通过，当前size/mtime仍匹配，CPU/NPU两点未执行。
  Qwen30B-A3B第5/16片11次URLError耗尽10次重试；前4片/完整文件14.899GiB保留，partial2.875GiB。

## 新调度故障与下一步

运行时在固定检出留下648字节`fusion_result.json`（graph_fusion，mtime10:59:01），
跟踪源码无diff，唯一未跟踪文件触发14:40:20的submit_lm_job干净源码保护，dispatcher退出。
因此Qwen7B CPU/NPU2项、Qwen MoE CPU2/NPU6项根本未提交；后者资产后来也失败。
这是运行工作目录隔离缺口，不是用户源码修改或模型失败。文件保留，不删除或忽略来掩盖问题。

下一步优先修复：模型子进程cwd改为各case输出/scratch目录，固定源码/输入用绝对路径，
保留clean guard；测试写cwd后延迟提交仍成功、超时/停止后代清理。提交固定新实现后，
只补Qwen7B两点与其他未执行项，不重跑完成曲线。Qwen MoE继续保留分片/断点，另列有界网络方案；
llama B32需先区分构造/warmup/实际生成阶段再决定新时限或实现优化。原数值/token失败不放宽。
本次只审阅与更新证据，未自动重启。唯一待办见[ROADMAP](ROADMAP.md)。

## 来源、材料与命令

数学0770593838a37c790a568226930567ed60f99b83本次校验通过；实现/安装固定
79197ad557b33bfa82c1914a02038f8944351247，118项设施资格已通过，不能当模型资格。
固定检出`source/restart-g`目前仅上述运行时输出未跟踪；主工作区独立。
公共Torch2.12/TorchNPU2.12/CANN9.1只读；私有Transformers4.57.1/Accelerate1.11.0/PEFT0.17.1。
llama.cpp固定2cbca389f515df455412dafa07e2b1359cc72a6d；实际native ARM/OpenMP、BLAS OFF，
构建320 affinity线程、测量16线程，GGUF MOSTLY_BF16/1D FP32及BF16 KV显式身份。

外部根`/mi/data2T/zlong/home/llm/gef-lm-experiments/20261102`：
`jobs/assets-small-g`成功；`assets-moe-g`与`dispatch-g`失败；`cpu-ready-20261104-g`、
`llama-g`和`placement-g`外层处理结束；`qualify-g`通过，全部终态。
原计划`recovery/plans-20261104-g/all.json`21项；本批`runs/*20261104-g*`只有11项实际记录，
其余10项未提交事实在本次审阅中明确列出，未伪造执行记录。
`engines/llama/ready.json`及`qwen15/olmoe/conversion.json`含build/GGUF身份，保留可复用资产。

```sh
git status --short --branch
python -B scripts/status.py
systemctl --user list-units 'gef-lm-*' --all
quota -vs
lm_root=/mi/data2T/zlong/home/llm/gef-lm-experiments/20261102
# 仅读取；生成清单不等于自动重跑：
"$lm_root/env/transformers/bin/python" -I -m tidegraph lm retry-matrix --group all
"$lm_root/env/transformers/bin/python" -I -m tidegraph lm results "$lm_root/runs" --format csv
cat "$lm_root/recovery/20261104-g/dispatch.json"
git -C "$lm_root/source/restart-g" status --short
```

每job保留status.json/task.log；每case保留配置、phase/process、result/check及所选resume。
新增运行用新固定检出、新attempt；当前生成器仍包含已完成项，不能不经筛选直接整轮重启。
NPU原48小时单卡共享策略不自动展期/重绑，未来先检查实际PID/start及资源。
账户约136GiB、实验根du约84GiB；project160/account300GiB新增写入停止线、800GiB硬限保持。
无需扩盘、tmpfs或gpu3090。公共运行入口见[external-lm](external-lm.md)。

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
