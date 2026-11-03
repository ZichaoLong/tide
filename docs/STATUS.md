# 当前交接 — Dense/MoE剩余任务已放后台（2026-11-03）

用户授权完成剩余事项并放到后台。当前7个服务都实际active，生命周期独立于本会话；
[后继证据](evidence/llm-continuation-20261103.md)/[JSON](evidence/llm-continuation-20261103.json)保存本次观测。
主工作区可继续编辑。活动输入/固定源码/环境位于仓库外，不得编辑或清理；未使用gpu3090。

## 作业、源码与剩余动作

外部根`/mi/data2T/zlong/home/llm/gef-lm-experiments/20261102`，下表路径均相对该根。
所有日志在job目录的`task.log`，持久状态`status.json`；外层完成不代表内部通过。

| service（省略.service） | 固定源码 | 记录与当前阶段 |
| --- | --- | --- |
| gef-lm-assets-small-20261103-d | 8b45cb4 / source/recovery-d | jobs/assets-small-d；OLMoE下载，随后7B |
| gef-lm-assets-moe-20261103-d | 8b45cb4 / source/recovery-d | jobs/assets-moe-d；30B MoE下载 |
| gef-lm-dispatch-20261103-e | 472d94f / source/runner-e | jobs/dispatch-e；recovery/20261103-e/dispatch.json，等待small/moe后继提交 |
| gef-lm-cpu-ready-20261103-e | 472d94f | jobs/cpu-ready-20261103-e；CPU B512后继计算中 |
| gef-lm-npu-ready-20261103-e | 472d94f | jobs/npu-ready-20261103-e；micro1显存诊断等compute.lock |
| gef-lm-llama-20261103-f | 983da72 / source/engines-f | jobs/llama-f；等待small下载终态后自动构建、转换、CPU检查与测量 |
| gef-lm-placement-20261103-f | 983da72 | jobs/placement-f；Qwen1.5B双NPU点已进入协作/项目锁队列 |

- 43项明确后继：32下载依赖点原配置、4个CPU原超时点的新7200秒上限、7个小诊断。已完成旧曲线不重跑。
  ready计算组已提交；small/moe资产齐备或下载失败终态后，dispatcher分别自动提交各24小时CPU/NPU服务。
- 引擎/放置后备另含4个CPU llama.cpp点、6个NPU跨层点和4小检查。后备是实际Transformers/Accelerate复制，
  不称vLLM或TP/EP。官方A3路径需NNAL和另配Torch2.10原生栈，本机NNAL缺失；本次有界停止及重启条件见报告。
- 32MiB分段/每文件6次临时重试已实际推进并恢复一次连接错误；完整模型hash尚待完成。无无限重试。
- 下载各30小时；计算队列各24小时；引擎服务42小时（含依赖），构建2小时/转换1小时，设备/锁等待6小时。
  所有重型构建、转换和正式CPU/NPU计时共用`compute.lock`，小型设施检查不计入性能。
- llama完整构建/链接、BF16 KV实际支持、GGUF及模型结果仍待后台；仅固定头文件C++语法检查已通过。

源码资格：8b45cb4固定相关27项；472d94f私有wheel安装和固定94项；983da72固定48项及C++语法检查。
94/48有重叠，不累计成模型资格。当前安装wheel为472d94f；引擎服务通过固定983da72 PYTHONPATH运行，
未把旧wheel冒称最新安装。公共Torch2.12/TorchNPU2.12/CANN9.1只读，Transformers4.57.1等原增量不升级；
llama转换另用私有overlay。数学固定0770593838a37c790a568226930567ed60f99b83，重入/交付校验通过。

## 已有结果与严格边界

[首次80项终态](evidence/llm-terminal-audit-20261103.md)原样保留：43执行完成、32依赖失败、4CPU超时、1预算拒绝。
36完成测量/观察不都是正式吞吐；7已执行独立小检查全部超差，6个选择的新进程精确恢复通过。
CPU4/NPU5训练配置各3次计时更新；不是模型效果资格。
Qwen1.5B BF16 NPU B128/B512分别6.339/24.924秒、2584.46/2629.45输出token/s，峰值5.591/13.668GiB。
CPU B128为580.328秒，原B512 1200秒超时；新后继不覆盖原失败。Qwen0.5B/1.5B/3B资产完整hash通过。

6个新数值诊断已完成：两平台两模型的同eager缓存/完整前缀从logits.1超差但token相同；
完整/最后logit投影对照通过。CPU同cached eager/SDPA超差、token相同；NPU两者数值/token均不同。
CPU FP32/eager完整梯度/第一步Adam小检查通过，NPU对应仍失败；不修饰旧BF16检查。
逐字段记录在`runs/*-ready-20261103-e/*/run/check.json`，小诊断不外推大点。
NPU0.5B L2048/micro4的73.23GiB保守准入拒绝保持，micro1诊断不是豁免。
MoE尚无完成的真实模型结果；自身恢复不能代替独立数值比较。

## 后续检查命令

```sh
git status --short --branch
python -B scripts/status.py --all-jobs
systemctl --user list-units 'gef-lm-*' --all
quota -vs
lm_root=/mi/data2T/zlong/home/llm/gef-lm-experiments/20261102
cat "$lm_root/recovery/20261103-e/dispatch.json"
cat "$lm_root/recovery/llama-20261103-f/status.json"
cat "$lm_root/recovery/placement-20261103-f/status.json"
# 只读现有结果，不启动模型（旧/新尝试分行，不能只数outer passed）：
"$lm_root/env/transformers/bin/python" -I -m tidegraph lm results "$lm_root/runs" --format html --output "$lm_root/recovery/results.html"
"$lm_root/env/transformers/bin/python" -I -m tidegraph lm results "$lm_root/runs" --format csv --output "$lm_root/recovery/results.csv"
# 最新引擎清单/预算入口：
PYTHONPATH="$lm_root/source/engines-f/python" "$lm_root/env/transformers/bin/python" -B -m tidegraph lm engine-matrix
# 停止时选定实际unit，随后核对ControlGroup、status.json及remaining_live_pids：
# systemctl --user stop gef-lm-llama-20261103-f.service
```

dispatcher只负责提交后继：停止dispatcher不会停止已提交的独立计算服务；如需全停须逐个停止上述unit及其已提交后继。
同一attempt不可重用，失败恢复先审阅逐case原因再生成选择性清单。后台不会自动唤醒AI，用户下次唤醒后审阅终态与证据。
从干净仓库重新生成材料、安装/单次运行见[external-lm](external-lm.md)；原始矩阵与后继清单均由跟踪代码生成。
旧失败、原始必要复现和partial尚未获准退役。首组分段c服务目录准备失败已修，原失败记录保留且无遗留后代。

## 资源事实

2026-11-03本次账户63,166MiB（约61.7GiB），实验根du约27GiB，硬配额800GiB。
项目160GiB、账户300GiB停止新增大写入；当前瓶颈是下载和CPU计算，无扩盘或tmpfs需求。
静态主存384GiB；systemd MemoryMax虽请求该值，v1 controller仍共用user service，不称逐unit硬隔离。
NPU按两次采样/无其他进程/整组锁分配，实际映射在case allocation.json；不停止其他项目任务。

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
