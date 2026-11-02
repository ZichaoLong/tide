# 当前交接 — Dense/MoE后台实测（2026-11-02）

用户已授权[有限计划](evidence/llm-experiment-preparation-20261102.md)，要求提交后台后稍后唤醒检查。
长下载/安装/实测使用独立background.slice；不连接或重租gpu3090。
主工作区可继续开发，活跃任务不读取其后续修改。本文是当前交接，待办只见ROADMAP。

## 固定身份与已验证范围

- 数学固定`0770593838a37c790a568226930567ed60f99b83`，校验通过，未改core-5语义。
- 资产bootstrap：`3360eafd3f5f94dc993d5d8cefdc428186d50b02`。
- 运行实现、已安装非editable wheel及CPU/条件后批：`6bbc7e2240c2a20958c1bb5206030f4d2ade2ac4`，固定81项相关检查通过。
- NPU队列后继：`15dfabe55ea8f0608cf6fc795513c6487eb664c4`，改计算锁为有界内核等待，固定19项队列/清单检查通过。未升级活跃环境或重建wheel；通过固定PYTHONPATH读取自身源码，实际包内容身份分别记录。
- 私有增量：Transformers4.57.1、Accelerate1.11.0、PEFT0.17.1；公共Python3.12/Torch2.12/TorchNPU2.12/CANN9.1保持只读。
- [启动证据](evidence/llm-background-execution-20261102.md)区分设施、独立小检查与原失败；随机tiny检查不替代预训练模型/NPU资格。

## 活动作业与材料

统一外部根：`/mi/data2T/zlong/home/llm/gef-lm-experiments/20261102`。
固定源码在`source/bootstrap-3360eaf`、`source/runner-6bbc7e2`、`source/runner-15dfabe`；
环境`env/transformers`，模型`models`，输出`runs`，作业`jobs`。这些目录不得清理、编辑或升级。
仓库`artifacts/jobs/lm-*`仅链接外部状态；短审计/代码检查仍可前台完成。

| unit | 当前事实 | 状态/日志相对外部根 |
| --- | --- | --- |
| gef-lm-cpu-20261102-b.service | 运行中；Qwen1.5B线程校准后选16线程，继续batch/长度/训练 | jobs/cpu-b；runs/cpu-b/queue.json |
| gef-lm-npu-20261102-c.service | 首个真实NPU小检查已完成（数值超差、token一致），继续首批；协作队列分配设备 | jobs/npu-c；runs/npu-c/queue.json |
| gef-lm-qwen05-recovery-20261102-b.service | 已exit0，全部Qwen0.5B资产验证完成，无遗留后代 | jobs/qwen05-recovery-b |
| gef-lm-olmoe-recovery-20261102-b.service | Qwen0.5B完成后显式续传OLMoE，原断流记录保留；无循环重试 | jobs/olmoe-recovery-b |
| gef-lm-assets2-20261102-b.service | 首批终态/有效Qwen1.5B B1校准为条件；获取3B/7B/30B MoE后提交后批 | jobs/assets2-b；stage2-download-b.json |

后批自动提交的unit是`gef-lm-cpu-stage2-20261102-b.service`与`gef-lm-npu-stage2-20261102-b.service`，
源码仍为6bbc7e2；不存在该unit时不写成已运行。首批68流程、条件后批12流程，
其中71测量/观察及9小检查。10个优化引擎/TP、EP点尚未排队，需配套适配，不能冒充完成。

Qwen0.5B/1.5B全部资产hash通过。OLMoE首分片在1,045,278,078/4,997,744,872 bytes处提前断流；
`jobs/assets1-a`终态exit1，partial和原失败保留。已提交一次显式OLMoE续传；如果再次失败，保留终态并只影响其依赖点。
Qwen0.5B恢复完成后，OLMoE续传与后批Qwen3B下载最多两路，不修改共享网络路线。

## 已观察的检查、成本与失败

Qwen1.5B CPU FP32 B2/L16/输出4：token一致，严格logits/KV有55项超差，最大1.83105e-4；
阈值atol1e-5/rtol1e-4不改。参考eager完整前缀，候选SDPA在线缓存；其他大点仍not_checked，
执行/自身恢复/数值/路由分别记录。CPU B1的同栈新进程精确续接通过，不代替独立正确性。
NPU-c已实际执行Qwen0.5B BF16 B2/L16/输出4：逻辑npu:0映射物理7，494,032,768参数位于NPU；
输出token一致，与独立CPU FP32的严格数值比较超差（最大绝对1.10617），保留精度差异，不写成数值通过。
CPU BF16 B8/L512/输出128线程校准单次完整窗：4线程161.754s、16线程67.664s、64线程184.702s；
这是校准成本，不是全部CPU模型/尺度结论。

原9cdf7d9的CPU/NPU `*-a` 清单初始化TypeError失败，未执行模型；assets2-a仅等待，已停止且无遗留后代。
6bbc7e2修复并保留原失败。npu-b因轮询计算锁饥饿，确认还未创建计算子进程后取消，MainPID=0、
ControlGroup为空、remaining_live_pids=[]；后继npu-c使用内核等待。活跃CPU任务和原下载源码未改动。

## 下一次检查与继续开发

```sh
git status --short --branch
python -B scripts/status.py
systemctl --user list-units 'gef-lm-*' --all
quota -vs
# 这些查询不导入Torch、不启动模型：
/mi/data2T/zlong/home/llm/gef-lm-experiments/20261102/env/transformers/bin/python -I -m tidegraph lm results /mi/data2T/zlong/home/llm/gef-lm-experiments/20261102/runs
# 持久日志例：
tail -n 50 /mi/data2T/zlong/home/llm/gef-lm-experiments/20261102/jobs/cpu-b/task.log
```

先审阅各run/check/resume和队列内层状态，不把外层完成当全部通过。优先检查NPU首个小例、
B128/B512、训练与精确续接，以及OLMoE/后批分片下载。下载失败应显式续传原固定资产，
只对依赖失败/未跑点安排新run，不重复已完成测量。
安装、预算、单点、独立检查/恢复、报告与示例命令见[外部LM手册](external-lm.md)。

可以编辑/提交主仓库。正式计时期间，本项目重型构建应等待同一锁：
`flock /mi/data2T/zlong/home/llm/gef-lm-experiments/20261102/compute.lock YOUR_BUILD_COMMAND`。
不修改共享工具链，不删活跃worktree、partial和环境；不自动停止其他项目任务。
停止某个明确unit用`systemctl --user stop UNIT`，随后核对status.json和实际ControlGroup/后代。
取消全部实验时先停止assets2条件服务，防止以后再提交后批。后台不承诺自动唤醒AI或跨主机重启续跑。

## 资源事实

账户最近约36GiB，硬配额800GiB；项目首批80GiB/全阶段160GiB、账户300GiB停止新增大写入。
每计算/恢复1200秒，设备/计算锁等待6小时，计算service24小时，下载service30小时；Qwen0.5B恢复3小时。
主存静态准入384GiB；systemd也请求MemoryMax=384GiB，但本机v1 memory controller仍落在共同user service，
MemoryCurrent未设置，不能声称每unit已硬隔离。实际生命周期控制组在background.slice、与focus.service分离；
计时串行并记录RSS/设备峰值。tmpfs未作为当前必需依赖，不修改公共控制器配置。

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
