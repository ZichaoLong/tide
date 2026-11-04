# 当前交接 — Dense/MoE失败重启准备（2026-11-04）

用户明确授权重启失败作业。当前实现正在固定和验证，尚未提交新模型作业；
旧11个后继service全部终态、没有活动项目后代。旧结果保留于
[57项终态](evidence/llm-terminal-audit-20261104.md)/[JSON](evidence/llm-terminal-audit-20261104.json)。
实现基于a96ee0d，数学0770593838a37c790a568226930567ed60f99b83本次校验通过。

## 本次选择、预算与依赖

`lm retry-matrix`从跟踪生成器产生21项：compute11、llama6、placement4。
19个未执行依赖点保持原配置；OLMoE CPU B512整流程1200→7200秒；
OLMoE原单NPU B512估算63.080GiB仍拒绝，另列双NPU显式跨层B512，每卡约36.8GiB。
后者不改逻辑batch、输入512/输出128、BF16、prefill_chunk64或Attention可见性。
已完成的数值失败不重跑，不放宽阈值；原OLMoE双NPU小检查/精确恢复按原身份复用。

实际1MiB探测：权重直连及当前proxy入口15秒超时，c28返回精确206（3.681秒）；
llama源码直连1.125秒、c28 2.210秒、当前proxy 6.572秒，均200且不支持Range。
这是首块可达性观察，含连接开销，不外推稳定大文件带宽。
模型仍用c28，8MiB Range、每文件10次临时错误重试；两个模型服务串行，保留所有已验分片和partial。
llama源码改用已测直连；构建/转换与模型计时共用compute.lock。

项目根`/mi/data2T/zlong/home/llm/gef-lm-experiments/20261102`，模型/私有环境继续复用。
公共Torch2.12/TorchNPU2.12/CANN9.1只读；Transformers4.57.1/Accelerate1.11.0/PEFT0.17.1不升级。
当前账户约106GiB、实验根约55GiB；剩余权重/GGUF/构建约70GiB仍在项目160GiB与账户300GiB停止线内，
800GiB硬配额不变。主存约1.6TiB可用，计算服务请求上限384GiB，不称v1控制器逐unit硬隔离。
构建按实际affinity320线程，模型计时16 ATen线程，项目重型计时串行。

NPU当前均有其他项目进程；普通队列要求空闲卡。若复用既有低占用共享授权，
只对重新观察并绑定PID/start/UID的一张低占用卡设置有期限的策略，其他卡仍需空闲；
策略或占用变化拒绝，不停止他人任务，不称共享计时独占。
设备/compute锁等待最多6小时，计算队列24小时；下载/依赖等待与整体服务超时在正式提交记录中固定。

## 下一步

完成相关检查后提交实现，建立仓库外新固定检出，安装私有wheel并执行固定资格；
随后提交独立下载、dispatcher、llama及放置服务，核对真实cgroup、状态和日志，再单独提交启动证据。
不得编辑活跃固定源码或重跑已完成曲线。所有新attempt必须新目录；旧失败保留。
公共生成/运行/恢复和站点适配见[external-lm](external-lm.md)。

```sh
git status --short --branch
python -B scripts/status.py
systemctl --user list-units 'gef-lm-*' --all
quota -vs
# 从当前已安装版本还不能假定存在retry-matrix；新安装完成后使用：
# tidegraph lm retry-matrix --group all
```

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
