# 当前交接 — Dense/MoE已授权，实施与后台准备（2026-11-02）

用户已授权按[有限计划](evidence/llm-experiment-preparation-20261102.md)执行，并要求后台提交后以后唤醒检查。
所有长下载、环境准备、构建和模型任务使用独立background.slice；短代码检查可以前台完成。
主工作区可继续开发，活跃任务只读仓库外固定提交；不自动采用后续提交或修改公共软件。

## 当前实施

- 新增`tidegraph lm catalog/fetch`，固定六模型revision、文件大小与SHA256/Git blob身份，断点读取拒绝错误Range，失败保留partial，完成后原子发布。入口不导入Torch。
- 项目/账户/文件系统三层写入准入，默认首批80GiB、账户300GiB停止线；复用原持久记录和子进程监督器。
- 下载、环境和运行源码将分别冻结在`/mi/data2T/zlong/home/llm/gef-lm-experiments/20261102/source/`。
- 共享资产、私有环境、日志位于同一仓库外根的`models/`、`env/`、`jobs/`。这些活动目录不得清理、编辑或升级。
- 开发检查：Python3.12下资产续传/错误Range/哈希拒绝/固定目录4项通过；无Torch导入检查通过。系统Python3.9不满足项目>=3.10，原开发启动失败已定位，任务显式选公共Python3.12。

当前源码尚在提交边界；首批下载和环境服务待本实现提交后启动，不能写成运行或通过。
计划unit：`gef-lm-assets1-20261102-a.service`、`gef-lm-env-20261102-a.service`。
LM推理/训练/检查/报告适配继续实施，尚未排计算矩阵。优化引擎配套、TP/EP仍按计划分别验证。
用户已撤回另机下载安排，本机将获取Qwen MoE；它仍属于后批，不提前绕过资源校准。

## 接续与协作边界

授权分支`graph-execution-foundation`，启动HEAD `58b68b11aa394749805f3121299edf5e9ae5f638`；固定数学校验通过。
每次提交后以固定检出运行，再单独提交小型证据；普通推送已授权。
用户可编辑/提交主工作区。不要删除活动worktree、修改其环境或清理模型partial；正式计时避免同时进行本项目重型构建。
其他工作不停止。后台提交不承诺自动通知，未知错误保留终态与日志，不能把外层完成当模型通过。

```sh
git status --short --branch
python -B scripts/status.py
quota -vs
systemctl --user status gef-lm-assets1-20261102-a.service gef-lm-env-20261102-a.service
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
