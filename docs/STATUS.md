# 当前交接 — 日常入口已迁至 CANN 9.2.0（2026-11-06）

本轮已完成，当前无排队/运行作业。公共工具链和旧9.1环境未修改，GPU仍已释放。
日常开发使用`ascend/dev-workspace-9.2.0`与调用者选择的私有venv，下面以`TIDE_VENV`传入位置；
新LM环境默认`EXPERIMENT_ROOT/env/transformers-cann9.2.0`，旧环境不会自动迁移。
新venv/旧环境误用拒绝、CLI配置预览及安装检查已实际验证。

## 已实现、已构建、实际通过及严格失败

入口实现`f1480a0`；实际wheel、Python/native、独立LibTorch-NPU core/客户端构建
`9df0ecf8a5f8a98e4144f45161cffd5054b04002`；恢复驱动修复
`60e19755b04f09d676f0636dd32a5cabc19e6560`。后者复用经身份核对的构建，没有声称重新编译。
数学仍为`0770593838a37c790a568226930567ed60f99b83`，固定快照通过。

[迁移报告](evidence/cann92-project-migration-20261106.md)/[JSON](evidence/cann92-project-migration-20261106.json)：
SettleGraph/FC/KV四节点、8190可训练参数、B2/P3/窗12、FP32、prefill、packed完整返回，
三客户端CPU/NPU mixed-c完整推理、声明loss训练/梯度/SGD槽和新进程精确续接全部通过。
CPU设施52项、随机小型LM CPU设施22项、固定驱动设施9项通过，部分重叠不合计成数学配置。

独立综合all根仍有三项严格失败：CPU/native、NPU PyTorch/native，input.0.1.0.1第4坐标。
NPU PyTorch差2.86102e-6与20261031历史同值。原容差/失败保持；完整门禁仍failed。
CPU PyTorch八组根及CPU/NPU standalone所列16个根案例通过，各客户端目标不能互相替代。
本轮不扩展数值工程；重启条件和其他未解除边界只见[ROADMAP](ROADMAP.md)。
resident、多卡、FP16、大batch、规模性能和公开权重没有重新验证，旧资格不自动升级。

## 作业、环境与接续命令

三服务`gef-cann92-migration-{a,b}`、`gef-cann92-recovery-c`全部终态exit1、MainPID=0、
ControlGroup为空、无匹配后代；a为已修复哈希入口失败，b/c保留综合根失败和原身份拒绝。
源码分别固定于`.local/cann92-source-{a,b,c}`，均干净；job记录在`artifacts/jobs/`。
主要结果/资格wheel环境/实际两套构建在`artifacts/cann92-20261106-b`，恢复补验在`...-c`。
日常venv独立于资格wheel环境；实际环境/构建位置保存在迁移证据，本机site文件不作为固定接口。
只读核对显示日常venv增量为pip25.0.1和editable tidegraph0.3.0（约13MiB），Torch/TorchNPU继承公共基线。
该环境绑定当前检出，不推荐给其他项目共用；下游按各自锁安装固定wheel。此次仅澄清文档路径，不改环境或资格。
账户约180GiB（软300/硬800），本轮材料约0.4GiB，无需tmpfs。

```sh
: "${TIDE_VENV:?请选择与该基线匹配的私有环境目录}"
module load ascend/dev-workspace-9.2.0
source "$TIDE_VENV/bin/activate"
python scripts/ascend_environment.py --require-venv
tidegraph plan configs/checks/ascend-migration.json
# 需要指定本机构建时追加：--site "$TIDE_SITE"
python scripts/status.py
```

从干净仓库按[development](development.md)重建环境/客户端。固定源码下通过有界background.slice运行
`bash scripts/qualify_ascend_migration.sh NEW_OUTPUT /absolute/use-local-ascend/scripts/run_on_free_npu.py all`，
材料全部来自跟踪配置；已知综合根仍预期严格失败。无需旧运行目录或权重下载。

## 原任务保留范围

旧LM根`/mi/data2T/zlong/home/llm/gef-lm-experiments/20261102`的9.1环境、固定源码与材料保持。
[15项旧终态](evidence/llm-isolation-terminal-20261105.md)、[B512入口](evidence/llm-capacity-entry-20261105.md)
及原core-5 204条结果继续按原身份解释；不自动重新入队。
远端未提交源码包仍在仓库外`~/.local/state/graph-execution-foundation/preserved-source/20261031-cuda3090/`，本次未分析。
旧历史身份按`python scripts/resolve_history_commit.py OLD_SHA`定位，不把映射当作验证。
