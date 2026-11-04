# 当前交接 — LM目录隔离修复、准备固定源码后继（2026-11-04）

用户已授权执行剩余事项。当前无活动项目作业；本次实现尚待提交后的固定源码资格，
不能把开发测试当模型通过。主工作区修改为本轮隔离/下载/llama进度及相关测试文档。
固定数学0770593838a37c790a568226930567ed60f99b83校验通过；开发相关38项通过。

## 本轮范围和顺序

- 模型/恢复子进程使用每case独立work目录与绝对路径；不放宽干净源码保护。
  原`source/restart-g/fusion_result.json`保留，不把旧固定目录当新资格源码。
- `lm retry-matrix --round isolation`只生成15项：compute10（Qwen7B2、QwenMoE8）、
  四NPU placement3、llama B32两项。前13项配置不变；B32仅显式延长1200→7200秒。
  已完成B1/OLMoE B512与严格失败按[上一批终态](evidence/llm-restart-terminal-20261104.md)复用。
- 下载只续传MoE，已验4/16片和第5片partial保留。c28当前2MiB精确Range试读通过，
  direct/proxy各20秒无数据；新预算单流32MiB、I/O120秒、每文件24次临时重试、服务30小时。
  Range/hash/容量错误仍立即失败，不修改公共路由。
- llama阶段日志覆盖构造、warmup、正式prefill/decode和保存；日志成本包含于相应时间。
  新客户端hash区分build目录，保留旧binary/shared libraries供原checkpoint使用。
  新旧两模型B2/L16/O4的完整输出/序列化状态与各自新进程恢复为B32成本前置门禁。

实现提交后：建立`source/isolation-h`固定检出，先`qualify_lm_runner.sh`安装/相关资格；
通过后冻结15项分组清单及旧llama build JSON。单独启动MoE下载、compute dispatcher、
llama与四NPU后继。Qwen7B完整资产可立即执行，不等MoE；所有重型构建/计时共用compute.lock。
资源等待、状态/日志和停止由background.slice及现有container_job记录，NPU沿用两次采样、整组锁。
当前有多张无进程NPU，先使用普通空闲策略，不自动续签原单卡共享策略。

## 资源与精确身份

外部根：`/mi/data2T/zlong/home/llm/gef-lm-experiments/20261102`。
账户占用约136GiB，主存可用约1.7TiB；项目160GiB/账户300GiB停止新增大写入，硬限800GiB。
无需扩盘或tmpfs，不连接已释放gpu3090。公共Torch2.12/TorchNPU2.12/CANN9.1只读，
私有Transformers4.57.1/Accelerate1.11.0/PEFT0.17.1保持匹配；新安装须在启动模型前完成。
原实现/安装79197ad557b33bfa82c1914a02038f8944351247，118项资格和旧模型结果保留原身份。
llama上游2cbca389f515df455412dafa07e2b1359cc72a6d，旧binary SHA256
818dc14af81b059a48523691d2f183c5e0121616605428073f1aadd72018c119；GGUF不重转。

```sh
git status --short --branch
python -B scripts/status.py
python -B scripts/check_upstream.py
systemctl --user list-units 'gef-lm-*' --all
quota -vs
# 提交并固定检出后，使用该检出的源码和匹配私有Python：
# bash scripts/qualify_lm_runner.sh /mi/data2T/zlong/home/llm/gef-lm-experiments/20261102
# tidegraph lm retry-matrix --round isolation --group all
```

当前15项尚未提交运行。下一次交接须替换为实际固定commit、unit、状态和精确检查/停止命令。
旧21项：6执行完成（4成本/2检查）、2llama超时、3依赖失败、10调度失败未提交，
详见[终态JSON](evidence/llm-restart-terminal-20261104.json)。外层完成不覆盖内层失败，
未做独立比较的成本点仍not_checked；旧数值/token差异不因本修复解除。
唯一待办见[ROADMAP](ROADMAP.md)，公共运行与复现入口见[external-lm](external-lm.md)。

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
