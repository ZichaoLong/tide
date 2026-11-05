# 当前交接 — LM选择性后继全部终态（2026-11-05）

截至2026-11-05 07:51 CST，本轮15项后继全部终态：13项推理成本测量完成，
1项独立小检查执行完成但数值/生成token失败，1项双NPU B512在模型分配前容量拒绝。
没有中断、待下载、排队或运行的本批作业；最后一个服务于11月5日05:43 CST结束。
9个外层service均inactive、MainPID=0、ControlGroup为空，24份job/case记录无遗留后代，
48个PID/start身份实查无匹配存活进程。外层passed不等于每项严格通过。

实现固定`694eb605b6769f8d3f208710a920dbfbf504abf4`，检出`source/isolation-h`仍干净。
固定设施125项通过、小型旧/新llama输出/状态及恢复回归保留原范围；本次没有重做这些检查。
数学`0770593838a37c790a568226930567ed60f99b83`本次校验通过。
[本轮15项终态](evidence/llm-isolation-terminal-20261105.md)/[JSON](evidence/llm-isolation-terminal-20261105.json)
保存逐项配置、原结果hash、耗时、真实工作量、物理放置和失败；[启动证据](evidence/llm-isolation-launch-20261104.md)不改写。

## 当前新增：供用户自行启动的B512试跑入口

用户希望直接尝试原双NPU B512并索取运行命令。本次补充显式
`execution.device_memory_admission=observe`及`lm queue --config`，默认strict和旧配置身份保持。
公开配置为`examples/external_lm/qwenmoe-two-npu-b512-observe.json`；7200秒计算上限、外层9小时，
保持原模型、BF16、B512/L512/O128、prefill64及16线程。超限预算先记录，OOM/其他异常仍失败。
当前仅准备和检查入口，没有替用户提交该模型作业；固定源码/设施资格接续记录将在提交后补全。
本次开发相关37项通过，尚不表示该B512已实际分配或执行。具体人类命令见[external-lm](external-lm.md#显式容量试跑与单配置后台运行)。

## 终态、结论与下一步

| 范围 | 已核对事实 |
| --- | --- |
| Qwen7B CPU/NPU B8 | 两项成本完成；输出吞吐5.798/274.072 token/s |
| Qwen MoE CPU B1/B8 | 两项成本完成；0.544/2.778 token/s |
| Qwen MoE双NPU B1/B8/B32/B128，L512 | 四项成本完成；2.059/9.704/23.912/53.138 token/s |
| Qwen MoE双NPU B8/L2048 | 成本完成；6.274 token/s |
| Qwen MoE四NPU B32/B128 | 两项成本完成；21.312/43.458 token/s，实际跨层显式复制 |
| llama CPU Qwen1.5B/OLMoE B32 | 两项成本完成；3.074/3.928 token/s，原1200秒超时另存 |
| Qwen MoE四NPU B2/L16/O4检查 | 100组logits/KV比较均超原容差，生成8个token中2个不同 |
| Qwen MoE双NPU B512 | 静态估算56.719GiB，空闲61.137GiB减6.4GiB余量后54.737GiB，分配前拒绝 |

成本点均为16线程、输入512（另标2048）/输出128，吞吐分母含prefill+decode，构造/warmup另计。
13项成本的numeric/discrete/checkpoint均not_checked；小检查没有选择checkpoint。
本轮没有新增完整恢复资格，旧精确恢复不能替代独立数值或当前规模检查。
用户要求有中断则修复；实查无中断，无需重启。容量拒绝和严格比较失败保留，不以相同配置重跑解决。
后续数值修复先定位首差；容量后继先校准KV/激活/workspace及reserved，再声明新配置。
未追加实验、修改固定源码/环境或清理材料；具体停止/重启条件只见[ROADMAP](ROADMAP.md)。

## 接续位置与只读命令

外部根`lm_root=/mi/data2T/zlong/home/llm/gef-lm-experiments/20261102`。
计划`recovery/plans-20261104-h/{all,compute,llama,placement,budgets,launch}.json`；
`lm retry-matrix --round isolation`可从干净仓库重建15项配置，不依赖旧目录。
终态job为`jobs/{qualify,assets-moe,llama,dispatch,placement}-h`及
`jobs/{cpu,npu}-{small,moe}-20261104-h`；各目录保留status.json/task.log/supervisor.log。
case结果位于`runs/*20261104-h/*/run/`，每项进程记录在对应`process/status.json`。

```sh
git status --short --branch
python -B scripts/status.py
lm_root=/mi/data2T/zlong/home/llm/gef-lm-experiments/20261102
systemctl --user list-units 'gef-lm-*' --state=running --no-pager
cat "$lm_root/recovery/20261104-h/dispatch.json"
cat "$lm_root/recovery/placement-20261104-h/status.json"
# 先按docs/external-lm.md激活现有公共软件基线与项目私有环境，再仅查询：
tidegraph lm results "$lm_root/runs" --model qwenmoe --format csv
quota -vs
```

## 环境、资源与保留

Qwen MoE16/16片及全部配套资产已于04:46:51完成发布hash校验，共61,084,262,963 bytes；
本次复查记录和size/mtime一致，未重复大文件hash扫描。两次临时读超时已自动恢复，旧失败不改写。
固定检出保持clean，NPU厂商相对输出实际进入各case的process/work，后继延迟提交成功。
旧`source/restart-g/fusion_result.json`与旧llama binary/shared libraries/GGUF仍保留；
checkpoint恢复固定原build身份，不随ready指针切换。

账户quota约176GiB、实验根du约122.54GiB；项目160/账户300GiB停止新增大写入，800GiB硬限不变。
无需扩盘/tmpfs，不连接已释放gpu3090。公共Torch2.12/TorchNPU2.12/CANN9.1只读；
私有Transformers4.57.1/Accelerate1.11.0/PEFT0.17.1，vLLM/TP/EP未由跨层放置认证。
新llama binary SHA256 `87377b5274eb8f93e2dc949dcfdb8ce6819afd6b13bf704a22bedb3db9922027`，
上游`2cbca389f515df455412dafa07e2b1359cc72a6d`；旧源码/构建资格不升级为当前复验。
原始材料继续保留，本次未进行大规模清理或重跑。公共入口见[external-lm](external-lm.md)。

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
