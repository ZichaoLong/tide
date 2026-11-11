# 当前交接 — 多owner张量窗口与后继迁移（2026-11-11）

继续已授权[ROADMAP P1—P6](ROADMAP.md#当前计划批量张量贯穿与架构验证已授权实施)。
数学固定0770593838a37c790a568226930567ed60f99b83，19快照通过。
生产实现796baafa，SDK peer入口后继efc2776b；推送状态以git status为准。
用户未跟踪的TL-REQ-004/005交接文件原样保留，不提交。

## 当前完成范围

- 五记忆/卷积、区域分配、消息及raw续接已接入Python/C++张量窗口；公共任务直接接收B×L，
  相关资格与代表成本从[证据索引](evidence/README.md)进入。
- [混合放置与共享伴随](evidence/column-placement-20261111.md)已验CPU、NPU安装和独立SDK。
  Python KV mixed-a原精确恢复失败由有序共享伴随修复；原失败保留。
- [多payload owner](evidence/column-owners-20261111.md)已实现：完整区域汇集、实际消息整列copy复用、
  目标owner逐边返回、跨输出owner投影/loss。保留参数共享共置约束，普通copy不宣称resident/P2P。
  固定CPU243 passed/13原数值xfail，SDK CPU通过；不同报告范围重叠，不相加。
- 新默认FC16/B8/L128/FP32完整训练单次CPU1：PyTorch1.234671秒、native1.225463秒，
  829.37/835.60输入位置每秒；两者16384候选/8344 Full/25060边消息。
  不新增完整尺度独立数值、恢复或TTFT资格。

## 作业与设备边界

当前没有活跃作业。column-owners-npu等待120秒、column-owners-host-npu等待60秒后均准入超时exit3，
没有进入模型计算；不把它们描述为数值/通信失败。原失败状态保持。
SDK CLI固定资格column-owners-cli-qualify已passed；各状态/日志在artifacts/jobs/。
用python scripts/status.py查看当前，--all-jobs审计历史。共享NPU仍须当前明确授权；GPU延后。

## 下一动作及复现

先完成本阶段证据提交/推送，再审查FP16新列迁移的真实算术边界；独立推进不需NPU的实现和CPU检查。
设备状态变化时，使用use-local-ascend空闲卡准入新建有界作业，不恢复旧队列。
待设备验证：双owner四项独立根/AdamW/恢复、八安装消费者、SDK五记忆×三放置、
新默认单卡回归及普通异步反向完成。通过后才选择有限跨卡成本。

```sh
TIDE_COLUMN_PEERS=npu:0,npu:1 python -m pytest -q tests/test_column_payload_owners.py --dtype float32
python examples/consumer_core5/tensor_window.py --device npu:0 --peer-device npu:1 --dtype float32 --client python --placement mixed-a --no-deterministic-algorithms --output-dir NEW
TIDE_COLUMN_PEER_DEVICE=npu:1 tidegraph-column-check --device=npu:0 --dtype=float32
```

现存固定源码/构建位于artifacts/columns-20261111：qualified-owners(796baafa)、
qualified-owners-cli(efc2776b)、owners-dev1/build及owners-cli-build，源码/二进制hash核验已归档。
安装install-owners、环境owners-runtime.sh，私有env artifacts/additive-work/env；Torch2.12/CANN9.2。
SDK peer后继必须用owners-cli-build；旧owners-sdk.sh未更新，保留原作业身份，新作业另写launcher。
恢复与测试均在匹配环境/冻结源码下执行，PYTHONPATH保留CANN环境路径。
正式计时避开项目重构建，实际CPU affinity用taskset并记录；构建可用全部320 CPU。
当前无磁盘阻塞，参考仓库和公共工具链不清理。

resident新列迁移、旧异步/容量边界、FP16和广泛源码审阅仍未完成，唯一待办见ROADMAP。
历史source定位：python scripts/resolve_history_commit.py OLD_SHA；正式复现不依赖旧raw目录。
