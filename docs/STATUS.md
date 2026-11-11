# 当前交接 — FP16张量窗口交付与resident迁移（2026-11-11）

继续已授权[ROADMAP P1—P6](ROADMAP.md#当前计划批量张量贯穿与架构验证已授权实施)。
数学固定0770593838a37c790a568226930567ed60f99b83，19快照再次通过。
当前实现ce50a2a5；阶段提交5816a4c4（half列）、5959bbc3（默认实验/loss）、ce50a2a5（SDK入口），
推送状态以git status为准。用户未跟踪TL-REQ-004/005交接文件原样保留，不提交。

## 本阶段完成与证据

[FP16与跨owner后继](evidence/column-half-20261111.md)分别固定源码、构建、运行和失败身份。
五记忆与位置KV新列接受half载荷/显式FP32评分，Attention内部FP32、原始版本守卫及int64时钟保持。
默认实验auto按实际窗口能力分派，half任务loss在乘法前提升FP32。
CPU固定188 passed/6原数值xfail；扩大回归112 passed/2原xfail/2旧sum FP32严格失败，
后两项在修改前796baafa同材料复现同值，原阈值和失败保持。
CPU/NPU共八安装消费者独立三步master AdamW及新进程精确恢复通过。
单NPU实际12项half五记忆/位置全根通过；同pytest3项CPU辅助单列。

双NPU普通异步反向仍报EE1017/invalid event：首例前向通过，8独立安装进程和SDK首例亦失败。
显式ASCEND_LAUNCH_BLOCKING=1的4全根/更新/raw恢复、2安装消费者与SDK FP32十五项通过，
不作为普通异步修复。ce50a2a5补SDK half入口标志后，新SDK构建及CPU、单NPU普通异步、
双NPU显式同步各16项half通过。没有新增half吞吐结论。
此前FC整链成本、mixed和多owner报告从[证据索引](evidence/README.md)进入，保持原实现身份。

## 作业与现场

当前无活跃项目作业。gef-column-half-sdk已passed；原gef-column-half-owners-npu、
gef-column-half-owners-blocking、gef-column-half-task-qualify仍为failed，内层通过项单独归档。
状态/日志在artifacts/jobs/；python scripts/status.py查看当前，--all-jobs审计历史。
实际NPU为合法空闲准入物理7/11，逻辑npu:0/1，host144–159；未使用共享卡。
共享NPU仍需当前明确授权，GPU延后。无磁盘阻塞，不清理参考仓库或公共环境。

固定源码/构建在artifacts/columns-20261111：qualified-half(5816)、qualified-half-task(5959)、
qualified-half-sdk(ce50)，half-dev1/build为Python-owned；half-sdk-build为独立SDK。
原half-dev1构建复用已核验完整源码/binary hash；保留其原开发manifest，不回写身份。
安装install-half-task，环境half-task-runtime.sh；项目私有env artifacts/additive-work/env，
Torch2.12/TorchNPU2.12.post3/CANN9.2。SDK使用libtorch-npu/2.12.0-cann9.2.0，不混wheel库。

## 下一动作

本阶段证据、ROADMAP/经验/复现导航整理后提交并正常推送当前授权分支，再继续P5/P6：
审阅resident的设备在线推进、局部工作区与反向journal，确定新张量列能接入的真实边界和独立反例。
不把host整窗提交冒称resident，也不再盲目加旧容量预算。普通异步跨卡按ROADMAP的有界停止条件保留。
已审具名清单与未审范围分别记录；当前P1—P6未全部完成。

```sh
python scripts/status.py
python scripts/check_upstream.py
TIDE_COLUMN_DEVICE=npu:0 python -m pytest -q tests/test_half_columns.py --dtype float32
TIDE_COLUMN_PEERS=npu:0,npu:1 python -m pytest -q tests/test_column_payload_owners.py --dtype float32
TIDE_COLUMN_PEER_DEVICE=npu:1 tidegraph-column-check --device=npu:0 --dtype=float16
```

设备命令须先依法准入且在对应冻结源码/环境执行。普通异步与显式launch blocking用独立输出目录；
详情及CPU/安装复现见本阶段报告，不依赖旧raw目录才能重建。
长任务固定源码、background.slice、有限资源/时长；构建可用全部320 CPU，计时避开项目自身争用。
历史source定位：python scripts/resolve_history_commit.py OLD_SHA。
