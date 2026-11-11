# 当前交接 — 混合张量窗口资格与共享伴随（2026-11-11）

继续已授权[ROADMAP P1—P6](ROADMAP.md#当前计划批量张量贯穿与架构验证已授权实施)。
数学固定0770593838a37c790a568226930567ed60f99b83，19快照通过。
实现540ccbd8、后继2ca00458已提交，固定证据见下；推送状态以git status为准。
用户两个未跟踪TL-REQ-004/005原样保留。

## 已完成

- b993f38公共张量任务与宽/多节点/1B/9B有限成本见[公共入口证据](evidence/experiment-columns-20261111.md)。
  原NPU卷积精确恢复问题已由8c72ca94的[有序tap转置](evidence/column-tap-transpose-20261111.md)修复。
- 540ccbd8实现Python/C++ mixed-a/b/c Read/control/selection列：CPU213通过/4原数值xfail，
  NPU安装11项中10项独立训练/恢复通过；SDK五记忆×三放置15项独立调度/raw/VJP/两SGD窗通过。
- Python KV mixed-a原精确恢复失败保留。2ca00458给共享Aggregate稳定使用槽与有序伴随，
  保留前向共享、全局并行及deterministic=false；CPU239通过/13原数值xfail，同图六次VJP逐位相等。
  原mixed-a两个独立流程，加KV mixed-b/c和SSD mixed-a共5个安装三步/新进程恢复通过。
- [本轮固定证据](evidence/column-placement-20261111.md)记录9项native mixed成本：全部completed及精确next通过，
  规模独立数值未测。PyTorch主FC16 CPU1/NPU2完整训练1.244159/0.736821秒，823.05/1389.75位置每秒；
  Full/消息不同，本轮未重验全尺度数值/TTFT，不把成本点称严格等价吞吐。

## 下一动作

继续多payload设备的张量边界：按节点保留raw状态与Read位置，按区域参数owner汇集完整描述后分配，
按目标owner复制整列物理消息并保留重复边和共享根。公共readout/loss需要跨输出owner整列归约。
具名入口：weighted/{window_execution,settle_columns,message_columns,column_placement}.py，
experiment/column_task.py及cpp/src/weighted_settle_columns.cpp。现有所有者约束与参数共享不能静默改写。
先做独立小图/三步/精确恢复，再有限跨卡成本；不能仅删除单设备检查。多owner列实现开发完成：94 passed/9原数值xfail，两套新构建与SDK CPU通过。
下一步提交实现，固定源码资格/安装，再两张空闲NPU独立调度、训练/新进程恢复及SDK五记忆。
其后resident新列、旧异步/资源/容量边界与广泛源码审阅按ROADMAP有限推进；GPU仍延后。

## 现场、作业和复现

作业gef-column-owners-dev1已passed，dirty快照owners-dev1/source与两套build保持不动；
日志artifacts/jobs/column-owners-dev1。本轮作业均已终态；python scripts/status.py。原column-placement-npu保持failed，后继成功单列。
固定worktree：artifacts/columns-20261111/qualified-placement(540ccbd8)、qualified-shared(2ca00458)。
两套placement-dev2/{build,build-sdk}源码/binary hash已核验，原开发manifest不改写。
安装install-shared，环境shared-runtime.sh，私有env artifacts/additive-work/env；CANN9.2/Torch2.12。
同图诊断：python scripts/check_column_adjoint_order.py --device npu:0 --placement mixed-a --include-columns --output-dir NEW。
训练恢复：examples/consumer_core5/tensor_window.py --device npu:0 --dtype float32 --client python --placement mixed-a --no-deterministic-algorithms --output-dir NEW。
成本配置/归档placement-cost-config.json、placement-cost-results.json，源码和日志在artifacts/columns-20261111及artifacts/jobs。
NPU物理11，诊断另用空闲7；正式NPU进程实际affinity144–159，CPU80–95，ATen2/1。
共享NPU仍须本次明确授权；正式计时避开重构建。当前无磁盘阻塞，不清理参考仓库或公共工具链。

历史source定位：python scripts/resolve_history_commit.py OLD_SHA。正式成果从仓库证据与生成器复现，无需旧raw目录。
