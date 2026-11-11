# 当前交接 — resident局部资格与原预算复验终态（2026-11-12）

已执行[ROADMAP P1—P6](ROADMAP.md#当前计划批量张量贯穿与架构验证已授权实施)中的本次resident有限阶段。
数学固定0770593838a37c790a568226930567ed60f99b83，19份快照再次通过离线核验。
当前实现e12aec0c；3e926ede交付单bank矩阵/块伴随，72c1dc02和d068fbdf修正新增容量见证，
c9775479交付实际公共step诊断。证据见[resident报告](evidence/resident-shared-bank-20261112.md)
及其机器记录；推送状态以git status为准。P1—P6总体未全部完成。

## 本阶段结果

- 同bank输入使用整块GEMM及参数转置，减少逐行矩阵复制和外积；组合预留与实际storage union分别检查。
  固定CPU32通过/6未指定设备skip，实际设备补验独立根。原两次容量见证map/scalar形状失败保留。
- e12aec0c让父服务递归停止直接子树。独立SDK六项全部通过：services、flow两dtype、window FP32、
  training两dtype；安装后10项公共raw根及KV-FC/SSD-PDG三步训练/磁盘恢复通过。
- 原预算SSD（B4/L16/d128，实际61位置）训练与精确下一窗通过，13.328988秒，4.57649位置/s；
  该规模独立数值未验，没有从两次接近的成本声称提速。
- 原预算FC16成功构造，但warmup仍EE1019/507002，未取得吞吐或恢复结论。
  旧单次trace定位反向program构造提交失败；新运行只到公共warmup边界，不套用旧stream/task编号。
  停止重复这个候选，不增加预算、不缩小输入、不静默启用blocking。
- 完成21文件hash绑定审阅及证据/复现/性能经验导航。与既有审阅有重叠，自动2157文件清单不等于全仓已审。

## 作业与现场

`gef-resident-stop-qualified.service`最终failed（仅代表FC16未通过，不覆盖内部各项独立通过）；
MainPID=0、ControlGroup为空，当前没有本任务活跃作业或遗留后代。
固定源码/构建/逐项结果：`artifacts/columns-20261111/resident-stop-qualified`；
作业状态与日志：`artifacts/jobs/resident-stop-qualified/{status.json,task.log}`。
总2700秒/640GiB；顺序jobs all（320）构建，设备host144–159、ATen2/interop1；
空闲物理11映射npu:0，普通异步，未使用共享卡。最终运行约761秒。
Python-owned与独立SDK分别构建；复用core保留原manifest且完整C++ hash匹配e12aec0c。
原shared-qualified/client/final与phase-trace均保留failed身份，stop-dev为开发快照通过。
磁盘配额约254GiB/软300GiB/硬800GiB，无存储阻塞；未清理参考仓库或公共环境。

## 接续边界与下一动作

本次共享矩阵/计费方案采用，小型服务停止修正通过，但原FC16规模限制未解除。
后继先从[预算与服务契约](resident-storage-accounting.md)、`cann_task_services.cpp`、
`weighted_flow_adjoint.cpp`与区域/节点伴随提交处设计有界服务提交或数值阶段合批；
在新独立嵌套/空窗/共享workspace/根与跨窗见证前，不重跑相同FC16或仅调高预算。
新resident完整B×L张量贯穿、普通异步双NPU、全仓未审范围按ROADMAP继续保留。
此前host张量窗口、区域/KV/五记忆、mixed、half及规模成本从[证据索引](evidence/README.md)按原身份进入；
GPU真机继续延后，共享NPU须当前明确授权。不得把原失败或待验范围改为完成。

```sh
python scripts/status.py
python scripts/check_upstream.py
python scripts/review_inventory.py --revision e12aec0c --reviews docs/evidence/resident-shared-bank-20261112-reviews.json --output artifacts/NEW_REVIEW.json
```

机器记录保存两个代表的完整configuration和展开局部绑定，干净checkout无需旧raw目录即可重建。
重现命令与两种运行时构建见本阶段报告；已有源码资格不需原样重跑。
用户未跟踪的`docs/tide-learning-lab-TL-REQ-004-handoff.md`与`TL-REQ-005-handoff.md`原样保留，不纳入本轮提交。
历史source定位用`python scripts/resolve_history_commit.py OLD_SHA`。
