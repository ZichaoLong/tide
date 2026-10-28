# 当前交接 — 2026-10-29

授权分支`graph-execution-foundation`；数学固定`0770593838a37c790a568226930567ed60f99b83`。
参考仓库/公共环境只读；不使用子代理。实现、有限验证、阶段提交和普通推送已获授权。

## 本轮完成范围与结论

实现`d8f71327f2c1515d78c2ccf3e355fc427673708f`新增仓库算术边界诊断、精确事件差异、
命名梯度/参数误差统计与局部干预下的真实checkpoint恢复；**没有改变生产算术或默认精度**。
[固定报告](evidence/core5-arithmetic-isolation-20261029.md)及同名JSON记录完整结果；
[诊断契约/复现入口](weighted-arithmetic-diagnostics.md)无需旧运行目录。

- 固定CPU设施检查**19 passed、0 skipped**，含独立相消/literal Jacobian、int64、None/零、版本/拒绝/清理，
  小图探针恢复及原Python/native两dtype训练/恢复。开发dirty冻结快照也19通过。
- e75e1f5上的9种有限探针均未通过原D64两步完整训练。只提升矩阵乘加使首步坏梯度张量2393→2；
  三组乘加/归约/RMS内部FP64首步全通过，第二步仍失败。仅反向提升几乎无效，未认定唯一瓶颈。
- d8f7132用重新生成的同hash原材料复现最佳局部方案：首步479条事件逐值相同，4186个梯度和1383个
  更新参数张量有容差内差异；第二步keys首见1.3411045e-6、2个梯度张量与动量槽超容差。
  参数更新仍在容差内，不能替代完整训练验收。两条调度各自换owner后下一窗完整记录/梯度/参数/槽精确恢复。
- 不采用局部探针作为生产精度策略。已有显式整条CPU FP64完整训练/恢复和有限计时仍有效，
  见[1f1ec57证据](evidence/core5-stable-preparation-precision-20261028.md)；原FP32失败未解除。

## 源码、作业与终态

原计划hash`6c8cc36ae97287133d40e67ccc8e137574b80e2df420feff6adfef3395397bb3`，
480节点/2208边/D64/B1/每窗1位置、9,076,975主体参数；原SGD及容差保持。
固定资格只读树`artifacts/worktrees/core5-arithmetic-qualified`位于d8f7132。
C++未变，复用前后完整核对C++ hash `8c942fa686ed276204b3bd5a04a026513a3ca21d0a480506c2ae3ba32211d7b2`
和23个binary hash；旧缓存manifest的1850434来源不改，不冒充本次资格来源。

| unit（均后缀`-20261029`） | 持久目录 | 终态 |
| --- | --- | --- |
| core5-d64-precision-stage1 | artifacts/jobs/core5-d64-precision-stage1 | failed/2，7种原严格失败 |
| core5-d64-precision-stage2 | artifacts/jobs/core5-d64-precision-stage2 | failed/2，2种追加严格失败 |
| core5-arithmetic-dev | artifacts/jobs/core5-arithmetic-dev | passed/0，19项设施检查 |
| core5-arithmetic-qualified | artifacts/jobs/core5-arithmetic-qualified | failed/2；设施19项通过，D64子作业strict_mismatch/2，精确恢复通过 |

**无活动作业**。各目录保留status.json/task.log；探索另有study.json及各variant/result.json；
固定资格另有qualification.json/tests.xml/live-unit.json/plan/workload.json，
D64子作业在d64-local64/{status.json,task.log,check/result.json}，实际checkpoint在check/。
background.slice/Nice10，算子1；stage1 32G/25min，stage2/固定资格32G/10min，开发16G/10min；
探索子进程180s、固定D64子任务240s。实时PID/Transient/slice已核对，终态以持久记录为准。

## 仍然有效的前阶段资格与限制

ad6d1e7的FP32 GDN/SSD normalized VJP及原AdamW/恢复修复已完成，
[固定资格](evidence/core5-fp32-normalization-20261028.md)为CPU453 passed/2原FP16 strict xfail，
SDK NPU四组、纯Python NPU两组训练/恢复和CUDA构建通过。本轮不重复无关设备资格。
1f1ec57的D64 FP64四种接口/调度训练、D2048匹配92.62亿主体参数的CPU FP64推理已通过，
后者151.416s/窗、测量后RSS峰值约90.35GiB；同参数不表示同core-3完整训练工作量。

尚未解除：D64/D2048 KV原FP32、FP16 GDN默认AdamW、SDK FP32 affine、FP32反馈PDG、
默认异步双NPU VJP；CUDA真机/x86_64、大型设备训练/跨卡吞吐待验。没有新增吞吐收益声明。
跨不同输入私有合批、设备小算子提交、resident组合矩阵/窗口反向仍有空间，全部待办见[ROADMAP](ROADMAP.md)。
历史LH/core-3及本轮正负经验在[performance-lessons](performance-lessons.md)，不另维护任务日记。

## 重入与下一步

本轮有限诊断、设施实现与固定验收均终态，不恢复旧队列或重复9种探针。
实现与本次证据/追溯文档分别提交；提交和普通推送的实际对齐以`git status --short --branch`核对。
重入先运行`git status --short --branch`、`python scripts/status.py`、`python scripts/check_upstream.py`。
后续数值工作需针对反向中间伴随量及共享贡献在汇总前后的舍入设计具体方案，
先独立公式/根与完整恢复，再原D64两步及有限完整计时；不能从当前证据宣称仅改最终leaf累加就能修复。
新工作按ROADMAP选择具体边界，保持原失败、显式精度与设备资格分列。
