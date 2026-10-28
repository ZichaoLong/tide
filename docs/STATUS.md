# 当前交接 — 2026-10-28

授权分支`graph-execution-foundation`，本次增量基提交1850434（实际HEAD由git记录）。数学固定
`0770593838a37c790a568226930567ed60f99b83`；19/5份新/旧快照核验通过。
参考仓库/公共环境只读，不使用子代理；实现、有限实验、提交和普通推送继续获授权。

## 当前增量

- native共享准备按首个事件排序；含Tensor地址的key只判断相等。
  原地址排序使checkpoint换owner后的梯度出现微小差异，现有双精度恢复反例已通过。
- CPU Python/native/standalone可显式FP64，保留同一FP32原始材料、别名与参数量，按生效dtype计费。
  不改默认FP32和容差。benchmark现在实际比较最后参数梯度和SGD槽；旧optimizer标志只比参数。
  resident本benchmark未导出这些字段，标志为false，原resident完整训练资格仍单独引用。
- 工作负载诊断每步核对optimizer，并在第一次更新后保存/换owner/精确解码，
  对比恢复后的下一窗口完整记录、梯度、参数与optimizer，容差均为零。

## 开发与下一动作

`core5-benchmark-precision-dev-20261028`：56 passed、2 native精确恢复失败，exit1，原终态保留。
`core5-stable-preparation-dev-20261028`：CPU核心/standalone重建，210 passed，exit0；
Python/native×FP32/FP64换owner恢复通过。冻结树`artifacts/worktrees/core5-stable-preparation-dev`，
报告/日志`artifacts/jobs/core5-stable-preparation-dev/{development.json,task.log,status.json}`。

新增C++反例任务首次因launcher预建输出目录被job.py拒绝，workload未启动；原failed/1保留。
`core5-stable-order-regression-dev-v2-20261028`已终态passed/0：增量构建，6项通过。
新增C++反例与双dtype/Python/native精确恢复已验证。原冻结输入/日志分别保留，
实现与契约已准备提交；固定资格将采用本次实现提交，不将dirty开发基提交当资格来源。

实现提交后建立固定资格：CPU相关门禁与D64 FP64两步checkpoint；
NPU eager KV训练/恢复和CUDA构建/CPU核验分别记录，不重跑未改动的resident核矩阵。
全部重型构建终态后运行四个D64 FP64接口/调度训练对照；均通过才执行一次D2048 FP64推理。
同一原始二进制不变，仅新manifest把其64GiB静态预算显式提高到128GiB以容纳约90GB FP64静态材料；
进程192GiB/1800s，原FP32大图不重跑。driver`.local/core5-explicit-precision-study.py`尚未启动。
固定任务入口`.local/core5-stable-preparation-qualify.sh`及`core5-stable-preparation-{npu,cuda}.sh`；
输出`artifacts/jobs/core5-stable-preparation-{qualify,npu,cuda}`，background.slice/Nice10，64G/20min。
CPU固定树将为`artifacts/worktrees/core5-stable-preparation-qualified`；设备固定树复用已终态且干净的
`artifacts/worktrees/core5-private-qualified`，顺序切换到本次提交后只读。此前build身份保留在新报告previous_build。

## 保留问题与证据

既有数值/低秩资格、原失败见[evidence/core5-local-allocation-factorized-kv-20261028.md](evidence/core5-local-allocation-factorized-kv-20261028.md)。
D64 FP32完整训练、D2048完整keys仍严格失败；显式FP64通过不能替代。
本轮动态算术探针未采用；row探针曾误覆盖首个arithmetic-trace输出，原stdout/partial恢复及artifact-issue保留，
不把它当完整资格。后续正式证据需记录这些诊断边界。
FP16 GDN AdamW、SDK affine、反馈PDG、默认异步双NPU见证继续保留；CUDA真机、x86_64、
大型设备训练/跨卡吞吐未验。私有合批覆盖及resident窗口反向仍有优化空间。
唯一待办见[ROADMAP](ROADMAP.md)。重入：`git status --short --branch`、`python scripts/status.py`、
`python scripts/check_upstream.py`；任务结果须同时核对unit、status和验收报告。
