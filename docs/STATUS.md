# 当前交接 — 2026-10-28

用户授权的三项审查修复已实现，等待固定源码资格。分支`graph-execution-foundation`，
数学固定`0770593838a37c790a568226930567ed60f99b83`；实现基线`2c46aa7`。
本页随修复实现提交；当前工作路径`/mi/data2T/zlong/home/llm/graph-execution-foundation`。

## 修复与开发证据

- ClockedState递归捕获带权参数，按实际局部时间透传分支分组与prepared数据；五记忆独立根、参数版本、共享delta/KV三步SGD回归。
- Python/C++ scalar/batch quad显式拒绝数学正支持下溢；独立dyadic闭式解与可表示正数/真实零边界。
- 参数bank在reshape/转换/拷贝前取得并逐次扣减FP32额度，来源表/填充及Full输出预留纳入预算；原参数身份与角色存储分开。
- 原提交上新增Python测试19失败、4通过；首次baseline命令受pytest路径配置影响读到当前源码，已排除，修正命令/日志/哈希见`artifacts/review-fixes-baseline.json`。
- `core5-review-fixes-dev` passed/exit0，442项开发门禁；冻结快照`artifacts/worktrees/core5-review-fixes-dev`是2c46aa7加补丁，原始归档/哈希与命令在`artifacts/jobs/core5-review-fixes-dev/`。
- 主工作区C++核心哈希与开发构建相同。冻结开发快照没有随后补齐的resident Full输出预留；组件资格必须从本次实现提交新建干净快照构建。

## 活动任务与下一步

两个核心构建仍在运行，均读取上述冻结开发快照、Nice10、两个workers、ATen/BLAS1、2400秒限时：
- `tide-core5-review-fixes-sdk-dev-20261028.service`，`build/npu-core`，libtorch-npu/2.10.0-cann9.0.0。
- `tide-core5-review-fixes-cuda-dev-20261028.service`，`build/cuda-core`，账户activate-cuda工具链。

任务记录为`artifacts/jobs/core5-review-fixes-{sdk,cuda}-dev/{status.json,task.log}`；启动器为`.local/JOB.sh`。
检查：`python scripts/status.py`，`tail -n 30 artifacts/jobs/JOB/task.log`。
不要修改活动任务的快照，不将live任务计为通过。

实现提交后：新建`artifacts/worktrees/core5-review-fixes-qualify`于该提交并冻结；
CPU按相同442项门禁复用经源码/二进制哈希核验的核心构建。
NPU用新组件build、Ascend910_9392，限定flow/config/training/node-training，两dtype及必要双卡；
CUDA同组件sm80/90/100构建/loader与CPU适配器检查，GPU真机仍未验。
资格使用新的`core5-review-fixes-{cpu,npu,cuda}-qualify`任务，不恢复旧队列。
完成后单独提交证据与最终STATUS/ROADMAP，再推送授权分支。

NPU组件dev因SoC误写9391在依赖等待中取消；正确目标已核对9392。
NPU组件dev2及CUDA组件dev因补齐Full输出预算预留，在依赖等待中取消。
三项均exit143，未开始组件编译或设备运行，旧记录保留。

## 保留边界

参考仓库和公共环境只读。现行规范为`/opt/software/ascend-locks/dev-workspace-9.1.0/README.md`与`/etc/modulefiles`；
Python私有环境`.local/venv-core5`，CPU禁止vendor autoload。长任务使用独立background.slice及有限时长。
只做相关资格，不重跑历史性能矩阵。当前导航见[ROADMAP](ROADMAP.md)、[语义对应](semantic-map.md)、[证据索引](evidence/README.md)。
既有严格限制保留：FP16 GDN默认AdamW约.001984；默认异步双NPU eager VJP CANN invalid event；
SDK2.10 FP32 fc/affine第三轮evidence差3.09944e-6；FP32反馈PDG两项4.11272e-6/2.71201e-6。
CUDA真机未验、x86_64未资格，本次CPU/构建结果不解除这些限制。
