# 当前交接 — 2026-10-28

三项审查修复及本机相关资格已完成。授权分支`graph-execution-foundation`，工作路径
`/mi/data2T/zlong/home/llm/graph-execution-foundation`；固定实现
`1d19c5998b96689b3620ecb69c7f712a9dd84ad0`已推送。本页与资格证据单独提交。
数学仍为`0770593838a37c790a568226930567ed60f99b83`，19份快照及5份legacy已与上游Git对象核对。
[审查修复报告](evidence/core5-review-fixes-20261028.md)提供范围、独立反例和干净源码复现命令；
[同名JSON](evidence/core5-review-fixes-20261028.json)保存哈希、源码/构建身份及全部任务终态。

## 完成范围

- ClockedState递归捕获带权参数，按实际局部时间透传分组/prepared；五记忆独立根、版本及共享KV/delta三步SGD通过。
- Python/C++ scalar/batch quad拒绝数学正支持的除法下溢；独立闭式反例与可表示正数/真实零边界通过。
- resident参数bank在展开/转换/拷贝前准入，逐角色扣减FP32额度，来源表/填充及Full输出预留；CPU与NPU真实边界通过。
- CPU正式442项全部通过；CUDA工具链CPU定向31项及sm80/90/100四组件构建/loader/help通过，无GPU真机资格。
- standalone NPU FP32/FP16各21图、15配置、单设备与双卡各五配置20次更新通过；含完整记录、反向、共享参数、optimizer、恢复及拒绝。

核心复用冻结开发快照，完整C++哈希与1d19c59一致；resident组件含最终Full预留，
从干净1d19c59重新构建。详细来源与复用范围在报告中分列。
首次NPU任务构建通过，但CANN因任务PATH中的Python入口不匹配而初始化失败，exit1保留；
修正后的独立runtime任务复用相同二进制完成8组。公共工具链未修改，任务入口规范已补入[开发文档](development.md)。
三项早期等待依赖时取消的任务仍为exit143；旧失败不改称通过。

## 任务终态与接续

无活动任务；本轮修复无需继续运行或恢复旧队列。
记录位于`artifacts/jobs/core5-review-fixes-NAME/{status.json,task.log}`：
`cpu-qualify`、`cuda-qualify`、`npu-runtime-qualify`均passed/exit0，`npu-qualify`仍failed/exit1。
对应unit为`tide-core5-review-fixes-NAME-20261028.service`；NPU另有`queue-state.json`和`checks/result.json`。
新NPU任务物理1/13→逻辑0/1、3300秒任务/3600秒unit，独立background.slice；最终状态、全部用例及哈希已核验。

下一次接续先运行`git status --short --branch`、`python scripts/status.py`、`python scripts/check_upstream.py`，
再按新需求与[ROADMAP](ROADMAP.md)选择范围。仓库导航与复现不依赖旧运行目录。
本次最终改动仅为证据、当前交接与文档链接；实现仍为上述固定提交。

## 保留边界

参考仓库与公共环境只读；本机工具链入口为现行`/etc/modulefiles`与
`/opt/software/ascend-locks/dev-workspace-9.1.0/README.md`，不同Torch/CANN环境分别构建。
CUDA真机和x86_64未资格；本轮未新增性能测量或Python-owned NPU资格。
既有GDN FP16默认AdamW约.001984、默认异步双NPU eager VJP invalid event、
SDK FP32 affine第三轮evidence差3.09944e-6、FP32反馈PDG两项4.11272e-6/2.71201e-6继续保留。
条件和独立证据见[ROADMAP](ROADMAP.md)、[语义对应](semantic-map.md)、[证据索引](evidence/README.md)。
