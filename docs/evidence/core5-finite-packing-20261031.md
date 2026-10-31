# 三平台有限值诊断packing试验（2026-10-31）

固定实现`bd2ff2ed6786c57f492b6ac8b83b4b3a4b0d821e`，安装内容hash
`afc35286eae8dca7744134dcc966cde82ddf5b30e07720cd0ef491ee30783ff8`；C++核心未修改、未重建，原实际构建来源保留在归档。
模型数学、dtype、输入、局部packing、完整返回、阈值和调度均不变。该试验只合并运行时有限值诊断输入。
三平台共18条终态：15正式流程、3独立profile；6个小训练严格通过，其余12项保留原数值超差并探索接纳。
18项执行、离散和精确续接通过；不解除任何历史严格失败。

## 实现与验证边界

[finite.py](../../python/tidegraph/experiment/finite.py)按device/dtype合并兼容连续张量，每组至多8MiB；
大/非连续/共轭/特殊视图保持原逐张量检查。只读detach，不缓存owner、journal或autograd图。
NaN/Inf仍明确失败，None/零梯度与所有连接关系不变；独立参考始终用scalar。
每卡和host另预留32MiB工作区。默认scalar身份不变，显式packed有新execution_id；
standalone和resident拒绝该覆盖项，继续使用自己的原检查。实现适用Python eager两客户端，
本报告实际性能客户端均为Python/native；CPU纯PyTorch共享参数/三窗/恢复另有定向测试。

旧实现已经按device只取回一个bool，本修改的目标是减少小isfinite/all kernel，不是减少逐tensor主机回传。
固定源码78项相关检查通过，覆盖NaN/Inf后续chunk、复杂/非连续tensor、独立梯度与alias、共享参数训练、完整状态/optimizer/磁盘恢复、配置/CLI/job/观察。
源码、日志摘要和对应CPU开发范围见[首组证据](core5-platform-baseline-20261031.md)。

## 同材料成本

B128/P2的FC4/d128/KV，1,134,054可训练参数，FP32 payload/scoring；CPU ATen4，NPU/CUDA ATen1，interop1/worker1。
局部packed、返回scalar、完整诊断返回；CPU placement=cpu，NPU/CUDA=mixed-c，GPU IEEE/TF32关闭。
两策略同模型/任务/时间窗，warmup1+连续2窗，各一次独立进程。下表推理仍为数值超差成本，不称严格等价加速。

| 平台 | B128 scalar诊断秒 | B128 packed诊断秒 | scalar/packed成本比 | B512 packed秒 | 小训练scalar/packed秒 |
| --- | ---: | ---: | ---: | ---: | ---: |
| cpu | 1.018292 | 0.853316 | 1.193 | 3.589865 | 0.093865/0.089891 |
| npu | 4.043991 | 3.374952 | 1.198 | 16.680769 | 0.338689/0.325406 |
| cuda | 2.317225 | 2.066858 | 1.121 | 8.011378 | 0.285838/0.291717 |

本轮B128三平台均观察到降低成本，小训练变化很小、GPU略反增。保留显式选择，不改默认、不外推其他dtype/大参数/多卡或resident。
B512沿用[先前基线](core5-platform-baseline-20261031.md)作背景；它没有在bd2ff2e重复scalar，不能冒称同源码交错统计或稳定收益。
构造、warmup、checkpoint出版、独立参考与恢复成本均在归档单列；profile成本不计入正式吞吐。

三平台B512第一计时窗均有下列实际独立样例差异；Full的时间行不能当独立B：

| 节点 | candidate事件 | candidate独立样例 | Full时间行 | active独立样例 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 640 | 512 | 319 | 298 |
| 1 | 512 | 384 | 473 | 358 |
| 2 | 768 | 512 | 646 | 450 |
| 3 | 768 | 512 | 265 | 218 |

[既有产物精确比较](core5-finite-packing-payloads-20261031.json)逐平台比较B128及小训练两策略：完整checkpoint payload（alias/参数/槽/state/history/pending/ledger）、task state及3窗输出在atol=rtol=0下相同；各对state.pt/task.pt文件hash也相同。
这不代替独立参考或完整梯度测试；后两者由各正式运行的独立训练检查和定向测试分别建立。

## Profile与六项问题的新增依据

三个profile都真实执行硬件；未观察到意外host fallback。NPU包含73个AICPU硬件任务，不能把它们写成AI Core任务或隐藏不报。
CUDA与原独立B128 profile具有相同model/task/workload和逐项相同packing counters；仅诊断策略改变。

| CUDA观测 | 原scalar诊断（1b3e39f） | packed诊断（bd2ff2e） |
| --- | ---: | ---: |
| kernel事件 | 147,380 | 111,268 |
| kernel union ms | 245.382 | 189.002 |
| isfinite inclusive ms | 958.840 | 469.843 |
| all inclusive ms | 331.609 | 185.836 |
| 完整窗range union ms | 4,900.623 | 4,820.113 |
| 显式同步API / union ms | 39,868 / 107.931 | 39,869 / 109.198 |

新profile显式同步占捕获跨度约2.265%，原约2.202%；没有同步调用减少的结论。
finite-check自身range union约259.806ms，其中显式同步仅1次；exclusive关键路径不能由inclusive相加推导。
新CUDA的index/nonzero/stack/cat inclusive仍约982.892/748.660/151.748/160.339ms，值得后续按实际因果定位；本轮不继续泛化搬运层。
NPU新profile98,433个kernel，union299.980ms；完整窗4326.763ms，finite-check324.610ms。
CPU profile保留按输入shape分组的算子表；其导出/观察开销很大，不拿profile总秒数比较吞吐。

三平台本材料packing counters相同：private BMM904次/2112行（864次仅2行），同权重weighted_mm36次/2915行。
这里weighted_mm是指定投影计数，不是全部ATen MM调用。实际shape/调用表见archive；大逻辑B并未消除小私有BMM。
CUDA原bmm inclusive18.719ms、新22.990ms，NPU13.236ms，相对几秒完整窗并非当前首要热点；因此不据2行标签重写跨节点调度。
KV可见Read12桶/672行、Full12桶/427行、最小Full桶6行，无单行桶；没有该材料上长度padding改造的依据。
反向模板和resident布局的决定仍需后续训练/设备证据，不能用本推理诊断改动替代。

## 清洁复现与查询

[三平台有限配置](../platform-matrix.md)的`*-finite.json`包含所有请求值；先从固定源码安装、核对真实库/site。

```sh
tidegraph plan configs/matrices/cpu-finite.json --matrix --site site.json
tidegraph matrix configs/matrices/cpu-finite.json --site site.json --output-dir jobs/finite-new --detach
tidegraph results docs/evidence/core5-finite-packing-20261031.json.gz --platform npu --format table
tidegraph results docs/evidence/core5-finite-packing-20261031.json.gz --format html --output finite.html
```

[完整v2归档](core5-finite-packing-20261031.json.gz)保留18项来源、配置、逐窗、首差、峰值与profile摘要。
CPU/NPU两个外层服务及GPU包含本组的9项任务均终态passed、无后代；这不覆盖内部数值严格失败。

归档SHA256：`b7ab91cc09730a02f2173983311f51bd18e1c753270c43eb94645f8cab5eb796`。
