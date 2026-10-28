# 当前交接 — 2026-10-29

授权分支`graph-execution-foundation`；数学固定0770593838a37c790a568226930567ed60f99b83。
用户批准继续本机非GPU工作，GPU限制延后。参考仓库/公共环境只读；不使用子代理。
阶段提交和普通推送已授权。当前基线ead6f347fa91d8d20b5ad3e5ed742bccfe8936e2。

## 当前工作

正在开发KV不同输入/基座的同形矩阵批；先保持原共同操作数复用，再合批未准备行，
保留各行参数、epsilon、None/零根和私有状态。8MiB局部packing预算；不改变dtype/图/检查点。
新Python独立公式及相关图检查26 passed；C++和完整开发门禁待运行，不宣称已交付。
实现及对应契约尚未提交；ROADMAP的GPU延期修改保留。

开发冻结树`artifacts/worktrees/core5-independent-dev`（精确tree hash见job的snapshot.json）；
unit `core5-independent-dev-20261029`，入口`.local/core5-independent-dev.sh`；
输出`artifacts/jobs/core5-independent-dev/{status.json,task.log,development.json,tests.xml}`。
独立background.slice/Nice10/24G/20min，构建2线程、ATen/BLAS1；源码只读，无运行中编辑。
检查`python scripts/status.py`；停止`systemctl --user stop core5-independent-dev-20261029`。
下一步：核对开发终态，修复真实失败；通过后提交实现、固定源码定向资格，再有限同材料性能对照。

## 本轮已取得的结果（均ead6f34，尚待整理证据提交）

- `core5-contribution-v2-20261029` passed/0，`artifacts/jobs/core5-contribution-v2/result.json`：
  最佳局部FP64探针下首步完整事件逐值相同，中间content伴随量先不同；未证明最终leaf为唯一根因。
  同Full输出伴随量的私有Full参数梯度相同。descriptor边界别名差异须谨慎解释。
- `core5-wide-adjoint-v3-20261029` failed/2，`artifacts/jobs/core5-wide-adjoint-v3/check/result.json`：
  外层汇合点FP64累加、局部VJP原dtype及原三组探针，原D64两步仍严格失败；
  手算/广播/小型packed图通过，内部私有图仍原autograd。该私有接口只供诊断，未接入生产。
  原v1/v2分别因广播缩减/Python Function调用失败，保留failed/1，不产生数学结论。
- `core5-async-boundaries-20261029` failed/2，`artifacts/jobs/core5-async-boundaries/study.json`：
  stock和device guard仍EE1017；关闭autograd多线程/CPU桥小反例通过。
  `core5-async-full-20261029` failed/134：单线程完整图仍invalid event，小反例不能认证完整异步。
- `core5-affine-current-20261029` **passed/0**，`artifacts/jobs/core5-affine-current/result.json`：
  当前C++与22个binary hash完整相符，复用ad6d1e7 SDK构建；原fc/affine FP32 mixed-c默认AdamW
  四次更新/完整checkpoint/CPU交接通过。旧470ce06失败保留；mixed-a尚未重验，不外推全24图。
  C++ hash 8c942fa686ed276204b3bd5a04a026513a3ca21d0a480506c2ae3ba32211d7b2。

上述探索原始驱动在.local/core5-*，固定树`artifacts/worktrees/core5-contribution-baseline`。
所有探索任务均终态，无需恢复旧队列。正式复现入口/独立见证待随本轮证据归入仓库。

## 仍有效的已提交资格

- d8f7132算术设施19项通过，原D64两步局部FP64仍严格失败，各调度自身精确恢复通过；
  [算术报告](evidence/core5-arithmetic-isolation-20261029.md)。生产默认未改。
- ad6d1e7 FP32 GDN/SSD normalized VJP修复：CPU453通过/2原FP16 strict xfail、相关NPU训练恢复通过；
  [报告](evidence/core5-fp32-normalization-20261028.md)。
- 1f1ec57显式CPU FP64原D64四接口完整训练/恢复及D2048同参数推理通过；
  [报告](evidence/core5-stable-preparation-precision-20261028.md)。原FP32失败不被替换。

剩余：D64/D2048 FP32、FP16 GDN默认AdamW、原FP32反馈PDG、默认异步双NPU；
CUDA真机延后、x86_64无硬件证据、大型设备训练/吞吐未验。SDK affine mixed-c已取得本轮通过，
其余仍按实际范围核验。性能不因矩阵批调用减少就宣称改善，经验写入performance-lessons。
当前全部待办以[ROADMAP](ROADMAP.md)为准；重入运行status、固定上游核验。

补充两项已声明的定向核对（同ead6f34只读树；默认参数与原容差不变）：
- `core5-affine-current-a-20261029` / `artifacts/jobs/core5-affine-current-a`，SDK同binary只检查原mixed-a，
  入口`.local/core5-affine-current-a.sh`；队列180s/计算180s/16G/7min。
- `core5-remaining-numeric-20261029` / `artifacts/jobs/core5-remaining-numeric`，
  原FP16 GDN Python/native及原W32/B4反馈PDG四窗完整梯度/SGD槽/换owner恢复；
  入口`.local/core5-remaining-numeric.sh`，冻结新反馈记录器hash写入result.json；
  CPU算子1/16G/8min，每项150s。原失败保留，开发探针不冒充新实现资格。

补充终态：SDK affine mixed-a passed/0；两条原FP16 GDN仍step1严格失败，maxabs .001983642578125。
`core5-remaining-numeric`总体failed/1是反馈驱动缺scripts导入路径，未运行反馈数学检查。
仅补反馈新unit `core5-feedback-current-20261029`，`artifacts/jobs/core5-feedback-current`，
入口`.local/core5-feedback-current.sh`已明确PYTHONPATH；同ead6f34，16G/3min，计算150s。
不重跑已终态的两条half诊断。

阶段提交准备：仓库化外层FP64汇合诊断（2项机制/小图精确恢复测试通过）、
原反馈四窗/梯度/SGD槽/逐值恢复入口及纯Torch异步政策开关。它们均为诊断，不改生产默认。
原反馈新驱动在ead6f34通过执行并保留strict_mismatch/2；记录见core5-feedback-current。
独立投影性能增量仍在冻结开发构建中，不能把本诊断提交当成其资格。
