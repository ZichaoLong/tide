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

## 本轮已提交诊断与待提交证据

诊断实现aadd3b4已普通推送；固定资格全部终态，报告：
[非GPU边界](evidence/core5-nongpu-boundaries-20261029.md)及同名JSON。
- core5-join-aadd3b4：failed/2；设施14 passed/0 skipped。原D64外层FP64汇合仍严格失败，
  两条自身精确恢复通过；反馈FP32四窗记录/梯度/槽失败、参数通过，显式FP64全通过，
  三条自身第四窗恢复逐值通过。
- core5-peer-aadd3b4：failed/2；stock/device guard纯Torch失败，single_thread/CPU桥小反例通过。
  早期ead6f34完整图single_thread仍failed/134，默认异步限制未解除。
- core5-affine-current及-a：passed/0，ead6f34原SDK fc/affine mixed-c/mixed-a FP32默认AdamW
  四次更新/完整checkpoint/CPU交接通过。C++及22个binary hash完整核对；不外推全24图。
- core5-remaining-numeric：failed/1；两个FP16 GDN子任务各strict_mismatch，原值.001983642578125；
  反馈导入路径错误另开core5-feedback-current，后者正常完成strict_mismatch/2。

各原始目录为artifacts/jobs/同名；固定诊断树artifacts/worktrees/core5-join-aadd3b4。
早期join v1/v2诊断器错误和一次启动目录碰撞保留原failed，报告明确区分基础设施与数学结果。
没有采用新的生产精度/optimizer，也没有新增吞吐声明。可复现入口均已归入scripts。

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

## 下一步

数值/异步诊断及SDK具体旧见证的本轮证据准备单独提交；原默认FP32/FP16和异步限制如上保留。
唯一活跃任务是页首core5-independent-dev（冻结tree53afed324c793b8585ea6df39b2303f784fc0cbd）。
新的节点私有来源模式三步AdamW/精确恢复Python4项已通过，稍后用新C++检查native同项。
开发门禁通过后提交KV合批实现、固定源码定向资格及一次有限对照；不恢复旧队列。
