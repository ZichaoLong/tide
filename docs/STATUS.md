# 当前交接 — 2026-10-29

授权分支`graph-execution-foundation`；数学固定0770593838a37c790a568226930567ed60f99b83。
用户要求继续本机非GPU工作，GPU限制延后。参考仓库/公共环境只读；不使用子代理。
阶段提交及普通推送已授权。当前HEAD为38ebf422b198d7cc041b6238fb954cf9775c02c0。
本轮数值/异步诊断aadd3b4已推送；38ebf42是已提交的对应证据，待随本阶段普通推送。

## 当前实现与开发验证

KV新增不同输入/基座的同形矩阵批：先保持合法共同操作数复用，再合批未准备行；
逐行参数/epsilon、独立None/零根、私有状态保留。8MiB局部矩阵packing预算，不改变dtype/图/checkpoint。
本提交交付该实现；公共接口、独立公式、完整图/恢复检查及契约同步。
纯Python26项及节点/区域三步AdamW/逐值恢复Python4项、native4项已通过。

- `core5-independent-dev`：C++构建完成，构建+测试共1170秒到限，failed/124；不计测试通过。
- `core5-independent-dev-tests`：复用经hash核对的构建，235 passed、2旧FP16 strict xfail、2 failed。
  两个失败为旧C++检查把独立输入的“无共享”写成“无准备值”。现改为零计算复用、真实独立批、
  手算输入梯度及另一行输入/adapter为None；生产实现及原容差不改。
  原始source.tar.gz、manifest与failure-explanation.json保留在该job目录。

修正开发任务`core5-independent-fix-dev-20261029`已终态passed/0，50项通过；
包括C++独立输入手算/None根、节点/区域三步AdamW精确续接及设备检查入口。
完整source archive、构建manifest与tests.xml位于`artifacts/jobs/core5-independent-fix-dev`。
先前failed/124及failed/1记录均保留，不回填为通过。当前没有活跃任务。
下一步以本实现提交冻结源码做定向资格；不得把开发archive身份当正式提交资格。

## 下一步

1. 建立本实现提交的`core5-independent-qualified`固定干净树；baseline固定38ebf42。
2. `.local/core5-independent-qualify.sh`：固定CPU定向资格，完整核对新C++/binary hash后复用。
3. `.local/core5-independent-npu.sh`：新SDK core及相关resident ABI重建和设备门禁；
   `.local/core5-independent-python-npu.sh`：纯Python mixed-c KV FC/HB/一般端口两dtype训练恢复。
   以上尚未启动，实际源码写`.local/core5-independent-source`；不得把旧设备二进制用于新ABI。
4. 资格终态后有限性能对照：CPU native中型与同参数D64显式FP64，纯Python NPU mixed中型，
   baseline/current每配置一次独立进程；另一个D64 profile进程。项目重型计时串行。
   入口`.local/core5-independent-study-{cpu,npu}.sh`，尚未启动；输出新job目录。
5. 固定证据、经验、当前契约和ROADMAP同步，证据另提交并普通推送；不恢复旧队列或GPU任务。

## 数值和异步边界

[已提交非GPU边界证据](evidence/core5-nongpu-boundaries-20261029.md)保存全部终态、见证和复现入口。
aadd3b4设施14项通过。原D64外层FP64汇合仍严格失败，局部私有VJP仍原精度；未采用生产探针。
原FP32反馈四窗记录/梯度/动量槽失败，参数通过，各调度自身恢复逐值通过；显式FP64四窗全通过。
FP16 GDN原AdamW差约.001984仍保留。默认异步双NPU纯Torch stock/guard失败，单线程只通过小例，
完整图仍abort；没有公共环境改动或已验证的非阻塞替代。
原SDK fc/affine FP32 mixed-a/c在ead6f34已通过四更新/完整checkpoint/CPU交接，不外推全矩阵。
D2048 FP32、CUDA真机延后、x86_64无本机硬件、大型设备训练/吞吐仍分列。
当前待办与适用范围以[ROADMAP](ROADMAP.md)为准。既有资格从[证据索引](evidence/README.md)导航。
