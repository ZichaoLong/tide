# 当前交接：存储收尾完成，数值资格边界保留

本次从 `660f3d0f` 接续，剩余51 GiB的存储收尾已完成；归档还原工具固定 `45cb5dad`。
当前大小、保留范围、还原入口与清理后安装检查见[存储收尾](evidence/storage-finalization-20261119.md)
和[审阅JSON](evidence/storage-finalization-20261119.json.gz)。本次不改生产树，不新增模型资格。

此前数值定位从 `c5bfebaf` 起步，修复实现与资格固定 **`97c52960`**；
诊断/重放工具固定 `2360728d`、`e1c6aaa0`、`aab9a4d2`、`72c1a9d0`。
原事实见[数值证据](evidence/cleanup-numerics-20261119.md)及其配套JSON，
保留原测量身份和失败，不以文档HEAD追认旧测量。当前有限计划无待运行测点。

数学固定 `4f5b9f20a4030d68ee66ae2d26e34659a5e37234`，108文件原字节核验通过。
持续FC/HB实例路线见[scope](project-scope.md)；通用基座未启动。

## 清理

前次约91.150→51.045 GiB；本次再从 **51.045降至约2.264 GiB**，净减少约48.782 GiB。
283个旧.pt和6669个.f32明确退役，释放45.591274 GiB；352份JSON无损归档再省3.213501 GiB。
新增索引/收据/Git证据计入净占用，不将本次临时安装/还原产物的删除重复算为原有配额减少。
账户quota约42.287/300 GiB软限（硬限700 GiB）；账户目录du有不可读Docker子目录。

保留约0.520 GiB的78个张量文件、可读汇总/小记录、压缩原JSON、74个失败profiler及独有源码。
近期六份训练/局部见证、六个旧小见证和native原输入/输出包继续在线；82份保护文件逐字节核验。
旧anchor/update1/VJP1和BF16 anchor已退役，旧整图续更不再有在线输入可直接重放；
此前exact重做是历史观察，原失败及阈值不改。JSON按[索引](evidence/storage-archive-index-20261119.json.gz)
使用 `scripts/materialize_record.py` 精确还原；解压不会重建已退休张量/trace。

清理后新隔离wheel的23包文件匹配 `97c52960`，公开CPU FP32训练、checkpoint、
配对更新及新进程实际恢复更新通过。还原工具8项检查及两份真实记录的精确还原通过。
下游运行不需要实验产物；主math/.git、所有注册工作树源码、私有环境、公共环境和下游保留。
四份用户未跟踪文档原字节保留、不提交。本次无NPU任务，三个服务及10个记录后代全部终态。

## 数值定位与修复

- **FP32 AdamW。** 原0/392/533超差字段完整重现。真实update1→update2的两个超阈bias坐标，
  独立80位Decimal预测与实测差仅约3.26e-11/−3.94e-12；确认小梯度、epsilon与历史的更新放大。
  只解释所选坐标，不宣称全部上游误差均为舍入，不改epsilon或阈值。
- **真实FFN局部VJP。** 原baseline主干、输入梯度和所选参数行exact；同X/W/dY的两路径
  输出/输入VJP exact，dW差5.09e-11，均在独立FP64公式原阈值内。全图成因仍有限定。
- **共享9B求解错误。** 原region43/node817/int64时间509的同输入正量被sorted quad归零。
  根因是前缀相消选错face而总质量检查仍通过。97c52960补最终face KKT检查，失败走原同设备
  同dtype回退，不改数学、不加epsilon。原错误也使应为(2/3,−1/3,−1/3)的score VJP变零。
- **BF16。** 第三更新见证是上游评分跨过真实支持边界，各自Decimal支持与生产一致；
  与上述solver错误区分。完整上游评分/梯度/历史成因仍未全部定位。

## 修复后的资格

冻结CPU全套 **1043 passed、1 strict xfailed、1已知profiler warning**。
原完整NPU形状的两份值/support/独立score VJP见证通过。
小X/d192/B2/L8/chunk4的baseline/combined各三次继续更新及新解释器恢复完成；
各自saved data/next forward exact，实际step4→5更新阈内非exact。
combined对baseline仍有1/1/8个字段超差，没有支持、int64、结构或非有限失败。

修复后9B共享B8/prompt128/chunk32/decode16的region1/4 Full均为180209，
内部支持与公开support/time一致；仍有298个内部标量元素比较超阈，不是全数值等价。
新隔离wheel的23个包文件匹配修复源码，公开CPU训练、checkpoint及实际恢复后更新冒烟通过。
原资格失败全部保留；修复不认证所有FP32边界，也没有新增9B训练、CUDA、多NPU或任务质量资格。

## 既有能力、性能与使用建议

五类记忆绑定仍全部保留：X、全SSD、反向混合、共同窄Q/K/V总宽、仅记忆区域共享rank8。
局部扩展与归属见[memory candidates](memory-candidates.md)；参数/性能/训练恢复见
[五类证据](evidence/memory-candidates-20261119.md)。FFN独立及全部历史状态私有的约定不变。
默认训练入口和gather/rectangular保持，优先FP32；建议包含97c52960的solver修复。

[KV128与TF32细拆](evidence/kv128-transformer-breakdown-20261119.md)及
[FFN/CPU/NPU对照](evidence/ffn-execution-devices-20261119.md)已完成，保留原源码身份。
W/P/WP中尺度省显存但未提高同batch训练吞吐，跨路径失败仍在；不默认启用或扩至9B。
新修复生产树没有新的性能测量，旧吞吐不能直接贴到当前HEAD。
共同32层参考不替代A–D；工程吞吐不证明下游模型效果。
具体入口、推荐设置与限制见[训练建议](execution-recommendations.md)。

## 资源与接续

此前数值任务15个服务、110个记录后代全部终态；CPU服务0.263/4小时，
NPU服务0.116/2卡时（含等待）；峰值进程树RSS约28.789/128 GiB。
新产物清理前观察约3.522/4 GiB，收尾后约0.560 GiB；单NPU并发、协作队列独占空闲物理11，
没有共享占用卡。服务时长不是模型性能计时。
本轮有限工作已经收尾，剩余可选定位/优化列于[ROADMAP](ROADMAP.md)，不自动重启旧矩阵。
