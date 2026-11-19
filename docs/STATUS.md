# 当前交接：清理、数值定位与quad修复已完成

本轮以 `c5bfebaf` 起步，修复实现与资格固定 **`97c52960`**；诊断/重放工具固定
`2360728d`、`e1c6aaa0`、`aab9a4d2`、`72c1a9d0`。证据单独提交，
不以文档HEAD追认旧测量。完整事实及复现见[本轮证据](evidence/cleanup-numerics-20261119.md)
和[审阅JSON](evidence/cleanup-numerics-20261119.json)。当前有限计划无待运行测点。

数学固定 `4f5b9f20a4030d68ee66ae2d26e34659a5e37234`，108文件原字节核验通过。
持续FC/HB实例路线见[scope](project-scope.md)；通用基座未启动。

## 清理

70个完成用途载荷、六组成功profiler及两份冗余update2释放 **40.637272 GiB**旧产物。
两份update2删除前与原件完整树exact；删除后又从anchor+真实update1重做并验证完整摘要exact。
12份新冻结源码的6036文件逐字节匹配已推送Git提交，连同缓存、成功CPU测试临时文件、
两份本地恢复bundle和隔离安装临时产物，再释放 **2.990738 GiB**。
本轮累计删除43.628010 GiB，其中后者是本轮新生成产物，不能重复当原有配额净减少。

仓库实际占用约91.150→**51.045 GiB**；
账户quota实测约91.006/300 GiB软限（硬限700 GiB）。
账户目录du有不可读Docker子目录，仅作部分扫描，账户总量采用quota。
原290个.pt、6734个.f32约45.604 GiB及74个失败profiler身份不变；
它们主要是已结束实验的失败复现输入，不代表这些实验仍待运行。
保留本轮有效局部见证、新训练anchor/checkpoints和小结果；没有整组删除失败材料。
四份用户未跟踪文档原字节保留、不提交，主math/.git、唯一ignored源码、公共环境和下游未改。

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

15个服务、110个记录后代全部终态；CPU服务0.263/4小时，
NPU服务0.116/2卡时（含等待）；峰值进程树RSS约28.789/128 GiB。
新产物清理前观察约3.522/4 GiB，收尾后约0.560 GiB；单NPU并发、协作队列独占空闲物理11，
没有共享占用卡。服务时长不是模型性能计时。
本轮有限工作已经收尾，剩余可选定位/优化列于[ROADMAP](ROADMAP.md)，不自动重启旧矩阵。
