# 当前交接：小目录清理完成，推送待网络恢复

本次从 `611e8d45` 接续，整个仓库含共享Git和主私有环境现约 **144.380 MB**，
低于用户200 MB目标；详见[小目录收尾](evidence/compact-repository-20261119.md)及
[审阅JSON](evidence/compact-repository-20261119.json)。本地提交完整，普通push因SSH网络超时尚未完成。
当前工具固定 `bef26abb`，历史源码保存 `9b270280`；生产树仍为 **`97c52960`**，本次不改模型。

数学固定 `4f5b9f20a4030d68ee66ae2d26e34659a5e37234`，108文件原字节核验通过。
持续FC/HB实例路线见[scope](project-scope.md)；通用基座未启动。此前数值定位及原失败见
[数值证据](evidence/cleanup-numerics-20261119.md)，不以清理或文档HEAD扩大资格。

## 当前存储与可用性

全部旧artifact载荷、原报告gzip、诊断见证及重复环境已退役；18个实验worktree已由Git移除。
旧整图和局部载荷都不再承诺直接重放。原分支refs、18个tracked树、6,171份独有源码/配置/
小结果及作业身份保存在Git历史；注册worktree只剩主仓和下游。
47份审阅JSON无损压缩并逐字节对照原Git文件，原失败、阈值和测量身份完整保留。
当前还原入口为 `scripts/materialize_record.py` 配合[索引](evidence/archive-index.json)；
旧352条artifact索引只留历史身份。Git、固定math、当前源码和主私有环境均保留。

清理后隔离wheel的23包文件匹配 `97c52960`，公开小CPU FP32训练、checkpoint及
新进程实际恢复更新通过。还原工具10项检查和真实审阅记录还原通过。
下游运行不依赖已退役产物；四份用户文档原字节保留、不提交，公共环境和参考仓库未动。
下游并发新增的未跟踪文件原样保留；一次删除后状态相等断言失败另有独立核验，不掩盖退出码。
本次7个服务及全部记录后代终态，无NPU任务。当前没有待跑实验；只剩网络恢复后的普通push。

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
