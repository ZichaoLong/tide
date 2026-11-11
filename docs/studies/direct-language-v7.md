# direct-v7：直接图模型的配方与扩大曝光

2026-11-11用户明确批准执行后继学习实验，并允许在上游不可用期间使用直接实现。
本表替代此前“仅性能实测、后继训练未授权”的当前限制；旧作业与旧失败不改。

## 问题与解释边界

先判断初始化/读出归一化是否使HB40/HB88的直接图模型稳定学习，再在相同配方、
数据曝光下比较容量与深度。使用已独立检查的普通Torch因果Attention/SwiGLU图模型：
输入来自固定导出的FC/HB邻接，节点独立参数，同位置入边等权平均，输出节点等权平均、绑定词嵌入读出。
它没有TIDE payload、transport、Read/Describe、区域分配、可学习发送或公共continuation。
不导入/复制基座执行核心，不伪造quad/proportional。**这是不同函数的学习研究，不是完整TIDE实现或旧HB失败的修复认证。**

固定数学0770593838a37c790a568226930567ed60f99b83与安装基座74e7bbd身份不变；
本轮模型不调用基座执行。旧基座与上游工作树只读，不接入尚未交付的优化。
它服务A路线的拓扑/配方诊断；B预训练路线本轮没有新提交，也不由本轮直接图结果替代。

## 有界执行表

| 组 / Lens group | 候选与对照 | 改变量 | 曝光、重复 | 主要指标与判定 |
| --- | --- | --- | --- | --- |
| 短校准 / TL-DIRECT-RECIPE-v7 | NPU HB88 B16/32/64/128，HB40/TF28 B32；CPU8线程HB88 B16、HB40 B32 | 物理batch/设备 | 各3 warmup + 5完整更新；合成输入不产生质量结论 | 同步完整forward/loss/backward/AdamW时间、实际参数/内存/工作；选定物理B，保持全局B32及更新数，不为吞吐自动增大全局B |
| 配方 / TL-DIRECT-RECIPE-v7 | HB40/88 × std0.15或fan-in × 最终无norm或RMSNorm，共8；TF12/28 × 原sin/cos或尺度平衡，共4 | 每个受控格只改声明因素 | seed11，各262,144有效目标；所有LR日程从开始就是4,194,304 | 完整32,768目标validation NLL、初始值、梯度范数、有限性；共同配方按两尺寸平均NLL选择，0.02内优先fan-in+RMSNorm；TF同理优先平衡位置尺度 |
| 扩大曝光 / TL-DIRECT-SCALE-v7 | 选定同配方HB40、HB88、FC16，加校准TF12、TF28 | 模型容量/拓扑；背景均从零 | seeds11/22，各到1,048,576与4,194,304；共10轨迹 | 同目标序列/同验证块的NLL/PPL，文档配对差、种子方向，实测token/s与累计成本；有限点趋势，不外推scaling law |
| 条件结构 / TL-DIRECT-STRUCTURE-v7 | 选定配方HB40/88加固定残差系数，原配方为对照 | 两条局部残差乘1/sqrt(2D)，D为最长rank | 最多2条seed11，各4,194,304 | 仅当两种子平均HB88−HB40 NLL在完整1M和4M均>0.05触发；同曝光验证传播干预，不再扩展架构 |

若某组没有合格共同配方，仅阻塞依赖它的候选；独立对照继续。
选择和条件判断由固定程序执行，**不自动生成科学支持、非劣或迁移结论**。
262K筛选与4M主训练使用同一冻结配置/日程；选中seed11的4条HB/TF续训原checkpoint。
其余配置新建且哈希固定，失败不改身份、不删记录。screen和main是同轨迹的阶段，不是独立重复。

## 共同设置与不等同之处

- 数据复用`runs/v6-data-final`：WikiText103训练池20,971,520 token，独立validation；4096 train-only BPE。
  训练均document-block L128、全局独立样本B32、FP32、AdamW(lr0.001, eps1e-6, wd0.01)、clip1，
  65,536 token warmup，余下cosine到lr的0.1；1M/4M是有效监督目标，不是输入padding或独立语料总量。
  同种子共享目标序列，里程碑尾部舍弃显式记录。test仍密封，不通过筛选验证集声称独立test资格。
- Direct：width128、H2、FFN512、节点参数独立；无额外位置编码，因果attention和BOS仍提供序列结构。
  fan-in指每个普通投影W的std=1/sqrt(d_in)，std0.15是直接模型干预，不是旧TIDE整体复刻。
  RMSNorm位于最终hidden与绑定head之间，epsilon1e-6。
- Transformer：width256、H4、FFN1024、12或28层；原固定位置编码RMS约0.707，
  平衡尺度乘sqrt(2)*0.02使其RMS约0.02。近参数对照不是相同FLOP或相同每层功能。
  直接FC16约4.73M、HB40约11.02M、HB88约23.62M；TF12约10.53M、TF28约23.17M，实际inventory为准。
- 实际所有节点均稠密执行；报告节点/边/attention位置和矩阵行数，无可学习稀疏资格。
  不把本轮稠密学习效果与历史TIDE稀疏观测拼成同一模型的联合证明。

## 正确性、资源与停止

复用普通节点独立公式/VJP检查，新增残差/norm/初始化与位置干预检查。
每个新训练配置先运行B2/L3 CPU对照、CE解析VJP、因果干预、共享embedding/head及optimizer检查，
再通过三个独立进程的3步精确恢复（model/optimizer/RNG/cursor/gradient）。
CPU/NPU严格阈值仍atol1e-6/rtol1e-5，因果零阈值；有限超差单列，不强制改为CPU FP64。
非有限、因果/共享/optimizer机制错误、恢复失败阻塞该配置。完整长轨迹精确重放未检；
推理验收仅16步greedy有限输出和执行，不等于生成质量。

同一12h wall预算，最多8累计NPU卡时、32累计CPU核时（含NPU host/等待/监督），
最多4张独立NPU并行、16声明线程、128GiB聚合RSS、32GiB新增产物；
账号硬配额800GiB下至少保留100GiB，主机至少留128GiB可用RAM。
screen每case≤900秒、main/条件case≤3600秒，达到总体预算立即停止并保留checkpoint/终态。
具体deadline固化在`configs/studies/language/direct-v7/budget.json`，不随阶段重置。
已授权轻载共享卡；两次观察、健康/≥32GiB HBM余量及生命周期锁保留。
不修改活动输入，使用background.slice、冻结git源码和独立run目录。
成功恢复后的临时probe checkpoint可按hash清单退役，正式科学checkpoint全部保留。

## 记录与复现

项目`tide-learning-lab`。配置在`configs/studies/language/direct-v7/`；固定选择器
`scripts/direct_campaign.py`，复用`campaign_worker.py`和`campaign_stage.py`。
run ID为`tl-v7-screen-<config-id>` / `tl-v7-main-<config-id>` / `tl-v7-residual-<config-id>`；
控制记录`runs/v7-campaign/state.json`，训练原始JSONL、逐块validation、依赖及checkpoint是权威。
Lens私有入口仅`.local/LENS.md`，短记录确认正确实例/项目后再提交学习。
结束后审阅实际质量、反证、仍未区分的机制和迁移条件，再更新研究图结果。
后继到TIDE的最低桥接是基座恢复后在相同配方/数据上复验，不能据直接模型结果跳过。

## 已完成的执行校准

源码3455c7a的CPU/NPU校准均完成，两个Lens投影verified/readback_matches，已核对无存活后代。
[实际计时、安装与batch选择](../evidence/direct-v7-calibration.json)：HB88 NPU B16/32/64/128
完整更新中位数0.0966/0.0942/0.1236/0.2034秒，约21.2K/43.5K/66.3K/80.6K输入位置每秒。
CPU8线程HB88 B16约2.157秒，HB40 B32约1.537秒；NPU HB40 B32约0.0458秒。
本轮选NPU、物理B32=全局B32、不累积、四独立worker，兼顾吞吐和固定更新数；并非声称B32是吞吐上限。
校准保持稳态optimizer连续更新，其计时口径不同于前一轮每窗口重置optimizer的成本锚。
19项模型/数据/生命周期测试通过；新增等待资源服务预算回归也通过。
新监督将资源等待算为活动服务，可统一预算停止；校准监督退出后再固定新源码提交学习，未修改活动快照。
