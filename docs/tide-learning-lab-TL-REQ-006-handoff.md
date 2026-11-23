# TL-REQ-006 issue：HB 训练的 AICore / AIVector / AICPU 占用与执行成本

2026-11-12，用户明确要求在基座分支仓库留下本 issue，供基座后续分析、优化。
状态：**待基座评审；未实施本需求的优化，未开展下游验收。**
这是本地交接入口；请求原文由实验仓库维护，不在基座另建一套实验状态台账。

[完整请求 TL-REQ-006（固定提交 dc943cc7）](https://github.com/ZichaoLong/tide/blob/dc943cc7ce773e3dac4d62736ab524953573decb/docs/requests/TL-REQ-006.md)
包含实际配置、依赖锁hash、计时边界、源代码定位、只读复核命令、有限排查建议与验收条件。
本机原文：`/mi/data2T/zlong/home/llm/tide-learning-lab/docs/requests/TL-REQ-006.md`。

## 为什么提出

本批 HB 语言训练经常显示较高总 NPU 利用率，但 AICore 很低；用户指出不能将前者解释为充分利用矩阵算力。
请求按实际完整训练成本分辨必要数学、索引/反向、细碎算子与同步、真实矩阵形状等因素，
帮助后续在同一个模型上推进语言质量、容量/依赖深度、低活动及局部性。
允许稠密计算加遮罩继续用于学习；不要求先兑现活动比例对应的吞吐。

**这些数据来自实验侧独立直接数学实现，不是当前基座执行器的实测。**
因此这是工作负载与机制线索，不能据此认定最新基座回归，也不能把下述直接路径称为完整公共执行资格。
历史 TL-REQ-004 的旧 eager trace 可参考，但旧 kernel 数量/耗时不能套到此次观测。

## 已有实测

北京时间2026-11-12 09:28:20–09:28:56，8轮只读 `npu-smi info -t usages` 采样，程序总耗时37.67秒。
以下为单卡计数器中位数；完整8卡每轮/min/median/max与job/config/step身份见原始记录。

| 物理卡 / case | AICore | AIVector | AICPU | HBM带宽 | 总NPU |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 / hb88-pool-s22 | 3% | 17.5% | 30% | 7% | 94.5% |
| 6 / hb88-pool-s11 | 3% | 21% | 31% | 7% | 94.5% |
| 8 / hb184-quad-s22 | 1% | 27% | 21% | 3% | 48% |
| 14 / hb184-quad-s11 | 1.5% | 26% | 14% | 4.5% | 56.5% |
| 4 / tree-r32-h2-bern-s22 | 2% | 14.5% | 33% | 11.5% | 100% |
| 10 / tree-r32-h2-bern-s11 | 2% | 15.5% | 33% | 11.5% | 96.5% |
| 2 / tree-r64-h2-q3-s11 | 2% | 18% | 33.5% | 11.5% | 92.5% |

物理1另有其他进程，其计数器不用于单任务归因。全部样本都是设备级计数，不是进程级kernel耗时。
不同百分比各有分母，不能相加、当作耗时占比或理论FLOP/s利用率。设备侧AICPU不等于host CPU fallback。
AIVector和带宽工作也可能是必要数学；不能用单独拉高AICore读数代替端到端性能改善。
持续存活的服务所属host PID约占1.6–1.9核，短进程和共享v1控制器限制归因，未证明host瓶颈或限流。

另外，已完成64M轨迹（每seed 67,108,864有效目标，双seed合并）的同步训练**更新**吞吐：

| 模型 / 参数量 | microbatch / global | 更新token/s |
| --- | ---: | ---: |
| HB184 pooled / 51.438M | 8 / 64 | 2,057 |
| 浅宽HB40 pooled / 49.885M | 32 / 64 | 14,851 |
| Transformer512×16 / 52.540M | 64 / 64 | 94,102 |

L256、FP32；计时含前向/loss/反向、当前模型观测回读、有限检查、clip、AdamW和终端同步；
不含外部batch准备/上传、构造、评价、checkpoint及日志写入。不是完整作业吞吐。
这些是不同模型与物理batch/累积、不同运行时段的成本，不是等价实现加速；也不是上表采样窗口的配对计时。
特别是吞吐表HB184为pooled，而资源表HB184为quad。尚无本批kernel trace或同窗口TF硬件计数器。

## 固定身份和取证入口

- 数学：`0770593838a37c790a568226930567ed60f99b83`。
- 实际训练：`6e7bd49eb5e35c15445826cac3a9b6ddda75aff4`，本机下游 `.local/source-v9-execution`。
- 报告、观察脚本与原始数据：`8995958728167eee7a4bcc2139f5e1794cbf82ab`。
- 已安装基座元数据：`74e7bbd62341d2ccd382194a2c52ad4fea033665`；不是本次直接实现的执行源码。
- 当前基座checkout：`a8a4ee028a5d79696d49bc47242a29c4c39aee4d`；本次没有在该提交上重测。
- 实际栈：私有 `.venv-positive-floor-v1`，Torch2.12.0+cpu、TorchNPU2.12.0.post3、CANN9.2基线、FP32。
- 配置：下游 `runs/v9-campaign/main/configs/<case>.json`；记录：`runs/v9-main-<case>/`；
  作业命令/固定输入hash：`artifacts/jobs/v9-main-w*/job.json`；数据锁：`runs/v9-data-r3/data-lock.json`。

固定公开材料：

- [已审中期报告](https://github.com/ZichaoLong/tide/blob/8995958728167eee7a4bcc2139f5e1794cbf82ab/docs/studies/direct-v9-interim-20261112.md)
- [设备原始采样及对应配置](https://github.com/ZichaoLong/tide/blob/8995958728167eee7a4bcc2139f5e1794cbf82ab/docs/evidence/direct-v9-resources-20261112.json)
- [逐运行、参数、实际成本与正确性分项](https://github.com/ZichaoLong/tide/blob/8995958728167eee7a4bcc2139f5e1794cbf82ab/docs/evidence/direct-v9-interim-20261112.json)

读取上述材料不需要运行训练、访问Lens或下载模型。活动源码、配置、依赖和checkpoint继续保持固定。

## 建议先回答什么

按[本仓库优化skill](../.agents/skills/optimize-tide-execution/SKILL.md)及其加法/减法方法，
先复用已有张量列、区域VJP与性能证据，再用有限代表区分：

1. `direct_semantic.py:159`的class高级索引，以及265–280行的节点select/stack，
   是否在反向形成大量Index/IndexPut/Scatter、梯度清零或重装配，哪些实际运行在AICPU。
2. `direct_semantic_ops.py:87`的28轮分配求根与active-set修正中，小归约、where、
   启动和临时张量各占多少；pooled不走该求根路径却有类似低AICore，不能将全部差距归给求解器。
3. `direct_semantic.py:288`每microbatch的观测回读，以及`campaign_runtime.py:83`之后的
   标量loss/有限检查、clip和AdamW，分别造成多少设备同步和实际工作。
4. 实际GEMM/bmm形状、布局、区域依赖与累积次数是否限制吞吐；global64不是每算子B64。
   R64入口树的micro4每更新需要16次前后向；模型总参数量不能代替设备算子形状。

这些是静态排查线索，尚未被本批trace证实。建议从HB88 pooled、HB184 quad、R64入口树及TF四代表的
固定权重/输入短trace入手；分前向/反向/optimizer，并另做无插桩完整步计时。详细有限预算建议见权威请求。
基座公共路径与直接v9职责不同，桥接时明确状态、独立根、调度与恢复范围，不能直接算一个“包装开销比”。

验收以可复核的成本归因、相关语义/共享梯度/optimizer/恢复检查和代表完整步收益为依据，
不预设AICore门槛、与TF的速度比或稀疏比例兑现门槛。执行、严格数值、离散语义、恢复、学习、性能分列；
v9原CPU/NPU有限超差及atol1e-6/rtol1e-5保持，不因为学习或性能结果而改成通过。

本次只按用户授权落文档。基座接受后的排程与实现仍由本仓库STATUS/ROADMAP管理；
本文件没有启动作业、接入新基座、修改执行代码或改变活动训练。旧TL-REQ-004/005交接文件保持原样。
基座回复及下游验收均待后续。
