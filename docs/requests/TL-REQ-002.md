# TL-REQ-002 — 全位置语言序列的有界执行成本

状态：下游需求已在实验仓库登记；未声称基座接受、修复或下游验收。基座保持只读，没有发送外部消息。

## 研究目的与固定身份

在同一模型上评价语言next-token质量、实际稀疏和多区域容量，而不退回单答案任务。执行、严格数值、离散语义、恢复和学习分别报告。

- 数学`0770593838a37c790a568226930567ed60f99b83`；foundation `85d161d5c9148eac2b164f89a0ba4204b4331898`，安装248文件hash未变。
- 下游初始实现`64493f6c996e7b6dd47d03a4b59cb524dbf44215`，配置`configs/studies/language/initial.json`，数据来源与锁见`docs/evidence/language-data-ready.json`。
- 公共`prepare/materialize`、`layered_inputs`、`eager_runner`、`Payload.decode/numerical`；实验只拥有embedding、全位置公共输出池化、tied head、CE、optimizer和应用恢复。没有导入benchmark或复制执行核心。
- Torch2.12.0+cpu / TorchNPU2.12.0.post3 / CANN9.2，NPU FP32，4 CPU线程，公共Python packed单host worker、mixed-c placement。

## 观察与未知

HB16/4区域/width32/KV-qry/regional_total k2：A自然文本B8、L32的NPU训练warmup为57.992s，首个测量更新54.041s，约4.74有效token/s；包含同步、前后向、clip及AdamW，排除进程/模型启动。三更新加验证的180s阶段限时失败，完整两次测量未取得。没有把这个不完整窗口称为正式性能基准。

同任务CPU有完整小型B8/L32校准，两个测量更新均值20.362s；其FP32独立严格超差仍保留，且不是对齐完整数值/工作量后的后端加速比较。两次NPU稠密/稀疏校准时限、日志与原始指标位于`runs/lm-a-language-v1/`；汇总及hash见`docs/evidence/language-initial-attempts.json`。HB16时限失败不说明模型学不会语言。

目前未做该长序列的完整算子profile，尚不能确定host调度、状态拼装、设备拷贝/同步、返回记录或算术各占多少。已有大逻辑batch本身不能证明大算子batch。下游选择有名字的FC4/FC8先导以继续学习研究，另保留HB问题。

协调器重复字段错误及B错误解冻固定传输参数属于下游缺陷，已另修，**不作为本需求的基座错误**。

## 希望的公共能力或指导

在固定语义和安装契约内，给出可用于下游全位置head/CE训练的公共执行选项或成本定位入口。优先确定目前接口的实际算子batch、候选/Upd/Read/Full/消息、host/device数据流与返回成本，并说明可支持的优化范围。可以是公共已有能力的明确接入方法，不预设必须新增专用kernel，也不要求一次交付所有后端。

## 有界验收建议

1. 用相同HB16、B8/L32材料保留完整独立前向/状态/VJP、共享参数和唯一optimizer检查；保留现有严格阈值及有限失败事实，不把全部浮点一致作为授权学习探索的前置门槛。
2. 同返回契约、数据、工作量、batch、线程与精度比较一小段包含同步/后向/optimizer的更新；报告初始化、warmup、验证与checkpoint成本。不同发送支持或近似算法是另一个配置，不能直接称等价加速。
3. 再测少量较大真实batch，给出算子shape和内存，而非只报告逻辑batch；保留失败与预算。
4. 若交付改变执行/状态/梯度边界，下游使用独立候选环境做新进程恢复与同任务小型验收，再决定是否更新安装锁。当前作业和公共环境不变。

这条需求只约束依赖HB长序列成本的后续规模工作；不阻塞当前已获准的FC语言先导或原模型控制。
