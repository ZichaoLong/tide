# direct-v7终态交付

2026-11-11 16:51北京时间，**12格筛选、10条4M主轨迹全部完成**。
这是22个训练阶段、18个独立神经初始化（4条seed11主轨迹从筛选续训），另有2个短校准与解析计数参照。
条件残差两格未触发、未运行。15个后台服务全部终态且无存活后代；没有后继训练在后台等待。

- [配方一页结论](direct-v7-recipe-results.md)：直接HB40/88共同选择fan-in、无最终RMSNorm。
- [扩大曝光一页结论](direct-v7-scale-results.md)：两seed的4M点都出现HB88优于HB40，平均改善0.07837 NLL。
- [研究表](direct-language-v7.md)、[运行/检查/统计/Lens证据](../evidence/direct-v7-observations.json)、
  [可复现图](../evidence/direct-v7-learning.svg)、[状态/日志/停止/恢复](direct-v7-operations.md)。

## 身份、验收与成本

训练源码`d30559c8bce74d5b6ea392bb18f53e8ba3b90cf1`，冻结tree
`d7627b018fc5a888f8b70f498b404a2ca0565cc0d41ff7190da00f653c6c794f`。
校准3455c7a，解析计数bc58bef→b463f80（原失败保留），各job.json有命令、输入SHA、实际物理设备与环境。
私有`.venv-positive-floor-v1`、Torch2.12.0+cpu / TorchNPU2.12.0.post3 / CANN9.2.0；
数学0770593838a37c790a568226930567ed60f99b83、安装基座74e7bbd保持，模型训练不调用基座执行。
`runs/v6-data-final/data-lock.json`固定训练/验证，test未读。模型引用已在前一轮迁移到`/mi/data2T/models/huggingface`，本轮从零不读取公开checkpoint。

| 轴 | 实际结果 |
| --- | --- |
| 执行 | 22/22训练阶段到各自262K或4M目标；全32,768目标validation与16步greedy有限输出完成 |
| 独立严格数值 | 原atol1e-6/rtol1e-5保留，所有阶段关联的参数VJP有有限失败；不能标等价通过 |
| 机制 / 离散 | 普通节点独立公式/VJP测试通过；每配置B2/L3 CE解析VJP、零阈值因果、绑定embedding/head与optimizer检查通过；静态稠密图，无可学习路由离散语义 |
| 恢复 | 18套独立配置的新进程3步精确model/optimizer/RNG/cursor/gradient恢复通过；4条真实262K父checkpoint续训到4M。完整长轨迹独立重放未检 |
| 学习 | 直接图有限容量收益经双seed/同文档验证审阅；不能外推完整TIDE、稀疏、长期scaling或可靠Transformer优越性 |
| Lens | 22训练+2校准+2修正解析参照均verified/readback_matches；另外2个原解析投影保留，其中seed22样本身份错误单列 |

有限数值排查保留了前向/VJP逐参数max_abs和relative_L2；因果干预均为零差，CPU独立节点公式/VJP通过，
数据/恢复/共享/optimizer机制检查通过。最坏参数VJP relative_L2约4.24e-4
（TF28原尺度的`blocks.3.norm2.weight`，max_abs约0.00243），完整根因未定位，不标为无害舍入。
因此继续的是已授权FP32学习探索，不是严格数值资格。未用CPU FP64限制正式NPU训练。

实测从首校准服务到监督结束跨度13.23分钟，终态保守扣账约**0.685 NPU卡时、2.351 CPU核时**
（全部服务实际分配1.851核时，含NPU host、等待、两代监督；另加0.5核时开发预留），明显低于8/32小时上限。
监督最后采样产物14.19GiB、配额余量632.88GiB、主机可用RAM1857.99GiB；未触发预算停止。
正式训练记录最大单进程RSS约2.65GiB、NPU allocator峰值约5.35GiB，不含驱动/他人进程，也不是聚合峰值。
并发四独立NPU；screen物理11/15/1/14，main物理11/10/14/1，均记录两次共享卡准入。

HB88正式更新中位约0.098秒/4K目标，约41.5K目标/s；HB40约0.047秒、FC16约0.021秒。
这是直接普通图的本机学习吞吐，不能外推为完整TIDE等价加速。

## 查看与下一批选项

Lens项目`tide-learning-lab`，groups `TL-DIRECT-RECIPE-v7`、`TL-DIRECT-SCALE-v7`；
`TL-DIRECT-STRUCTURE-v7`没有实际run。入口/SSH方式在本机`.local/LENS.md`，未改服务访问策略。
图中记录本轮对A的条件启示，B预训练路线与旧TIDE证据保持各自身份。

| 下一批可选工作 | 依据 | 建议有界申请（尚未提交） |
| --- | --- | --- |
| 直接HB40/88更多曝光 | 4M两seed容量收益一致，远未遍历20.97M训练池 | 同尺寸、同两seed、各到20.97M；先导校准完整LR日程，≤4 NPU卡时/8h wall |
| 可靠Transformer对照与上下文检查 | TF28容量倒挂且落后固定二元参照，当前位置缩放未解决 | 2–3个具名LR/残差初始化先导及位置保持的上下文干预；≤2 NPU卡时/8h wall，筛选后再定正式对照 |
| TIDE桥接与真实稀疏 | 直接模型缺TIDE的关键执行/分配职责 | 上游可用后先HB40/88小接入和共同曝光稠密桥接，再有限稀疏对照；先申请≤2 NPU卡时校准后再定长训 |

以上预算是供下一次决策的单项上限，不是已批准叠加作业，也不自动再扩大模型/词表/任务。
