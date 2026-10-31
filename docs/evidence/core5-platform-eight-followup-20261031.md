# 八张NPU共享资源后继的终态审阅（2026-10-31）

三项均完成执行、完整离散检查和精确续接。FC16/d8小训练严格通过；约1.014B推理及
新loss训练均保留独立FP32超差，只有探索成本结论。原八卡等待超时、原loss非有限、
任意独立根EE1017/SIGSEGV、B512下一窗失败和多卡resident门禁均未解除。

## 来源、资源与执行

launcher/配置固定`d3de18fb1ee5b849ed88d5de82be938972808833`，实际安装包来源4485697，
安装内容SHA256 `0312c34e47639eb4d7c6287f9b4263ed2d5cca128f31952b5e48b7f40eb8cb84`；
实际NPU native库构建源`eca070f73c760bf159bbe5c8e3b6c1bf1a9060c3`。未称较新Python提交
重新构建了旧库。全部逐文件、二进制、配置身份在[归档](core5-platform-eight-followup-20261031.json.gz)。
固定数学仍为`0770593838a37c790a568226930567ed60f99b83`。

物理4–11映射逻辑npu:0–7；物理9与用户明确允许的低占用其他任务共用。
准入绑定原PID/启动身份，HBM上限8192MiB、AICore上限5%、剩余至少24576MiB，
保留priority/FIFO、两次采样及八锁整组持有；未停止其他任务。队列等待6.936秒。
该资源条件不认证独占或统计稳定性能，也不是八卡安全容量上限。

Torch2.12.0+cpu/TorchNPU2.12.0.post2/CANN9.1；Python/native、mixed-c、ordinary显式复制，
payload/scoring均FP32、HF32关闭，ATen/interop/worker各1，prefill、packed/scalar完整返回。
每项warmup1窗、测量连续2窗；训练warmup也执行更新。profile未混入计时。
窗口计时含准备、调度、packing、前向、必要复制、完整返回与同步；训练另含反向及SGD。
构造、独立参考、checkpoint保存与精确恢复另列。大checkpoint仅写已准入的tmpfs，
小记录持久保存；因此本报告的恢复是实际文件序列化/新owner恢复，不证明重启后的耐久性。

## 同一模型与实际结果

大模型为FC16节点、8区域/不可拆共享组、KV/d2048/FFN4d、history8，1,014,118,392个
去重可训练参数（另60冻结元素）；推理与训练共享model_id/workload_id，task_id区分loss。
推理model/workload/task身份与原七卡材料逐项相同；八卡每卡2节点，未将共享参数复制为
独立变量。实际state_slots与逐节点state设备在全部warmup/测量窗按映射核对；详细参数
owner、消息目标、原生调用统计在归档。小模型同为FC16/KV，32,784个可训练参数。

| 配置 | B / 时间位置 / span | 窗口中位秒 | candidate / Full / 消息（每窗） | 独立数值 / 离散 / 精确续接 |
| --- | --- | ---: | --- | --- |
| 约1.014B推理 | 1 / 2 / 20 | 0.861879 | 31 / 16 / 28 | 超差 / 通过 / 通过 |
| d8小训练 | 2 / 3 / 30 | 1.861679 | 79 / 79 / 138 | 通过 / 通过 / 通过 |
| 约1.014B训练，loss系数1e-9 | 1 / 2 / 20 | 1.820309 | 31 / 16 / 28 | 超差 / 通过 / 通过 |

小训练与大训练采用新loss系数1e-9、SGD lr0.001/momentum0.3/weight_decay0.01。
大训练在小训练的8份独立记录/梯度/参数/槽比较及精确续接通过后才开始。
这不是原0.001配方的修复，也不是任意独立VJP根资格。
两个大配置首差均为window0/sample0/node0的KV keys：190/2048坐标超差，最大绝对差
4.43459e-5，原atol1e-6/rtol1e-5保持。恢复成功不能替代独立正确性。

原生跨卡计数：推理7组/229488逻辑bytes；小训练14组/9936bytes；大训练14组/458976bytes。
ordinary显式端点同步计数0，物理tensor-leg bytes与逻辑相同；这些计数不包含厂商隐藏
复制或全部Python边界返回，不能称无同步或认证直接P2P、HCCL及resident跨卡协议。
推理实际最大State样例批1、时间序列2、私有投影批2；增加卡数没有增加逻辑batch。

| 配置 | 逐卡全流程allocated峰值GB（逻辑0→7） | 最大reserved GB | RSS峰值GB | 最后checkpoint GB |
| --- | --- | ---: | ---: | ---: |
| 约1.014B推理 | 每卡约0.591053 | 0.641729 | 17.125310 | 4.607425 |
| d8小训练 | 0.034077 / 0.034165 / 0.034151 / 0.034180 / 0.034151 / 0.034151 / 0.034151 / 0.034195 | 0.037749 | 2.719527 | 0.000645 |
| 约1.014B训练 | 2.062520 / 2.062184 / 1.355787 / 1.355787 / 1.355787 / 2.062800 / 1.355788 / 2.062817 | 2.157969 | 45.139886 | 7.721324 |

逐卡峰值取各阶段最大值，包含恢复；不含其他账户及驱动隐藏内存。不能只取最后阶段
瞬时量。大训练host仍有45.14GB峰值，不能按八卡把CPU快照预算除以8。
构造/恢复/独立检查秒分别为推理10.448/28.531/13.383，小训练0.556/4.385/2.972，
大训练10.523/57.966/68.627。总流程86.105/21.934/194.518秒，额外存储对照未运行。

## 终态与干净复现

`gef-npu-eight-d3de18f.service`于15:07:41Z外层exit0，三个内层监督与workload均exit0。
审阅时MainPID0、ControlGroup空，三个监督的启动身份核对无存活workload或后代。
没有把外层exit0提升为两项大模型严格通过。本机配额约312GiB，低于400GiB软限与
800GiB硬限；没有新增大型持久盘材料。原输入、失败见证及GPU dirty-dev保全包未删除。

从干净安装按[人类手册](../human-workflow.md)配置当前机器的site，然后：

```sh
tidegraph plan configs/capacity-training/npu-eight-followup.json --matrix --site site.json
tidegraph results docs/evidence/core5-platform-eight-followup-20261031.json.gz --format table
```

按门禁顺序复现整组，使用[矩阵文档](../platform-matrix.md#八卡共享资源的有界后继)中的
固定源码driver及当前有效的资源许可。不得复用已过期的共享policy或旧PID。
无须旧运行目录即可从受版本控制的配置生成输入；归档可直接查询完整配置/身份/结果。
原始结果hash保留于每一行，归档SHA256：
`2f6f001b6d8d55ac382ebb4c031a200d1c2403bb4d991d230843f92eb7264e75`。
