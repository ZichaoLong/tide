# 第一轮试运行报告：pilot-r1-20261119

生成时间：2026-11-19T12:19:39.850029+00:00。原始记录：`/mi/data2T/zlong/home/llm/tide-learning-lab/.local/pilot-r1/round`。

计划 40 个任务，报告覆盖 40 个；调度器状态 `completed_with_failures`，观察时长 1,914.4 秒，全部任务进程已终止：`True`。
执行状态：`{"completed": 39, "failed": 1}`。恢复状态：`{"not_attempted": 1, "within_tolerance": 39}`。

执行完成只表示该运行的执行结果。浮点误差、结构/整数、恢复数据与学习观察仍按各自记录判断。

## 资源与并发

| 后端 | 任务数 | 最大同时执行 worker | 累计训练 token | 累计 worker 秒 | 启动耗时中位数/最大值（秒） |
|---|---:|---:|---:|---:|---:|
| cpu | 8 | 4 | 2,914,304 | 2,095.5 | 3.3 / 3.4 |
| npu | 32 | 9 | 40,534,528 | 13,923.6 | 3.8 / 9.0 |

观察到的物理 NPU：`[0, 2, 4, 5, 7, 8, 10, 11, 14]`。跨后端合计最大同时执行 worker：13。
worker 区间包括训练、评估、保存和子进程恢复。设备等待、解释器启动及实际执行在逐任务 JSON 中分别保留。
本轮没有独占与并发的配对测量；能够同时执行不等于没有争用，也不证明同卡并发能力。

## 全部任务

吞吐单位为有效文本 token/s。恢复 `within_tolerance` 表示记录的恢复检查在既定容差内；逐位一致标志另存 JSON。unknown 表示未记录。

| 任务 | 后端/物理卡 | 参数 M | B×L | 训练步 | 训练吞吐 | Prefill | Decode | 执行 | 恢复 |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| npu-fc-k-independent | npu:11 | 2.861 | 8×64 | 3,057 | 3,740.9 | 5,803.1 | 73.8 | completed | within_tolerance |
| npu-fc-k-d128 | npu:7 | 10.383 | 8×64 | 3,010 | 3,685.0 | 5,942.9 | 77.7 | completed | within_tolerance |
| npu-fc-s-shared | npu:5 | 1.321 | 8×64 | 2,055 | 2,514.2 | 4,074.1 | 65.2 | completed | within_tolerance |
| npu-fc-k-incoming64 | npu:4 | 1.522 | 8×64 | 2,253 | 2,754.6 | 4,937.7 | 61.7 | completed | within_tolerance |
| npu-lattice-k | npu:8 | 2.520 | 8×64 | 1,102 | 1,347.2 | 2,454.4 | 29.4 | completed | within_tolerance |
| npu-lattice-s | npu:10 | 2.509 | 8×64 | 881 | 1,078.1 | 1,793.3 | 29.7 | completed | within_tolerance |
| npu-exchange-k | npu:14 | 2.122 | 8×64 | 1,418 | 1,736.6 | 3,496.3 | 36.8 | completed | within_tolerance |
| npu-exchange-k-independent | npu:2 | 5.193 | 8×64 | 1,509 | 1,844.7 | 3,473.4 | 40.0 | completed | within_tolerance |
| npu-exchange-x | npu:0 | 3.106 | 8×64 | 737 | 902.0 | 1,771.4 | 20.3 | completed | within_tolerance |
| npu-exchange-s | npu:11 | 3.098 | 8×64 | 627 | 766.5 | 1,340.2 | 19.5 | completed | within_tolerance |
| npu-exchange-x-layer-shared | npu:7 | 2.620 | 8×64 | 741 | 905.5 | 1,707.7 | 20.1 | completed | within_tolerance |
| npu-exchange-x-independent | npu:5 | 7.320 | 8×64 | 685 | 835.1 | 1,189.9 | 13.9 | completed | within_tolerance |
| npu-exchange-x-incoming64 | npu:10 | 3.731 | 8×64 | 613 | 749.6 | 1,624.0 | 18.1 | completed | within_tolerance |
| npu-exchange-x-incoming128 | npu:4 | 4.165 | 8×64 | 578 | 706.6 | 1,513.3 | 18.4 | completed | within_tolerance |
| npu-exchange-x-incoming-wide | npu:8 | 3.922 | 8×64 | 611 | 746.8 | 1,469.9 | 16.5 | completed | within_tolerance |
| npu-exchange-x-shannon | npu:2 | 3.106 | 8×64 | unknown | unknown | unknown | unknown | failed | not_attempted |
| npu-exchange-x-pool | npu:14 | 3.106 | 8×64 | 841 | 1,028.8 | 2,235.9 | 25.5 | completed | within_tolerance |
| npu-exchange-x-budget4 | npu:0 | 3.106 | 8×64 | 765 | 934.3 | 1,908.9 | 20.7 | completed | within_tolerance |
| npu-multiscale-k | npu:2 | 2.520 | 8×64 | 1,077 | 1,316.8 | 2,461.8 | 28.7 | completed | within_tolerance |
| npu-multiscale-x-compact9 | npu:11 | 5.820 | 8×64 | 341 | 414.6 | 778.8 | 9.4 | completed | within_tolerance |
| npu-timed-x-group | npu:7 | 3.785 | 8×64 | 326 | 398.4 | 786.1 | 14.6 | completed | within_tolerance |
| npu-timed-s | npu:5 | 2.509 | 8×64 | 493 | 602.2 | 929.1 | 28.0 | completed | within_tolerance |
| npu-timed-x-singleton | npu:10 | 5.500 | 8×64 | 193 | 235.3 | 466.7 | 9.7 | completed | within_tolerance |
| npu-timed-x-mixed | npu:4 | 3.710 | 8×64 | 377 | 460.4 | 945.5 | 20.4 | completed | within_tolerance |
| npu-exchange-x-reverse | npu:8 | 3.095 | 8×64 | 669 | 817.1 | 1,427.8 | 20.6 | completed | within_tolerance |
| npu-exchange-x-kv32 | npu:14 | 2.942 | 8×64 | 742 | 906.6 | 1,665.2 | 20.0 | completed | within_tolerance |
| npu-exchange-x-memory-shared | npu:0 | 6.366 | 8×64 | 693 | 845.4 | 1,568.0 | 18.6 | completed | within_tolerance |
| npu-exchange-x-d128 | npu:2 | 8.696 | 8×64 | 674 | 823.1 | 1,544.1 | 19.4 | completed | within_tolerance |
| npu-transformer-d128-l4 | npu:11 | 1.902 | 8×64 | 20,001 | 28,330.1 | 107,216.9 | 823.8 | completed | within_tolerance |
| npu-transformer-d256-l6 | npu:7 | 7.212 | 8×64 | 16,613 | 20,421.1 | 80,844.6 | 591.9 | completed | within_tolerance |
| npu-transformer-d384-l8 | npu:5 | 17.308 | 8×64 | 13,179 | 16,181.9 | 62,505.5 | 461.7 | completed | within_tolerance |
| npu-exchange-x-b16-l128 | npu:4 | 3.106 | 16×128 | 577 | 2,819.6 | 5,244.6 | 42.2 | completed | within_tolerance |
| cpu-fc-k-independent-b2-l32-t8 | cpu | 2.861 | 2×32 | 2,287 | 614.2 | 1,642.5 | 62.1 | completed | within_tolerance |
| cpu-fc-k-independent-b8-l64-t40 | cpu | 2.861 | 8×64 | 346 | 739.9 | 5,771.8 | 135.6 | completed | within_tolerance |
| cpu-exchange-x-b2-l32-t40 | cpu | 3.106 | 2×32 | 127 | 33.3 | 97.6 | 3.6 | completed | within_tolerance |
| cpu-exchange-x-b8-l64-t40 | cpu | 3.106 | 8×64 | 69 | 142.2 | 1,068.6 | 33.6 | completed | within_tolerance |
| cpu-transformer-d128-l4-t16 | cpu | 1.902 | 8×64 | 3,429 | 7,394.8 | 11,925.4 | 385.2 | completed | within_tolerance |
| cpu-transformer-d256-l6-t40 | cpu | 7.212 | 8×64 | 925 | 1,975.7 | 11,598.4 | 702.9 | completed | within_tolerance |
| cpu-timed-x-group-b2-l32 | cpu | 3.785 | 2×32 | 66 | 17.6 | 93.3 | 5.3 | completed | within_tolerance |
| cpu-fc-s-shared-t16 | cpu | 1.321 | 8×64 | 613 | 1,306.5 | 5,566.0 | 174.5 | completed | within_tolerance |

## 资源成本和学习探针

以下峰值为各任务记录范围内的峰值。主进程及恢复进程分开保存，不能简单相加为同时占用。

| 任务 | 初始化秒 | 排队/启动秒 | Worker 秒 | 训练峰值 GiB（NPU / RSS） | Checkpoint MiB / 保存秒 | 验证 NLL 前→后（变化） |
|---|---:|---:|---:|---:|---:|---|
| npu-fc-k-independent | 3.7 | 7.7 | 436.9 | 0.485 / 2.300 | 42.0 / 0.1 | 9.0230→6.0220 (-3.0010) |
| npu-fc-k-d128 | 8.6 | 14.3 | 443.5 | 0.916 / 2.296 | 155.3 / 0.3 | 9.0435→5.6487 (-3.3948) |
| npu-fc-s-shared | 8.6 | 21.1 | 443.7 | 0.485 / 2.286 | 19.7 / 0.1 | 9.0178→6.2746 (-2.7432) |
| npu-fc-k-incoming64 | 3.7 | 27.8 | 439.1 | 0.726 / 2.285 | 26.7 / 0.1 | 9.0161→6.0824 (-2.9337) |
| npu-lattice-k | 8.5 | 34.7 | 450.2 | 0.966 / 2.251 | 75.0 / 0.2 | 9.0226→6.5343 (-2.4883) |
| npu-lattice-s | 3.7 | 41.5 | 445.5 | 0.930 / 2.257 | 41.0 / 0.2 | 9.0190→6.7189 (-2.3001) |
| npu-exchange-k | 8.9 | 48.0 | 448.0 | 0.859 / 2.259 | 61.2 / 0.2 | 9.0266→6.4035 (-2.6231) |
| npu-exchange-k-independent | 8.6 | 54.9 | 447.7 | 0.774 / 2.229 | 78.0 / 0.3 | 9.0124→6.3821 (-2.6303) |
| npu-exchange-x | 9.0 | 62.0 | 456.8 | 1.171 / 2.223 | 58.1 / 0.2 | 9.0227→6.8265 (-2.1962) |
| npu-exchange-s | 3.8 | 455.8 | 452.5 | 1.117 / 2.231 | 48.5 / 0.2 | 9.0109→6.9071 (-2.1038) |
| npu-exchange-x-layer-shared | 3.9 | 479.1 | 451.2 | 1.159 / 2.229 | 52.5 / 0.2 | 9.0231→6.7922 (-2.2309) |
| npu-exchange-x-independent | 3.9 | 492.4 | 459.6 | 1.187 / 2.192 | 106.3 / 0.4 | 9.0435→6.8495 (-2.1941) |
| npu-exchange-x-incoming64 | 3.9 | 499.1 | 452.9 | 1.526 / 2.223 | 65.3 / 0.3 | 9.0212→6.8597 (-2.1615) |
| npu-exchange-x-incoming128 | 3.8 | 505.8 | 454.5 | 1.784 / 2.222 | 88.6 / 0.4 | 9.0293→6.8688 (-2.1605) |
| npu-exchange-x-incoming-wide | 3.8 | 512.6 | 455.3 | 1.621 / 2.222 | 85.8 / 0.3 | 9.0210→6.8629 (-2.1580) |
| npu-exchange-x-shannon | 3.8 | 519.1 | 5.4 | unknown / unknown | unknown / unknown | unknown→unknown (unknown) |
| npu-exchange-x-pool | 3.7 | 77.6 | 448.0 | 1.214 / 2.171 | 58.1 / 0.2 | 9.0225→6.6638 (-2.3588) |
| npu-exchange-x-budget4 | 3.7 | 78.2 | 450.8 | 1.205 / 2.226 | 58.1 / 0.2 | 9.0271→6.7523 (-2.2748) |
| npu-multiscale-k | 3.8 | 79.9 | 445.4 | 1.013 / 2.231 | 52.0 / 0.2 | 9.0216→6.5702 (-2.4514) |
| npu-multiscale-x-compact9 | 3.8 | 453.5 | 474.1 | 2.631 / 2.199 | 115.1 / 0.4 | 9.0304→7.1317 (-1.8987) |
| npu-timed-x-group | 3.7 | 433.6 | 464.4 | 6.155 / 2.194 | 230.5 / 0.6 | 9.0319→7.1205 (-1.9113) |
| npu-timed-s | 3.7 | 457.4 | 449.5 | 2.953 / 2.215 | 35.4 / 0.2 | 9.0127→6.9692 (-2.0435) |
| npu-timed-x-singleton | 3.8 | 450.8 | 481.1 | 5.854 / 2.215 | 157.3 / 0.4 | 9.0271→7.3714 (-1.6557) |
| npu-timed-x-mixed | 3.8 | 457.8 | 456.0 | 6.160 / 2.191 | 137.7 / 0.4 | 9.0247→7.0804 (-1.9443) |
| npu-exchange-x-reverse | 3.7 | 464.3 | 451.9 | 1.129 / 2.234 | 52.6 / 0.2 | 9.0200→6.8612 (-2.1589) |
| npu-exchange-x-kv32 | 3.8 | 468.4 | 451.8 | 1.070 / 2.225 | 58.6 / 0.2 | 9.0228→6.8094 (-2.2134) |
| npu-exchange-x-memory-shared | 3.8 | 91.2 | 453.3 | 1.182 / 2.212 | 95.4 / 0.3 | 9.0264→6.8425 (-2.1838) |
| npu-exchange-x-d128 | 3.8 | 76.6 | 454.7 | 2.117 / 2.210 | 179.3 / 0.7 | 9.0104→6.5216 (-2.4889) |
| npu-transformer-d128-l4 | 3.7 | 465.0 | 377.3 | 0.148 / 2.156 | 23.8 / 0.1 | 9.0427→4.5370 (-4.5057) |
| npu-transformer-d256-l6 | 3.8 | 470.4 | 434.3 | 0.295 / 2.160 | 88.6 / 0.3 | 9.0710→4.3923 (-4.6787) |
| npu-transformer-d384-l8 | 4.0 | 470.1 | 435.7 | 0.525 / 2.209 | 222.0 / 0.5 | 9.0997→4.5030 (-4.5968) |
| npu-exchange-x-b16-l128 | 3.8 | 446.6 | 452.5 | 4.237 / 2.259 | 117.2 / 0.4 | 9.0258→6.5954 (-2.4304) |
| cpu-fc-k-independent-b2-l32-t8 | 3.2 | 0.1 | 249.9 | unknown / 1.305 | 36.3 / 0.1 | 8.9966→6.5205 (-2.4761) |
| cpu-fc-k-independent-b8-l64-t40 | 3.3 | 0.1 | 256.0 | unknown / 1.878 | 51.2 / 0.3 | 9.0230→7.0852 (-1.9378) |
| cpu-exchange-x-b2-l32-t40 | 3.2 | 0.1 | 277.9 | unknown / 1.412 | 43.7 / 0.3 | 9.0334→7.6662 (-1.3672) |
| cpu-exchange-x-b8-l64-t40 | 3.2 | 0.1 | 275.4 | unknown / 2.536 | 58.1 / 0.3 | 9.0227→8.2448 (-0.7779) |
| cpu-transformer-d128-l4-t16 | 3.3 | 0.1 | 250.3 | unknown / 1.552 | 23.8 / 0.1 | 9.0427→5.4744 (-3.5684) |
| cpu-transformer-d256-l6-t40 | 3.4 | 0.1 | 253.4 | unknown / 1.690 | 88.6 / 0.2 | 9.0710→5.9314 (-3.1396) |
| cpu-timed-x-group-b2-l32 | 3.4 | 0.1 | 281.2 | unknown / 2.137 | 67.4 / 0.2 | 9.0308→8.2894 (-0.7413) |
| cpu-fc-s-shared-t16 | 3.2 | 0.1 | 251.5 | unknown / 1.813 | 19.7 / 0.1 | 9.0178→6.8939 (-2.1239) |

验证仅使用小型固定文章探针，训练 token 曝光与完成更新数也不同；表中损失变化不用于架构质量排名。

## 机制覆盖观察

以下仅列值得后续审阅的观察，不将其自动归为 bug、不可训练或执行失败。None 梯度只反映最后一个训练更新；未变化参数不能证明不可训练，参数变化也可能包含 AdamW decay。活动计数来自 B=1 冷状态探针，不认证训练全程或完整离散轨迹。

参数变化记录只有总数，没有逐参数未变名单；末步 None 梯度名单不能当作整轮未变参数清单。

| 任务 | 发生变化/总参数张量 | 最后一步 None 梯度数 | 选定输出空位（初始 / 最终） |
|---|---:|---:|---|
| npu-timed-x-singleton | 152 / 158 | 6 | 0 of 64 / 0 of 64 |

`npu-timed-x-singleton` 的最后一步 None 梯度：`core.memories.S.summary.weight`, `core.memories.S.summary.left`, `core.memories.S.summary.right`, `core.memories.S.read.weight`, `core.memories.S.read.left`, `core.memories.S.read.right`。

## 失败与未完成检查

失败类别计数（可重叠）：`{"execution": 1}`。
阶段覆盖：`{"checkpoint": {"recorded": 39, "unknown": 1}, "inference": {"recorded": 39, "unknown": 1}, "parameter_updates": {"recorded": 39, "unknown": 1}, "training": {"recorded": 39, "unknown": 1}}`。

| 任务 | 类别 / 字段 | 原始信息摘录 |
|---|---|---|
| npu-exchange-x-shannon | execution / initial_validation_before_training | allocation multiplier did not meet the total constraint |

## 解释范围

- 本轮没有独占运行与并发运行的配对比较；重叠区间证明同时执行的事实，不能证明并发无吞吐损失或同卡并发能力。
- 吞吐来自各任务实际配置与并发环境；不同参数量、batch、序列长度和算子路径不可据此直接排名架构效率。
- 验证数据是 WikiText 上游训练文章的固定试运行留出；每次仅有少量样本，验证损失变化不构成模型质量排名。
- 执行完成、浮点容差、结构/整数检查、checkpoint 数据、输入流与 RNG 恢复分别报告；缺失阶段保持 unknown。
- 已保存的恢复比较仅覆盖记录的 logits、梯度、参数、优化器及私有状态；不外推全部组合、尺度或后端。
- 因果探针仅在初始化、空私有状态下使用 1 个 16-token 样本，改动后 8 个 token 并比较前 8 个 logits；不认证训练后或其他状态、长度及 batch 的因果性。
- 恢复梯度在全局范数裁剪后捕获；其一致性不等同于未裁剪梯度一致性。
- 参数变化与最后一步非 None 梯度不能证明每个模块有效学习；参数变化可能包含 AdamW weight decay，零梯度与未连通梯度也需分别解释。
- 活动量来自 B=1、空初始状态探针的聚合计数；它不认证完整离散轨迹或训练全程活动量。
- 当前 resume-verify 仅检查新进程的一次实际优化器更新及私有状态延续，尚非长程断点任务队列功能。
- 训练前失败的任务没有可测训练吞吐，也没有恢复资格；缺测保留 unknown，不作为 0 吞吐参与统计。
- 训练吞吐按 worker 的测量边界统计；推理为驻留模型、包含任务头的真实文本 teacher-forced 继续，排除分词与主机传输。

本报告的 JSON 保留每个任务原始配置、执行与恢复记录、各字段比较、输入身份及缺测信息，供下一轮排程与诊断使用。
